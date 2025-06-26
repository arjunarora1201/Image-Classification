import numpy as np
import cv2
import time
import os
import joblib
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def extract_color_histogram(image, bins=32):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)

        feature_vector = np.hstack([hist_h, hist_s, hist_v]).flatten()
        
        return feature_vector
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return np.zeros(bins * 3)

def extract_color_features(images, bins=32):
    return np.array(Parallel(n_jobs=-1)(delayed(extract_color_histogram)(img, bins) for img in images))

if __name__ == "__main__":
    start_time = time.time()

    print("Loading dataset...")
    data = np.load("C:/Projects/Object Recognition/data/dataset.npz")
    x_train, x_test = data["x_train"], data["x_test"]
    
    print(f"Loaded dataset with {len(x_train)} training images and {len(x_test)} test images")   
    print("Extracting color histogram features...")

    X_train_color = extract_color_features(x_train, bins=32)
    X_test_color = extract_color_features(x_test, bins=32)

    print("Scaling features using StandardScaler...")
    
    scaler = StandardScaler()
    X_train_color_scaled = scaler.fit_transform(X_train_color)
    X_test_color_scaled = scaler.transform(X_test_color)

    scalers_folder = "C:/Projects/Object Recognition/scalers"
    os.makedirs(scalers_folder, exist_ok=True)
    scaler_path = os.path.join(scalers_folder, "color_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    print(f"Scaler saved at: {scaler_path}")

    print(f"Training set: {X_train_color.shape[0]} samples, {X_train_color.shape[1]} features per sample")
    print(f"Testing set: {X_test_color.shape[0]} samples, {X_test_color.shape[1]} features per sample")

    np.savez_compressed("C:/Projects/Object Recognition/data/optimized_color_features.npz",
                        X_train_color=X_train_color, 
                        X_test_color=X_test_color)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print("Color histogram features extracted and saved!")