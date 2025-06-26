import numpy as np
import cv2
import os
import joblib
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
import time

def extract_lbp_features(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_uint8 = (gray * 255).astype('uint8')
    lbp = local_binary_pattern(gray_uint8, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P+2, range=(0, P+2), density=True)
    return hist

def extract_batch_lbp_features(images, P=8, R=1):
    # P (default=8): Number of circularly symmetric neighbor points.
    # R (default=1): Radius of the circular pattern.
    features = []
    total_images = len(images)
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"Processing image {i}/{total_images}...")
        features.append(extract_lbp_features(img, P, R))
    return np.array(features)

if __name__ == "__main__":
    start_time = time.time()

    print("Loading dataset...")

    data = np.load("C:/Projects/Object Recognition/data/dataset.npz")
    x_train, x_test = data["x_train"], data["x_test"]

    print(f"Loaded dataset with {len(x_train)} training images and {len(x_test)} test images")

    P = 8
    R = 1

    print(f"Extracting LBP features with P={P}, R={R}...")

    X_train_lbp = extract_batch_lbp_features(x_train, P, R)
    X_test_lbp = extract_batch_lbp_features(x_test, P, R)

    print("Scaling Texture features using StandardScaler...")
    scaler = StandardScaler()
    X_train_lbp_scaled = scaler.fit_transform(X_train_lbp)
    X_test_lbp_scaled = scaler.transform(X_test_lbp)

    scalers_folder = "C:/Projects/Object Recognition/scalers"
    os.makedirs(scalers_folder, exist_ok=True)

    scaler_path = os.path.join(scalers_folder, "texture_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    print(f"Training set: {X_train_lbp.shape} samples, {X_train_lbp.shape} features per sample")
    print(f"Testing set: {X_test_lbp.shape} samples, {X_test_lbp.shape} features per sample")

    np.savez_compressed("C:/Projects/Object Recognition/data/optimized_texture_features.npz", X_train_lbp=X_train_lbp, X_test_lbp=X_test_lbp)

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"LBP features extracted and saved!")