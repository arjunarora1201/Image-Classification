import numpy as np
from skimage.color import rgb2gray
from scipy import ndimage as ndi
import time
import joblib
from sklearn.preprocessing import StandardScaler
import os

def extract_gabor_features(images):
    features = []
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    sigmas = [1, 3]
    frequencies = [0.05, 0.25]
    
    total_images = len(images)
    
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"Processing image {i}/{total_images}...")
            
        gray = rgb2gray(img)
        gabor_features = []

        for theta in thetas:
            for sigma in sigmas:
                for frequency in frequencies:
                    x, y = np.meshgrid(np.arange(gray.shape[1]), np.arange(gray.shape[0]))
                    rotx = x * np.cos(theta) + y * np.sin(theta)
                    gabor_real = ndi.gaussian_filter(gray, sigma) * np.cos(2*np.pi*frequency*rotx)

                    mean_response = np.mean(gabor_real)
                    var_response = np.var(gabor_real)
                    energy = np.sum(gabor_real**2)

                    gabor_features.extend([mean_response, var_response, energy])

        features.append(gabor_features)
    return np.array(features)

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading dataset...")
    data = np.load("C:/Projects/Object Recognition/data/dataset.npz")
    x_train, x_test = data["x_train"], data["x_test"]
    
    print(f"Loaded dataset with {len(x_train)} training images and {len(x_test)} test images")
    print("Extracting Gabor features...")

    X_train_gabor = extract_gabor_features(x_train)
    X_test_gabor = extract_gabor_features(x_test)
    
    print("Scaling Gabor features using StandardScaler...")
    scaler = StandardScaler()
    X_train_lbp_scaled = scaler.fit_transform(X_train_gabor)
    X_test_lbp_scaled = scaler.transform(X_test_gabor)

    scalers_folder = "C:/Projects/Object Recognition/scalers"
    os.makedirs(scalers_folder, exist_ok=True)

    scaler_path = os.path.join(scalers_folder, "gabor_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    print(f"Training set: {X_train_gabor.shape} samples, {X_train_gabor.shape[1]} features per sample")
    print(f"Testing set: {X_test_gabor.shape} samples, {X_test_gabor.shape[1]} features per sample")

    np.savez_compressed("C:/Projects/Object Recognition/data/optimized_gabor_features.npz", X_train_gabor=X_train_gabor, X_test_gabor=X_test_gabor)
    
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)") 
    print(f"Gabor features extracted and saved!")
    