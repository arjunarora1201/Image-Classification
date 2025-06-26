import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

file_paths = {
    "hog": "C:/Projects/Object Recognition/data/optimized_hog_features.npz",
    "color": "C:/Projects/Object Recognition/data/optimized_color_features.npz",
    "texture": "C:/Projects/Object Recognition/data/optimized_texture_features.npz",
    "gabor": "C:/Projects/Object Recognition/data/optimized_gabor_features.npz"
}

for feature_type, path in file_paths.items():
    if not os.path.exists(path):
        print(f"Warning: {feature_type} feature file not found at {path}")
    else:
        print(f"Found {feature_type} features at {path}")

print("Loading feature datasets...")

hog_data = np.load(file_paths["hog"])
X_train_hog, X_test_hog = hog_data["X_train_hog"], hog_data["X_test_hog"]

color_data = np.load(file_paths["color"])
X_train_color, X_test_color = color_data["X_train_color"], color_data["X_test_color"]

texture_data = np.load(file_paths["texture"])
X_train_lbp, X_test_lbp = texture_data["X_train_lbp"], texture_data["X_test_lbp"]

gabor_data = np.load(file_paths["gabor"])
X_train_gabor, X_test_gabor = gabor_data["X_train_gabor"], gabor_data["X_test_gabor"]

if "y_train" in hog_data and "y_test" in hog_data:
    y_train, y_test = hog_data["y_train"], hog_data["y_test"]
    has_labels = True
else:
    has_labels = False
    print("Warning: No labels found in feature files")

print("All feature datasets loaded successfully!")

print(f"HOG features: {X_train_hog.shape[1]} dimensions")
print(f"Color features: {X_train_color.shape[1]} dimensions")
print(f"LBP features: {X_train_lbp.shape[1]} dimensions")
print(f"Gabor features: {X_train_gabor.shape[1]} dimensions")

print("Normalizing features...")

scaler_hog = StandardScaler()
X_train_hog_scaled = scaler_hog.fit_transform(X_train_hog)
X_test_hog_scaled = scaler_hog.transform(X_test_hog)

scaler_color = StandardScaler()
X_train_color_scaled = scaler_color.fit_transform(X_train_color)
X_test_color_scaled = scaler_color.transform(X_test_color)

scaler_lbp = StandardScaler()
X_train_lbp_scaled = scaler_lbp.fit_transform(X_train_lbp)
X_test_lbp_scaled = scaler_lbp.transform(X_test_lbp)

scaler_gabor = StandardScaler()
X_train_gabor_scaled = scaler_gabor.fit_transform(X_train_gabor)
X_test_gabor_scaled = scaler_gabor.transform(X_test_gabor)

X_train_combined = np.hstack((X_train_hog_scaled, X_train_color_scaled, X_train_lbp_scaled, X_train_gabor_scaled))
X_test_combined = np.hstack((X_test_hog_scaled, X_test_color_scaled, X_test_lbp_scaled, X_test_gabor_scaled))

print(f"Combined feature shape: {X_train_combined.shape}")

n_components = 200

print(f"Applying PCA with {n_components} components...")

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_combined)
X_test_pca = pca.transform(X_test_combined)

print(f"After PCA, feature shape: {X_train_pca.shape}")
print(f"Explained variance ratio sum: {sum(pca.explained_variance_ratio_):.4f}")

with open("C:/Projects/Object Recognition/models/pca_model(45k).pkl", "wb") as f:
    pickle.dump(pca, f)
print("PCA model saved for future use")

if has_labels:
    np.savez_compressed("C:/Projects/Object Recognition/data/optimized_feature_combination_pca.npz", X_train_pca=X_train_pca, X_test_pca=X_test_pca,y_train=y_train,y_test=y_test)
else:
    np.savez_compressed("C:/Projects/Object Recognition/data/optimized_feature_combination_pca.npz", X_train_pca=X_train_pca, X_test_pca=X_test_pca)

print(f"Combined all features, applied PCA and saved!")
