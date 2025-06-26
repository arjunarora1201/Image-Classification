import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from itertools import product
import time
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from joblib import Parallel, delayed

def process_image(img, orientations, pixels_per_cell, cells_per_block):
    gray = rgb2gray(img)
    hog1 = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False)
    resized = np.array(gray[::2, ::2])
    downsampled_pixels = (max(1, pixels_per_cell[0]//2), max(1, pixels_per_cell[1]//2))
    hog2 = hog(resized, orientations=orientations, pixels_per_cell=downsampled_pixels, cells_per_block=cells_per_block, visualize=False)
    feature_vector = np.hstack([hog1, hog2])
    print(f"Multi-scale HOG: Extracted {len(feature_vector)} features for one image.")
    
    return feature_vector

def extract_multiscale_hog_with_params(images, orientations, pixels_per_cell, cells_per_block, batch_size=100):
    features = []
    total_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images))
        batch = images[start_idx:end_idx]
    
        print(f"Processing batch {i+1}/{total_batches}...")
        batch_features = Parallel(n_jobs=-1)(delayed(process_image)(img, orientations, pixels_per_cell, cells_per_block) for img in batch)

        features.extend(batch_features)
    return np.array(features)

def extract_single_scale_hog(images, orientations, pixels_per_cell, cells_per_block, batch_size=100):
    features = []
    total_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images))
        batch = images[start_idx:end_idx]
    
        batch_features = Parallel(n_jobs=-1)(delayed(lambda img: hog(rgb2gray(img), orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block))(img) for img in batch)

        features.extend(batch_features)
        if features:
            print(f"Single-scale HOG: Extracted {len(features[0])} features per image.")
    return np.array(features)

def evaluate_hog_params(x_train, y_train, x_val, y_val, hog_params):
    best_accuracy = 0
    best_params = None
    results = []
    param_combinations = list(product(
    hog_params['orientations'], 
    hog_params['pixels_per_cell'], 
    hog_params['cells_per_block']))

    print(f"Testing {len(param_combinations)} HOG parameter combinations...")

    for i, (orientations, pixels_per_cell, cells_per_block) in enumerate(param_combinations):
        start_time = time.time()
    
        print(f"\nCombination {i+1}/{len(param_combinations)}:")
        print(f"  - orientations: {orientations}")
        print(f"  - pixels_per_cell: {pixels_per_cell}")
        print(f"  - cells_per_block: {cells_per_block}")

        X_train_hog = extract_single_scale_hog(
            x_train, orientations, pixels_per_cell, cells_per_block
        )
        X_val_hog = extract_single_scale_hog(
            x_val, orientations, pixels_per_cell, cells_per_block
        )

        clf = SVC(kernel='linear', C=1)
        clf.fit(X_train_hog, y_train)

        y_pred = clf.predict(X_val_hog)
        accuracy = accuracy_score(y_val, y_pred)

        elapsed_time = time.time() - start_time

        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Time: {elapsed_time:.2f} seconds")

        results.append({
            'orientations': orientations,
            'pixels_per_cell': pixels_per_cell,
            'cells_per_block': cells_per_block,
            'accuracy': accuracy,
            'time': elapsed_time
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'orientations': orientations,
                'pixels_per_cell': pixels_per_cell,
                'cells_per_block': cells_per_block
            }

    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\nTop 3 HOG parameter combinations:")
    for i, result in enumerate(results[:min(3, len(results))]):
        print(f"{i+1}. orientations={result['orientations']}, "
            f"pixels_per_cell={result['pixels_per_cell']}, "
            f"cells_per_block={result['cells_per_block']}, "
            f"accuracy={result['accuracy']:.4f}, "
            f"time={result['time']:.2f}s")

    return best_params, results

if __name__ == "__main__":
    start_time = time.time()

    print("Loading dataset...")
    data = np.load("C:/Projects/Object Recognition/data/dataset.npz")
    x_train, x_test = data["x_train"], data["x_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    val_size = int(0.1 * len(x_train))
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"Training set: {len(x_train)} images")
    print(f"Validation set: {len(x_val)} images")
    print(f"Test set: {len(x_test)} images")

    hog_params = {
        'orientations': [8, 9],
        'pixels_per_cell': [(8, 8), (16, 16)],  
        'cells_per_block': [(2, 2)]  
    }

    print("Evaluating HOG parameters...")
    best_params, all_results = evaluate_hog_params(x_train, y_train, x_val, y_val, hog_params)

    print("\nBest HOG parameters:")
    print(f"  - orientations: {best_params['orientations']}")
    print(f"  - pixels_per_cell: {best_params['pixels_per_cell']}")
    print(f"  - cells_per_block: {best_params['cells_per_block']}")

    print("\nExtracting multi-scale HOG features with best parameters...")
    X_train_hog = extract_multiscale_hog_with_params(
        np.concatenate([x_train, x_val]), 
        best_params['orientations'],
        best_params['pixels_per_cell'],
        best_params['cells_per_block']
    )

    X_test_hog = extract_multiscale_hog_with_params(
        x_test,
        best_params['orientations'],
        best_params['pixels_per_cell'],
        best_params['cells_per_block']
    )

    print("Scaling HOG features using StandardScaler...")
    scaler = StandardScaler()
    X_train_hog_scaled = scaler.fit_transform(X_train_hog)
    X_test_hog_scaled = scaler.transform(X_test_hog)

    scalers_folder = "C:/Projects/Object Recognition/scalers"
    os.makedirs(scalers_folder, exist_ok=True)

    scaler_path = os.path.join(scalers_folder, "hog_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at: {scaler_path}")

    output_file = "C:/Projects/Object Recognition/data/optimized_hog_features.npz"
    np.savez_compressed(
        output_file, 
        X_train_hog=X_train_hog, 
        X_test_hog=X_test_hog,
        y_train=np.concatenate([y_train, y_val]),
        y_test=y_test,
        hog_params=np.array([
            best_params['orientations'],
            best_params['pixels_per_cell'][0],
            best_params['pixels_per_cell'][1],
            best_params['cells_per_block'][0],
            best_params['cells_per_block'][1]
        ])
    )

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Optimized HOG features extracted and saved to {output_file}!")
    print(f"Feature dimensions: {X_train_hog.shape}")

