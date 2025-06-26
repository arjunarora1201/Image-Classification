import numpy as np
import pickle
import time
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="C:/Projects/Object Recognition/results/confusion_matrix(45k)_3fit_BestParam.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names if class_names else "auto", yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    start_time = time.time()
    
    print("Loading features...")
    data = np.load("C:/Projects/Object Recognition/data/optimized_feature_combination_pca.npz")
    X_train = data["X_train_pca"]
    X_test = data["X_test_pca"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    param_grid = {
        'C': [2,3],
        'gamma': [0.001,0.01],
        'kernel': ['rbf','linear']
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\nTraining SVM classifier with grid search...")
    grid_search = GridSearchCV(
        svm.SVC(probability=True, class_weight="balanced"),
        param_grid,
        cv=cv,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    print("\nGrid search complete!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    y_pred = grid_search.predict(X_test)
    test_accuracy = (accuracy_score(y_test, y_pred))*100
    print(f"\nTest accuracy: {test_accuracy:.2f}%")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    plot_confusion_matrix(y_test, y_pred, class_names)

    print("Saving the best model...")
    with open("C:/Projects/Object Recognition/models/svm_best_model(45k)_3fit_BestParam.pkl", "wb") as f:
        pickle.dump(grid_search.best_estimator_, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print("\nModel successfully saved!")
