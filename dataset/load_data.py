import numpy as np
from tensorflow.keras.datasets import cifar10

def load_cifar10(n_train=45000, n_test=5000):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Converts image data from integers (0-255) to floats (0.0-1.0) by dividing by 255
    # This is called "normalization" and helps neural networks train better
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshapes the label arrays from 2D (N,1) to 1D (N,) format -1 means "infer this dimension automatically"
    
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    x_train = x_train[:n_train]
    y_train = y_train[:n_train]
    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    np.savez_compressed("C:/Projects/Object Recognition/data/dataset.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    load_cifar10()
    print("Dataset saved in `data/dataset.npz`")
