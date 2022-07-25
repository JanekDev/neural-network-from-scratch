import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    # reshape so that each image is a flattened vector of 28x28 pixels
    images = images.reshape(images.shape[0], 1, 784)
    
    return images, labels
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def cross_entropy_prime(y, y_hat):
    return y_hat - y

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mse_prime(y, y_hat):
    return 2 * (y - y_hat) / y.size