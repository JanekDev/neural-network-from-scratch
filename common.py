#implement common functions
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
def plot_iris_data(iris):
    #copy iris
    iris_ = iris.copy()
    iris_['color'] = iris_['Species'].map({'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'})
    #plot iris data on different plots
    plt.figure(figsize=(20,10))
    plt.subplot(2,3,1)
    plt.scatter(iris_['SepalLength'], iris_['SepalWidth'], c=iris_['color'])
    plt.title('Sepal Length vs Sepal Width')
    plt.subplot(2,3,2)
    plt.scatter(iris_['PetalLength'], iris_['PetalWidth'], c=iris_['color'])
    plt.title('Petal Length vs Petal Width')
    plt.subplot(2,3,3)
    plt.scatter(iris_['SepalLength'], iris_['PetalLength'], c=iris_['color'])
    plt.title('Sepal Length vs Petal Length')
    plt.subplot(2,3,4)
    plt.scatter(iris_['SepalWidth'], iris_['PetalWidth'], c=iris_['color'])
    plt.title('Sepal Width vs Petal Width')
    plt.subplot(2,3,5)
    plt.scatter(iris_['SepalLength'], iris_['PetalWidth'], c=iris_['color'])
    plt.title('Sepal Length vs Petal Width')
    plt.subplot(2,3,6)
    plt.scatter(iris_['SepalWidth'], iris_['PetalLength'], c=iris_['color'])
    plt.title('Sepal Width vs Petal Length')
    plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x));

def softmax_prime(x):
    return softmax(x)*(1-softmax(x));

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true*np.log(y_pred));

def cross_entropy_prime(y_true, y_pred):
    return -y_true/y_pred;