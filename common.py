#implement common functions
import numpy as np
import matplotlib.pyplot as plt
def plot_iris_data(iris):
    iris['color'] = iris['Species'].map({'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'})
    #plot iris data on different plots
    plt.figure(figsize=(20,10))
    plt.subplot(2,3,1)
    plt.scatter(iris['SepalLength'], iris['SepalWidth'], c=iris['color'])
    plt.title('Sepal Length vs Sepal Width')
    plt.subplot(2,3,2)
    plt.scatter(iris['PetalLength'], iris['PetalWidth'], c=iris['color'])
    plt.title('Petal Length vs Petal Width')
    plt.subplot(2,3,3)
    plt.scatter(iris['SepalLength'], iris['PetalLength'], c=iris['color'])
    plt.title('Sepal Length vs Petal Length')
    plt.subplot(2,3,4)
    plt.scatter(iris['SepalWidth'], iris['PetalWidth'], c=iris['color'])
    plt.title('Sepal Width vs Petal Width')
    plt.subplot(2,3,5)
    plt.scatter(iris['SepalLength'], iris['PetalWidth'], c=iris['color'])
    plt.title('Sepal Length vs Petal Width')
    plt.subplot(2,3,6)
    plt.scatter(iris['SepalWidth'], iris['PetalLength'], c=iris['color'])
    plt.title('Sepal Width vs Petal Length')
    plt.show()

        


    