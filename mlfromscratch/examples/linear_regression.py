
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from mlfromscratch.supervised_learning.regression import LinearRegression
from mlfromscratch.utils.data_manipulation import train_test_split
from mlfromscratch.utils.data_operation import mean_squared_error

def main():

    X, y = make_regression(n_samples=100, n_features=1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) # test data: 40%
    n_samples, n_features = np.shape(X)

    # train
    n_iterations = 100
    model = LinearRegression(n_iterations, learning_rate=0.001, gradient_descent=True)
    model.fit(X_train, y_train)

    # Training error plot
    # NOTE: training_errors are calculated only if gradient_descent=True
    training, = plt.plot(range(n_iterations), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Iterations")
    plt.show()

    y_predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, y_predicted)
    print("Mean Squared Error: %s" % (mse))

    y_predicted_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10) # train data
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10) # test data
    plt.plot(366 * X, y_predicted_line, color='black', linewidth=2, label='Prediction') # predicted line
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

