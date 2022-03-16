from locale import normalize
from sklearn import datasets
from sklearn.metrics import accuracy_score

from mlfromscratch.utils.data_manipulation import normalize, train_test_split
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.misc import Plot
from mlfromscratch.utils.kernels import *
from mlfromscratch.supervised_learning.support_vector_machine import SupportVectorMachine

def main():
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0]) # get rid of target = 0
    y = data.target[data.target != 0] # get rid of target = 0
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = SupportVectorMachine(kernel=polynomial_kernel, power=4, coef=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Support Vector Machine", accuracy=accuracy)