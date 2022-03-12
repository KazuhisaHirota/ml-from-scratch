from sklearn import datasets

from mlfromscratch.utils.data_manipulation import normalize, train_test_split
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.misc import Plot
from mlfromscratch.supervised_learning.logistic_regression import LogisticRegression

def main():
    # Load dataset
    data = datasets.load_iris()
    # get rid of null rows
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    # transform labels
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1) # test data: 33%

    classifier = LogisticRegression(gradient_descent=True)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Reduce dimension to 2 using PCA
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)