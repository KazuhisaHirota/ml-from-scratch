from sklearn import datasets
from sklearn.metrics import accuracy_score

from mlfromscratch.utils.data_manipulation import train_test_split
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.misc import Plot
from mlfromscratch.supervised_learning.linear_discriminant_analysis import LDA

def main():

    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Three -> Two classes
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="LDA", accuracy=accuracy)