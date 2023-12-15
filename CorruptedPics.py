# Corrupts images and uses a lot of them
# plots the relationship between tested and accuracy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what
# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set
# print(len(data)) prints the length of the array (rows)


X, y = load_digits(return_X_y=True)
fig, ax = plt.subplots()
mlp = MLPClassifier()
for i in range(len(X)):
    for j in range(0, len(X[0]), 2):
        X[i, j] = 16


def loss():
    count_right = 0
    count_wrong = 0
    count_rows, count_cols = X_test.shape
    for i in range(count_rows):  # comparing the predicted to the actual
        if predicted_y[i] == y_test[i]:
            count_right = count_right + 1
        else:
            count_wrong = count_wrong + 1
    count_final = count_wrong / (count_wrong + count_right)  # retrieving 0, 1 loss
    return 'Loss:', count_final


def acc():  # get accuracy of the test
    accuracy = mlp.score(X_test, y_test)
    return accuracy


if __name__ == "__main__":
    for i in range(50):
        data = X
        # X = X[:300]
        target = y
        # y = y[:300]
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=((i * 2) + 1) / 100)  # test size inc, acc dec

        mlp.fit(X_train, y_train)  # trains the data
        predicted_y = mlp.predict(X_test)  # sets the predicted data from our training set
        ax.scatter(((i * 2) + 1) / 100, acc())

    plt.xlabel("% tested")
    plt.ylabel("Accuracy")
    plt.axis([0, 1, 0, 1])
    plt.savefig('output.png')
    plt.show()
    print(loss())  # loss of the last one calculated
    print(acc())  # acc of the last one calculated
