# plots the relationship between tested and accuracy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_classification

mc = make_classification(n_samples=1000, n_features=5)
# blood = fetch_openml(name='blood-transfusion-service-center', version=1, parser="auto") # gets the data
fig, ax = plt.subplots()
mlp = MLPClassifier(hidden_layer_sizes=10)  # default inputs for the mlp

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what
# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set
# print(len(data)) prints the length of the array (rows)


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
        data, target = mc
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=((i * 2) + 1) / 100)  # test size inc, acc dec

        mlp.fit(X_train, y_train)  # trains the data
        predicted_y = mlp.predict(X_test)  # sets the predicted data from our training set
        ax.scatter(((i * 2) + 1) / 100, acc())

    plt.xlabel("% tested")
    plt.ylabel("Accuracy")
    plt.axis([0, 1, 0, 1])
    plt.show()
    # plt.savefig('output.png')  # acc of the last one calculated