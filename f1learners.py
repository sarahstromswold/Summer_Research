# testing different learners, (svm, nn, trees, mlp) with f1 score

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

X, y = load_digits(return_X_y=True)  # gets the data
mlp = MLPClassifier()  # default inputs for the mlp
fig, ax = plt.subplots()

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what
# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set


if __name__ == "__main__":
    data = X
    target = y
    X_train, X_test, y_train, y_test = train_test_split(data, target)  # test size inc, acc dec
    pipe = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier())])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test) # the soft predictions, find out what it returns
    # print(y_pred)
    # predicted_y = mlp.predict(X_test)  # sets the predicted data from our training set

    print(f1_score(y_test, y_pred, average=None))