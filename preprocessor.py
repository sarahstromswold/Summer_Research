# trying different preprocessors testing graphs

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)  # gets the data
mlp = MLPClassifier()  # default inputs for the mlp
fig, ax = plt.subplots()

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what
# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set


if __name__ == "__main__":
    for i in range(50):
        data = X
        target = y
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=((i * 2) + 1) / 100)  # test size inc, acc dec
        pipe = Pipeline([('scaler', MinMaxScaler()), ('mlp', MLPClassifier())])

        pipe.fit(X_train, y_train)
        # predicted_y = mlp.predict(X_test)  # sets the predicted data from our training set

        # print(len(data))  # prints the length of the array (rows)
        ax.scatter(((i * 2) + 1) / 100, pipe.score(X_test, y_test))
        # print(pipe.score(X_test, y_test))

    plt.xlabel("% Tested")
    plt.ylabel("Score")
    plt.axis([0, 1, 0, 1])
    plt.savefig('output.png')



