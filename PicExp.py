import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()  # gets the data
data = np.concatenate([X_train, X_test], axis=0)[:500]
print(data)
target = np.concatenate([Y_train, Y_test], axis=0)[:500]
print(target)
n, _, _ = data.shape
d = target.shape[0]
data = data.reshape(n, -1)
target = target.reshape(d, -1)
target = target.ravel()
print("X_train:", data.shape)
print("y_train:", target.shape)
fig, ax = plt.subplots()
mlp = MLPClassifier()  # default inputs for the mlp

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
        # data = digits['data']
        # target = digits['target']
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=((i * 2) + 1) / 100)  # test size inc, acc dec
        mlp.fit(X_train, y_train)  # trains the data
        predicted_y = mlp.predict(X_test)  # sets the predicted data from our training set
        ax.scatter(((i * 2) + 1) / 100, acc())

    plt.xlabel("% tested")
    plt.ylabel("Accuracy")
    plt.axis([0, 1, 0, 1])
    # plt.show()
    plt.savefig('output.png')
    print(loss())  # loss of the last one calculated
    print(acc())  # acc of the last one calculated
