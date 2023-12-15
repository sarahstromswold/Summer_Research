# implement a multilayer perceptron on MNIST
# train on the first half and test on the second, experiment

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

digits = load_digits()  # gets the data

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what

mlp = MLPClassifier()  # default inputs for the mlp
data = digits['data']
target = digits['target']

mlp.fit(data, target)  # trains the data
predicted_y = mlp.predict(data)  # sets the predicted data from our training set

# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set
# print(len(data)) prints the length of the array (rows)


def loss():
    count_right = 0
    count_wrong = 0
    count_rows, count_cols = data.shape
    for i in range(count_rows):  # comparing the predicted to the actual
        if predicted_y[i] == target[i]:
            count_right = count_right + 1
        else:
            count_wrong = count_wrong + 1
    count_final = count_wrong / (count_wrong + count_right)  # retrieving 0, 1 loss
    return 'Loss:', count_final


print(loss())  # calls the loss function
