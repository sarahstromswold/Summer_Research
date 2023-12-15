# print picture of the image

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_digits(return_X_y=True)  # gets the data
fig, ax = plt.subplots()
mlp = MLPClassifier()  # default inputs for the mlp

for i in range(len(X)):
    for j in range(0, len(X[0]), 3):
        X[i, j] = 16

# X[0,0] = 16

ax.imshow(X[0].reshape(8, 8), cmap='gray')
plt.show()
# plt.savefig('corrupted.png')
