# trying different preprocessors with pics

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)  # gets the data
mlp = MLPClassifier()  # default inputs for the mlp
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what
# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set


if __name__ == "__main__":
    data = X
    target = y
    X_train, X_test, y_train, y_test = train_test_split(data, target, shuffle=False)  # test size inc, acc dec
    ax.imshow(X[0].reshape(8, 8), cmap='gray')
    preprocessor = StandardScaler()
    pipe = Pipeline([('scaler', preprocessor), ('mlp', MLPClassifier())])
    pipe.fit(X_train, y_train)
    preprocessor.transform(X_train)
    preprocessed_X_train = preprocessor.transform(X_train)
    ax2.imshow(preprocessed_X_train[0].reshape(8, 8), cmap='gray')
    plt.axis([0, 1, 0, 1])
    plt.show()
    plt.savefig('output.png')
