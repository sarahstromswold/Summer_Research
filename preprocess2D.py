# trying different processors with 2D Data

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


mc = make_classification(n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, scale=1, random_state=123)
mlp = MLPClassifier()  # default inputs for the mlp
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

# print(help(digits)) says what this "bunch" (type) has
# print(digits.keys()) says what
# print(mlp.t_) number of training samples
# print(predicted_y) prints the predicted data from our training set


if __name__ == "__main__":
    data, target = mc
    X_train, X_test, y_train, y_test = train_test_split(data, target, shuffle=False)  # test size inc, acc dec
    ax.scatter(data[:, 0], data[:, 1], c=target)
    preprocessor = MinMaxScaler()
    pipe = Pipeline([('scaler', preprocessor), ('mlp', MLPClassifier())])
    pipe.fit(X_train, y_train)
    preprocessor.transform(X_train)
    preprocessed_X_train = preprocessor.transform(X_train)
    ax2.scatter(preprocessed_X_train[:, 0], preprocessed_X_train[:, 1], c=y_train)

    ax.set_xlabel("1st Feature")
    ax.set_ylabel("2nd Feature")
    ax2.set_xlabel("1st Feature")
    ax2.set_ylabel("2nd Feature")
    plt.axis([min(data[:, 0]), max(data[:, 0]), min(data[:, 1]), max(data[:, 1])])
    # plt.show()
    plt.savefig('output.png')

