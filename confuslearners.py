# testing different learners, (svm, nn, trees, mlp) with confusion matrix

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    X_train, X_test, y_train, y_test = train_test_split(data, target, shuffle=False)  # test size inc, acc dec
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier())])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test) # the soft predictions, find out what it returns
    # print(y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap='YlGnBu')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    # plt.show()
    plt.savefig('output.png')
