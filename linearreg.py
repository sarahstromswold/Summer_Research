import numpy as np
import matplotlib.pyplot as plt
import math
# logistic regression


def make_pred(x, z): # pred pos or neg
    return np.sign((z[0] * x[0]) + (z[1] * x[1]))


def loss(x, y, z):
    countright = 0
    countwrong = 0
    numrows, numcols = x.shape
    for i in range(numrows):
        mypred = make_pred(x[i, :], z)
        if mypred == y[i]:
            countright = countright + 1
        else:
            countwrong = countwrong + 1
    numlos = countwrong / (countwrong + countright)
    return numlos


def surrloss(x, y, z):
    total = 0
    numrows, numcols = x.shape
    for i in range(numrows):
        surrloss = math.log(1 + math.exp((-1 * y[i]) * np.dot(z, x[i])))
        total = surrloss + total
    return total


def gradloss(x, y, z):
    s1 = 0
    s2 = 0
    numrows, numcols = x.shape
    for i in range(numrows):
        s1 += (1 / (1 + math.exp(-1 * y[i] * (np.dot(z, x[i]))))) * math.exp(-1 * y[i] * np.dot(z, x[i])) * (-1 * y[i] * x[i, 0])
        print(np.dot(z, x[i]))
        print(-y[i])
        print(z)
        print(math.exp(-y[i] * (np.dot(z, x[i]))))
        s2 += (1 / (1 + math.exp(-1 * y[i] * (np.dot(z, x[i]))))) * math.exp(-1 * y[i] * np.dot(z, x[i])) * (-1 * y[i] * x[i, 1])
        return np.array([s1, s2])


def graddesc(z):
    for i in range(500):
        z = z - (0.5 * gradloss(xarr, signarr, z))
        print("Loss", loss(xarr, signarr, z))
        print("Surrloss", surrloss(xarr, signarr, z))
        print("Model is currently ", z)
        ax.scatter(z[0], z[1], color='#0000FF')
    return z


if __name__ == "__main__":
    xarr = np.array([[2, 5], [3, 8], [6, 10], [12, 8], [10, 14], [-1, -5], [-3, -10], [-10, -4], [-13, -12], [-7, -9]])
    signarr = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    # xarr = np.array([[15, 15], [-15, -15]])
    # signarr = ([-1, 1])
    model = np.array([-5.0, 3.0])  # theta1, theta2 number of features

    fig, ax = plt.subplots()
    for i in range(len(signarr)):
        if signarr[i] == 1:
            ax.scatter(xarr[i, 0], xarr[i, 1], color='hotpink')
        else:
            ax.scatter(xarr[i, 0], xarr[i, 1], color='#88c999')

    plt.axis([-15, 15, -15, 15])
    print("pred", make_pred([.5, 3.5], model))
    print("Loss", loss(xarr, signarr, model))
    print("Surrloss", surrloss(xarr, signarr, model))
    model = graddesc(model)
    print("pred", make_pred([.5, 3.5], model))
    print("Loss", loss(xarr, signarr, model))
    print("Surrloss", surrloss(xarr, signarr, model))
    plt.grid(True)
    plt.show()

