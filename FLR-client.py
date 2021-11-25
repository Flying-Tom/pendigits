import ctypes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

libc = ctypes.CDLL("./libsecureLR_test.so")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_data():
    data = pd.read_csv('data.csv', header=None)
    features = np.array(data.iloc[:, 0:-1], dtype=np.float64)
    labels = np.array(data.iloc[:, -1], dtype=np.float64)
    # features = np.insert(features, 0, 1, axis=1)
    return features, labels


def loss_funtion(features, labels, weights):
    m, n = np.shape(features)
    loss = 0.0
    for i in range(m):
        sum_theta_x = 0.0
        for j in range(n):
            sum_theta_x += features[i, j] * weights.T[0, j]
        propability = sigmoid(sum_theta_x)
        loss += -labels[i, 0] * np.log(propability) - (1 - labels[i, 0]) * np.log(1 - propability)
    return loss


def ABY_grad_compute(x_theta, Y, x, alpha):
    x_theta, Y = x_theta.T, Y.T
    x_theta, Y = x_theta.tolist()[0], Y.tolist()[0]
    len = x_theta.__len__()
    x_theta = (ctypes.c_double * len)(*x_theta)
    Y = (ctypes.c_double * len)(*Y)
    res = (ctypes.c_double * len)(*[])
    libc.client_request(len, x_theta, Y, res)
    for i in range(len):
        res[i] = res[i] / len
    res = np.array(res)

    return res * x


def grad_descent(features, labels):
    features = np.mat(features)  # (m,n)
    labels = np.mat(labels).T

    weights = np.ones((np.shape(features)[1], 1))
    alpha = 1e-3
    maxstep = 1000
    eps = 0.0001
    count = 0
    loss_array = []

    for i in range(maxstep):
        print(f"Client Epoch: {i}")
        loss = loss_funtion(features, labels, weights)

        grad = ABY_grad_compute(features * weights, labels, features, alpha)

        # print(features.shape)
        # print(grad)
        # assert(False)
        new_weights = weights - alpha * grad

        new_loss = loss_funtion(features, labels, new_weights)
        loss_array.append(new_loss)
        if abs(new_loss - loss) < eps:
            break
        else:
            weights = new_weights
            count += 1

    print("count is: ", count)
    print("loss is: ", loss)
    print("weights is: ", weights)

    return weights, loss_array


def plotloss(loss_array):
    n = len(loss_array)
    plt.xlabel("iteration num")
    plt.ylabel("loss")
    plt.scatter(range(1, n+1), loss_array)
    plt.show()


# %%
data, labels = parse_data()
r, loss_array = grad_descent(data, labels)
r = np.mat(r).transpose()
plotloss(loss_array)
