# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X = np.arange(-10, 10, 20 / 500)
    X = np.delete(X, [100, 250, 425])
    X_bar = np.reshape(np.array([-6.0, 0.0, 7.0]), (3, 1))
    K_X_X = np.full((497, 497), .2)
    K_X_bar_X = np.full((3, 497), .2)
    K_X_bar_X_bar = np.full((3, 3), .2)
    Y_bar = np.reshape(np.array([3, -2, 2]), (3, 1))
    mu_n = np.full((497, 1), 0.0)
    mu_m = np.full((3, 1), 0.0)

    for i in range(0, 497):
        for j in range(0, 497):
            K_X_X[i, j] += math.exp(-2 * (math.sin(math.pi * abs(X[i] - X[j]) / 4)) ** 2 / (1))

    for i in range(0, 3):
        for j in range(0, 497):
            K_X_bar_X[i, j] += math.exp(-2 * (math.sin(math.pi * abs(X_bar[i] - X[j]) / 4)) ** 2 / (1))

    for i in range(0, 3):
        for j in range(0, 3):
            K_X_bar_X_bar[i, j] += math.exp(-2 * (math.sin(math.pi * abs(X_bar[i] - X_bar[j]) / 4)) ** 2 / (1))

    dog = K_X_X

    b = mu_n + np.matmul(np.matmul(np.transpose(K_X_bar_X), np.linalg.pinv(K_X_bar_X_bar)), (Y_bar - mu_m))
    b = np.reshape(b, 497)
    A = K_X_X - np.matmul(np.matmul(np.transpose(K_X_bar_X), np.linalg.pinv(K_X_bar_X_bar)), K_X_bar_X)

    Y = np.full((4, 497), 0.0)
    Y[0, :] = np.random.multivariate_normal(b, A, 1)
    Y[1, :] = np.random.multivariate_normal(b, A, 1)
    Y[2, :] = np.random.multivariate_normal(b, A, 1)
    Y[3, :] = np.random.multivariate_normal(b, A, 1)
    plt.figure()
    plt.scatter(X, Y[0], color='red', linewidths=.10, linestyle=":")
    plt.scatter(X, Y[1], color='yellow', linewidths=.001)
    plt.scatter(X, Y[2], color='green', linewidths=.001)
    plt.scatter(X, Y[3], color='purple', linewidths=.005)
    #plt.plot([-6, 0, 7], [3, -2, 2])

    plt.plot(-6, 3, 'ro')
    plt.text(-6, 3.5, "(-6,3)")
    plt.plot(0, -2, 'ro')
    plt.text(0, -2.5 , "(0,-2)")
    plt.plot(7, 2, 'ro')
    plt.text(7, 2.5, "(7,2)")
    #plt.annotate((0, -2), marker='.')
    #plt.annotate((7, 2), marker='.')
    plt.show()

    mu_n = np.full((497, 1), 0.0)
    mu_m = np.full((3, 1), 0.0)
    X = np.arange(-10, 10, 20 / 500)
    X = np.delete(X, [100, 250, 425])
    X_bar = np.reshape(np.array([-6.0, 0.0, 7.0]), (3, 1))
    K_X_X = np.full((497, 497), 0.0)
    K_X_bar_X = np.full((3, 497), 0.0)
    K_X_bar_X_bar = np.full((3, 3), 0.0)
    h = 5
    Y = np.full((4, 497), 0.0)
    Y_bar = np.reshape(np.array([3, -2, 2]), (3, 1))
    for i in range(0, 497):
        for j in range(0, 497):
            K_X_X[i, j] = math.exp(-((X[i] - X[j]) * (X[i] - X[j])) / h)

    for i in range(0, 3):
        for j in range(0, 497):
            K_X_bar_X[i, j] = math.exp(-((X_bar[i] - X[j]) * (X_bar[i] - X[j])) / h)

    for i in range(0, 3):
        for j in range(0, 3):
            K_X_bar_X_bar[i, j] = math.exp(-((X_bar[i] - X_bar[j]) * (X_bar[i] - X_bar[j])) / h)

    abc = K_X_X

    b = mu_n + np.matmul(np.matmul(np.transpose(K_X_bar_X), np.linalg.inv(K_X_bar_X_bar)), (Y_bar - mu_m))
    b = np.reshape(b, 497)
    A = K_X_X - np.matmul(np.matmul(np.transpose(K_X_bar_X), np.linalg.inv(K_X_bar_X_bar)), K_X_bar_X)

    Y[i, :] = b
    plt.scatter(X, Y[i])
    plt.scatter([-6, 0, 7], [3, -2, 2])
    plt.annotate("Point 1", (-6, 3))
    plt.annotate("Point 2", (0, -2))
    plt.annotate("Point 3", (7, 2))
    plt.show()

    X = np.arange(-10, 10, 20 / 500)
    X = np.delete(X, [100, 250, 425])
    X_bar = np.reshape(np.array([-6.0, 0.0, 7.0]), (3, 1))
    K_X_X = np.full((497, 497), 0.0)
    K_X_bar_X = np.full((3, 497), 0.0)
    K_X_bar_X_bar = np.full((3, 3), 0.0)
    Y_bar = np.reshape(np.array([3, -2, 2]), (3, 1))

    for i in range(0, 497):
        for j in range(0, 497):
            K_X_X[i, j] += math.exp(-2 * (math.sin(math.pi * abs(X[i] - X[j]) / 4)) ** 2 / (2))

    for i in range(0, 3):
        for j in range(0, 497):
            K_X_bar_X[i, j] += math.exp(-2 * (math.sin(math.pi * abs(X_bar[i] - X[j]) / 4)) ** 2 / (2))

    for i in range(0, 3):
        for j in range(0, 3):
            K_X_bar_X_bar[i, j] += math.exp(-2 * (math.sin(math.pi * abs(X_bar[i] - X_bar[j]) / 4)) ** 2 / (2))

    b = mu_n + np.matmul(np.matmul(np.transpose(K_X_bar_X), np.linalg.pinv(K_X_bar_X_bar)), (Y_bar - mu_m))
    b = np.reshape(b, 497)
    A = K_X_X - np.matmul(np.matmul(np.transpose(K_X_bar_X), np.linalg.pinv(K_X_bar_X_bar)), K_X_bar_X)

    Y[i, :] = b
    plt.scatter(X, Y[i])
    plt.scatter([-6, 0, 7], [3, -2, 2])
    plt.annotate("Point 1", (-6, 3))
    plt.annotate("Point 2", (0, -2))
    plt.annotate("Point 3", (7, 2))
    plt.show()


