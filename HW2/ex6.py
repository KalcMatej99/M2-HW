import numpy as np
from time import time
import random

def sgd(x_1, eps, n, der_f_stohastic, X, y):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(x_k - eps * der_f_stohastic(x_k, X, y))
    return x_k

def gd(x_1, eps, n, der_f, X, y):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(x_k - eps * der_f(x_k, X, y))
    return x_k



def newton(x_1, n, der_f, hess_f, time_max = None):
    x_k = np.array(x_1)
    start_time = time()
    for i in range(n):
        x_k = np.array(x_k - np.linalg.inv(hess_f(x_k)).dot(der_f(x_k)))
        if not time_max == None and time() - start_time > time_max:
            return x_k
    return x_k


def bfgs(x_1, n, der_f, X, y):
    number_of_elements = len(x_1)
    x_k = np.array(x_1)
    B_k = np.identity(number_of_elements) / 100000
    for i in range(n):
        x_k_new = np.array(x_k - B_k.dot(der_f(x_k, X, y)))
        sigma = np.nan_to_num(x_k_new - x_k)
        gamma = np.nan_to_num(der_f(x_k_new, X, y) - der_f(x_k, X, y))
        d = sigma.dot(gamma)
        if d == 0:
            return x_k
        B_k_1 = B_k.copy()
        B_k = B_k_1 - (
            np.matmul(np.outer(sigma, gamma), B_k_1)
            + np.matmul(B_k_1, np.outer(gamma, sigma))
        ) / (d)
        B_k += (
            (np.outer(sigma, sigma))
            / (np.dot(sigma, gamma))
            * (1 + (gamma.dot(B_k_1).dot(gamma)) / (np.dot(sigma, gamma)))
        )
        B_k = np.nan_to_num(B_k)
        x_k = x_k_new
    return x_k

def l_bfgs(x_1, n, der_f, X, y):
    number_of_elements = len(x_1)
    x_k = np.array(x_1)
    B_k = np.identity(number_of_elements) / 100000


    gamma = []
    omega = []

    for k in range(n):
        q = der_f(x_k, X, y)
        for i in range(k):
            p_i = np.linalg.inv(np.dot(omega[i], gamma[i]))
            alfa_i = np.dot(np.dot(p_i, omega[i]), q)
            q -= np.dot(alfa_i, gamma[i])

        if k > 0:
            B_k = np.dot(omega[k], gamma[k])/np.dot(gamma[k], gamma[k]) * np.identity(len(x_1))
        r = np.dot(B_k, q)

        for i in range(k):
            p_i = np.linalg.inv(np.dot(omega[i], gamma[i]))
            alfa_i = np.dot(np.dot(p_i, omega[i]), q)
            b = np.dot(np.dot(p_i, gamma[i]), r)
            r += np.dot(omega[i], alfa_i - b)

        x_k_new = x_k - r
        omega.append(np.nan_to_num(x_k_new - x_k))
        gamma.append(np.nan_to_num(der_f(x_k_new, X, y) - der_f(x_k, X, y)))
        x_k = x_k_new

    return x_k

def generateN(N):
    X = []
    y = []
    for i in range(1, N + 1):
        X.append([i, 1])
        y.append(i + np.random.normal(0.5, 0.1, 1)[0])
    X = np.array(X)
    y = np.array(y)
    return X, y

def loss(k, n, X):
    loss = 0
    for x in X:
        g = k * x[0] + n
        loss += (g - x[1]) ** 2
    return loss

def der_f(x_k, X, y):
    return 2 * np.dot((np.dot(np.transpose(x_k), np.transpose(X)) - y), X)

def der_f_stohastic(x_k, X, y):
    j = random.randint(0, len(X)-1)
    return 2 * np.dot((np.dot(np.transpose(x_k), np.transpose(X[j, :])) - y[j]), X[j, :])


print("L-BFGS")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 2]
    n = 100000
    x_k = l_bfgs(x_1, n, der_f, X, y)
    print(x_k)


print("BFGS")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 2]
    n = 100000
    x_k = bfgs(x_1, n, der_f, X, y)
    print(x_k)

print("SGD")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 2]
    eps = 0.01/(N ** 2)
    n = 100000
    x_k = sgd(x_1, eps, n, der_f_stohastic, X, y)
    print(x_k)

print("GD")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 2]
    eps = 0.01/(N ** 3)
    n = 1000
    x_k = gd(x_1, eps, n, der_f, X, y)
    print(x_k)