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

    eps = 1

    gamma = []
    omega = []

    m = 10

    for k in range(n):
        q = der_f(x_k, X, y)
        for i in range(k - 1, max(0, k - m - 1), -1):
            if np.dot(omega[i], gamma[i]) == 0:
                return x_k
            p_i = np.nan_to_num(1/np.dot(omega[i], gamma[i]))
            alfa_i = p_i * np.dot(omega[i], q)
            q -= alfa_i * gamma[i]

        if k > 0:
            B_k = np.dot(omega[k-1], gamma[k-1])/np.dot(gamma[k-1], gamma[k-1]) * np.identity(len(x_1))
        r = np.dot(B_k, q)

        for i in range(max(0, k - m), k):
            p_i = np.nan_to_num(1/np.dot(omega[i], gamma[i]))
            alfa_i = p_i * np.dot(omega[i], q)
            b = p_i * np.dot(gamma[i], r)
            r += omega[i] * (alfa_i - b)

        x_k_new = x_k - eps * r
        omega.append(np.nan_to_num(x_k_new - x_k))
        gamma.append(np.nan_to_num(der_f(x_k_new, X, y) - der_f(x_k, X, y)))
        if np.array_equal(x_k, x_k_new):
            return x_k_new
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
    return 2 * np.dot(np.transpose(np.dot(x_k, np.transpose(X)) - np.transpose(y)), X)
def hess_f(X, y):
    return 2 * np.matmul(np.transpose(X), X)

def der_f_stohastic(x_k, X, y):
    j = random.randint(0, len(X)-1)
    return 2 * np.dot((np.dot(np.transpose(x_k), np.transpose(X[j, :])) - y[j]), X[j, :])

def rmse(x_k, X, y):
    return np.sqrt(np.sum(np.square(np.dot(x_k, np.transpose(X)) - y))/len(X))

print("GD")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 1]

    eigenvals = np.linalg.eigvals(hess_f(X, y))
    alfa = np.min(eigenvals)
    beta = np.max(eigenvals)
    eps = 2/(alfa + beta)
    n = 10000
    x_k = gd(x_1, eps, n, der_f, X, y)
    print(N, np.round(x_k, 3), "RMSE", np.round(rmse(x_k, X, y), 3))

print("SGD")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 1]
    eps = 0.01/(N ** 2)
    n = 10000
    x_k = sgd(x_1, eps, n, der_f_stohastic, X, y)
    print(N, np.round(x_k, 3), "RMSE", np.round(rmse(x_k, X, y), 3))

print("BFGS")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 1]
    n = 1000
    x_k = bfgs(x_1, n, der_f, X, y)
    print(N, np.round(x_k, 3), "RMSE", np.round(rmse(x_k, X, y), 3))

print("L-BFGS")
for N in [50, 100, 1000, 10000, 100000, 1000000]:
    X, y = generateN(N)
    x_1 = [2, 1]
    n = 1000
    x_k = l_bfgs(x_1, n, der_f, X, y)
    print(N, np.round(x_k, 3), "RMSE", np.round(rmse(x_k, X, y), 3))




