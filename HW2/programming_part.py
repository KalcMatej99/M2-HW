import numpy as np


def gd(x_1, eps, n, der_f):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(x_k - eps * der_f(x_k))
    return x_k


def pgd(x_1, x_2, eps, mom, n, der_f):
    x_k_1 = np.array(x_1)
    x_k = np.array(x_2)
    for i in range(n):
        x_k_s = np.array(x_k - eps * der_f(x_k) + mom * (x_k - x_k_1))
        x_k_1 = x_k
        x_k = x_k_s
    return x_k


def ngd(x_1, x_2, eps, mom, n, der_f):
    x_k_1 = np.array(x_1)
    x_k = np.array(x_2)
    for i in range(n):
        x_k_s = np.array(
            x_k - eps * der_f(x_k + mom * (x_k - x_k_1)) + mom * (x_k - x_k_1)
        )
        x_k_1 = x_k
        x_k = x_k_s
    return x_k


def ada_gd(x_1, eps, n, der_f):
    x_k = np.array(x_1)
    x = [x_k]
    for i in range(n):
        d_i_k = [
            np.sum([der_f(x_j)[i] ** 2 for x_j in x]) ** -0.5 for i in range(len(x_1))
        ]
        D_k = np.nan_to_num(np.diag(d_i_k))
        x_k = np.array(x_k - eps * D_k.dot(der_f(x_k)))
        x.append(x_k)
    return x_k


def newton(x_1, n, der_f, hess_f):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(x_k - np.linalg.inv(hess_f(x_k)).dot(der_f(x_k)))
    return x_k


def bfgs(x_1, n, der_f):
    number_of_elements = len(x_1)
    x_k = np.array(x_1)
    B_k = np.identity(number_of_elements)
    for i in range(n):
        x_k_new = np.array(x_k - B_k.dot(der_f(x_k)))
        sigma = np.nan_to_num(x_k_new - x_k)
        gamma = np.nan_to_num(der_f(x_k_new) - der_f(x_k))
        print(sigma, gamma)
        B_k_1 = B_k.copy()
        d = sigma.dot(gamma)
        if d == 0:
            return x_k
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


def f1(X):
    x = X[0]
    y = X[1]
    z = X[2]

    return (x - z) ** 2 + (2 * y + z) ** 2 + (4 * x - 2 * y + z) ** 2 + x + y


def der_f1(X):
    x = X[0]
    y = X[1]
    z = X[2]

    d_x = 2 * (x - z) + 8 * (4 * x - 2 * y + z) + 1
    d_y = 4 * (2 * y + z) - 4 * (4 * x - 2 * y + z) + 1
    d_z = -2 * (x - z) + 2 * (2 * y + z) + 2 * (4 * x - 2 * y + z)

    return np.array([d_x, d_y, d_z])


def hess_f1(X):

    d_x_x = 2 + 8 * 4
    d_x_y = -16
    d_x_z = -2 + 8
    d_y_x = -16
    d_y_y = 16
    d_y_z = 0
    d_z_x = -2 + 8
    d_z_y = 0
    d_z_z = 6

    return np.array(
        [[d_x_x, d_x_y, d_x_z], [d_y_x, d_y_y, d_y_z], [d_z_x, d_z_y, d_z_z]]
    )


start_f1_a = [0, 0, 0]
start_f1_b = [1, 1, 0]


def f2(X):
    x = X[0]
    y = X[1]
    z = X[2]

    return (
        (x - 1) ** 2 + (y - 1) ** 2 + 100 * (y - x**2) ** 2 + 100 * (z - y**2) ** 2
    )


def der_f2(X):
    x = X[0]
    y = X[1]
    z = X[2]

    d_x = 2 * (200 * x**3 - 200 * x * y + x - 1)
    d_y = -200 * x**2 + 400 * y ** 3 + y * (202 - 400 * z) - 2
    d_z = 200 * (z - y**2)

    return np.array([d_x, d_y, d_z])

def hess_f2(X):
    x = X[0]
    y = X[1]
    z = X[2]

    d_x_x = 2 - 400 * y + 400 * 3 * x**2
    d_x_y = -400 * x
    d_x_z = 0
    d_y_x = -200 * 2 * x
    d_y_y = 2 + 200 - 400 * z + 400 * 3 * y ** 2
    d_y_z = -400 * y
    d_z_x = 0
    d_z_y = -200 * 2 * y
    d_z_z = 200

    return np.array(
        [[d_x_x, d_x_y, d_x_z], [d_y_x, d_y_y, d_y_z], [d_z_x, d_z_y, d_z_z]]
    )

start_f2_a = [1.2, 1.2, 1.2]
start_f2_b = [-1, 1.2, 1.2]


def f3(X):
    x = X[0]
    y = X[1]

    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * (y**2)) ** 2
        + (2.625 - x + x * (y**3)) ** 2
    )


def der_f3(X):
    x = X[0]
    y = X[1]

    d_x = (
        2 * (1.5 - x + x * y) * (-1 + y)
        + 2 * (2.25 - x + x * (y**2)) * (-1 + y**2)
        + 2 * (2.625 - x + x * (y**3)) * (-1 + y**3)
    )
    d_y = (
        2 * (1.5 - x + x * y) * x
        + 2 * (2.25 - x + x * (y**2)) * 2 * x * y
        + 2 * (2.625 - x + x * (y**3)) * 3 * x * (y**2)
    )

    return np.array([d_x, d_y])


start_f3_a = [1, 1]
start_f3_b = [4.5, 4.5]

print("Function 1")
for n in [2, 5, 10, 100]:
    print("n = ", n)
    x_k = gd(start_f1_a, 0.01, n, der_f1)
    print("GD", x_k, f1(x_k))
    x_k = gd(start_f1_b, 0.01, n, der_f1)
    print("GD", x_k, f1(x_k))

    x_k = pgd(start_f1_a, start_f1_b, 0.01, 0.01, n, der_f1)
    print("PGD", x_k, f1(x_k))

    x_k = ngd(start_f1_a, start_f1_b, 0.01, 0.01, n, der_f1)
    print("Nesterov GD", x_k, f1(x_k))

    x_k = ada_gd(start_f1_a, 0.01, n, der_f1)
    print("Adagrad GD", x_k, f1(x_k))
    x_k = ada_gd(start_f1_b, 0.01, n, der_f1)
    print("Adagrad GD", x_k, f1(x_k))

    x_k = newton(start_f1_a, n, der_f1, hess_f1)
    print("Newton", x_k, f1(x_k))
    x_k = newton(start_f1_b, n, der_f1, hess_f1)
    print("Newton", x_k, f1(x_k))

    x_k = bfgs(start_f1_a, n, der_f1)
    print("BFGS", x_k, f1(x_k))
    x_k = bfgs(start_f1_b, n, der_f1)
    print("BFGS", x_k, f1(x_k))

print("Function 2")
eps = 0.01
for n in [2, 5, 10, 100]:
    print("n = ", n)
    x_k = gd(start_f2_a, eps, n, der_f2)
    print("GD", x_k, f2(x_k))
    x_k = gd(start_f2_b, eps, n, der_f2)
    print("GD", x_k, f2(x_k))

    x_k = pgd(start_f2_a, start_f2_b, eps, eps, n, der_f2)
    print("PGD", x_k, f2(x_k))

    x_k = ngd(start_f2_a, start_f2_b, eps, eps, n, der_f2)
    print("Nesterov GD", x_k, f2(x_k))

    x_k = ada_gd(start_f2_a, eps, n, der_f2)
    print("Adagrad GD", x_k, f2(x_k))
    x_k = ada_gd(start_f2_b, eps, n, der_f2)
    print("Adagrad GD", x_k, f2(x_k))

    x_k = newton(start_f2_a, n, der_f2, hess_f2)
    print("Newton", x_k, f2(x_k))
    x_k = newton(start_f2_b, n, der_f2, hess_f2)
    print("Newton", x_k, f2(x_k))

    x_k = bfgs(start_f2_a, n, der_f2)
    print("BFGS", x_k, f2(x_k))
    x_k = bfgs(start_f2_b, n, der_f2)
    print("BFGS", x_k, f2(x_k))

