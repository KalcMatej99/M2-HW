import numpy as np

a = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
c = [1, 1.2, 3, 3.2]
p = np.array([[0.36890, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])

def f(z):
    return -np.sum([c[i] * np.exp(-np.sum([a[i, j] * (z[j] - p[i,j]) ** 2 for j in range(3)])) for i in range(4)])

def der_f(z):
    return np.array([-np.sum([c[i] * np.exp(-np.sum([a[i, j] * (z[j] - p[i,j]) ** 2 for j in range(3)])) * (-2* a[i, d] * (z[d] - p[i, d])) for i in range(4)]) for d in range(3)])


def project_in_area(x_k):
    return x_k
        
def pgd(x_1, eps, n, project_in_domain):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(x_k - eps * der_f(x_k))
    return x_k

x1 = np.array([0.5, 0.5, 0.5])
x_k = pgd(x1, 0.01, 10000, project_in_area)

print("x_k", x_k)
print("Minimum found", f(x_k))
print("Abs difference between real and found min", np.abs(-3.86278214782076 - f(x_k)))

