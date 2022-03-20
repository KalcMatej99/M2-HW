import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def der_f(x):
    if x < 1:
        return 25*x
    if 1 <= x and x <= 2:
        return x + 24
    return 25 * x - 24

def der_f2(x):
    return 2 * x

def der_f3(x, k, n):
    if x > 0:
        return 2*x
    return k * x + n
    
def gd(x_1, eps, n, k, nu):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(x_k - eps * der_f3(x_k, k, nu))
    return x_k
def pgd(x_1, x_2, eps, mom, n):
    x_k_1 = np.array(x_1)
    x_k = np.array(x_2)
    for i in range(n):
        x_k_s = np.array(x_k - eps * der_f(x_k)) + mom * (x_k - x_k_1)
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

def like(params):
    eps = params[3]
    x_1 = 1
    k = params[1]
    nu = params[2]
    x_2 = params[0]
    x_3 = gd(x_1, eps, 2, k, nu)
    x_4 = gd(x_1, eps, 3, k, nu)
    x_5 = gd(x_1, eps, 4, k, nu)

    loss = (x_1 - x_4) ** 2 + (x_2 - x_5) ** 2 + np.sign(x_2 * x_3)
    print(params, loss)

    return loss


p = fmin_l_bfgs_b(like, [1,1 ,1, 0.5], approx_grad=True)
print(p[0])
x_3 = gd(p[0][0], p[0][3], 1, p[0][1], p[0][2])
print(x_3)
x_3 = gd(p[0][0], p[0][3], 2, p[0][1], p[0][2])
print(x_3)
x_3 = gd(p[0][0], p[0][3], 3, p[0][1], p[0][2])
print(x_3)

'''
p = fmin_l_bfgs_b(like, [-3, 1.5], approx_grad=True)
print(p)
eps = 1/9
mom = 4/9
x_3 = pgd(p[0], p[1], eps, mom, 1)
print(x_3)
'''