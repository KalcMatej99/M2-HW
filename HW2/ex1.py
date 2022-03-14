import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def der_f(x):
    if x < 1:
        return 25*x
    if 1 <= x and x <= 2:
        return x + 24
    return 25 * x - 24
    
def gd(x_1, x_2, eps, mom, n):
    x_k_1 = np.array(x_1)
    x_k = np.array(x_2)
    for i in range(n):
        x_k_s = np.array(x_k - eps * der_f(x_k)) + mom * (x_k - x_k_1)
        x_k_1 = x_k
        x_k = x_k_s
    return x_k

def like(params):
    eps = 1/9
    mom = 4/9
    x_1 = params[0]
    x_2 = params[1]
    x_3 = gd(x_1, x_2, eps, mom, 1)
    x_4 = gd(x_1, x_2, eps, mom, 2)
    x_5 = gd(x_1, x_2, eps, mom, 3)

    print(x_3)
    loss = (x_1 - x_3) ** 2 + (x_2 - x_4) ** 2
    print(params, loss)

    return loss

p = fmin_l_bfgs_b(like, [-3, 1.5], approx_grad=True)
print(p)
eps = 1/9
mom = 4/9
x_3 = gd(p[0], p[1], eps, mom, 1)
print(x_3)