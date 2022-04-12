import numpy as np
import subprocess

def nelder_mead(f, x_0, displacement = 1):

    def nelder_mead_internal(f, X, tol = 1e-5, alpha = 1, gamma = 2, p = 0.5, sigma = 0.5):


        Y = []
        for x in X:
            Y.append(f(x))
        print(Y)
        
        X = X[np.argsort(Y)]

        if np.std(Y) < tol:
            return X[np.argmin(Y)]

        centroid = np.mean(X[:len(X) - 1], axis=0)

        x_r = centroid + alpha*(centroid - X[-1])
        f_x_r = f(x_r)
        f_x_0 = f(X[0])
        f_x_m2 = f(X[-2])

        if f_x_0 <= f_x_r and f_x_r < f_x_m2:
            return nelder_mead_internal(f, np.concatenate((X[:-1], [x_r]), axis = 0), tol=tol, alpha=alpha, gamma=gamma, p=p, sigma=sigma)

        if f_x_r < f_x_0:
            x_e = centroid + gamma * (x_r - centroid)
            if f(x_e) < f_x_r:
                return nelder_mead_internal(f, np.concatenate((X[:-1], [x_e]), axis = 0), tol=tol, alpha=alpha, gamma=gamma, p=p, sigma=sigma)
            else:
                return nelder_mead_internal(f, np.concatenate((X[:-1], [x_r]), axis = 0), tol=tol, alpha=alpha, gamma=gamma, p=p, sigma=sigma)

        f_x_m1 = f(X[-1])
        if f_x_r < f_x_m1:
            x_c = centroid + p * (x_r - centroid)
            if f(x_c) < f_x_r:
                return nelder_mead_internal(f, np.concatenate((X[:-1], [x_c]), axis = 0), tol=tol, alpha=alpha, gamma=gamma, p=p, sigma=sigma)
        else:
            x_c = centroid + p * (X[-1] - centroid)
            if f(x_c) < f_x_m1:
                return nelder_mead_internal(f, np.concatenate((X[:-1], [x_c]), axis = 0), tol=tol, alpha=alpha, gamma=gamma, p=p, sigma=sigma)

        X_i = [X[0] + sigma * (x - X[0]) for x in X[1:]]
        return nelder_mead_internal(f, np.concatenate(([X[0]], X_i), axis = 0), tol=tol, alpha=alpha, gamma=gamma, p=p, sigma=sigma)

    X = [x_0]
    n = len(x_0)
    for i in range(n):
        x_i = x_0.copy()
        x_i[i] += displacement
        X.append(x_i)

    print(X)
    return nelder_mead_internal(f, np.array(X))



def f1(X):
    x = X[0]
    y = X[1]
    z = X[2]

    return (x - z) ** 2 + (2 * y + z) ** 2 + (4 * x - 2 * y + z) ** 2 + x + y

def f2(X):
    x = X[0]
    y = X[1]
    z = X[2]

    return (
        (x - 1) ** 2 + (y - 1) ** 2 + 100 * (y - x**2) ** 2 + 100 * (z - y**2) ** 2
    )

def f3(X):
    x = X[0]
    y = X[1]

    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * (y**2)) ** 2
        + (2.625 - x + x * (y**3)) ** 2
    )

def ex2():
    start_f1_a = [0, 0, 0]
    start_f1_b = [1, 1, 0]
    x = np.array([start_f1_a, start_f1_b, [0,0,5]])
    optimal_x = nelder_mead(f1, x)
    print("f1", "x = ", optimal_x, "f(x) = ", f1(optimal_x))

    start_f2_a = [1.2, 1.2, 1.2]
    start_f2_b = [-1, 1.2, 1.2]
    x = np.array([start_f2_a, start_f2_b, [-2,-1,-1]])
    optimal_x = nelder_mead(f2, x)
    print("f2", "x = ", optimal_x, "f(x) = ", f2(optimal_x))

    start_f3_a = [1, 1]
    start_f3_b = [4.5, 4.5]
    x = np.array([start_f3_a, start_f3_b, [-1,2.5]])
    optimal_x = nelder_mead(f3, x)
    print("f3", "x = ", optimal_x, "f(x) = ", f3(optimal_x))

def ex3():
    def f1(X):
        r = subprocess.run(["./hw4_nix", "63180368", "1", str(X[0]), str(X[1]), str(X[2])], stdout=subprocess.PIPE)
        return float(r.stdout)
    def f2(X):
        r = subprocess.run(["./hw4_nix", "63180368", "2", str(X[0]), str(X[1]), str(X[2])], stdout=subprocess.PIPE)
        return float(r.stdout)
    def f3(X):
        r = subprocess.run(["./hw4_nix", "63180368", "3", str(X[0]), str(X[1]), str(X[2])], stdout=subprocess.PIPE)
        return float(r.stdout)

    x = np.array([[1,-1,1], [0,0,0], [-1,2.5, 7]])
    #optimal_x = nelder_mead(f1, x)
    #print("f1", "x = ", optimal_x, "f(x) = ", f1(optimal_x)) #1.7

    x = np.array([[1,-1,1], [0,0,0], [-1,2.5, 7]])
    #optimal_x = nelder_mead(f2, x)
    #print("f2", "x = ", optimal_x, "f(x) = ", f2(optimal_x)) #1.33

    x = [0,0,0]
    optimal_x = nelder_mead(f3, x)
    print("f3", "x = ", optimal_x, "f(x) = ", f3(optimal_x)) #0.863
ex3()