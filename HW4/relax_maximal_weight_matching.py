from operator import index
from cvxopt import solvers, matrix
import numpy as np
from sklearn.linear_model import LinearRegression

def maximalWeightMatching(e, G, h):

    M = e * 0.0
    n_iters = 1000
    k = 1
    old_cost = np.sum(M * e)

    print(M)

    for i in range(n_iters):

        new_M = M.copy()

        amount_of_edges_to_remove = np.random.random_integers(1, k)
        amount_of_edges_to_add = np.random.random_integers(1, k)
        if np.sum(new_M) >= amount_of_edges_to_remove:
            indexes = np.where(new_M == 1)
            edges_to_remove = np.random.choice(indexes[0], size=amount_of_edges_to_remove)
            new_M[edges_to_remove] = 0
        
        indexes = np.where(new_M == 0)
        edges_to_add = np.random.choice(indexes[0], size=amount_of_edges_to_add)
        new_M[edges_to_add] = 1

        new_cost = np.sum(new_M * e)

        if new_cost > old_cost:
            M = new_M
            new_cost = old_cost
    

    print("ccost", old_cost)
    return old_cost, M

def relaxMaximalWeightMatching(e, G, h):

    c = matrix(-e)
    G_ = matrix(G)
    h_ = matrix(h)

    solvers.options["silent"] = True
    sol = solvers.lp(c = c, G = G_, h = h_, solver = "glpk")

    x_sol = sol["x"]
    print(x_sol)

    return sol["primal objective"]



if __name__ == "__main__":

    X = []
    y = []
    for s in [3]:
        X.append([s])
        print(s)

        e = []
        for i in range(s):
            e.extend(np.random.uniform(1, 2, s - 1))

        allVertical = len(e)
        for i in range(s - 1):
            e.extend(np.random.uniform(1, 2, s))

        e = np.array(e)


        G = []

        for i in range(s):
            for j in range(s):
                e_v = np.zeros(len(e))
                if j > 0:
                    e_v[i * (s - 1) + j - 1] = 1
                if j < s - 1:
                    e_v[i * (s - 1) + j] = 1

                if i > 0:
                    e_v[allVertical + (i - 1) * s + j] = 1
                if i < s - 1:
                    e_v[allVertical + i * s + j] = 1

                e_v *= e

                G.append(e_v)
        
        G = np.array(G)
        
        h = np.ones((s * s,1))

        primal_obj = maximalWeightMatching(e, G, h)
        cost, M = relaxMaximalWeightMatching(e, G, h)

        print("cost", cost)

        y.append(-primal_obj)


    from sklearn.preprocessing import PolynomialFeatures
    poly_reg=PolynomialFeatures(degree=2)
    X=poly_reg.fit_transform(X)
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
            