from cvxopt import solvers, matrix
import numpy as np

def maximalWeightMatching(e, G, h, k = 2, n_iters = 10000, use_prob = True):

    M = e * 0.0
    cost = np.sum(e[M == 1])

    h = np.transpose(h)[0]


    for i in range(n_iters):

        new_M = M

        amount_of_edges_to_remove = np.random.randint(0, k + 1)
        amount_of_edges_to_add = np.random.randint(0, k + 1)
        
        if amount_of_edges_to_remove > 0 and np.sum(new_M) >= amount_of_edges_to_remove:
            indexes = np.where(new_M == 1)[0]
            if len(indexes) > amount_of_edges_to_remove:
                if use_prob:
                    probs = (2 - e[indexes])/np.sum((2 - e[indexes]))
                    edges_to_remove = np.random.choice(indexes, size=amount_of_edges_to_remove, p = probs)
                else:
                    edges_to_remove = np.random.choice(indexes, size=amount_of_edges_to_remove)
                new_M[edges_to_remove] = 0
        
        if amount_of_edges_to_add > 0:
            indexes = np.where(new_M == 0)[0]
            if len(indexes) > amount_of_edges_to_add:
                if use_prob:
                    probs = (e[indexes] - 1)/np.sum((e[indexes] - 1))
                    edges_to_add = np.random.choice(indexes, size=amount_of_edges_to_add, p = probs)
                else:
                    edges_to_add = np.random.choice(indexes, size=amount_of_edges_to_add)
                new_M[edges_to_add] = 1

        new_cost = np.sum(e[new_M == 1])

        if new_cost > cost and (np.dot(G, new_M) <= h).all():
            M = new_M
            cost = new_cost
    
    return cost, M

def relaxMaximalWeightMatching(e, G, h):

    c = matrix(-e)
    G_ = matrix(G)
    h_ = matrix(h)


    solvers.options['show_progress']=False
    sol = solvers.lp(c = c, G = G_, h = h_)

    x_sol = sol["x"]

    return x_sol, sol["primal objective"]



if __name__ == "__main__":

    np.random.seed(2)

    s = 20

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

    G[G > 0] = 1

    number_of_edges = G.shape[1]
    G = np.concatenate((G, np.eye(number_of_edges)), axis = 0)
    G = np.concatenate((G, -np.eye(number_of_edges)), axis = 0)

    
    h = np.ones((s * s,1))
    h = np.concatenate((h, np.ones((number_of_edges, 1))), axis = 0)
    h = np.concatenate((h, np.zeros((number_of_edges, 1))), axis = 0)


    sol, cost = relaxMaximalWeightMatching(e, G, h)
    print("Relax Maximal Weight Matching with cost", -cost)

    
    n_iters = 100000
    cost, M = maximalWeightMatching(e, G, h, k = 1, n_iters=n_iters)
    print("k = 1, cost", cost)
    cost, M = maximalWeightMatching(e, G, h, k = 2, n_iters=n_iters)
    print("k = 2, cost", cost)
    cost, M = maximalWeightMatching(e, G, h, k = 3, n_iters=n_iters)
    print("k = 3, cost", cost)
            