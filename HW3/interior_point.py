import numpy as np
from cvxopt import solvers, matrix

def interior_point(c, A, b):
    rankA = np.linalg.matrix_rank(A)

    n = len(c)
    U = 1000
    m = len(c)
    W = (m * U)**m
    R = 1/W**2 * (1/2 * n * ((m + 1)*U)**(3*(m + 1)))
    M = 4 * n * U/R

    d = 1/W * b
    e = np.ones(len(A[0]))
    p = d - A * e

    ## Construct (P')

    A_ = np.concatenate((np.concatenate((A, np.zeros((len(A), 1))), axis = 1),p), axis = 1)
    A_ = np.concatenate((A_, np.concatenate((np.transpose(e), [1, 1]), axis = 1)), axis = 0)

    b_ = np.concatenate((d, n + 2), axis = 0)

    ## Construct (D')

    d_ = np.concatenate((d, [n + 2]), axis = 0)
    At = np.transpose(A)
    At_ = np.concatenate((np.concatenate((At, e), axis = 1), np.concatenate((np.zeros((1, len(A))), [1]), axis = 1)), axis = 0)
    pt = np.transpose(p)
    At_ = np.concatenate((At_, np.concatenate((pt, [1]), axis = 1)), axis = 0)
    c_= np.concatenate((c, [0]), axis = 0)
    c_= np.concatenate((c_, [M]), axis = 0)



