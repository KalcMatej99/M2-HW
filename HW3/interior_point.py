from cv2 import sqrt
import numpy as np
from cvxopt import solvers, matrix
import pulp as p


def interior_point(c, A, b):
    rankA = np.linalg.matrix_rank(A)
    '''
    lambdas, V =  np.linalg.eig(A.T)
    # The linearly dependent row vectors
    lambdas = lambdas.real
    print(lambdas)
    lambdas[np.abs(lambdas) < 1e-7] = 0
    print(lambdas)
    A = A[lambdas != 0,:]
    b = b[lambdas != 0]
    '''
    
    At = np.transpose(A)

    n = len(c)
    U = max(np.abs(b).max(), np.abs(A).max(), np.abs(c).max())
    reducer = 10 ** (len(str(int(U))) - 1)


    if reducer >= 100:
        x, s, y = interior_point(c/reducer, A/reducer, b/reducer)
        return x, s, y

    m = len(A)
    W = (m * U)**m
    R = 1/(W**2 * 2 * n * ((m + 1)*U)**(3*(m + 1)))
    M = 4 * n * U/R
    d = b/W
    e = np.ones((len(A[0]), 1))
    p = d - np.matmul(A, e)

    ## Construct (P')

    A_ = np.concatenate((np.concatenate((A, np.zeros((len(A), 1))), axis = 1),p), axis = 1)
    A_ = np.concatenate((A_, np.concatenate((np.transpose(e), [[1.0, 1.0]]), axis = 1)), axis = 0)
    At_ = np.transpose(A_)
    b_ = np.concatenate((d, [[n + 2]]), axis = 0)

    ## Initial solution (x_0, y_0, s_0, mu_0)
    mu = 2 * (np.square(M) + np.sum(np.square(c))) ** 0.5

    x_0 = np.concatenate((np.ones((n, 1)), [[1.0], [1.0]]), axis = 0)
    y_0 = np.concatenate((np.zeros((m, 1)), [[-mu]]), axis = 0)
    s_0 = np.concatenate((c + e * mu, [[mu], [M + mu]]), axis = 0)

    e_ = np.ones((len(A_[0]), 1))
    ## Solve P' using S in iterations
    while mu >= R**2/(32 * n**2):
        S = np.zeros((len(s_0), len(s_0)))
        np.fill_diagonal(S, s_0)
        X = np.zeros((len(x_0), len(x_0)))
        np.fill_diagonal(X, x_0)
        S_inv = np.linalg.inv(S)
        k = np.matmul(np.matmul(np.matmul(A_, S_inv), X), At_)
        k = np.linalg.inv(k)
        k = np.dot(k, b_ - np.dot(np.matmul(mu * A_, S_inv), e_))

        f = np.dot(-At_, k)

        h = np.dot(np.matmul(-X, S_inv), f) + np.dot(np.dot(mu, S_inv), e_) - x_0


        x_0 = x_0 + h
        s_0 = s_0 + f
        y_0 = y_0 + k
        mu = (1-0.15) * mu
        #mu = (1-1/(6 * np.sqrt(n + 2))) * mu


    x_0[x_0 < R/(4*n)] = 0
    s_0[s_0 < R/(4*n)] = 0

    optimal_x = x_0[:len(x_0) - 2]
    optimal_s = s_0[:len(s_0) - 2]
    optimal_y = y_0[:len(y_0) - 1]

    B = [i for i in range(n) if optimal_s[i] == 0]
    N = [i for i in range(n) if optimal_x[i] == 0]

    optimal_x *= W
    optimal_s *= W
    optimal_y *= W


    x_b = optimal_x[B]
    x_n = optimal_x[N]
    s_b = optimal_s[B]
    s_n = optimal_s[N]
    c_b = c[B]
    c_n = c[N]
    A_b = A[:, B]
    A_n = A[:, N]

    if len(B) < m:
        optimal_x, optimal_s, optimal_y = interior_point(c_b, A_b, b)
    elif len(B) == m:
        optimal_x_b = np.dot(np.linalg.inv(A_b), b)
        optimal_x[B] = optimal_x_b

    return optimal_x, optimal_s, optimal_y

def bread():
    c = np.transpose(np.array([[10.0, 22.0, 15.0, 45.0, 40.0, 20.0, 87.0, 21.0]]))
    A = np.array([[-18.0, -48.0,-5.0,-1.0,-5.0,-0.0,-0.0,-8.0], [-2.0, -11.0, -3.0, -1.0, -3.0, -0.0, -15.0,-1.0], [-0.0,-6.0,-3.0,-10.0,-3.0,-100.0,-30.0,-1.0], [-77.0,-270.0,-60.0,-140.0,-61.0,-880.0,-330.0,-32.0], [18.0, 48.0,5.0,1.0,5.0,0.0,0.0,8.0], [2.0, 11.0, 3.0, 1.0, 3.0, 0.0, 15.0,1.0], [0.0,6.0,3.0,10.0,3.0,100.0,30.0,1.0], [77.0,270.0,60.0,140.0,61.0,880.0,330.0,32.0]])
    b = np.transpose(np.array([[-250.0, -50.0, -50.0, -2200.0, 370.0, 170.0, 90.0, 2400.0]]))
    ce = np.concatenate((c, np.zeros((len(A), 1))), axis = 0)
    Ae = np.concatenate((A, np.identity(len(A))), axis = 1)


    xe, se, y = interior_point(ce, Ae, b)

    x = xe[:len(c)]
    s = se[:len(A[0])]

    print("Primal Problem")
    print("Optimal x", x)
    print("Min val", np.dot(np.transpose(c), x))
    print("A*x=", np.dot(A, x))
    print("Ae*xe=", np.dot(Ae, xe))
    print("Original b", b)


    print("Dual problem")
    print("Max value", np.dot(np.transpose(b), y))
    print("A^T * y + s =", np.dot(np.transpose(A), y) + s)
    print("Ae^T * y + s =", np.dot(np.transpose(Ae), y) + se)
    print("Original c", c)


    Lp_prob = p.LpProblem('Problem', p.LpMinimize) 
  
    # Create problem Variables 
    pot = p.LpVariable("p", lowBound = 0)
    b = p.LpVariable("b", lowBound = 0)
    m = p.LpVariable("m", lowBound = 0)
    e = p.LpVariable("e", lowBound = 0)
    y = p.LpVariable("y", lowBound = 0)
    v = p.LpVariable("v", lowBound = 0)
    be = p.LpVariable("be", lowBound = 0)
    s = p.LpVariable("s", lowBound = 0)

    # Objective Function
    Lp_prob += 10 * pot + 22 * b + 15* m + 45 * e + 40 * y + 20 * v + 87 * be + 21 * s

    # Constraints:
    Lp_prob += 18 * pot + 48 * b + 5* m + 1 * e + 5 * y + 0 * v + 0 * be + 8 * s <= 370
    Lp_prob += 2 * pot + 11 * b + 3* m + 13 * e + 3 * y + 0 * v + 15 * be + 1 * s <= 170
    Lp_prob += 0 * pot + 5 * b + 3* m + 10 * e + 3 * y + 100 * v + 30 * be + 1 * s <= 90
    Lp_prob += 77 * pot + 270 * b + 60* m + 140 * e + 61 * y + 880 * v + 330 * be + 1 * s <= 2400
    Lp_prob += 18 * pot + 48 * b + 5* m + 1 * e + 5 * y + 0 * v + 0 * be + 8 * s >= 250
    Lp_prob += 2 * pot + 11 * b + 3* m + 13 * e + 3 * y + 0 * v + 15 * be + 1 * s >= 50
    Lp_prob += 0 * pot + 5 * b + 3* m + 10 * e + 3 * y + 100 * v + 30 * be + 1 * s >= 50
    Lp_prob += 77 * pot + 270 * b + 60* m + 140 * e + 61 * y + 880 * v + 330 * be + 1 * s >= 2200
    
    # Display the problem
    print(Lp_prob)
    
    status = Lp_prob.solve()   # Solver
    print(p.LpStatus[status])   # The solution status
    
    # Printing the final solution
    print(p.value(pot), p.value(b), p.value(m), p.value(e), p.value(y), p.value(v), p.value(be), p.value(s), p.value(Lp_prob.objective)) 


if __name__ == "__main__":
    bread()

