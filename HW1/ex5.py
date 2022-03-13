import numpy as np

def f(x, y):
    return x ** 2 + np.exp(x) + y ** 2 - x * y

def der_f(x, y):
    return np.array([2 * x + np.exp(x) - y, 2 * y - x])

def project_in_circle(x_k):
    x = x_k[0]
    y = x_k[1]
    if np.sqrt(x ** 2 + y ** 2) < np.sqrt(1.5):
        return x_k
    else:
        return x_k * np.sqrt(1.5/ (x ** 2 + y ** 2))

def project_in_square(x_k):
    x = x_k[0]
    y = x_k[1]
    
    if x >= -1 and x <= 1 and y >= -1 and y <= 1:
        return x_k
    elif y >= -1 and y <= 1:
        return [np.sign(x), y]
    elif x >= -1 and x <= 1:
        return [x, np.sign(y)]
    else:
        return [np.sign(x), np.sign(y)]

def project_in_triangle(x_k):
    x = x_k[0]
    y = x_k[1]
    
    if x >= -1 and x <= 1.5 and y < -1:
        return [x, -1]
    elif (x >= 1.5 and y <= -1) or (x >= 1.5 and y >= -1 and y <= x + 2.5):
        return [1.5, -1]
    elif y >= x - 2.5 and y <= x + 2.5 and y >= -x + 1:
        return [x - y + 1, y - x + 1]/2
    elif (x <= -1 and y >= 1.5) or (x >= -1 and y >= 1.5 and y >= x + 2.5):
        return [-1, 1.5]
    elif x <= -1 and y >= -1 and y <= 1.5:
        return [-1, y]
    elif x <= -1 and y <= -1:
        return [-1, -1]
    else:
        return x_k
        
def pgd(x_1, eps, n, project_in_domain):
    x_k = np.array(x_1)
    for i in range(n):
        x_k = np.array(project_in_domain(x_k - eps(i + 1) * der_f(x_k[0], x_k[1])))
    return x_k

B = 9.522
def eps_B(k):
    return 1/B
def eps_a_B(k):
    return 1/B
a = 1.07
def eps_a_L(k):
    return 2/(a * (k + 1))
L = 15

x1 = np.array([-1, 1])
for domain in [project_in_circle, project_in_square, project_in_triangle]:
    if domain == project_in_circle:
        print("Circle")
    if domain == project_in_square:
        print("Square")
    if domain == project_in_triangle:
        print("Triangle")
    x_star = np.array([-0.432, -0.216])
    x_k = pgd(x1, eps_B, 10, domain)

    norm_x1_minus_x_star = np.linalg.norm(x1 - x_star)
    print("PGD output", f(x_k[0], x_k[1]) - f(x_star[0], x_star[1]))
    print("Theorem garantee", (3 * B * (norm_x1_minus_x_star ** 2) + f(x1[0], x1[1]) - f(x_star[0], x_star[1])) / (10))
    print("Theoretical garantee 1 satisfied", f(x_k[0], x_k[1]) - f(x_star[0], x_star[1]) <= (3 * B * (norm_x1_minus_x_star ** 2) + f(x1[0], x1[1]) - f(x_star[0], x_star[1])) / (10))
    
    x_k_1 = pgd(x1, eps_a_B, 11, domain)
    k = B/a

    print("PGD output", f(x_k_1[0], x_k_1[1]) - f(x_star[0], x_star[1]))
    print("Theorem garantee", (B/2) * (((k - 1)/k) ** (2 * 11)) * (norm_x1_minus_x_star ** 2))
    print("Theoretical garantee 2 satisfied", f(x_k_1[0], x_k_1[1]) - f(x_star[0], x_star[1]) <= (B/2) * (((k - 1)/k) ** (2 * 11)) * (norm_x1_minus_x_star ** 2))
    
    
    x_k = pgd(x1, eps_a_L, 10, domain)

    s0 = np.sum([2 * i * pgd(x1, eps_a_L, i - 1, domain)[0] for i in range(1, 10)])/(110)
    s1 = np.sum([2 * i * pgd(x1, eps_a_L, i - 1, domain)[1] for i in range(1, 10)])/(110)
    
    print("PGD output", f(s0, s1) - f(x_star[0], x_star[1]))
    print("Theorem garantee", (2 * L ** 2)/(a * (11)))
    print("Theoretical garantee 3 satisfied", f(s0, s1) - f(x_star[0], x_star[1]) <= (2 * L ** 2)/(a * (11)))
    
