import matplotlib.pyplot as plt
from NM_1_1 import LUP_solve
from NM_1_1 import LUP_decompose
from NM_1_1 import get_LU

target = '''
i:        0        1        2     3       4       5
Xi:    -3.0     -2.0     -1.0   0.0     1.0     2.0
Yi: 0.04979  0.13534  0.36788   1.0  2.7183  7.3891
'''


def func(x, values):
    return sum([c * (x ** i) for i, c in enumerate(values)])


def sse(f, y):
    return sum([f_i - y_i for f_i, y_i in zip(f, y)])


def mls(n, x, y):
    matrix = [[] for _ in range(n + 1)]
    size = len(matrix)
    for i in range(n + 1):
        for j in range(n + 1):
            matrix[i].append(sum([x_j ** (i + j) for x_j in x]))

    b = [0 for _ in range(n + 1)]
    for i in range(n + 1):
        b[i] = sum([y_j * (x_j ** i) for x_j, y_j in zip(x, y)])

    P = LUP_decompose(matrix, size)
    L, U = get_LU(matrix)
    new_b = LUP_solve(L, U, P, b, size)
    return [round(i, 5) for i in new_b]


def f_printer(coefs):
    n = len(coefs)
    f = f'F{n - 1}(x) = '
    for i in range(n):
        f += f'{coefs[i]}x^{i} + '
    f = f[:-2]
    return f


if __name__ == '__main__':
    x = [-3., -2.0, -1.0, 0.0, 1.0, 2.0]
    y = [0.04979, 0.13534, 0.36788, 1.0, 2.7183, 7.3891]
    F = []
    err = []
    coefs = []

    for degree in [1, 2]:
        print(f'Degree = {degree}')
        coefs.append(mls(degree, x, y))
        print(f_printer(coefs[degree - 1]))
        F.append([func(i, coefs[degree - 1]) for i in x])
        err.append(sse(F[0], y))

    plt.scatter(x, y, color='r')
    plt.plot(x, F[0], color='m')
    plt.plot(x, F[1], color='g')
    plt.grid()
    plt.savefig('3_3.png')
    plt.show()
