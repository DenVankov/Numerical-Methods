import tools
from math import sqrt
import numpy as np
import pprint


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def QR_decomp(A):
    n = len(A)
    Ak = [rows.copy() for rows in A]

    E = tools.diagonal(n)
    Q = tools.diagonal(n)

    for i in range(n):
        V = [0 for _ in range(n)]

        V[i] = Ak[i][i] + sign(A[i][i]) * sqrt(sum([Ak[j][i] ** 2 for j in range(i, n)]))

        for j in range(i + 1, n):
            V[j] = Ak[j][i]

        V_trans = [V]

        V = [[V[i]] for i in range(n)]

        L = tools.mm_mult(V,V_trans) # Matrix nxn
        R = tools.mm_mult(V_trans,V) # After multi it become a const

        for j in range(n):
            for k in range(n):
                L[j][k] /= R[0][0]
                L[j][k] *= 2
        Hk = tools.mm_substr(E, L)
        Ak = tools.mm_mult(Hk, Ak)
        Q = tools.mm_mult(Q, Hk)

    return Q,Ak


def normal(x):
    return sum(k**2 for k in x) ** 0.5


def columns(A, i):
    return [A[j][i] for j in range(i+1, len(A))]


def small(A, eps = 0.01):
    n = len(A)
    array = []

    i = 0
    while i < n:
        if normal(columns(A, i)) <= eps:
            array.append('R')
        elif normal(columns(A, i + 1)) <= eps:
            array.append('I')
            i += 1
        else:
            array.append(None)
        i += 1
    return array


def solve(A, array):
    n = len(A)
    solution = []
    Ak = np.array(A)
    k = 0
    for piece in array:
        if piece == 'R':
            solution.append(Ak[k, k])
        else:
            A11 = Ak[k, k]
            A12 = A21 = A22 = 0

            if k + 1 < n:
                A12 = Ak[k, k + 1]
                A21 = Ak[k + 1, k]
                A22 = Ak[k + 1, k + 1]

            solution.extend(np.roots((1, -A11 -A22, A11 * A22 - A12 * A21)))
            k += 1
        k += 1
    return solution


def QR_method(A, eps = 0.01):
    n = len(Q)
    Ak = [rows.copy() for rows in A]

    stop = True

    for i in range(100):
        Qk, Rk = QR_decomp(Ak)
        Ak = tools.mm_mult(Rk, Qk)
        tools.matrix_print(Ak, header=f"A on {i} iteration")

        array = small(Ak, eps)
        if all(array):
            if stop:
                stop = False
            else:
                return solve(Ak, array)

    return None


if (__name__ == '__main__'):
    print("Input demention of matrix: ")
    n = int(input())
    A = []
    print("Input matrix: ")
    for i in range(n):
        A.append(list(map(float, input().split())))

    print("Input eps:")
    eps = float(input())

    print("Start:")
    tools.matrix_print(A, header= "A")
    Q, R = QR_decomp(A)
    tools.matrix_print(Q, header="Q")
    tools.matrix_print(R, header="R")
    tools.matrix_print(tools.mm_mult(Q, R), header="Again A")
    print("Solution X: ",QR_method(A, eps))
    print("With numpy library:", end = '')
    X, U = np.linalg.eig(A)
    pprint.pprint(X)
