import sys

import numpy as np
from numpy.linalg import norm

from matrix import Matrix, Vector
from numpy.linalg import eig

eps = 0.01


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def matrix_print(matrix, header=None):
    if header:
        print(header, '\n')
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print('{:6.2f}'.format(float(matrix[i][j])), end=' ')
        print()
    print()


def get_ans(matrix, my_values):
    print("Answer:")
    print(my_values)

    a = np.array(matrix.get_data())
    eig_np = eig(a)
    print("Answer with numpy:")
    print(eig_np[0].round(3))


def householder(a, sz, k):
    v = np.zeros(sz)
    a = np.array(a.get_data())
    v[k] = a[k] + sign(a[k]) * norm(a[k:])
    for i in range(k + 1, sz):
        v[i] = a[i]
    v = v[:, np.newaxis]
    H = np.eye(sz) - (2 / (v.T @ v)) * (v @ v.T)
    return Matrix.from_list(H.tolist())


def get_QR(A):
    sz = len(A)
    Q = Matrix.identity(sz)
    A_i = Matrix(A)

    for i in range(sz - 1):
        col = A_i.get_column(i)
        H = householder(col, len(A_i), i)
        Q = Q.multiply(H)
        A_i = H.multiply(A_i)

    return Q, A_i


def get_roots(A, i):
    sz = len(A)
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < sz else 0
    a21 = A[i + 1][i] if i + 1 < sz else 0
    a22 = A[i + 1][i + 1] if i + 1 < sz else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def finish_iter_for_complex(A, eps, i):
    Q, R = get_QR(A)
    A_next = R.multiply(Q)
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return True if abs(lambda1[0] - lambda2[0]) <= eps and \
                   abs(lambda1[1] - lambda2[1]) <= eps else False


def get_eigenvalue(A, eps, i):
    A_i = Matrix(A)
    while True:
        Q, R = get_QR(A_i)
        A_i = R.multiply(Q)
        a = np.array(A_i.get_data())
        if norm(a[i + 1:, i]) <= eps:
            res = (a[i][i], False, A_i)
            break
        elif norm(a[i + 2:, i]) <= eps and finish_iter_for_complex(A_i, eps, i):
            res = (get_roots(A_i, i), True, A_i)
            break
    return res


def QR_method(A, eps):
    res = Vector()
    i = 0
    A_i = Matrix(A)
    while i < len(A):
        eigenval = get_eigenvalue(A_i, eps, i)
        if eigenval[1]:
            res.extend(eigenval[0])
            i += 2
        else:
            res.append(eigenval[0])
            i += 1
        A_i = eigenval[2]
    return res, i


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("use {} <matrix_file>")
        exit(0)

    matrix_file = sys.argv[1]

    data = []
    with open(matrix_file) as m:
        for line in m:
            data.append(list(map(int, line.split())))

    A = Matrix()
    A.data = data
    matrix_print(A, header='A')
    tmp, count_iter = QR_method(A, eps)

    get_ans(A, tmp)
