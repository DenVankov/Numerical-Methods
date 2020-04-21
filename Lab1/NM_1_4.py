import numpy as np
import copy
import math

class MatrixException(Exception):
    pass

def show(A, n):
    for i in range(0, n):
        for j in range(0, n):
            print("\t", A[i][j], " ", end='')
        print("\n")

def matrixsum(A, B):
    out = [[0] * len(A[0]) for _ in range(len(A))]
    if len(B) != len(A) and len(B[0]) != len(A[0]):
        raise MatrixException('Matrix can\'t be compared')
    for i in range(len(A)):
        for j in range(len(A[0])):
            out[i][j] = A[i][j] + B[i][j]
    return out


def multi(M1, M2):
    sum = 0  # сумма
    tmp = []  # временная матрица
    ans = []  # конечная матрица
    if len(M2) != len(M1[0]):
        raise MatrixException('Matrix can\'t be multiplied')
    else:
        row1 = len(M1)  # количество строк в первой матрице
        col1 = len(M1[0])  # Количество столбцов в 1
        row2 = col1  # и строк во 2ой матрице
        col2 = len(M2[0])  # количество столбцов во 2ой матрице
        for k in range(0, row1):
            for j in range(0, col2):
                for i in range(0, col1):
                    sum = sum + M1[k][i] * M2[i][j]
                tmp.append(sum)
                sum = 0
            ans.append(tmp)
            tmp = []
    return ans

def getMaxInd(A):
    i_max = j_max = 0
    a_max = A[0][0]
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > a_max:
                a_max = abs(A[i][j])
                i_max, j_max = i, j
    return i_max, j_max

def rotation(A, eps = 0.01):
    n = len(A)
    Ak = [row.copy() for row in A]

    U = [[0. if i != j else 1. for i in range(n)] for j in range(n)]

    cov = False

    while not cov:
        ik, jk = 0, 1

        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(Ak[i][j]) > abs(Ak[ik][jk]):
                    ik, jk = i, j
        if Ak[ik][ik] == Ak[jk][jk]:
            phi = math.pi / 4
        else:
            phi = math.atan(2 * A[ik][jk] / (A[ik][ik] - A[jk][jk])) * 0.5

        Uk = [[0. if i != j else 1. for i in range(n)] for j in range(n)]

        Uk[ik][jk] = math.sin(phi)
        Uk[jk][ik] = -math.sin(phi)

        Uk[ik][ik] = math.cos(phi)
        Uk[jk][jk] = math.cos(phi)

        tmp = multi(Uk, Ak)
        Uk[ik][jk], Uk[jk][ik] = Uk[jk][ik], Uk[ik][jk]

        Ak = multi(tmp, Uk)
        U = multi(U, Uk)

        count = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                count += Ak[i][j] ** 2

        average = math.sqrt(count)
        if average < eps:
            cov = True

    return [Ak[i][i] for i in range(n)], U



if __name__ == '__main__':
    print("Input demention of matrix: ")
    n = int(input())
    A = []
    print("Input matrix: ")
    for i in range(n):
        A.append(list(map(float, input().split())))


    print("Start:")
    show(A, n)
    print("Input epsilon:")
    eps = float(input())

    X, U = rotation(A, eps)
    print('X:\n', X)
    print('U:\n')
    show(U, len(U))

    print("With linalg:")
    X, U = np.linalg.eig(A)
    print('X:\n', X)
    print('U:\n')
    show(U, len(U))
