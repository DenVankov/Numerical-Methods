import copy
from math import sqrt

class MatrixException(Exception):
    pass

def seidel(A, b, eps):
    n = len(A)
    beta = [0 for _ in range(n)]

    for i in range(n):
        beta[i] = b[i] / A[i][i]

    x = [beta[i] for i in range(n)]

    stop = False
    count = 0
    while not stop:
        x_new = copy.deepcopy(x)
        print("Iteration", count, "Ans ", x)
        for i in range(n):
            tmp_1 = 0
            tmp_2 = 0
            for j in range(i):
                tmp_1 += A[i][j] * x_new[j] # Sum for x_(k+1)
            for j in range(i+1, n):
                tmp_2 += x[j] * A[i][j] # Sum for x_k

            #s1 = sum(A[i][j] * x_new[j] for j in range(i)) # Sum for x_(k+1)
            #s2 = sum(A[i][j] * x[j] for j in range(i + 1, n)) # Sum for x_k
            x_new[i] = (b[i] - tmp_1 - tmp_2) / A[i][i]


        stop = sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        if stop == False:
            x = x_new
        count += 1

    x = [round(x[i], 4) for i in range(n)]
    return x


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


def simpleIteration(A, n, b):
    beta = [0 for _ in range(n)]
    alpha = [[0] * n for _ in range(n)]

    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = - A[i][j] / A[i][i]
            else:
                alpha[i][j] = 0

    # Simple iteration
    x = [[beta[i]] for i in range(n)]
    t = [[beta[i]] for i in range(len(beta))]
    x = multi(alpha,t)
    x = matrixsum(x,t)
    for i in range(n):
        x = multi(alpha, x)
        tmp = [[beta[i]] for i in range(len(beta))]
        x = matrixsum(x, tmp)

    x = [round(x[i][0], 7) for i in range(n)]
    print(x)


def show(A, n):
    for i in range(0, n):
        for j in range(0, n):
            print("\t", A[i][j], " ", end='')
        print("\n")


if __name__ == '__main__':
    print("Input demention of matrix: ")
    n = int(input())
    A = []
    print("Input matrix: ")
    for i in range(n):
        A.append(list(map(float, input().split())))

    print("Input answer: ")
    b = list(map(float, input().split()))
    print("Start:")
    show(A, n)
    #simpleIteration(A, n, b)
    print(seidel(A, b, 0.01))
