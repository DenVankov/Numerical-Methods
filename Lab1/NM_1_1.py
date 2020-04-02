from random import randint

class MatrixException(Exception):
    pass

def getCofactor(A, tmp, p, q, n): # Function to get cofactor of A[p][q] in tmp[][]
    i = 0
    j = 0
    # Copying into temporary matrix only those element which are not in given row and column
    for row in range(n):
        for col in range(n):
            if (row != p and col != q):
                tmp[i][j] = A[row][col]
                j += 1

                if (j == n - 1):
                    j = 0
                    i += 1

def determinant(A, n):
    D = 0
    if (n == 1):
        return A[0][0]

    tmp = [[0]*n for _ in range(n)] # Cofactors
    sign = 1 # Plus or minus

    for i in range(n):
        getCofactor(A, tmp, 0, i, n)
        D += sign * A[0][i] * determinant(tmp, n - 1)

        sign = -sign

    return D

def adjoin(A, n): # Сопряженная
    adj = [[0] * n for _ in range(n)]
    if (n == 1):
        adj[0][0] = 1
        return

    tmp = [[0] * n for _ in range(n)]  # Cofactors
    sign = 1 # Plus or minusn
    for i in range(n):
        for j in range(n):
            getCofactor(A, tmp, i, j, n) # Cofactor A[i][j]

            if ((i + j)%2 == 0):
                sign = 1
            else:
                sign = -1

            adj[j][i] = (sign) * (determinant(tmp, n - 1)) # transpose of the cofactor matrix

    return adj

def inverse(A, B, n):
    det = determinant(A, n)
    if (det == 0):
        print("Singular matrix, can't find its inverse")
        return False

    adj = adjoin(A, n)

    for i in range(n):
        for j in range(n):
            B[i][j] = adj[i][j] / det

    return True

def transpose(A, n):
    B = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            B[i][j] = A[j][i]
    return B

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
                    sum = round(sum + M1[k][i] * M2[i][j],8)
                tmp.append(sum)
                sum = 0
            ans.append(tmp)
            tmp = []
    return ans

def show(A, n):
    for i in range(0, n):
        for j in range(0,n):
            print("\t",A[i][j]," ", end="")
        print("\n")


def LUP_solve(L, U, pi, b, n):
    x = [0 for i in range(n)]
    y = [0 for i in range(n)]

    for i in range(n):
        summ = 0
        for j in range(i):
            summ += L[i][j] * y[j]

        y[i] = b[pi[i]] - summ

    for i in range(n - 1, -1, -1):
        sec_summ = 0
        for j in range(i+1, n):
            sec_summ += U[i][j] * x[j]

        x[i] = (y[i] - sec_summ) / U[i][i]

    x = [round(x[i], 5) for i in range(len(x))]
    return x

def LUP_decompose(A, n):
    pi = [i for i in range(n)]

    for k in range(n):
        p = 0
        for i in range(k, n):
            if (abs(A[i][k]) > p):
                p = abs(A[i][k])
                tmp_k = i
        if p == 0:
            raise MatrixException('Matrix is degenerate')
        pi[k], pi[tmp_k] = pi[tmp_k], pi[k]

        for i in range(n):
            A[k][i], A[tmp_k][i] = A[tmp_k][i], A[k][i]
        for i in range(k+1, n):
            A[i][k] = A[i][k] / A[k][k]
            for j in range(k+1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
    return pi

if (__name__ == '__main__'):
    print("Input demention of matrix: ")
    n = int(input())
    A = [[] * n for i in range(0,n)]

    for i in range(0,n):
        for j in range(0,n):
            A[i].append(int(input()))

    print("Start:")
    show(A,n)
    print("The Adjoint is :\n")


    inv = [[0] * n for _ in range(n)]

    adj = adjoin(A, n)
    show(adj, n)

    print("The Inverse is :\n")
    if (inverse(A, inv, n)):
        show(inv, n)

    pi = LUP_decompose(A,n)
    print("A after LUP:")
    show(A,n)

    L = [[0] * n for i in range(0,n)]
    U = [[0] * n for i in range(0,n)]

    for i in range(n):
        L[i][i] = 1
        for j in range(n):
            if (j < i):
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]
    print("L:")
    show(L,n)
    print("U:")
    show(U,n)
    print("A:\n")
    R = multi(L, U)
    show(R, n)
    print("Solving:")
    b = [-23,39,-7,30]
    print(LUP_solve(L,U,pi,b,n))

# 4
# 2
# 3
# 1
# 5
# 6
# 13
# 5
# 19
# 2
# 19
# 10
# 23
# 4
# 10
# 11
# 31
###
# 4
# 1
# 2
# -1
# -7
# 8
# 0
# -9
# -3
# 2
# -3
# 7
# 1
# 1
# -5
# -6
# 8
