def tdma(matrix, n, d):
    size = int(n)
    a = []
    b = []
    c = []
    cnt = 0
    for i in range(size):
        for j in range(size):
            if (i == j):
                b.append(float(matrix[i][j]))
        if (cnt != size - 1):
            c.append(float(matrix[cnt][cnt + 1]))
            a.append(float(matrix[cnt + 1][cnt]))
            cnt += 1

    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)

    y     = [0 for i in range(size)]
    alpha = [0 for i in range(size)]
    beta  = [0 for i in range(size)]

    # Pre phaze
    y[0] = b[0]
    alpha[0] = -c[0] / y[0]
    beta[0]  = d[0] / y[0]

    # Main cycle
    for i in range(1, size):
        y[i] = b[i] + alpha[i - 1] * a[i - 1]
        if (i != size -1):
            alpha[i] = -c[i] / y[i]
        beta[i]  = (d[i] - beta[i - 1] * a[i -1]) / y[i]

    # Reverse cycle
    x = [0 for i in range(n)]
    x[-1] = beta[-1]
    for i in range(size - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    # Result of equation
    print("x =",x)

def show(A, n):
    print("Start matrix")
    for i in range(n):
        for j in range(n):
            print("\t", A[i][j], " ", end = "")
        print("\n")

if __name__ == '__main__':
    n = int(input())
    matrix = []
    for i in range(n):
        matrix.append(list((input().split())))

    d = list(map(float, input().split()))

    show(matrix, n) #Start matrix
    tdma(matrix, n, d) #Start algo
# 2 -1 0
# 5 4 2
# 0 1 -3
# 3 6 2

# 6 -5 0 0 0
# -6 16 9 0 0
# 0 9 17 -3 0
# 0 0 8 22 -8
# 0 0 0 6 -13
# -58 161 -114 -90 -55
