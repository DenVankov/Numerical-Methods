from math import sqrt

def matrix_print(matrix, header=None):
    if header:
        print(header, '\n')
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print('{:6.2f}'.format(float(matrix[i][j])), end=' ')
        print()
    print()


def vector_print(vec, header=None):
    if header:
        print(header, '\n')
    for i in range(len(vec)):
        print('{:6.2f}'.format(float(vec[i])), end=' ')
    print('\n')


def mv_mult(size, matrix, vector):
    R = [0.0 for _ in range(size)]
    for i in range(size):
        for j in range(size):
            R[i] += matrix[i][j] * vector[j]
    return R


def mm_mult(matrix1, matrix2):
    C = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix1[0])):
                C[i][j] += matrix1[i][k] * matrix2[k][j]
    return C

def vv_substr(size, vec1, vec2):
    return [vec1[i] - vec2[i] for i in range(size)]


def mm_substr(matrix1, matrix2):
    return [[matrix1[i][j] - matrix2[i][j] for i in range(
        len(matrix1[j]))] for j in range(len(matrix1))]

def vv_add(size, vec1, vec2):
    return [vec1[i] + vec2[i] for i in range(size)]


def v_norm(size, vec):
    return sum(abs(vec[i]) for i in range(size))


def euclid_v_norm(vec):
    return sqrt(sum([x**2 for x in vec]))


def m_norm(size, matrix):
    v = [abs(sum(matrix[i][j] for i in range(size))) for j in range(size)]
    return max(v)


def diagonal(size):
    return [[0 if i != j else 1 for i in range(size)] for j in range(size)]