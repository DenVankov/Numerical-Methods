import math
import numpy as np
import matplotlib.pyplot as plt

"""
x1 - cos(x2) = 1
x2 - lg(x1 - 1) = a
a = 3
"""


# First func
def f1(x1, x2):
    return x1 - math.cos(x2) - 1


def x1(x2):
    return math.cos(x2) + 1


def df1_x1(x1, x2):
    return 1


def df1_x2(x1, x2):
    return math.sin(x2)


# Second func
def f2(x1, x2):
    return x2 - math.log10(x1 + 1) - 3


def x2(x1):
    return math.log10(x1 + 1) + 3


def df2_x1(x1, x2):
    return -1 / ((x1 + 1) * math.log(10))


def df2_x2(x1, x2):
    return 1


function = {
    "f1": f1,
    "f2": f2,

    "df1_dx1": df1_x1,
    "df1_dx2": df1_x2,
    "df2_dx1": df2_x1,
    "df2_dx2": df2_x2
}


def norm(x, x_last):
    return max(abs(x[0] - x_last[0]), abs(x[1] - x_last[1]))


# determinant of a 2x2 matrix
def det2(m):
    return m[0][0] * m[1][1] - m[1][0] * m[0][1]


# recursive part
def recursive(m, some=None, prod=1):
    if some is None:
        some = []

    if len(m) == 1:
        some.append(m[0][0])
    elif len(m) == 2:
        some.append(det2(m) * prod)
    else:
        for index, elem in enumerate(m[0]):
            transpose = [list(a) for a in zip(*m[1:])]
            del transpose[index]
            minor = [list(a) for a in zip(*transpose)]
            some = recursive(minor, some, prod * m[0][index] * (-1) ** (index + 2))

    return some


def compute(x1, x2, func):
    a1 = [
        [func["f1"](x1, x2), func["df1_dx2"](x1, x2)],
        [func["f2"](x1, x2), func["df2_dx2"](x1, x2)]
        ]
    a2 = [
        [func["df1_dx1"](x1, x2), func["f1"](x1, x2)],
        [func["df2_dx1"](x1, x2), func["f2"](x1, x2)]
    ]

    jacobi = [
        [func["df1_dx1"](x1, x2), func["df1_dx2"](x1, x2)],
        [func["df2_dx1"](x1, x2), func["df2_dx2"](x1, x2)]
    ]

    det_j = recursive(jacobi)

    return x1 - recursive(a1)[0] / det_j[0], x2 - recursive(a2)[0] / det_j[0]


def printer(data):
    for title, value in data.items():
        print(f"| {title} == {value:7.4f} ", end='')
    print('|')


def newton(x1, x2, eps=0.01):
    x1_last, x2_last = x1, x2
    k = 0
    while True:
        k += 1
        x1_cur, x2_cur = compute(x1_last, x2_last, function)

        cur_eps = norm((x1_cur, x2_cur), (x1_last, x2_last))
        printer({"x1": x1_cur, "x2": x2_cur, "eps": cur_eps})

        if cur_eps <= eps:
            break
        x1_last = x1_cur
        x2_last = x2_cur

    return x1_cur, x2_cur


def phi1(x2):
    return math.cos(x2) + 1


def phi2(x1):
    return math.log10(x1 + 1) + 3


def dphi1_dx1(x1,x2):
    return 0


def dphi1_dx2(x2):
    return -math.sin(x2)


def dphi2_dx1(x1):
    return -1 / ((x1 + 1) * math.log(10))


def dphi2_dx2(x1, x2):
    return 0


phi = {
    "phi1": phi1,
    "phi2": phi2,

    "dphi1_dx1": dphi1_dx1,
    "dphi1_dx2": dphi1_dx2,
    "dphi2_dx1": dphi2_dx1,
    "dphi2_dx2": dphi2_dx2
}


def getQ(x1, x2, phi):
    return max(
        abs(phi["dphi1_dx1"](x1, x2)) + abs(phi["dphi1_dx2"](x2)),
        abs(phi["dphi2_dx1"](x1)) + abs(phi["dphi2_dx2"](x1, x2)))


def simpleIteration(x1, x2, eps = 0.01):
    x1_last, x2_last = x1, x2
    k = 0

    q = getQ(x1, x2, phi)
    print("q:", q)
    if q >= 1:
        print("Leave field of G")
        printer({"q":q})
    while True:
        k += 1
        x1_cur, x2_cur = phi1(x2_last), phi2(x1_last)

        cur_eps = abs((q / (1 - q) * norm((x1_cur, x2_cur), (x1_last, x2_last))))
        printer({"x1": x1_cur, "x2": x2_cur, "eps": cur_eps})

        if cur_eps <= eps:
            break

        x1_last = x1_cur
        x2_last = x2_cur
    return x1_cur, x2_cur


def plot_show(f1, f2, x, file = None, step = 0.1):
    X = np.arange(x[0], x[-1], step)
    Y1 = [f1(val) for val in X]
    Y2 = [f2(val) for val in X]

    fig, axis = plt.subplots()
    axis.set_title(f'Scatter x1 from x2')
    axis.plot(X, Y1, label='func1')
    axis.plot(Y2, X, label='func2')
    axis.legend(loc='upper right')
    axis.grid()

    if file:
        fig.savefig(file)
        print(f'Saved in {file}')
        plt.close(fig)

    plt.show()


if __name__ == '__main__':
    plot_show(x1, x2, [0, 5])
    plot_show(dphi1_dx2, dphi2_dx1, [0, 5])
    print("Iteration:")
    simpleIteration(2, 4, 0.001)
    print("Newton method:")
    newton(2, 4, 0.001)
