import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.exp(x) - 2 * x - 2

def df(x):
    return math.exp(x) - 2

def ddf(x):
    return math.exp(x)

def phi(x):
    return math.log(2 * x + 2)

def dphi(x):
    return 2 / (2 * x + 2)

def simpleIteration(phi, dphi, a, b, eps = 0.001):
    q = min(abs(dphi(a)), abs(dphi(b)))
    x = (a + b) / 2
    k = 0
    go = True

    while go:
        k += 1
        x_cur = phi(x)

        print(f'x: {x_cur}, k: {k}, q/(1-q)*|x_cur - x|: {q * abs(x_cur - x) / (1 - q)}')
        if (q * abs(x_cur - x) / (1 - q)) <= eps:
            go = False

        x = x_cur
        if (k == 10):
            break

def newton(f, df, x0, eps = 0.001):
    x = x0
    k = 0
    go = True
    while go:
        k += 1
        x_cur = x - f(x) / df(x)
        print(f'x: {x_cur}, k: {k}, |x_cur - x|: {abs(x_cur - x)}')
        if abs(x_cur - x) <= eps:
            go = False

        x = x_cur

def show(f, df, x, file = None, step = 0.5, ddf = None):
    X = np.arange(x[0], x[-1], step)
    Y = [f(i) for i in X]
    dY = [df(i) for i in X]

    if ddf:
        ddY = [ddf(i) for i in X]

    fig, axis = plt.subplots()
    axis.plot(X, Y, label='f')
    axis.plot(X, dY, label='df')

    if ddf:
        axis.plot(X, ddY, label='ddf')

    axis.legend(loc='upper right')
    axis.grid()

    if file:
        fig.savefig(file)
        print(f'File {file} was saved correctly')
        plt.close(fig)

    plt.show()


if __name__ == '__main__':
    print("My function is: e^x - 2*x - 2")
    print("Simple iteration:")
    simpleIteration(phi, dphi, 0, 2)
    print("Newton method:")
    newton(f, df, 1.1)
    show(f, df, [0, 2], step=0.1, ddf=ddf)
    show(phi, dphi, [0, 2], step=0.1)
    # e^x - 2*x - 2 = 0