import math
import numpy as np
import matplotlib.pyplot as plt


target = '''
y(0) = 1
y'(0) = 0
x = [0, 1]
h = 0.1
Exact function: (1 + x) * math.exp(-x^2)
Cauchy problem: y'' + 4*x*y' + (4*x^2 + 2)*y = 0
'''


def cauchyProblem(x, y, y_der):
    return -4 * x * y_der - (4 * x ** 2 + 2) * y


def pureFunction(x):
    return (1 + x) * math.exp(-x ** 2)


def g(x, y, k):
    return k


def analytical(f, a, b, h):
    x = [i for i in np.arange(a, b + h, h)]
    y = [f(i) for i in x]
    return x, y


def euler(f, a, b, h, y0, y_der):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    y = [y0]
    k = y_der
    for i in range(n):
        k += h * f(x[i], y[i], k)
        y.append(y[i] + h * g(x[i], y[i], k))

    return x, y


def rungeKut(f, a, b, h, y0, y_der):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    y = [y0]
    k = [y_der]
    for i in range(n):
        K1 = h * g(x[i], y[i], k[i])
        L1 = h * f(x[i], y[i], k[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, k[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, k[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, k[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, k[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, k[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, k[i] + L3)
        y.append(y[i] + (K1 + 2 * K2 + 2 * K3 + K4) / 6)
        k.append(k[i] + (L1 + 2 * L2 + 2 * L3 + L4) / 6)
    return x, y, k


def adams(f, x, y, k, h):
    n = len(x)
    x = x[:4]
    y = y[:4]
    k = k[:4]
    for i in range(3, n - 1):
        k.append(k[i] + h * (55 * f(x[i], y[i], k[i]) -
                             59 * f(x[i - 1], y[i - 1], k[i - 1]) +
                             37 * f(x[i - 2], y[i - 2], k[i - 2]) -
                              9 * f(x[i - 3], y[i - 3], k[i - 3])) / 24)

        y.append(y[i] + h * (55 * g(x[i], y[i], k[i]) -
                             59 * g(x[i - 1], y[i - 1], k[i - 1]) +
                             37 * g(x[i - 2], y[i - 2], k[i - 2]) -
                              9 * g(x[i - 3], y[i - 3], k[i - 3])) / 24)
        x.append(x[i] + h)
    return x, y


def rungeRobert(dict_):

    k = dict_[0]['h'] / dict_[1]['h']
    euler = []
    for i in range(len(dict_[0]['Euler']['y'])):
        euler.append(round(abs(dict_[0]['Euler']['y'][i] - dict_[1]['Euler']['y'][i]) / (k ** 1 - 1), 6))

    runge = []
    for i in range(len(dict_[0]['Runge']['y'])):
        runge.append(round(abs(dict_[0]['Runge']['y'][i] - dict_[1]['Runge']['y'][i]) / (k ** 4 - 1), 6))

    adams = []
    for i in range(len(dict_[0]['Adams']['y'])):
        adams.append(round(abs(dict_[0]['Adams']['y'][i] - dict_[1]['Adams']['y'][i]) / (k ** 4 - 1), 6))

    return {'Euler': euler, 'Runge': runge, 'Adams': adams}


def show(res, pure, h):
    n = len(res)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.scatter(res[i]["Euler"]["x"], res[i]["Euler"]["y"], color='r', alpha=0.4, label='Euler method')
        plt.plot(res[i]["Euler"]["x"], res[i]["Euler"]["y"], color='r', alpha=0.4)
        plt.scatter(res[i]["Runge"]["x"], res[i]["Runge"]["y"], color='b', alpha=0.4, label='Runge-Kutta method')
        plt.plot(res[i]["Runge"]["x"], res[i]["Runge"]["y"], color='b', alpha=0.4)
        plt.scatter(res[i]["Adams"]["x"], res[i]["Adams"]["y"], color='g', alpha=0.4, label='Adams method')
        plt.plot(res[i]["Adams"]["x"], res[i]["Adams"]["y"], color='g', alpha=0.4)
        plt.scatter(pure[i][0], pure[i][1], color='k', alpha=0.4, label='Pure function')
        plt.plot(pure[i][0], pure[i][1], color='k', alpha=0.4)

        plt.legend(loc='best')
        plt.title('h{0} = '.format(i + 1) + str(h[i]))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    plt.show()


if __name__ == '__main__':
    a = 0
    b = 1
    h = 0.1
    y0 = 1
    y_der = 1
    print(target)
    res = []
    pure = []
    steps = [h, h/2]
    for h in steps:
        print("Euler method:")
        x_eul, y_eul = euler(cauchyProblem, a, b, h, y0, y_der)
        for x, y in zip(x_eul, y_eul):
            print(f'x = {round(x, 4)}, y = {y}')

        print("Runge-Kutta method:")
        x_rung, y_rung, k_rung = rungeKut(cauchyProblem, a, b, h, y0, y_der)
        for x, y in zip(x_rung, y_rung):
            print(f'x = {round(x, 4)}, y = {y}')

        print("Adams method:")
        x_ad, y_ad = adams(cauchyProblem, x_rung, y_rung, k_rung, h)
        for x, y in zip(x_ad, y_ad):
            print(f'x = {round(x, 4)}, y = {y}')

        print("Analytical method:")
        x_anal, y_anal = analytical(pureFunction, a, b, h)
        for x, y in zip(x_anal, y_anal):
            print(f'x = {round(x, 4)}, y = {y}')

        pure.append((x_anal, y_anal))
        res.append({
                    "h": h,
                    "Euler": {'x': x_eul, 'y': y_eul},
                    "Runge": {'x': x_rung, 'y': y_rung},
                    "Adams": {'x': x_ad, 'y': y_ad},
                    })

    err = rungeRobert(res)
    print("Euler error: {0}".format(err['Euler']))
    print("Runge error: {0}".format(err['Runge']))
    print("Adams error: {0}".format(err['Adams']))

    show(res, pure, steps)
