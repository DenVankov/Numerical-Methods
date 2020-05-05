import numpy as np

target = '''
y = x / ((2 * x + 7) * (3 * x + 4))
x0 = -1
x_k = 1
h1 = 0.5
h2 = 0.25
'''


def func(x):
    return x / ((2 * x + 7) * (3 * x + 4))


def get_x(x0, x, step):
    return [i for i in np.arange(x0, x + step, step)]


def get_y(x):
    return [func(i) for i in x]


def rectangle(x, h):
    return h * sum([func((x[i] + x[i + 1]) / 2) for i in range(len(x) - 1)])


def trapeze(x, h):
    y = get_y(x)
    return h * (y[0] / 2 + sum([y[i] for i in range(1, len(y) - 2)]) + y[len(y) - 1] / 2)


def simpson(x, h):
    y = get_y(x)
    return h / 3 * (y[0] +
                    sum([4 * y[i] for i in range(1, len(y) - 1, 2)]) +
                    sum([2 * y[i] for i in range(2, len(y) - 2, 2)]) +
                    y[len(y) - 1])


def runge_Romberg(res, true_value):
    k = res[1]['h'] / res[0]['h']
    val_rec = [res[0]['rec'] + (res[0]['rec'] - res[1]['rec']) / (k ** 2 - 1),
               abs(res[0]['rec'] + (res[0]['rec'] - res[1]['rec']) / (k ** 2 - 1) - true_value)]

    val_trp = [res[0]['trp'] + (res[0]['trp'] - res[1]['trp']) / (k ** 2 - 1),
               abs(res[0]['trp'] + (res[0]['trp'] - res[1]['trp']) / (k ** 2 - 1) - true_value)]

    val_smp = [res[0]['smp'] + (res[0]['smp'] - res[1]['smp']) / (k ** 4 - 1),
               abs(res[0]['smp'] + (res[0]['smp'] - res[1]['smp']) / (k ** 4 - 1) - true_value)]
    return {'rec': val_rec, 'trp': val_trp, 'smp': val_smp}


if __name__ == '__main__':
    x0 = -1
    x = 1
    h = [0.5, 0.25]
    true_value = -0.04133027217305138
    res = []
    for h_i in h:
        print('Current h =', h_i)
        X = get_x(x0, x, h_i)
        print(f'x = {X}')
        y = get_y(X)
        print(f'y = {y}')

        print('Rectangle method:')
        res_rec = rectangle(X, h_i)
        print(f'Value = {res_rec}')

        print('Trapeze method:')
        res_trp = trapeze(X, h_i)
        print(f'Value = {res_trp}')

        print('Simpson method:')
        res_smp = simpson(X, h_i)
        print(f'Value = {res_smp}')
        print()

        res.append({"h": h_i,
                    "rec": res_rec,
                    "trp": res_trp,
                    "smp": res_smp})

    err = runge_Romberg(res, true_value)

    print('All errors:')
    print(f'True value of integral: {true_value}')
    print('Rectangle specified value: {}, rectangle absolute error: {}'.format(err['rec'][0], err['rec'][1]))
    print('Trapeze   specified value: {}, trapeze absolute error {}'.format(err['trp'][0], err['trp'][1]))
    print('Simpson   specified value: {}, Simpson absolute error {}'.format(err['smp'][0], err['smp'][1]))
