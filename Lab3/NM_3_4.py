target = '''X* = 0.2
i:         0    1        2         3       4
Xi:     -0.2  0.0      0.2       0.4     0.6
Yi: -0.20136  0.0  0.20136   0.41152  0.64350
'''


def find_interval(x_i, x0):
    for i in range(len(x_i) - 1):
        if x_i[i] <= x0 <= x_i[i + 1]:
            return i


def first_derivative(x_i, y_i, x0):
    i = find_interval(x_i, x0)
    left = (y_i[i + 1] - y_i[i]) / (x_i[i + 1] - x_i[i])
    right = ((y_i[i + 2] - y_i[i + 1]) / (x_i[i + 2] - x_i[i + 1]) - left) / \
            (x_i[i + 2] - x_i[i]) * (2 * x0 - x_i[i] - x_i[i + 1])
    return left + right


def second_derivative(x_i, y_i, x0):
    i = find_interval(x_i, x0)
    left = (y_i[i + 1] - y_i[i]) / (x_i[i + 1] - x_i[i])
    right = 2 * ((y_i[i + 2] - y_i[i + 1]) / (x_i[i + 2] - x_i[i + 1]) - left) / \
            (x_i[i + 2] - x_i[i])
    return right


if __name__ == '__main__':
    print(target)
    x0 = 0.2
    x_i = [-0.2, 0.0, 0.2, 0.4, 0.6]
    y_i = [-0.20136, 0.0, 0.20136, 0.41152, 0.64350]
    print(f'First derivative = {round(first_derivative(x_i, y_i, x0), 6)}')
    print(f'Second derivative = {round(second_derivative(x_i, y_i, x0), 6)}')
