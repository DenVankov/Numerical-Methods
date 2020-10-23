import numpy as np
from tools import tma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def get_zeros(N, K):
    lst = [np.zeros(N) for _ in range(0, 4)]
    lst.append(np.zeros((K, N)))
    return lst


class Data:
    def __init__(self, params):
        self.l = params['l']
        self.f = params['f']
        self.psi = params['psi']
        self.phi0 = params['phi0']
        self.phi1 = params['phi1']
        self.bound_type = params['bound_type']
        self.solve = params['solution']


class ParabolicSolver:
    def __init__(self, params, equation_):
        self.h = 0
        self.tau = 0
        self.sigma = 0
        self.data = Data(params)
        try:
            self.solve_aux = getattr(self, f'{equation_}_solver')
        except:
            raise Exception("This type does not exist")

    def solve(self, N, K, T):
        self.h = self.data.l / N
        self.tau = T / K
        self.sigma = self.tau / (self.h ** 2)
        return self.solve_aux(N, K, T)

    def analyticSolve(self, N, K, T):
        self.h = self.data.l / N
        self.tau = T / K
        u = np.zeros((K, N))
        for i in range(K):
            for j in range(N):
                u[i][j] = self.data.solve(j * self.h, i * self.tau)
        return u

    def calculate(self, a, b, c, d, u, N, k):
        for j in range(1, N - 1):
            a[j] = self.sigma
            b[j] = -(1 + 2 * self.sigma)
            c[j] = self.sigma
            d[j] = -u[k - 1][j] - self.tau * self.data.f(j * self.h, k * self.tau)

        if self.data.bound_type == 'a1p1':
            a[0] = 0
            b[0] = -(1 + 2 * self.sigma)
            c[0] = self.sigma
            d[0] = -(u[k - 1][0] + self.sigma * self.data.phi0(k * self.tau))
            a[-1] = self.sigma
            b[-1] = -(1 + 2 * self.sigma)
            c[-1] = 0
            d[-1] = -(u[k - 1][-1] + self.sigma * self.data.phi1(k * self.tau))
        elif self.data.bound_type == 'a1p2':
            a[0] = 0
            b[0] = -(1 + 2 * self.sigma)
            c[0] = self.sigma
            d[0] = -(u[k - 1][0] + self.sigma * self.data.phi0(k * self.tau)) - \
                   self.tau * self.data.f(0, k * self.tau)
            a[-1] = self.sigma
            b[-1] = -(1 + 2 * self.sigma)
            c[-1] = 0
            d[-1] = -(u[k - 1][-1] + self.sigma * self.data.phi1(k * self.tau)) - \
                    self.tau * self.data.f((N - 1) * self.h, k * self.tau)
        elif self.data.bound_type == 'a1p3':
            a[0] = 0
            b[0] = -(1 + 2 * self.sigma)
            c[0] = self.sigma
            d[0] = -((1 - self.sigma) * u[k - 1][1] + self.sigma / 2 * u[k - 1][0]) - self.tau \
                   * self.data.f(0, k * self.tau) - self.sigma * self.data.phi0(
                k * self.tau)
            a[-1] = self.sigma
            b[-1] = -(1 + 2 * self.sigma)
            c[-1] = 0
            d[-1] = self.data.phi1(k * self.tau) + self.data.f((N - 1) * self.h, k * self.tau) \
                    * self.h / (2 * self.tau) * u[k - 1][-1]

    def implicit_solver(self, N, K, T):
        lst = get_zeros(N, K)
        a = lst[0]
        b = lst[1]
        c = lst[2]
        d = lst[3]
        u = lst[4]

        for i in range(1, N - 1):
            u[0][i] = self.data.psi(i * self.h)
        u[0][-1] = 0

        for k in range(1, K):
            self.calculate(a, b, c, d, u, N, k)
            u[k] = tma(a, b, c, d)

        return u

    def explicit_solver(self, N, K, T):
        u = np.zeros((K, N))
        for j in range(1, N - 1):
            u[0][j] = self.data.psi(j * self.h)

        for k in range(1, K):
            u[k][0] = self.data.phi0(k * self.tau)
            for j in range(1, N - 1):
                u[k][j] = self.sigma * u[k - 1][j + 1] + \
                          (1 - 2 * self.sigma) * u[k - 1][j] + \
                          self.sigma * u[k - 1][j - 1] + \
                          self.tau * self.data.f(j * self.h, k * self.tau)

            if self.data.bound_type == 'a1p1':
                u[k][-1] = u[k][-2] + self.data.phi1(k * self.tau) * self.h
            elif self.data.bound_type == 'a1p2':
                # (self.data.phi1(k * self.tau) * 2 * self.h - u[k][-3] + 4 * u[k][-2]) / 3
                u[k][-1] = self.data.phi1(k * self.tau)
            elif self.data.bound_type == 'a1p3':
                u[k][-1] = (self.data.phi1(k * self.tau) + u[k][-2] / self.h + 2 * self.tau * u[k - 1][-1] / self.h) / \
                           (1 / self.h + 2 * self.tau / self.h)
        return u

    def crank_nicolson_solver(self, N, K, T):
        theta = 0.5
        lst = get_zeros(N, K)
        a = lst[0]
        b = lst[1]
        c = lst[2]
        d = lst[3]
        u = lst[4]
        for i in range(1, N - 1):
            u[0][i] = self.data.psi(i * self.h)

        for k in range(1, K):
            self.calculate(a, b, c, d, u, N, k)

            tmp_imp = tma(a, b, c, d)

            tmp_exp = np.zeros(N)
            tmp_exp[0] = self.data.phi0(self.tau)
            for j in range(1, N - 1):
                tmp_exp[j] = self.sigma * u[k - 1][j + 1] + (1 - 2 * self.sigma) * u[k - 1][j] + \
                             self.sigma * u[k - 1][j - 1] + self.tau * self.data.f(j * self.h, k * self.tau)
            tmp_exp[-1] = self.data.phi1(self.tau)

            for j in range(N):
                u[k][j] = theta * tmp_imp[j] + (1 - theta) * tmp_exp[j]

        return u


def draw(dict_, N, K, T, save_file="plot.png"):
    fig = plt.figure(figsize=plt.figaspect(0.7))

    # Make data
    x = np.arange(0, np.pi, np.pi / N)
    t = np.arange(0, T, T / K)
    x, t = np.meshgrid(x, t)
    z1 = np.array(dict_['numerical'])
    z2 = np.array(dict_['analytic'])

    # Plot the surface.
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.title('numerical')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('t', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    ax.plot_surface(x, t, z1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=True)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('t', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    plt.title('analytic')
    surf = ax.plot_surface(x, t, z2, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    # # Customize the z axis
    # ax.set_zlim(-1.01, 1.01)

    # # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15)

    plt.savefig(save_file)
    plt.show()


def compare_error(dict_):
    error = [[abs(i - j) for i, j in zip(x, y)] for x, y in zip(dict_['numerical'], dict_['analytic'])]
    return error


data = {'equation_type': 'explicit', 'N': 15, 'K': 500, 'T': 1}

if __name__ == '__main__':
    equation_type = data['equation_type']
    N, K, T = int(data['N']), int(data['K']), int(data['T'])

    params = {
        'l': np.pi,
        'psi': lambda x: np.sin(x),
        'f': lambda x, t: 0.5 * np.exp(-0.5 * t) * np.cos(x),
        'phi0': lambda t: -np.exp(-0.5 * t),  # Wrong graphic if "+"
        'phi1': lambda t: -np.exp(-0.5 * t),
        'solution': lambda x, t: np.exp(-0.5 * t) * np.sin(x),
        'bound_type': 'a1p2',
    }

    var7 = ParabolicSolver(params, equation_type)
    ans = {
        'numerical': var7.solve(N, K, T).tolist(),
        'analytic': var7.analyticSolve(N, K, T).tolist()
    }

    # print(ans['numerical'])
    # print(ans['analytic'])

    draw(ans, N, K, T)

    error = compare_error(ans)
    avg_err = 0.0
    for i in error:
        for j in i:
            avg_err += j
        avg_err /= N

    print(error[0])
    print(error[int(K/2)])
    print(error[-1])
    print(f'Average error in each N: {avg_err}')
    print(f'Average error\t\t   : {avg_err / K}')

