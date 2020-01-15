import AFT
import Solver
import torch
import numpy as np
from Model import Model


""" 激励加载 """
class Force_EX(object):
    def __init__(self,DOF, Nh, Nf):

        self.Nh = Nh
        self.Nf = Nf
        self.DOF = DOF


    def get_f(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        F[2, self.DOF] = 4
        return F.view(1, -1)

    def get_dfdw(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        # F[2, self.DOF] = 0
        return F.view(1, -1)


""" 非线性力加载 cub """
class Force_cubic(object):
    def __init__(self, DOF, k):
        self.k = k
        self.DOF = DOF

    def get_f(self, x, dx, d2x):
        f = torch.zeros_like(x, dtype=torch.double, requires_grad=True)
        f[0, :] = - self.k[0] * (x[0, :]) ** 3 - self.k[1] * (x[0, :] - x[1, :]) ** 3
        f[1, :] = - self.k[1] * (x[1, :] - x[0, :]) ** 3 - self.k[2] * (x[1, :] - x[2, :]) ** 3
        f[2, :] = - self.k[2] * (x[2, :] - x[1, :]) ** 3
        return f


""" 非线性力加载 cub """
class Force_gap(object):
    def __init__(self, DOF, g, k):
        self.g = g
        self.k = k
        self.DOF = DOF

    def get_f(self, x, dx, d2x):
        f = torch.zeros_like(x, dtype=torch.double, requires_grad=True)
        f[0, :] = self.gap_double(x[0], self.k[0], self.g[0])
        f[1, :] = self.gap_double(x[1], self.k[1], self.g[1])
        f[2, :] = self.gap_double(x[2], self.k[2], self.g[2])
        return f

    def gap_double(self, x, k, g):
        f = k * torch.relu(x-g) - k * torch.relu(-x-g)
        return f




""" 主程序"""

if __name__ == "__main__":
    Nt = 512
    Nh = 3
    Nf = 3

    K = torch.DoubleTensor([[7e3, -2e3, 0], [-2e3, 6e3, -4e3], [0, -4e3, 4e3]])
    M = torch.DoubleTensor([[2, 0, 0], [0, 1, 0],[0, 0, 1.5]])
    C = torch.DoubleTensor([[2.8, -0.3, 0], [-0.3, 0.38, 0.08], [0, -0.08, 0.08]])

    model = Model(M, C, K, Nh)

    import scipy.linalg as scp
    freq = np.sqrt(scp.eigh(K.numpy(), M.numpy(), eigvals_only=True)) / 2 / np.pi
    print("frequencies:   " + str(freq))

    # force_nl = Force_cubic(DOF=(0, 1, 2), k=[40000, 40000, 40000])
    force_nl = Force_gap(DOF=(0, 1, 2), k=[10000, 10000, 10000], g=[2e-3, 4e-3, 6e-3])
    force_ex = Force_EX(DOF=(0, 1, 2), Nh=Nh, Nf=Nf)
    aft = AFT.AFT(Nh=Nh, Nt=Nt, Nf=Nf, force=force_nl)


    w1=(20+0)*2*np.pi
    w2=(20+0.2)*2*np.pi

    N_solver = Solver.Newton_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, max_iter=20)

    X1 = N_solver.solve(w=w1)
    X2 = N_solver.solve(w=w2)
    ws = [w1, w2]
    Xs = [X1, X2]

    aft.process(X1)
    q10 = float(torch.max(aft.x_time[0, :].detach().abs()))
    q11 = float(torch.max(aft.x_time[1, :].detach().abs()))
    q12 = float(torch.max(aft.x_time[2, :].detach().abs()))
    aft.process(X2)
    q20 = float(torch.max(aft.x_time[0, :].detach().abs()))
    q21 = float(torch.max(aft.x_time[1, :].detach().abs()))
    q22 = float(torch.max(aft.x_time[2, :].detach().abs()))
    Qs0 = [q10, q20]
    Qs1 = [q11, q21]
    Qs2 = [q12, q22]

    weight = 1.e-3
    ds = torch.sqrt((weight * (w2 - w1)) ** 2 + torch.norm(X2 - X1) ** 2)

    A_solver = Solver.Arclen_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, arc_len=ds,
        direction=-1, max_step=10, max_iter=10, weight_w=weight, )


    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set(color_codes=True)
    plt.ion()
    plt.figure(figsize=(15, 12))

    for i in range(5000):
        # w,  X = A_solver.solve(ws[-2], ws[-1], Xs[-2], Xs[-1])

        w, X = A_solver.solve(ws[-1], Xs[-1])
        aft.process(X)

        ws.append(w)
        Xs.append(X)
        Qs0.append(float(torch.max(aft.x_time[0, :].detach().abs())))
        Qs1.append(float(torch.max(aft.x_time[1, :].detach().abs())))
        Qs2.append(float(torch.max(aft.x_time[2, :].detach().abs())))

        plt.clf()
        plt.subplot(311)
        plt.plot(np.array(ws)/np.pi/2, Qs0)
        plt.scatter(ws[-1]/np.pi/2, Qs0[-1], c='r')
        plt.semilogy()
        plt.grid(True)
        plt.pause(0.001)

        plt.subplot(312)
        plt.plot(np.array(ws) / np.pi / 2, Qs1)
        plt.scatter(ws[-1] / np.pi / 2, Qs1[-1], c='r')
        plt.semilogy()
        plt.grid(True)
        plt.pause(0.001)

        plt.subplot(313)
        plt.plot(np.array(ws) / np.pi / 2, Qs2)
        plt.scatter(ws[-1] / np.pi / 2, Qs2[-1], c='r')
        plt.semilogy()
        plt.grid(True)
        plt.pause(0.001)


        print(str(i) + " :   " + str(w.detach().numpy()) + "   " + str(Qs0[-1]))

        if w <= 0:
            break

    plt.savefig("result_gap.svg")
    plt.ioff()
    plt.show()