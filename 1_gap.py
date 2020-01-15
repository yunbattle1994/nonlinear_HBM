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
        F[2, self.DOF] = 1
        return F.view(1, -1)

    def get_dfdw(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        F[2, self.DOF] = 0
        return F.view(1, -1)


""" 非线性力加载 gap """
class Force_gap(object):
    def __init__(self, DOF, g, k):
        self.g = g
        self.k = k
        self.DOF = DOF

    def get_f(self, x):
        f = torch.relu(self.k * (x - self.g))
        return f


""" 非线性力加载 duffin """
class Force_cubic(object):
    def __init__(self, DOF, k):
        self.k = k
        self.DOF = DOF

    def get_f(self, x):
        f = self.k * x ** 3
        return f





""" 主程序"""

if __name__ == "__main__":
    Nt = 2048
    Nh = 5
    Nf = 2

    K = torch.DoubleTensor([[3.0e4, -1.5e4], [-1.5e4, 1.5e4]])
    M = torch.DoubleTensor([[1, 0], [0, 1]])
    C = torch.DoubleTensor([[2, -1], [-1, 1]])

    model = Model(M, C, K, Nh)

    force_nl = Force_gap(DOF=(0, ), g=0.0001, k=1e7)
    # force_nl = Force_cubic(DOF=(0, ), k=5e9)
    force_ex = Force_EX(DOF=(0, ), Nh=Nh, Nf=Nf)
    aft = AFT.AFT(Nh=Nh, Nt=Nt, Nf=Nf, force=force_nl)


    w1=(50+0)*2*np.pi
    w2=(50+1)*2*np.pi

    N_solver = Solver.Newton_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, max_iter=20)

    X1 = N_solver.solve(w=w1)
    X2 = N_solver.solve(w=w2)
    ws = [w1, w2]
    Xs = [X1, X2]

    aft.process(X1)
    q1 = float(torch.max(aft.x_time.detach()))
    aft.process(X2)
    q2 = float(torch.max(aft.x_time.detach()))
    Qs = [q1, q2]

    weight = 1.e-7
    ds = torch.sqrt((weight * (w2 - w1)) ** 2 + torch.norm(X2 - X1) ** 2)

    A_solver = Solver.Arclen_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, arc_len=ds,
        direction=-1, max_step=10, max_iter=10, weight_w=weight, )


    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set(color_codes=True)
    plt.ion()
    plt.figure(figsize=(15, 8))

    for i in range(5000):
        # w,  X = A_solver.solve(ws[-2], ws[-1], Xs[-2], Xs[-1])

        w, X = A_solver.solve(ws[-1], Xs[-1])
        aft.process(X)

        ws.append(w)
        Xs.append(X)
        Qs.append(float(torch.max(aft.x_time.detach())))

        plt.clf()
        plt.plot(ws, Qs)
        plt.scatter(ws[-1], Qs[-1], c='r')
        plt.grid(True)
        plt.pause(0.001)

        print(str(i) + " :   " + str(w.detach().numpy()) + "   " + str(Qs[-1]))

        if w <= 0:
            break

    plt.savefig("result_gap.svg")
    plt.ioff()
    plt.show()