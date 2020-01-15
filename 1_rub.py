import AFT
import Solver
import torch
import numpy as np
from Model import Model

""" 非线性力加载 rub """
class Force_rub(object):
    def __init__(self, DOF, g, k, miu):
        self.g = g
        self.k = k
        self.miu = miu
        self.DOF = DOF

    def get_f(self, x, dx, d2x):
        f = torch.relu(self.k * (torch.norm(x, dim=0, keepdim=True) - self.g)) / torch.norm(x, dim=0, keepdim=True)
        M = torch.zeros((2, 2), dtype=torch.float64)
        M[0, 0] = 1.
        M[1, 0] = self.miu
        M[0, 1] = -self.miu
        M[1, 1] = 1.
        f = f * torch.matmul(M, x)
        return f

""" 激励加载 """
class Force_EX_rub(object):
    def __init__(self,DOF, Nh, Nf):

        self.Nh = Nh
        self.Nf = Nf
        self.DOF = DOF


    def get_f(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        F[2, self.DOF[0]] = 2 * 4.e-5 * w ** 2
        F[3, self.DOF[1]] = 2 * 4.e-5 * w ** 2

        return F.view(1, -1)

    def get_dfdw(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        F[2, self.DOF[0]] = 2 * 4.e-5 * w * 2
        F[3, self.DOF[1]] = 2 * 4.e-5 * w * 2
        return F.view(1, -1)


""" 主程序"""

if __name__ == "__main__":
    Nt = 4096
    Nh = 5
    Nf = 2

    K = torch.DoubleTensor([[6.0e4, 0], [0, 6.0e4]])
    M = torch.DoubleTensor([[2, 0], [0, 2]])
    C = torch.DoubleTensor([[80, 0], [0, 80]])
    model = Model(M, C, K, Nh=Nh)


    force_nl = Force_rub(DOF=(0, 1), g=0.0001, k=8e8, miu=0.7)
    # force_nl = Force_cubic(DOF=0, k=5e9)
    force_ex = Force_EX_rub(DOF=(0, 1), Nh=Nh, Nf=Nf)
    aft = AFT.AFT(Nh=Nh, Nt=Nt, Nf=Nf, force=force_nl)


    w1=(1+0)*2*np.pi
    w2=(1+1)*2*np.pi

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

    weight = 1.e-6
    ds = torch.sqrt((weight * (w2 - w1)) ** 2 + torch.norm(X2 - X1) ** 2)

    A_solver = Solver.Arclen_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, arc_len=ds,
                                    max_step=10, max_iter=11, weight_w=weight)


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

    plt.savefig("result_rub.svg")
    plt.ioff()
    plt.show()