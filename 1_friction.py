import AFT
import Solver
import torch
import numpy as np
from Model import Model

import matplotlib.pyplot as plt
import seaborn

""" 激励加载 """
class Force_EX(object):
    def __init__(self,DOF, Nh, Nf):

        self.Nh = Nh
        self.Nf = Nf
        self.DOF = DOF


    def get_f(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        F[3, self.DOF[0]] = 100
        F[3, self.DOF[1]] = 100
        return F.view(1, -1)

    def get_dfdw(self, w):
        F = torch.zeros(2 * self.Nh, self.Nf, dtype=torch.float64)
        return F.view(1, -1)


from ForceNonlinear import Friction




""" 主程序"""

if __name__ == "__main__":
    Nt = 512
    Nh = 5
    Nf = 2

    K = torch.DoubleTensor([[40, 0], [0, 80]])
    M = torch.DoubleTensor([[1, 0], [0, 1]])
    C = torch.DoubleTensor([[0.4, 0], [0, 0.4]])

    model = Model(M, C, K, Nh)

    Cs = []
    for p in [-100, 50, 0, 100, 200, 300, 500, 1000, 2000]:

        # force_nl = Force_gap(DOF=(0, ), g=0.0001, k=1e6)
        force_nl = Friction(DOF=(0, 1), kx=100, ky=200, n0=p, mu=0.3)
        force_ex = Force_EX(DOF=(0, 1), Nh=Nh, Nf=Nf)
        aft = AFT.AFT(Nh=Nh, Nt=Nt, Nf=Nf, force=force_nl)


        w1 = (4+0.2)*2*np.pi
        w2 = (4+0.1)*2*np.pi

        N_solver = Solver.Newton_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, max_iter=20)

        X1 = N_solver.solve(w=w1)
        X2 = N_solver.solve(w=w2)
        ws = [w1, w2]
        Xs = [X1, X2]

        aft.process(X1)
        x00 = aft.f_time[0, :].detach().numpy()
        x01 = aft.f_time[1, :].detach().numpy()
        q01 = abs(float(torch.max(aft.x_time[0, :].detach().abs())))
        q11 = abs(float(torch.max(aft.x_time[1, :].detach().abs())))
        f00 = aft.f_time[0, :].detach().numpy()
        f01 = aft.f_time[1, :].detach().numpy()

        aft.process(X2)
        x10 = aft.f_time[0, :].detach().numpy()
        x11 = aft.f_time[1, :].detach().numpy()
        q02 = abs(float(torch.max(aft.x_time[0, :].detach().abs())))
        q12 = abs(float(torch.max(aft.x_time[1, :].detach().abs())))
        f10 = aft.f_time[0, :].detach().numpy()
        f11 = aft.f_time[1, :].detach().numpy()

        qs = [[q01, q02], [q11, q12]]
        xs = [[x00, x01], [x10, x11]]
        fs = [[f00, f01], [f10, f11]]

        weight = 1.e-1
        ds = torch.sqrt((weight * (w2 - w1)) ** 2 + torch.norm(X2 - X1) ** 2)

        A_solver = Solver.Arclen_Solver(model=model, force_ex=force_ex, force_nl=force_nl, aft_method=aft, arc_len=ds,
            direction=-1, max_step=10, max_iter=10, weight_w=weight, )


        seaborn.set(color_codes=True)
        plt.ion()
        plt.figure(1, figsize=(10, 6))
        plt.figure(2, figsize=(10, 6))

        for i in range(10000):
            # w,  X = A_solver.solve(ws[-2], ws[-1], Xs[-2], Xs[-1])

            w, X = A_solver.solve(ws[-1], Xs[-1])
            aft.process(X)

            ws.append(w)
            Xs.append(X)
            qs[0].append(abs(float(torch.max(aft.x_time[0, :].detach().abs()))))
            qs[1].append(abs(float(torch.max(aft.x_time[1, :].detach().abs()))))
            fs[0].append(aft.f_time[0, :].detach().numpy())
            fs[1].append(aft.f_time[1, :].detach().numpy())
            xs[0].append(aft.x_time[0, :].detach().numpy())
            xs[1].append(aft.x_time[1, :].detach().numpy())

            plt.figure(1)
            plt.clf()
            plt.plot(ws, qs[0])
            plt.scatter(ws[-1], qs[0][-1], c='r')
            plt.plot(ws, qs[1])
            plt.scatter(ws[-1], qs[1][-1], c='r')
            plt.grid(True)
            plt.pause(0.001)

            plt.figure(2)
            plt.clf()
            plt.subplot(121)
            plt.plot(xs[0][-1], fs[0][-1])
            plt.grid(True)
            plt.subplot(122)
            plt.plot(xs[1][-1], fs[1][-1])
            plt.grid(True)
            plt.pause(0.001)

            print(str(i) + " :   " + str(w.detach().numpy()) + "   " + str(qs[0][-1])+ "   " + str(qs[1][-1]))

            if w <= 0:
                Cs.append([ws, qs[0]])
                break

        plt.figure(3, figsize=(15, 8))
        # plt.clf()
        plt.plot(Cs[-1][0], Cs[-1][1])
        plt.grid(True)
        plt.pause(0.001)


    plt.savefig("result_friction_2d.svg")
    plt.ioff()
    plt.show()