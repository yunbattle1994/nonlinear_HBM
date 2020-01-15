import torch
from torch.autograd import grad
import time
class AFT(object):

    def __init__(self, Nh, Nt, Nf, force):
        self.Nh = Nh
        self.Nt = Nt
        self.Nf = Nf
        self.force = force


    def get_differ(self, Xs, Ys):

        return torch.stack([grad([Ys[:, i].sum()], [Xs],
                                 retain_graph=True, create_graph=True)[0] for i in range(Ys.size(1))], dim=-1)


    def get_x0(self):

        x_freq = self.x_freq.view(-1, self.Nf)[:, self.force.DOF].permute(1, 0)
        x = torch.zeros(x_freq.shape[0], int(self.Nt/2), 2, dtype=torch.float64)
        x[:, 0:self.Nh, 0] =  x_freq[:, 0: 2*self.Nh: 2] * self.Nt / 2    # ifft 变换系数
        x[:, 0:self.Nh, 1] = -x_freq[:, 1: 2*self.Nh: 2] * self.Nt / 2    # ifft 变换系数
        x[:, 0, :] = x[:, 0, :] * 2   # ifft 变换系数
        return x

    def get_x1(self):

        x_freq = self.x_freq.view(-1, self.Nf)[:, self.force.DOF].permute(1, 0)
        x = torch.zeros(x_freq.shape[0], int(self.Nt / 2), 2, dtype=torch.float64)
        x[:, 1:self.Nh, 0] = torch.linspace(1, self.Nh-1, self.Nh-1, dtype=torch.double) \
                             * x_freq[:, 3: 2 * self.Nh: 2] * self.Nt / 2  # ifft 变换系数
        x[:, 1:self.Nh, 1] = torch.linspace(1, self.Nh-1, self.Nh-1, dtype=torch.double)\
                             * x_freq[:, 2: 2 * self.Nh: 2] * self.Nt / 2  # ifft 变换系数
        return x

    def get_x2(self):

        x_freq = self.x_freq.view(-1, self.Nf)[:, self.force.DOF].permute(1, 0)
        x = torch.zeros(x_freq.shape[0], int(self.Nt / 2), 2, dtype=torch.float64)
        x[:, 1:self.Nh, 0] = -torch.linspace(1, self.Nh - 1, self.Nh - 1, dtype=torch.double) ** 2\
                             * x_freq[:, 2: 2 * self.Nh: 2] * self.Nt / 2  # ifft 变换系数
        x[:, 1:self.Nh, 1] = torch.linspace(1, self.Nh - 1, self.Nh - 1, dtype=torch.double) ** 2\
                             * x_freq[:, 3: 2 * self.Nh: 2] * self.Nt / 2  # ifft 变换系数
        return x

    def irfft(self, X):

        x = torch.zeros(X.shape[0], X.shape[1] * 2 - 1, dtype=torch.double, requires_grad=True)
        for i in range(X.shape[0]):
            x[i] = torch.irfft(X[i].unsqueeze(0), 2)
        return x


    def rfft(self, x):

        X = torch.zeros(x.shape[0], int(x.shape[1] / 2 + 1), 2, dtype=torch.double, requires_grad=True)
        for i in range(X.shape[0]):
            X[i] = torch.rfft(x[i].unsqueeze(0), 2)
        return X

    def process(self, X):


        self.x_freq = X
        x = self.get_x0()
        dx = self.get_x1()
        d2x = self.get_x2()
        self.x_time = self.irfft(x)
        self.u_time = self.irfft(dx)
        self.a_time = self.irfft(d2x)

        # start = time.time()
        self.f_time = self.force.get_f(self.x_time, self.u_time, self.a_time)
        # end = time.time()
        # print("force:  " + str(end - start))

        F = self.rfft(self.f_time) / self.Nt * 2    # fft 变换系数
        F[:, :, 1] = -F[:, :, 1]      # fft 变换系数
        F[:, 0, :] = F[:, 0, :] / 2   # fft 变换系数
        self.f_freq = torch.zeros(self.Nh, 2, self.Nf, dtype=torch.float64, )
        self.f_freq[:, :, self.force.DOF] = F.permute(1, 2, 0)[:self.Nh, :, :]
        self.f_freq = self.f_freq.view(1, -1)


    def get_vector(self):
        F = self.f_freq
        return F

    def get_jacobian(self):
        try:
            J = self.get_differ(self.x_freq, self.f_freq).transpose(2, 1)
        except:
            J = torch.zeros(1, 2*self.Nf*self.Nh, 2*self.Nf*self.Nh, dtype=torch.double)
        return J


""" 以下为测试 """
if __name__ == "__main__":


    Nf = 2
    Nh = 3
    Nt = 1024

    from ForceNonlinear import Friction
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.ion()
    plt.show()


    t = torch.linspace(0, 2*np.pi, Nt+1, dtype=torch.double)
    x = torch.zeros((2, Nt+1), dtype=torch.double)
    x[0, :] = torch.sin(t[:])
    x[1, :] = torch.sin(t[:] + 4*np.pi/4)

    X_ = torch.rfft(x, 2) / Nt * 2
    X = X_.permute(1, 2, 0)[:Nh, :, :]
    X = X.contiguous()


    X[:, 1, :] = -X[:, 1, :]
    X[0, :, :] = X[0, :, :] / 2
    X = X.view(1, -1)
    X.requires_grad_(True)

    force = Friction(DOF=(0, 1), kx=1, ky=1, n0=1.2, mu=0.4)
    aft = AFT(Nh, Nt, Nf, force)

    aft.process(X)
    aft.get_vector()
    aft.get_jacobian()



