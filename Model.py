import numpy as np
import torch

class Model(object):
    
    def __init__(self, M, C, K, Nh):
        
        self.Nh = Nh
        self.M, self.C, self.K = M, C, K
        self.Nf = self.K.shape[0]

    
    def get_A(self, w):

        A = torch.zeros((2 * self.Nh * self.Nf, 2 * self.Nh * self.Nf), dtype=torch.float64)

        for i in range(self.Nh):
            A[(2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf, (2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf] =\
                - self.M * (i*w)**2 + self.K
            A[(2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf, (2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf] =\
                self.C * (i*w)
            A[(2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf, (2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf] =\
                -self.C * (i*w)
            A[(2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf, (2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf] =\
                -self.M * (i*w)**2 + self.K


        return A.unsqueeze(0)
    
    def get_DADw(self, w):

        A = torch.zeros((2 * self.Nh * self.Nf, 2 * self.Nh * self.Nf), dtype=torch.float64)

        for i in range(self.Nh):
            A[(2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf, (2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf] = \
                - self.M * (i**2 * w) * 2
            A[(2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf, (2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf] = \
                self.C*i
            A[(2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf, (2 * i + 0) * self.Nf:(2 * i + 1) * self.Nf] = \
                -self.C*i
            A[(2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf, (2 * i + 1) * self.Nf:(2 * i + 2) * self.Nf] = \
                - self.M * (i**2 * w) * 2
    

        return A.unsqueeze(0)





if __name__ == '__main__':
    
    
    model = Model(10, 1024)
    A = model.get_A(2000.)
    B = model.get_DADw(2000.)
