import torch
import numpy as np
import time

class Arclen_Solver(object):

    def __init__(self, model, force_ex, force_nl, aft_method, arc_len, direction=1,
                 max_step=20, max_iter=10, weight_w=1e-4, max_err=1.e-8, display=False):
        self.model = model
        self.force_ex = force_ex
        self.force_nl = force_nl
        self.aft_method = aft_method

        self.direction = direction
        self.max_step = max_step
        self.max_iter = max_iter
        self.weight_w = weight_w
        self.ds = arc_len
        self.max_err = max_err
        self.display = display

        self.Nf = self.model.Nf
        self.Nh = self.model.Nh



    def get_tau(self, dRdX, dRdw):

        J = torch.cat((dRdX, dRdw), dim=-1)
        t = torch.zeros(1, J.shape[-1], dtype=torch.double)
        m = torch.max(J)
        for index in range(J.shape[-1]):
            t[0, index] = (-1) ** (index + 1) * torch.det(torch.cat((J[0, :, :index]/m, J[0, :, index+1:]/m), dim=-1))


        t = t / t.min().abs()
        t = t / torch.norm(t)

        if self.direction == 1:
            t = -t
            # t = t * (-1) ** J.shape[2] * torch.sign(torch.det(dRdX[0]))

        return t, J


    def predictor(self, w0, X0, ds):

        self.aft_method.process(X0)
        dFNdX = self.aft_method.get_jacobian().detach()

        A0 = self.model.get_A(w0)
        dAdw = self.model.get_DADw(w0) / self.weight_w
        dFEdw = self.force_ex.get_dfdw(w0) / self.weight_w

        dRdX = A0 + dFNdX
        dRdw = torch.matmul(dAdw, X0.unsqueeze(-1)).detach() - dFEdw.unsqueeze(-1)

        t, _ = self.get_tau(dRdX, dRdw)

        dw = ds * t[:, -1] / self.weight_w
        dX = ds * t[:, :-1]

        return dw, dX

    def corrector(self, w0, X0, dw, dX, ds):

        self.aft_method.process(X0)
        FE = self.force_ex.get_f(w0)
        FN = self.aft_method.get_vector().detach()

        # start = time.time()
        dFdX = self.aft_method.get_jacobian().detach()
        # end = time.time()
        # print("differ:  " + str(end - start))

        A0 = self.model.get_A(w0).detach()
        dAdw = self.model.get_DADw(w0).detach() / self.weight_w
        dFEdw = self.force_ex.get_dfdw(w0) / self.weight_w

        RX = torch.matmul(A0, X0.unsqueeze(-1)) + FN.unsqueeze(-1) - FE.unsqueeze(-1)
        err = float(torch.norm(RX, 1).detach().numpy())

        dRdX = A0 + dFdX
        dRdw = torch.matmul(dAdw, X0.unsqueeze(-1)).detach() - dFEdw.unsqueeze(-1)

        t, J = self.get_tau(dRdX, dRdw)

        J_ = torch.cat((J, t.unsqueeze(-1).permute(0, 2, 1)), dim=1)
        Rw = torch.sum(t * torch.cat((dX, self.weight_w * dw.view(-1, 1)), dim=1)) - ds

        D_lu = torch.lu(J_)
        d = torch.cat((RX, Rw.unsqueeze(0).unsqueeze(0).unsqueeze(0)), dim=1).detach().lu_solve(*D_lu)
        dw = dw - d[:, -1] / self.weight_w
        dX = dX - d[:, :-1].squeeze(-1)

        return dw.detach(), dX.detach(), err

    def solve(self, w0, X0):

        alf = 1.0
        converge = False
        err = 1.0

        for step in range(self.max_step):

            if converge == True:
                print("step:  " + str(step) + "   iter  " + str(iter) + "    error:   " + str(err))
                break
            else:
                if step >= 1:
                    alf = alf * 0.5
                elif step >= 8:
                    alf = 1.5

            X = torch.zeros((1, 2 * self.Nf * self.Nh), dtype=torch.float64, requires_grad=True)
            w = w0
            X[:, :] = X0.squeeze(-1)
            dw, dX = self.predictor(w0, X, alf*self.ds)
            X[:, :] = X0 + alf * dX[:, :]
            w = w + alf * dw

            for iter in range(self.max_iter):

                dw, dX, err = self.corrector(w, X, dw, dX, alf*self.ds)
                if self.display: print(err)
                if err < self.max_err:
                    converge = True
                    break

                w = w0 + dw
                X[:, :] = X0 + dX

        return w.detach(), X.detach()





class Newton_Solver(object):

    def __init__(self, model, force_ex, force_nl, aft_method,
                 max_iter=10, max_err=1e-8, display=False):
        self.model = model
        self.force_ex = force_ex
        self.force_nl = force_nl
        self.aft_method = aft_method

        self.max_iter = max_iter
        self.max_err = max_err
        self.display = display

        self.Nh = self.model.Nh
        self.Nf = self.model.Nf

    def solve(self, w):

        X = torch.zeros((1, 2 * self.Nf * self.Nh), dtype=torch.float64, requires_grad=True)
        # w = torch.zeros(1, requires_grad=True)

        A0 = self.model.get_A(w)
        D_lu = torch.lu(A0)
        FE = self.force_ex.get_f(w)
        X[0, :] = FE.unsqueeze(-1).lu_solve(*D_lu).squeeze()

        for iter in range(self.max_iter):

            self.aft_method.process(X)
            FN = self.aft_method.get_vector().detach()
            dFdX = self.aft_method.get_jacobian().detach()

            RX = torch.matmul(A0, X.unsqueeze(-1)) + FN.unsqueeze(-1) - FE.unsqueeze(-1)
            err = float(torch.norm(RX, 1).detach().numpy())

            if self.display: print("error:  " + str(err))
            if err < self.max_err:
                print("iter:  " + str(iter + 1) + "    error:   " + str(err))
                break

            D_lu = torch.lu(A0 + dFdX)
            dX = RX.lu_solve(*D_lu).squeeze(-1)
            X = X - dX

        return X.detach()



