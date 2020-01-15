import torch
import matplotlib.pyplot as plt



class Friction(object):

    def __init__(self, DOF, kx, ky, n0, mu, fy=None):
        self.DOF = DOF
        self.kx = kx
        self.ky = ky
        self.n0 = n0
        self.mu = mu
        self.fy = fy


    def get_f(self, u, du, d2u):

        if self.fy is None:
            fy = torch.relu(self.ky * u[1, :] + self.n0)
            x, y = u[0, :], u[1, :]
            dx, dy = du[0, :], du[1, :]
            d2x, d2y = d2u[0, :], d2u[1, :]

            dfy = self.ky * dy
            dfy[fy < 0] = 0
            d2fy = self.ky * d2y
            d2fy[fy < 0] = 0

        else:
            fy, dfy, d2fy = self.fy[0], self.fy[1],  self.fy[2]
            x, dx, d2x = u[0, :], du[0, :], d2u[0, :]


        if torch.min(fy) > 0:
            fx = self.get_no_sep(x, dx, d2x, fy, dfy, d2fy)
        else:
            fx = self.get_sep(x, dx, d2x, fy, dfy, d2fy)


        if self.fy is None:
            fy = fy - torch.relu(torch.DoubleTensor([self.n0]))
            f = torch.cat((fx.unsqueeze(0), fy.unsqueeze(0)), dim=0)
        else:
            f = fx.unsqueeze(0)
        return f



    def get_no_sep(self, x, dx, d2x, fy, dfy, d2fy):
        Nt = x.shape[0]
        I1 = -torch.ones(Nt, dtype=torch.long)
        fx = torch.zeros(Nt, dtype=torch.double, requires_grad=True)

        mx = (x.max() + x.min()) / 2.

        for j in range(Nt-1):
            if (x[j] - mx) * (x[j+1] - mx) <= 0: break

        T0 = j+1

        for j in range(T0, Nt+T0):
            k = j % Nt
            tmp = self.mu * fy[k] - torch.abs(self.kx*(x[k]-mx))
            if tmp <= 0: break

        T1 = k
        S0 = torch.sign(self.kx*(x[T1]-mx))

        if T0 == (T1+1) % Nt and tmp > 0:
            fx = self.kx * (x - mx)
            return fx

        for j in range(T1+1, Nt+T1):
            k = j % Nt
            tmp = S0 * self.kx * dx[k] - self.mu * dfy[k]
            dmp = S0 * self.kx * d2x[k] - self.mu * d2fy[k]
            if tmp <= 0 and dmp <= 0: break

        T0 = k
        x0 = x[T0]
        f0 = S0 * fy[T0] * self.mu

        I1[T0] = T0
        fx[T0] = f0

        times = [[T0,],]
        state = [0,]
        x0s = [x0,]
        f0s = [f0,]
        s0s = [S0,]

        for j in range(T0+1, Nt+T0):
            l, k, r = (j-1) % Nt, j % Nt, (j+1) % Nt
            if I1[l] != -1:
                tmp = fy[k] * self.mu - torch.abs(self.kx * (x[k] - x0) + f0)
                if tmp <= 0:   #  stick to slip
                    S0 = torch.sign(self.kx * (x[k] - x0) + f0)
                    I1[k] = -1
                    times[-1].append(k)
                    times.append([k, ])
                    state.append(1)
                    s0s.append(S0)
                    x0s.append(x0)
                    f0s.append(f0)
                else:          #  stick to stick
                    I1[k] = k
            else:
                tmp = S0 * self.kx * dx[k] - self.mu*dfy[k]
                dmp = S0 * self.kx * d2x[k] - self.mu*d2fy[k]

                if tmp <= 0 and dmp <= 0:   #  slip to stick
                    x0 = x[k]
                    f0 = S0 * fy[k] * self.mu
                    I1[k] = k
                    times[-1].append(k)
                    times.append([k, ])
                    state.append(0)
                    s0s.append(S0)
                    x0s.append(x0)
                    f0s.append(f0)
                else:                       #  slip to slip
                    I1[k] = -1


        times[-1].append(T0)

        for ind in range(len(times)):
            ind0 = times[ind][0]
            ind1 = times[ind][1]
            if state[ind] == 0:
                if ind1>ind0:
                    fx[ind0:ind1] = self.kx * (x[ind0:ind1] - x0s[ind]) + f0s[ind]
                else:
                    fx[ind0:] = self.kx * (x[ind0:] - x0s[ind]) + f0s[ind]
                    fx[:ind1] = self.kx * (x[:ind1] - x0s[ind]) + f0s[ind]

            elif state[ind] == 1:
                if ind1 > ind0:
                    fx[ind0:ind1] = s0s[ind] * fy[ind0:ind1] * self.mu
                else:
                    fx[ind0:] = s0s[ind] * fy[ind0:] * self.mu
                    fx[:ind1] = s0s[ind] * fy[:ind1] * self.mu

        return fx


    def get_sep(self, x, dx, d2x, fy, dfy, d2fy):
        Nt = x.shape[0]
        I1 = -torch.ones(Nt, dtype=torch.long)
        fx = torch.zeros(Nt, dtype=torch.double, requires_grad=True)

        for j in range(Nt):
            k, r = j % Nt, (j+1) % Nt
            if fy[k] == 0 and fy[r] > 0: break

        T0 = r
        S0 = torch.sign(dx[T0])

        if T0 == 0 and fy[r] == 0:
            return fx

        tmp = S0 * self.kx * dx[T0] - self.mu * dfy[T0]
        if tmp <= 0:
            I1[T0] = T0
            f0 = 0
            x0 = x[T0]
            fx[T0] = 0
            st = 0
        else:
            I1[T0] = -1
            fx[T0] = S0 * self.mu * fy[T0]
            f0 = 0
            x0 = 0
            st = 1

        times = [[T0, ], ]
        state = [st, ]
        x0s = [x0, ]
        f0s = [f0, ]
        s0s = [S0, ]

        for j in range(T0+1, Nt+T0):
            l, k, r = (j-1) % Nt, j % Nt, (j+1) % Nt

            if fy[l] > 0 and fy[k] > 0:
                if I1[l] != -1:
                    tmp = fy[k] * self.mu - torch.abs(self.kx * (x[k] - x0) + f0)
                    if tmp <= 0:  # stick to slip
                        S0 = torch.sign(self.kx * (x[k] - x0) + f0)
                        I1[k] = -1
                        times[-1].append(k)
                        times.append([k,])
                        state.append(1)
                        s0s.append(S0)
                        x0s.append(x0)
                        f0s.append(f0)
                    else:     # stick to stick
                        I1[k] = k
                else:
                    tmp = S0 * self.kx * dx[k] - self.mu * dfy[k]
                    dmp = S0 * self.kx * d2x[k] - self.mu * d2fy[k]

                    if tmp <= 0 and dmp <= 0:  # slip to stick
                        x0 = x[k]
                        f0 = S0 * fy[k] * self.mu
                        I1[k] = k
                        times[-1].append(k)
                        times.append([k, ])
                        state.append(0)
                        s0s.append(S0)
                        x0s.append(x0)
                        f0s.append(f0)
                    else:  # slip to slip
                        I1[k] = -1

            elif fy[l] == 0 and fy[k] > 0:
                S0 = torch.sign(dx[k])
                tmp = S0 * self.kx * dx[k] - self.mu * dfy[k]
                if tmp <= 0:
                    I1[k] = k
                    f0 = 0
                    x0 = x[k]
                    times[-1].append(k)
                    times.append([k, ])
                    state.append(0)
                    s0s.append(S0)
                    x0s.append(x0)
                    f0s.append(f0)
                else:
                    I1[k] = -1
                    times[-1].append(k)
                    times.append([k, ])
                    state.append(1)
                    s0s.append(S0)
                    x0s.append(0)
                    f0s.append(0)

            elif fy[l] > 0 and fy[k] == 0:
                I1[k] = k
                times[-1].append(k)
                times.append([k, ])
                state.append(2)
                s0s.append(0)
                x0s.append(0)
                f0s.append(0)



        times[-1].append(T0)
        for ind in range(len(times)):
            ind0 = times[ind][0]
            ind1 = times[ind][1]
            if state[ind] == 0:
                if ind1 > ind0:
                    fx[ind0:ind1] = self.kx * (x[ind0:ind1] - x0s[ind]) + f0s[ind]
                else:
                    fx[ind0:] = self.kx * (x[ind0:] - x0s[ind]) + f0s[ind]
                    fx[:ind1] = self.kx * (x[:ind1] - x0s[ind]) + f0s[ind]

            elif state[ind] == 1:
                if ind1 > ind0:
                    fx[ind0:ind1] = s0s[ind] * fy[ind0:ind1] * self.mu
                else:
                    fx[ind0:] = s0s[ind] * fy[ind0:] * self.mu
                    fx[:ind1] = s0s[ind] * fy[:ind1] * self.mu
        return fx


