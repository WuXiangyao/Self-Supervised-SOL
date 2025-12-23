import torch
import numpy as np
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import math
import functools

def sin_transform(u):
    Nx = u.shape[-1]
    V = torch.cat([u, -u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = -torch.fft.fft(V, dim=-1)[..., :Nx].imag# / Nx
    return a

def isin_transform(a):
    Nx = a.shape[-1]
    V = torch.cat([a, -a.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    u = torch.fft.ifft(V, dim=-1)[..., :Nx].imag# * Nx
    return u

def cos_transform(u):
    Nx = u.shape[-1]

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    a = torch.fft.fft(V, dim=-1)[..., :Nx].real# / Nx
    return a

def icos_transform(a):
    Nx = a.shape[-1]

    V = torch.cat([a, a.flip(dims=[-1])[..., 1:Nx-1]], dim=-1)
    u = torch.fft.ifft(V, dim=-1)[..., :Nx].real# * Nx
    return u

def dctII_old(u):
    Nx = u.shape[-1]

    v = torch.cat([u[..., ::2], u[..., 1::2].flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(Nx, dtype=u.dtype, device=u.device)
    W4 = torch.exp(-.5j * torch.pi * k / Nx)
    return 2 * (V * W4).real

def idctII_old(a):
    Nx = a.shape[-1]

    k = torch.arange(Nx, dtype=a.dtype, device=a.device)
    iW4 = 1 / torch.exp(-.5j * torch.pi * k / Nx); iW4[..., 0] /= 2

    V = torch.fft.ifft(a * iW4).real
    u = torch.zeros_like(V)
    u[..., ::2], u[..., 1::2] = V[..., :Nx - (Nx // 2)], V.flip(dims=[-1])[..., :Nx // 2]

    return u

def dctII(u):
    Nx = u.shape[-1]

    v = torch.cat([u[..., ::2], u[..., 1::2].flip(dims=[-1])], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(Nx, dtype=u.dtype, device=u.device)
    W4 = torch.exp(-.5j * torch.pi * k / Nx)
    return 2 * (V * W4).real / Nx

def idctII(a):
    Nx = a.shape[-1]

    k = torch.arange(Nx, dtype=a.dtype, device=a.device)
    iW4 = 1 / torch.exp(-.5j * torch.pi * k / Nx); iW4[..., 0] /= 2

    V = torch.fft.ifft(a * iW4).real
    u = torch.zeros_like(V)
    u[..., ::2], u[..., 1::2] = V[..., :Nx - (Nx // 2)], V.flip(dims=[-1])[..., :Nx // 2]

    return u * Nx



def dstII(u):
    v = u.clone()
    v[..., 1::2] = -v[..., 1::2]
    return dctII(v).flip(dims=[-1])

def idstII(a):
    v = idctII(a.flip(dims=[-1]))
    u = v.clone()
    u[..., 1::2] = -u[..., 1::2]
    return u


def WSWA(u):

    V = torch.cat([u, u.flip(dims=[-1])[..., 1:u.shape[-1]]], dim=-1)
    W = torch.cat([V, -V.flip(dims=[-1])[..., 1:V.shape[-1]-1]], dim=-1)
    a = -torch.fft.fft(W, dim=-1)[..., [0]+list(range(1, V.shape[-1], 2))].imag
    return a

def iWSWA(a):
    Nx = a.shape[-1]

    W = torch.zeros(*a.shape[:-1], a.shape[-1]*2-1, dtype=a.dtype, device=a.device)
    W[..., 1:2*Nx:2] = a[..., 1:]
    # W = torch.cat([temp, -temp.flip(dims=[-1])[..., 1:a.shape[-1]]], dim=-1)
    V = torch.cat([W, -W.flip(dims=[-1])[..., 1:W.shape[-1]-1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].imag
    return a

def fourier_partial(u, d=-1, T=sin_transform, iT=icos_transform):
    total_dim = u.dim()
    if (d != total_dim-1) and (d != -1):
        u = torch.transpose(u, d, -1)

    Nx = u.shape[-1]
    k = (torch.linspace(0, Nx-1, Nx))
    du_fft = iT(T(u) * k).real

    if (d != total_dim-1) and (d != -1):
        du_fft = torch.transpose(du_fft, d, -1)
    return du_fft

def fourier_partial2(u, T, iT):
    Nx = u.shape[-1]
    k = -(torch.linspace(1, Nx, Nx)) ** 2 
    k = k * torch.pi ** 2
    du_fft = iT(T(u) * k).real
    return du_fft

if __name__ == "__main__":

    def test_Dx():

        from torch import pi
        Nx = 128 + 1
        x = torch.linspace(0, pi, Nx, dtype=torch.float64)
        # testf = torch.sin(x)

        # test Dx
        testf = x ** 2 * (x - pi) ** 2
        true_df = 4 * x ** 3 - 6 * pi * x ** 2 + 2 * pi ** 2 * x
        true_ddf = 12 * x ** 2 - 12 * pi * x + 2 * pi ** 2

        testf = x*(x - pi)
        true_df = 2*x-pi
        true_ddf = 2 *torch.ones_like(x)

        # testf = torch.exp(-25*(x/pi*2-1)**2/(1-(x/pi*2-1)**2))
        testf = torch.tanh(25 * (x / pi * 2 - 1) / torch.sqrt(1 - (x / pi * 2 - 1) ** 2))

        qf = isin_transform(sin_transform(testf))
        df = fourier_partial(testf, T=sin_transform, iT=icos_transform)
        ddf = fourier_partial(-df, T=cos_transform, iT=isin_transform)


        import matplotlib.pyplot as plt

        # plt.plot(true_df, '*')
        # plt.plot(df.resolve_neg().numpy())
        plt.plot((true_df.numpy()-df.resolve_neg().numpy()))
        plt.show()
        k = (torch.linspace(0, Nx - 1, Nx))
        print(sin_transform(testf) * k / Nx)

        plt.loglog(((sin_transform(testf))[1::2] / Nx).abs())

        line = 1 / torch.linspace(1, Nx, Nx) ** 3
        plt.loglog(line, ':')
        plt.annotate('$N^{-3}$', xy=(10, line[10]), xytext=(10 + 5, line[10]))

        plt.xlabel('coefficients of basis')
        plt.yscale('log', base=10)
        plt.title('Non-exponential decay for the original func.' + 'x(x-1)')
        plt.show()

        ### difference
        plt.loglog(((cos_transform(df-true_df))[1::2] / Nx).abs())

        line = 1 / torch.linspace(1, Nx, Nx) ** 2
        plt.loglog(line, ':')
        plt.annotate('$N^{-2}$', xy=(10, line[10]), xytext=(10 + 5, line[10]))

        plt.xlabel('coefficients of basis')
        plt.yscale('log', base=10)
        plt.title('2nd-order decay for the derivatives $u_x$')
        plt.show()

        # plt.plot(df)
        # plt.plot(true_ddf - ddf.resolve_neg().numpy())
        # plt.plot(true_df - df)
        # plt.plot(true_ddf)
        # plt.plot(ddf[1:-1].resolve_neg().numpy())
        # plt.show()

    def test_DDx():

        from torch import pi
        Nx = 1280 + 1
        x = torch.linspace(0, pi, Nx, dtype=torch.float64)
        # testf = torch.sin(x)

        ## test Dx
        testf = x ** 2 * (x - pi) ** 2
        true_df = 4 * x ** 3 - 6 * pi * x ** 2 + 2 * pi ** 2 * x
        true_ddf = 12 * x ** 2 - 12 * pi * x + 2 * pi ** 2

        qf = isin_transform(sin_transform(testf))
        df = fourier_partial(testf, T=sin_transform, iT=icos_transform)
        ddf = fourier_partial(-df, T=cos_transform, iT=isin_transform)

        import matplotlib.pyplot as plt

        # plt.plot(true_ddf, '*')
        # plt.plot(ddf.resolve_neg().numpy())
        plt.plot(true_ddf.numpy()-ddf.resolve_neg().numpy())
        plt.show()

        # plt.plot(df)
        # plt.plot(true_ddf - ddf.resolve_neg().numpy())
        # plt.plot(true_df - df)
        # plt.plot(true_ddf)
        # plt.plot(ddf[1:-1].resolve_neg().numpy())
        # plt.show()

    def test_WSWA():
        Nx = 128 + 1
        x = torch.linspace(0, torch.pi, Nx, dtype=torch.float64)
        ## test Dx
        testf = x ** 2 * (x - torch.pi) ** 2
        # testf = torch.cos(x) - torch.torch.cos(2*x)

        a = WSWA(testf)
        f = torch.zeros_like(testf)
        for i in range(1, f.shape[-1]):
            f += a[i]*torch.sin((2*i-1)*x/2)
        # plt.plot(testf)
        # plt.plot(f/(Nx-1)/2)
        # plt.show()
        print("WSWA error", (testf-f/(Nx-1)/2).abs().max())
        print("iWSWA error", (testf-iWSWA(a)).abs().max())

    def test_dctII():

        def dct_ii_test(u):
            Nx = u.shape[-1]
            X = torch.zeros(Nx, dtype=u.dtype)
            for k in range(Nx):
                X[k] = 2 * torch.sum(
                    u * torch.cos(torch.pi * torch.arange(1, 2 * Nx + 1, 2, dtype=u.dtype) * k / (2 * Nx)))
            return X / Nx

        def idct_ii_test(u):
            Nx = u.shape[-1]
            X = torch.zeros(Nx, dtype=u.dtype)
            for n in range(Nx):
                X[n] = 1 / Nx * torch.sum(u[..., 1:] *
                                          torch.cos(-torch.pi * torch.arange(1, Nx, dtype=u.dtype) * (2 * n + 1) / (
                                                      2 * Nx))).real
            X += 1 / (2 * Nx) * u[..., 0].real

            return X * Nx

        u = torch.rand(17, dtype=torch.float64)

        print('dctII error:', (dctII(u) - dct_ii_test(u)).abs().max())
        print('idctII error:', (idctII(u) - idct_ii_test(u)).abs().max())
        print('DCT dual error:', (idctII(dctII(u)) - u).abs().max())


    def test_dstII():

        def dst_ii_test(u):
            Nx = u.shape[-1]
            X = torch.zeros(Nx, dtype=u.dtype)
            for k in range(Nx):
                X[k] = 2 * torch.sum(
                    u * torch.sin(torch.pi/Nx * torch.linspace(1/2, Nx-1/2, Nx, dtype=u.dtype) * (k+1)))
            return X / Nx

        def idst_ii_test(u): # DST-III
            Nx = u.shape[-1]
            X = torch.zeros(Nx, dtype=u.dtype)
            for k in range(Nx):
                X[k] = (-1)**k/2 * u[..., -1] + \
                       torch.sum(u[..., :-1] *
                            torch.sin(torch.pi/Nx * (k+1/2) * torch.arange(1, Nx, dtype=u.dtype)))  # .real
            # X += 1 / (2 * Nx) * u[..., 0]#.imag

            return X

        import matplotlib.pyplot as plt
        Nx = 16
        x = torch.linspace(0, 1, Nx+1, dtype=torch.float64)
        x += (x[-1] - x[0]) / (Nx) / 2
        x = x[:-1]
        u = torch.sin(torch.pi * x)

        print('spectrum of sin', dst_ii_test(u))
        print('spectrum of sin', dstII(u))

        u = torch.rand(Nx, dtype=torch.float64)

        print('dstII error:', (dstII(u) - dst_ii_test(u)).abs().max())
        print('idstII error:', (idstII(u) - idst_ii_test(u)).abs().max())

        print('--random test--')
        a = torch.rand(Nx, dtype=torch.float64);
        # a[0] = 0
        fst = idstII(a)
        st = idst_ii_test(a)

        print('DST randomize idst error', (fst - st).abs().max())
        print('DST dual error', (dst_ii_test(fst) - a).abs().max())


    test_Dx()
    # test_WSWA()
    # test_dctII()
    # test_dstII()
