import torch
import numpy as np

def sin_transform(u, unidtype=True):
    Nx = u.shape[-1]
    v = torch.cat([u, -u.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    a = -torch.fft.fft(v, dim=-1)[..., :Nx].imag
    if unidtype:
        a = a.to(u.dtype)
    return a


def isin_transform(a, unidtype=True):
    Nx = a.shape[-1]
    v = torch.cat([a, -a.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    u = torch.fft.ifft(v, dim=-1)[..., :Nx].imag
    if unidtype:
        u = u.to(a.dtype)
    return u


def cos_transform(u):
    Nx = u.shape[-1]

    V = torch.cat([u, u.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    a = torch.fft.fft(V, dim=-1)[..., :Nx].real  # / Nx
    return a


def icos_transform(a):
    Nx = a.shape[-1]

    V = torch.cat([a, a.flip(dims=[-1])[..., 1 : Nx - 1]], dim=-1)
    u = torch.fft.ifft(V, dim=-1)[..., :Nx].real  # * Nx
    return u


def WSWA(u):

    V = torch.cat([u, u.flip(dims=[-1])[..., 1 : u.shape[-1]]], dim=-1)
    W = torch.cat([V, -V.flip(dims=[-1])[..., 1 : V.shape[-1] - 1]], dim=-1)
    a = -torch.fft.fft(W, dim=-1)[..., [0] + list(range(1, V.shape[-1], 2))].imag
    return a


def iWSWA(a):
    Nx = a.shape[-1]

    W = torch.zeros(*a.shape[:-1], a.shape[-1] * 2 - 1, dtype=a.dtype, device=a.device)
    W[..., 1 : 2 * Nx : 2] = a[..., 1:]
    # W = torch.cat([temp, -temp.flip(dims=[-1])[..., 1:a.shape[-1]]], dim=-1)
    V = torch.cat([W, -W.flip(dims=[-1])[..., 1 : W.shape[-1] - 1]], dim=-1)
    a = torch.fft.ifft(V, dim=-1)[..., :Nx].imag
    return a

def fourier_partial(u, d=-1, T=sin_transform, iT=icos_transform):
    total_dim = u.dim()
    if (d != total_dim - 1) and (d != -1):
        u = torch.transpose(u, d, -1)

    Nx = u.shape[-1]
    k = torch.linspace(0, Nx - 1, Nx)
    # k = k * (torch.pi ** 2)
    du_fft = iT(T(u) * k).real

    if (d != total_dim - 1) and (d != -1):
        du_fft = torch.transpose(du_fft, d, -1)
    return du_fft


def fourier_partial2(u, T, iT):
    Nx = u.shape[-1]
    device = u.device
    k = -((torch.linspace(0, Nx - 1, Nx)) ** 2)
    k = k[..., None].repeat(1, Nx) + k[None, ...].repeat(Nx, 1)
    # print(k)
    k = k * (torch.pi**2)
    # print(T(u))
    k = k.to(device)
    du_fft = iT(T(u) * k).real
    return du_fft
