import torch
import functools
import sys

from . import fourierpack as sp
from SOL2_0.NOs_dict.models import Wrapper, FuncMat_Wrapper
from .utilities import LpLoss

## Parameters
kx = 0.01
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


## Spectral transforms
def div_Nx(u):
    return u / (u.shape[-1] - 1)


def mul_Nx(u):
    return u * (u.shape[-1] - 1)


cross_phi = functools.partial(Wrapper, [sp.sin_transform, div_Nx])
DST = cross_phi
iDST = functools.partial(Wrapper, [sp.isin_transform, mul_Nx])


def Div_Nx(u):
    return u / (2 * u.shape[-1] - 2)


def Mul_Nx(u):
    return u * (2 * u.shape[-1] - 2)


# i*j>0-> (i,j): torch.sin(i * torch.pi * X) * torch.sin((2 * j - 1) * np.pi * Y / 2)
DT = functools.partial(FuncMat_Wrapper, [[sp.WSWA, Div_Nx], [sp.sin_transform, div_Nx]])
iDT = functools.partial(
    FuncMat_Wrapper, [[sp.iWSWA, Mul_Nx], [sp.isin_transform, mul_Nx]]
)


## weak loss
def weak(u, f, T=DST, m=0, relative=True):
    batch_size, Nx, _ = f.shape
    device = u.device
    # matrices used in inner product when computing weak loss
    freq = torch.linspace(0, Nx - 1, Nx, dtype=torch.float64)
    A = torch.diag(freq**2) * (torch.pi**2) * kx
    B = torch.eye(Nx)
    A, B = A.double(), B.double()
    A, B = A.to(device), B.to(device)
    u = u
    U, b = T(u, dim=[-1, -2]), T(f, dim=[-1, -2])
    weak_err = (A @ U @ B + B @ U @ A.T) - b
    if m > 0:
        weak_err = weak_err[:, :m, :m]
        b = b[:, :m, :m]
    err_norm = torch.norm(weak_err.reshape(batch_size, -1), 2, 1)
    # f_norm = torch.norm(f.reshape(batch_size, -1), 2, 1)
    f_norm = torch.norm(b.reshape(batch_size, -1), 2, 1)
    if relative:
        return (err_norm / f_norm).sum()
    else:
        return err_norm.sum()


def weak_matrix(u, f, u0=0, T=DST):
    batch_size, Nx, _ = f.shape
    device = u.device
    # matrices used in inner product when computing weak loss
    freq = torch.linspace(0, Nx - 1, Nx, dtype=torch.float64)
    A = torch.diag(freq**2) * (torch.pi**2) * kx
    B = torch.eye(Nx)
    A, B = A.double(), B.double()
    A, B = A.to(device), B.to(device)
    u = u - u0
    U, b = T(u, dim=[-1, -2]), T(f, dim=[-1, -2])
    weak_err = (A @ U @ B + B @ U @ A.T) - b
    return weak_err


def weak_mix(u, f, u0=0, T=DT, relative=True):
    batch_size, Nx, _ = f.shape
    device = u.device
    # matrices used in inner product when computing weak loss
    freqx = torch.linspace(0, Nx - 1, Nx, dtype=torch.float64)
    freqy = (
        torch.tensor([0, *[2 * j - 1 for j in range(Nx)[1:]]], dtype=torch.float64) / 2
    )  # [0, 1, 3, ..., 2N-3] \in R^N
    freqX, freqY = torch.meshgrid(freqx, freqy, indexing="ij")
    MAT = (freqX**2 + freqY**2) * (torch.pi**2) * kx
    MAT = MAT.reshape(Nx, Nx).to(device)

    v = u - u0
    V, b = T(v, dim=[-1, -2]), T(f, dim=[-1, -2])
    weak_err = V * MAT - b
    err_norm = torch.norm(weak_err.reshape(batch_size, -1), 2, 1)
    # f_norm = torch.norm(f.reshape(batch_size, -1), 2, 1)
    f_norm = torch.norm(b.reshape(batch_size, -1), 2, 1)
    if relative:
        return (err_norm / f_norm).sum()
    else:
        return err_norm.sum()


## strong loss
def Laplacian(u):
    size = u.size(-1)
    u = u.reshape(-1, size, size)
    a = torch.ones(u.shape, device=u.device)
    dx = 1 / (size - 1)
    dy = dx

    uxx = (u[:, :-2, 1:-1] + u[:, 2:, 1:-1] - 2 * u[:, 1:-1, 1:-1]) / (dx**2)
    uyy = (u[:, 1:-1, :-2] + u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1]) / (dy**2)
    a = a[:, 1:-1, 1:-1]

    auxx = a * uxx
    auyy = a * uyy
    Du = auxx + auyy
    return Du


def fourier_p2(u, T, iT):
    Nx = u.shape[-1]
    k = -((torch.linspace(0, Nx - 1, Nx, dtype=torch.float64)) ** 2)
    k = k[..., None].repeat(1, Nx) + k[None, ...].repeat(Nx, 1)
    # print(k)
    k = k * (torch.pi**2)
    # print(T(u))
    k = k.to(u.device)
    du_fft = iT(T(u, dim=[-1, -2]) * k, dim=[-1, -2]).real
    return du_fft


D2 = functools.partial(fourier_p2, DST, iDST)


def strong(u, f, label="fourier"):
    batch_size = u.shape[0]
    Nx = u.shape[-1]
    myloss = LpLoss(size_average=False)
    if label == "fourier":
        DDu = D2(u)[:, 1:-1, 1:-1]
    else:
        DDu = Laplacian(u)

    err = myloss(
        -kx * DDu.reshape(batch_size, -1), f[:, 1:-1, 1:-1].reshape(batch_size, -1)
    )
    return err


def strong_matrix(u, f, label="fourier"):
    # output: The equation residual
    Nx = u.shape[-1]
    u = u.reshape(Nx, Nx)[None, ...]
    f = f.reshape(Nx, Nx)[None, ...]
    if label == "fourier":
        DDu = D2(u)[:, 1:-1, 1:-1]
    else:
        DDu = Laplacian(u)
    err = -kx * DDu - f[:, 1:-1, 1:-1]
    err = err.reshape(Nx - 2, Nx - 2)
    er = torch.zeros(Nx, Nx)
    er[1:-1, 1:-1] = err

    return er


def D2_mix(u, T=DT, iT=iDT):
    Nx = u.shape[-1]
    kx = torch.linspace(0, Nx - 1, Nx, dtype=torch.float64)
    ky = torch.tensor([0, *[2 * j - 1 for j in range(Nx)[1:]]], dtype=torch.float64) / 2
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    MAT = -(KX**2 + KY**2) * torch.pi**2
    MAT = MAT.to(device)
    du_fft = iT(T(u, dim=[-1, -2]) * MAT, dim=[-1, -2]).real
    return du_fft


def strong_mix(u, f, u0=0, label="fourier"):
    batch_size = u.shape[0]
    Nx = u.shape[-1]
    myloss = LpLoss(size_average=False)
    u = u - u0
    if label == "fourier":
        DDu = D2_mix(u)[:, 1:-1, 1:-1]
    else:
        DDu = Laplacian(u)
    err = myloss(
        -kx * DDu.reshape(batch_size, -1), f[:, 1:-1, 1:-1].reshape(batch_size, -1)
    )
    return err

## prediction on point cloud
def evaluate_semi(F, x, y):
    m = F.shape[0]
    u = torch.zeros_like(x)      
    for i in range(m):
        for j in range(m):
            u += F[i,j] * torch.sin(i*torch.pi*x)*torch.sin(j*torch.pi*y) 

    return u
