"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""

import os

# from utilities3 import *

import sys

sys.path.append("..")
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from copy import deepcopy
import h5py
from scipy.io import loadmat
import fourierpack as sp
import functools
from math import ceil
from NOs_dict.models import Wrapper, FuncMat_Wrapper

import matplotlib

# dstn_pad = functools.partial(Wrapper, [sp.sin_pad, sp.sin_transform])
dstn = functools.partial(Wrapper, [sp.sin_transform])
idstn = functools.partial(Wrapper, [sp.isin_transform])
dctn = functools.partial(Wrapper, [sp.cos_transform])
idctn = functools.partial(Wrapper, [sp.icos_transform])
device = torch.device("cuda:1")

T_pipe = functools.partial(FuncMat_Wrapper, [[sp.WSWA], [sp.sin_transform]])
iT_pipe = functools.partial(FuncMat_Wrapper, [[sp.iWSWA], [sp.isin_transform]])

# T_pipe = functools.partial(FuncMat_Wrapper, [[sp.WSWA2], [sp.sin_transform]])
# iT_pipe = functools.partial(FuncMat_Wrapper, [[sp.iWSWA2], [sp.isin_transform]])


class SinNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, degree1, degree2, bandwidth):
        super(SinNO2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree1 = degree1
        self.degree2 = degree2
        self.bandwidth = bandwidth

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels * bandwidth * bandwidth,
                out_channels,
                degree1 * degree2,
                dtype=torch.float32,
            )
        )
        # self.weights = nn.Parameter(
        #     self.scale * torch.rand(in_channels*bandwidth*bandwidth, out_channels, degree1*degree2, dtype=torch.complex64))

        # self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth), padding=(self.bandwidth-1)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth, self.bandwidth))

    def quasi_diag_mul2d(self, input, weights):
        xpad = self.unfold(input)
        # print(xpad.shape, input.shape, weights.shape)
        return torch.einsum("bix, iox->box", xpad, weights)
        # return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx, Ny = u.shape

        a = dstn(u, [-1, -2])

        b = torch.zeros(
            batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float32
        )
        b[..., : self.degree1, : self.degree2] = self.quasi_diag_mul2d(
            a[
                ...,
                : self.degree1 + self.bandwidth - 1,
                : self.degree2 + self.bandwidth - 1,
            ],
            self.weights,
        ).reshape(batch_size, self.out_channels, self.degree1, self.degree2)

        u = idstn(b, [-1, -2])
        return u


class CosNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, degree1, degree2, bandwidth):
        super(CosNO2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree1 = degree1
        self.degree2 = degree2
        self.bandwidth = bandwidth

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels * bandwidth * bandwidth,
                out_channels,
                degree1 * degree2,
                dtype=torch.float32,
            )
        )
        # self.weights = nn.Parameter(
        #     self.scale * torch.rand(in_channels*bandwidth*bandwidth, out_channels, degree1*degree2, dtype=torch.complex64))

        # self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth), padding=(self.bandwidth-1)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth, self.bandwidth))

    def quasi_diag_mul2d(self, input, weights):
        xpad = self.unfold(input)
        # print(xpad.shape, input.shape, weights.shape)
        return torch.einsum("bix, iox->box", xpad, weights)
        # return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx, Ny = u.shape

        a = dctn(u, [-1, -2])

        b = torch.zeros(
            batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float32
        )
        b[..., : self.degree1, : self.degree2] = self.quasi_diag_mul2d(
            a[
                ...,
                : self.degree1 + self.bandwidth - 1,
                : self.degree2 + self.bandwidth - 1,
            ],
            self.weights,
        ).reshape(batch_size, self.out_channels, self.degree1, self.degree2)

        u = idctn(b, [-1, -2])
        return u


class WSWANO(nn.Module):
    def __init__(self, in_channels, out_channels, degree1, degree2, bw1, bw2):
        super(WSWANO, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree1 = degree1
        self.degree2 = degree2
        self.bw1, self.bw2 = bw1, bw2

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels * bw1 * bw2,
                out_channels,
                degree1 * degree2,
                dtype=torch.float32,
            )
        )
        # self.weights = nn.Parameter(
        #     self.scale * torch.rand(in_channels*bandwidth*bandwidth, out_channels, degree1*degree2, dtype=torch.complex64))

        # self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth), padding=(self.bandwidth-1)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(bw1, bw2))

    def quasi_diag_mul2d(self, input, weights):
        xpad = self.unfold(input)
        # print(xpad.shape, input.shape, weights.shape)
        return torch.einsum("bix, iox->box", xpad, weights)
        # return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx, Ny = u.shape

        a = T_pipe(u, [-1, -2])

        b = torch.zeros(
            batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float32
        )
        b[..., : self.degree1, : self.degree2] = self.quasi_diag_mul2d(
            a[..., : self.degree1 + self.bw1 - 1, : self.degree2 + self.bw2 - 1],
            self.weights,
        ).reshape(batch_size, self.out_channels, self.degree1, self.degree2)

        u = iT_pipe(b, [-1, -2])
        return u


class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(1, dtype=x.dtype, device=x.device)


class OPNO2d(nn.Module):
    def __init__(
        self,
        degree1,
        degree2,
        width,
        bw1,
        bw2,
        skip=True,
    ):
        super(OPNO2d, self).__init__()

        self.degree1 = degree1
        self.degree2 = degree2
        self.width = width

        self.conv0 = WSWANO(
            self.width, self.width, self.degree1, self.degree2, bw1, bw2
        )
        self.conv1 = WSWANO(
            self.width, self.width, self.degree1, self.degree2, bw1, bw2
        )
        self.conv2 = WSWANO(
            self.width, self.width, self.degree1, self.degree2, bw1, bw2
        )
        self.conv3 = WSWANO(
            self.width, self.width, self.degree1, self.degree2, bw1, bw2
        )

        self.convl = WSWANO(3, self.width - 3, self.degree1, self.degree2, bw1, bw2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)  # .double()
        self.w1 = nn.Conv2d(self.width, self.width, 1)  # .double()
        self.w2 = nn.Conv2d(self.width, self.width, 1)  # .double()
        self.w3 = nn.Conv2d(self.width, self.width, 1)  # .double()

        # self.w4 = nn.Conv2d(self.width, self.width, 1)  # .double()
        # self.w5 = nn.Conv2d(self.width, self.width, 1)  # .double()
        # self.w6 = nn.Conv2d(self.width, self.width, 1)  # .double()
        # self.w7 = nn.Conv2d(self.width, self.width, 1)  # .double()

        self.fc1 = nn.Linear(self.width, 128)  # .double()
        self.fc2 = nn.Linear(128, 1)  # .double()
        # self.fc0 = nn.Linear(3, self.width)  # .double()
        self.skip = nn.Identity() if skip else ZerosFilling()

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        x = torch.cat([x, self.acti(self.convl(x))], dim=1)

        x = self.skip(x) + self.acti(self.w0(x) + self.conv0(x))

        x = self.skip(x) + self.acti(self.w1(x) + self.conv1(x))

        x = self.skip(x) + self.acti(self.w2(x) + self.conv2(x))

        x = self.skip(x) + self.acti(self.w3(x) + self.conv3(x))
        # x = self.w3(x) + self.conv3(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # x = torch.fft.ifftn(x, dim=(1, 2))[:, :Nx, :Nx, :]#.imag
        # x = idstn(dstn(x, [1, 2]), [1, 2])
        # x = iT_pipe(T_pipe(x, [1, 2])[:, 1:x.shape[1]:2, ...], [1, 2])
        # x = iT_pipe(T_pipe(x, [1, 2]), [1, 2])
        # print(x.shape)
        # x = x + comp

        return x
