"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn / liuziyuan@pku.edu.cn
"""

import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

from SOL2_0.NOs_dict.models import SinNO2d
from SOL2_0.Adam import Adam

from packages.utilities3 import LpLoss, count_params
from packages.loss import weak, strong
from packages.datasets import LoadData, test_dataset, rand_loader

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import argparse


def get_args():
    parser = argparse.ArgumentParser("Spectral Operator Learning", add_help=False)

    ## data
    parser.add_argument(
        "--data-dict",
        default="/code/poisson/data/",
        type=str,
        help="dataset folder",
    )
    parser.add_argument("--sub", default=4, type=int, help="sub-sample on the data")
    parser.add_argument("--test-size", default=100, type=int, help="")
    parser.add_argument("--train-size", default=800, type=int, help="")
    parser.add_argument("--weak-size", default=0, type=int, help="")
    parser.add_argument("--strong-size", default=0, type=int, help="")
    parser.add_argument("--unfixed", default=True, type=bool, help="")
    parser.add_argument("--alpha", default=2, type=int, help="alpha for GRF")

    ## train
    parser.add_argument("--epochs", default=500, type=int, help="training iterations")
    parser.add_argument("--batch-size", default=20, type=int, help="batch size")
    parser.add_argument(
        "--same-batches",
        default=True,
        type=bool,
        help="whether to hold number of batchs per epoch",
    )
    # optimizer&scheduler
    parser.add_argument("--wd", default=-4, type=float, help="weight decay")
    parser.add_argument("--lr", default=4e-3, type=float, help="learning rate")
    parser.add_argument(
        "--step-size", default=40, type=int, help="step size for the StepLR (if used)"
    )
    parser.add_argument(
        "--patience", default=50, type=int, help="patience for the ReducedLR (if used)"
    )

    ## model
    parser.add_argument(
        "--model-dict",
        default="/code/poisson/models/",
        type=str,
        help="model folder",
    )
    parser.add_argument(
        "--bw", default=1, type=int, help="band width"
    ) 
    parser.add_argument("--modes", default=24, type=int, help="Fourier-like modes")
    parser.add_argument("--width", default=32, type=int, help="")
    parser.add_argument("--triL", default=0, type=int, help="")
    parser.add_argument("--suffix", default="", type=str, help="")
    parser.add_argument("--sol-skipflag", default=1, type=int, help="")

    return parser.parse_args()


#### parameters settings
args = get_args()

epochs = args.epochs  # default 500
step_size = args.step_size  # for StepLR, default 50
batch_size = args.batch_size  # default 20
same_batches = args.same_batches
sub = args.sub  # default 4
learning_rate = args.lr  # default 1e-3
bandwidth = args.bw  # default 1
modes = args.modes
triL = args.triL
suffix = args.suffix
sol_skipflag = args.sol_skipflag
width = args.width
wd = args.wd
weight_decay = 10**wd  # 1e-4
train_size, weak_size, strong_size = args.train_size, args.weak_size, args.strong_size
un_train_size = weak_size + strong_size
test_size = args.test_size
patience = args.patience
unfixed = args.unfixed  
alpha = args.alpha

device = torch.device("cuda:0")
ddtype = torch.float64
train_withdata = False

#### data settings
data_label = "res401alpha2"
data_name = "elli2d-Nx401-GRF-alpha2-0.pt"
data_PATH = args.data_dict + data_name

scheduler_label = "ReduLRpa" + str(patience)
if unfixed:
    file_name = (
        "model-data_"
        + data_label
        + "-sub"
        + str(sub)
        + "-datasize_labeled"
        + str(train_size)
        + "_weak"
        + str(weak_size)
        + "_strong"
        + str(strong_size)
        + "-modes"
        + str(modes)
        + "-"
        + scheduler_label
        + "-epochs"
        + str(epochs)
    )
else:
    file_name = (
        "model-data_"
        + data_label
        + "-sub"
        + str(sub)
        + "-datasize_labeled"
        + str(train_size)
        + "_fixed_weak"
        + str(weak_size)
        + "_strong"
        + str(strong_size)
        + "-modes"
        + str(modes)
        + "-"
        + scheduler_label
        + "-epochs"
        + str(epochs)
    )

notes = ""
if same_batches:
    notes = "-same_batches"  # +'-batch_size'+str(batch_size)

file_name = file_name + notes
result_PATH = args.model_dict + file_name + ".pkl"
print("model:", file_name)

if not os.path.exists(args.model_dict):
    print("----------Warning: model path does not exist:")
    print(args.model_dict)
    halt

if os.path.exists(result_PATH):
    print("----------Warning: pre-trained model already exists:")
    print(result_PATH)

#### main

## load data
if train_size > 0:
    train_withdata = True
print("Data set:", data_name)
data_size, Nx, train_loader, test_loader = LoadData(
    data_dir=args.data_dict,
    name=data_name,
    sub=sub,
    alpha=alpha,
    batch_size=batch_size,
    test_size=test_size,
    train_size=train_size,
    w=weak_size,
    s=strong_size,
    same_batches=same_batches,
    unfixed=unfixed,
)
print(f"Data size: {data_size}, resolution: {Nx}*{Nx}")


## model
model = (
    SinNO2d(3, modes, width, bandwidth, triL=triL, skip=sol_skipflag)
    .to(device)
    .double()
)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, threshold=1e-1, patience=patience
)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*((train_size+weak_size+strong_size)//batch_size))
print("--model parameters number =", count_params(model))
print("--scheduler:", scheduler_label)

myloss = LpLoss(size_average=False)

train_list, test_list, weak_list, test_weak_list = [], [], [], []
strong_list, test_strong_list = [], []

if un_train_size == 0:
    unfixed = False

# train
t0 = default_timer()
for ep in range(epochs):
    if unfixed:
        train_loader = rand_loader(
            size=un_train_size,
            Nx=Nx,
            batch_size=batch_size,
            alpha=alpha,
            same_batches=same_batches,
        )

    train_l2, weak_residual, strong_residual = 0, 0, 0
    scheduler_loss = 0

    model.train()
    t1 = default_timer()
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        out = model(x).reshape(batch_size, Nx, Nx)

        if train_withdata:
            data_l2 = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            train_l2 += data_l2.item() / batch_size
            data_l2.backward()
            optimizer.step()
            scheduler_loss += train_l2
            # scheduler.step()
        else:
            f = x[:, :, :, 0].reshape(batch_size, Nx, Nx)
            weak_loss = weak(out, f)
            strong_loss = strong(out, f)
            weak_residual += weak_loss.item() / batch_size
            strong_residual += strong_loss.item() / batch_size

            if weak_size > 0:
                weak_loss.backward()
                optimizer.step()
                scheduler_loss += weak_loss
                # scheduler.step()
            else:
                strong_loss.backward()
                optimizer.step()
                scheduler_loss += strong_loss
                # scheduler.step()

    # scheduler.step()
    scheduler.step(scheduler_loss)

    t2 = default_timer()

    # storing
    train_l2 /= len(train_loader)
    weak_residual /= len(train_loader)
    strong_residual /= len(train_loader)

    train_list.append(train_l2)
    weak_list.append(weak_residual)
    strong_list.append(strong_residual)

    model.eval()
    test_l2, test_weak, test_strong = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            b = x.shape[0]
            f = x[:, :, :, 0].reshape(b, Nx, Nx)
            out = model(x).reshape(b, Nx, Nx)

            test_l2 += myloss(out.reshape(b, -1), y.reshape(b, -1)).item()

            weak_loss = weak(out, f)
            test_weak += weak_loss.item()

            strong_loss = strong(out, f)
            test_strong += strong_loss.item()

    test_l2 /= (
        test_size  
    )
    test_list.append(test_l2)
    test_weak /= test_size
    test_weak_list.append(test_weak)
    test_strong /= test_size
    test_strong_list.append(test_strong)

    print(
        ep,
        str(t2 - t1)[:4],
        optimizer.state_dict()["param_groups"][0]["lr"],
        train_l2,
        test_weak,
        test_strong,
        test_l2,
    )
t_end = default_timer()
train_time = str(t_end - t0)[:4]

## save model
if epochs >= 500:
    import inspect

    current_code = inspect.getsource(inspect.currentframe())
    torch.save(
        {
            "model": model.state_dict(),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "width": width,
            "modes": modes,
            "bandwidth": bandwidth,
            "triL": triL,
            "train_time": train_time,
            "train_list": train_list,
            "weak_list": weak_list,
            "strong_list": strong_list,
            "code": current_code,
            "scheduler": scheduler,
            "test_list": test_list,
            "test_weak_list": test_weak_list,
            "test_strong_list": test_strong_list,
        },
        result_PATH,
    )
plt.cla()

print("End of traing !")
