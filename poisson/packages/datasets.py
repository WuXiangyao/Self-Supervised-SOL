import torch
from torch.utils.data import DataLoader

from .GRF_ref_16 import gaussian_random_field_batch
from .loss import weak, strong

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

## load data
def rand_loader(size, Nx, batch_size = 20, alpha = 2, same_batches = True):
    # combine input data with mesh
    x_unif = torch.linspace(0, 1, Nx)
    mesh = torch.stack(torch.meshgrid(x_unif, x_unif, indexing = 'ij'), dim=-1)
    f_data = torch.from_numpy(gaussian_random_field_batch(size, alpha, Nx))
    f_data = f_data[..., None]
    f_data = torch.cat([f_data, mesh[None, ...].repeat(f_data.shape[0], 1, 1, 1)], dim=-1)
    u_data = torch.zeros(f_data.shape[0], Nx, Nx)
    if same_batches:
        f_data = f_data.repeat(int(800/size),1,1,1)
        u_data = u_data.repeat(int(800/size),1,1)
    return  torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(f_data, u_data), batch_size=batch_size,
        shuffle=True)
    
def LoadData(data_dir = 'poisson/data/',
             name = '', 
             sub = 1, batch_size = 20, alpha = 2, test_size = 100,
             train_size = 0, w = 0, s = 0, 
             same_batches = True, unfixed = False, 
             ):
    data_PATH = data_dir +name
    un_train_size = w + s
    # raw_data = loadmat(data_PATH)
    raw_data = torch.load(data_PATH, weights_only=True)
    x_data, y_data = raw_data['f'], raw_data['u']
    x_data, y_data = x_data[:, ::sub, ::sub].clone().detach(), y_data[:, ::sub, ::sub].clone().detach()

    data_size, Nx, _ = x_data.shape

    # combine input data with mesh
    x_unif = torch.linspace(0, 1, Nx)
    mesh = torch.stack(torch.meshgrid(x_unif, x_unif, indexing = 'ij'), dim=-1)
    x_data = x_data[..., None] 
    x_data = torch.cat([x_data, mesh[None, ...].repeat(x_data.shape[0], 1, 1, 1)], dim=-1)

    if train_size > 0:
        f_data = x_data[:train_size]
        u_data = y_data[:train_size]
        if same_batches:
            f_data = f_data.repeat(int(800/train_size),1,1,1)
            u_data = u_data.repeat(int(800/train_size),1,1)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_data, u_data), batch_size=batch_size,
            shuffle=True)

    if un_train_size > 0:
        if unfixed:
            print('Unlabeled data will be generated at each epoch')
            train_loader = 0
        else:
            f_data = torch.from_numpy(gaussian_random_field_batch(un_train_size, alpha, Nx))
            f_data = f_data[..., None]
            f_data = torch.cat([f_data, mesh[None, ...].repeat(f_data.shape[0], 1, 1, 1)], dim=-1)
            u_data = torch.zeros(f_data.shape[0], Nx, Nx)
            if same_batches:
                f_data = f_data.repeat(int(800/un_train_size),1,1,1)
                u_data = u_data.repeat(int(800/un_train_size),1,1)
            train_loader = DataLoader(
                torch.utils.data.TensorDataset(f_data, u_data), batch_size=batch_size,
                shuffle=True)
            print('Unlabeled data loaded! GRF alpha = '+str(alpha))

    test_loader = DataLoader(
        torch.utils.data.TensorDataset(x_data[-test_size:], y_data[-test_size:]), batch_size=batch_size,
        shuffle=False)

    # print('--data set size = ', data_size, 'Nx = ', Nx, 'sub = ',sub)
    # print('--labeled: '+str(train_size),'weak: '+str(w),'strong:'+str(s))
    # print('--batch_size: '+str(batch_size),'number of batchs: '+str(len(train_loader)))
    return data_size, Nx, train_loader, test_loader

def test_dataset(train_loader, batch_size, Nx):
    data_weakloss, data_strongloss = 0, 0
    for x,y in train_loader:
        x, y = x.cuda(), y.cuda()
        x = x[:,:,:,0].reshape(batch_size, Nx, Nx)
        dw = weak(y, x)
        ds = strong(y, x)
        data_weakloss += dw
        data_strongloss += ds

    data_weakloss /= batch_size * len(train_loader)
    data_strongloss /= batch_size * len(train_loader)
    print('weak loss='+str(data_weakloss.item()),'strong loss='+str(data_strongloss.item()))

def Load_testdata(data_dir = 'poisson/data/',
             name = '', 
             sub = 1, batch_size = 1, test_size = 100,
             comb = True
             ):
    data_PATH = data_dir +name
    # raw_data = loadmat(data_PATH)
    raw_data = torch.load(data_PATH,weights_only=True)
    x_data, y_data = raw_data['f'], raw_data['u']
    x_data, y_data = x_data[:, ::sub, ::sub].clone().detach(), y_data[:, ::sub, ::sub].clone().detach()

    data_size, Nx, _ = x_data.shape

    # combine input data with mesh
    if comb:
        x_unif = torch.linspace(0, 1, Nx)
        mesh = torch.stack(torch.meshgrid(x_unif, x_unif, indexing = 'ij'), dim=-1)
        x_data = x_data[..., None] 
        x_data = torch.cat([x_data, mesh[None, ...].repeat(x_data.shape[0], 1, 1, 1)], dim=-1)
    else:
        x_data = x_data[..., None] 
        
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(x_data[-test_size:], y_data[-test_size:]), batch_size=batch_size,
        shuffle=False)

    return data_size, Nx, test_loader
