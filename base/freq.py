import numpy as np
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data
import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
RANDOM_SEED=2222
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) #  
    torch.backends.cudnn.deterministic = True #  
    torch.backends.cudnn.benchmark = False 
set_seed(RANDOM_SEED)

def transfft(x_input):  #4d
    if len(x_input.shape) == 3:
        x_input = x_input.unsqueeze(1)
    if len(x_input.shape) == 2:
        x_input = x_input.unsqueeze(2).unsqueeze(1)
  #  x_fft_init = torch.fft.rfft(x_input[..., 0],dim=2,norm="forward")
    x_fft_init = torch.fft.fft(x_input[..., 0],dim=2,norm="forward")
    x_fft = torch.stack((x_fft_init.real, x_fft_init.imag), -1)
    x_fft = x_fft.detach().cpu().numpy()

    for i in range(x_fft.shape[0]): 
        if i==0:
            ff = np.sqrt(x_fft[i,:,:,0]**2 + x_fft[i,:,:,1]**2) 
            ff=ff.reshape(1,x_fft.shape[1],x_fft.shape[2],1)
            continue 
        f = np.sqrt(x_fft[i,:,:,0]**2 + x_fft[i,:,:,1]**2).reshape(1,x_fft.shape[1],x_fft.shape[2],1)
        ff=np.concatenate([ff,f],0)
    x_fft=torch.from_numpy(ff[:,:,:-1,:]).to(device)
    return x_fft,x_fft_init


def freqf(x:Tensor):
    if len(x.shape) == 2:
        x=x.unsqueeze(1).unsqueeze(3)
    elif len(x.shape) == 3:
        x=x.unsqueeze(3)
    b, c, _ ,_= x.size()
    x_fft,x_fft_init=transfft(x)   
    rms = nn.AdaptiveAvgPool2d(1)(x_fft[:,:,:]*x_fft[:,:,:]).sqrt().view(b,c,1,1)
    max=nn.AdaptiveMaxPool2d(1)(abs(x_fft[:,:,:])).view(b,c,1,1)
    y_p=max/rms 
    peak=torch.where(torch.isnan(y_p), torch.full_like(y_p, 0), y_p)   
    k1=torch.Tensor(np.arange(1,x_fft.shape[2]+1)).to(device).repeat(x_fft.shape[0],x_fft.shape[1],1)
    fc=k1.unsqueeze(3)*x_fft
    centriod=(torch.sum(fc,dim=2)/torch.sum(x_fft,dim=2)).view(b,c,1,1) 

    return peak,centriod
