
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
from base.Muldataloader import process_ts_data
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder
from base.MyDataset import OneDataset
import pandas as pd
def readucr(filename):
    data= load_from_tsfile_to_dataframe(filename)
    return data
def ureaducr(filename):
    data = pd.read_csv(filename,sep="  ",header=None)
    Y = data.iloc[0:len(data),0]
    X = data.iloc[0:len(data),1:data.shape[1]]
    return X, Y
import sys
from .TSC_data_loader import TSC_multivariate_data_loader
def get_split_dataset(fname, batch_size, n_worker, val_size, path,
                      use_real_val=False, shuffle=True):
    try:      
        X_train, y_train,X_test,y_test = TSC_multivariate_data_loader(path, fname)
    except:
        X_train, y_train = readucr(path+fname+'/'+fname+'_TRAIN.ts')
        X_train = process_ts_data(X_train, normalise=False)
        X_test, y_test = readucr(path+fname+'/'+fname+'_TEST.ts')
        X_test = process_ts_data(X_test, normalise=False)
    class_le = LabelEncoder()
    y_train = class_le.fit_transform(y_train)
    y_test = class_le.fit_transform(y_test)
    n_class = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    channels=X_train.shape[1]
    data_column = 1
    train_data = OneDataset(X_train, y_train)
    test_data = OneDataset(X_test, y_test)
    if len(y_train) < batch_size:
        batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size= len(y_train), shuffle=False,
                                            num_workers=0,drop_last=True)

    val_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=len(y_test), shuffle=False,
                                            num_workers= 0,drop_last=True)
    return train_loader, val_loader, channels, n_class, X_train.shape[-1],X_train.shape[0]

