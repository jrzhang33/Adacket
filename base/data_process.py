
import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .MyDataset import MyDataset
from .Muldataloader import process_ts_data
from sktime.datasets import load_from_tsfile_to_dataframe
from .TSC_data_loader import TSC_multivariate_data_loader
from .ts_datasets import UCR_UEADataset
from regulator.TCE import Low2High
def readucr(filename, loader):
    if loader == "UCR":
        data = pd.read_csv(filename,sep="  ",header=None )
        Y = data.iloc[0:len(data),0]
        X = data.iloc[0:len(data),1:data.shape[1]]
        if X.shape[1] == 0:
            data = pd.read_csv(filename,sep=",",header=None )
            Y = data.iloc[0:len(data),0]
            X = data.iloc[0:len(data),1:data.shape[1]] 
            X[np.isnan(X)] = 0
        return X, Y
    else:
        data= load_from_tsfile_to_dataframe(filename)
        return data


def get_split_dataset(loader, each, root, batch_size):
    if loader == "UEA":
        path, fname = root, each
        if each in ["InsectWingbeat","Phoneme"]:      
            X_train, y_train,X_test,y_test = TSC_multivariate_data_loader(path, fname)
        else:        
            X_train, y_train = readucr(path+fname+'/'+fname+'_TRAIN.ts', loader)
            X_train = process_ts_data(X_train, normalise=False)
            X_test, y_test = readucr(path+fname+'/'+fname+'_TEST.ts', loader)
            X_test = process_ts_data(X_test, normalise=False)
            class_le = LabelEncoder()
            y_train = class_le.fit_transform(y_train)
            y_test = class_le.fit_transform(y_test)
        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        channels=X_train.shape[1]
        batch_size = min(int(X_train.shape[0]/10), batch_size)  
        train_index=np.array(range(len(y_train))).reshape(len(y_train),1)
        val_index=np.array(range(len(y_test))).reshape(len(y_test),1)
        train_data=MyDataset(X_train,y_train,train_index)
        validation_data=MyDataset(X_test,y_test,val_index)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0,drop_last=True)

        validate_loader = torch.utils.data.DataLoader(validation_data,
                                            batch_size=batch_size, shuffle=True,num_workers=0,drop_last=True)
        return train_loader, validate_loader, nb_classes, channels, X_train.shape[-1]
    else:
        path, fname = root, each
        try:
            x_train, y_train = readucr(path+fname+'/'+fname+'_TRAIN.txt', loader)
            x_train=x_train.to_numpy()
            y_train=y_train.to_numpy()
            x_test, y_test = readucr(path+fname+'/'+fname+'_TEST.txt', loader)
            x_test=x_test.to_numpy()
            y_test=y_test.to_numpy()
        except:
            x_train, y_train,x_test,y_test = [],[],[],[]
            train_dataset = UCR_UEADataset(fname, split="train",extract_path = "Univer")
            test_dataset = UCR_UEADataset(fname, split="test",extract_path = "Univer")
            for i in range(len(train_dataset)):
                x_train.append(train_dataset[i]['input'][:,0].numpy())
                y_train.append(train_dataset[i]['label'].numpy())
            for i in range(len(test_dataset)):
                x_test.append(test_dataset[i]['input'][:,0].numpy())
                y_test.append(test_dataset[i]['label'].numpy())
            x_train, y_train, x_test, y_test  = np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
        nb_classes = len(np.unique(y_test))
        y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
        channels=1
        batch_size = min(int(x_train.shape[0]/10), batch_size)
        batch_size_test = min(int(x_test.shape[0]/10), batch_size)
        x_train=x_train.reshape(x_train.shape[0],channels,x_train.shape[1],1)
        x_test=x_test.reshape(x_test.shape[0],channels,x_test.shape[1],1)
        train_index=np.array(range(len(y_train))).reshape(len(y_train),1)
        val_index=np.array(range(len(y_test))).reshape(len(y_test),1)
        train_data=MyDataset(x_train,y_train,train_index)
        validation_data=MyDataset(x_test,y_test,val_index)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0,drop_last=True)

        validate_loader = torch.utils.data.DataLoader(validation_data,
                                            batch_size=batch_size_test, shuffle=True,num_workers=0,drop_last=True)
        return train_loader, validate_loader, nb_classes, channels, x_train.shape[-2]