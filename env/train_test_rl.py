import pandas as pd
import numpy as np
import torch
import torch.utils.data
from ts2vec.utils import init_dl_program
import torch
import numpy as np
from .ConvModel import Net
from sktime.datasets import load_from_tsfile_to_dataframe
import sys
from ts2vec.ts2vec import TS2Vec
import argparse
sys.path.append('.')

def readucr(filename):
    data= load_from_tsfile_to_dataframe(filename)
    return data

def readucr(filename):
    data= load_from_tsfile_to_dataframe(filename)
    return data

from datetime import *
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='runs',help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', default='UEA',type=str,  help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', default=1,action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')

    return parser.parse_args()
def data_compare(each, train_loader, nb_classes, channels_id, channels_out, model_params ):
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    args = parse_args()
    rl_model=Net(channels_id, channels_out, model_params,nb_classes).to(device)
    
    print("Dataset:", each)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    loss=0
    for train_data in train_loader:
        train_data = np.array(np.transpose(train_data[0].cpu(), (0, 2, 1)))  # sample,len,dim
        model = TS2Vec(
        input_dims=train_data.shape[1],
        device=device,
        **config
        )
        loss +=  model.fit_our(rl_model,train_data,verbose=True)
    
    return loss.item()


        


