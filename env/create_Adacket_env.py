import time
import math
import torch
import torch.nn as nn
from lib.utils import prGreen
from lib.data import get_split_dataset
import math
import pandas as pd
from base.freq import freqf as behavior
import numpy as np
from lib.classification import eval_classification
from env.train_test_rl import data_compare
import random
from sklearn.linear_model import RidgeClassifierCV
import pickle
from base.rocket import _PPV
Conv1d_Adacket = nn.Conv1d
def same_padding1d(seq_len, ks, stride=1, dilation=1):
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2
class KernelSearch:
    """
    Env for convolutional kernel search
    """
    def __init__(self, data, preserve_ratio, args, n_data_worker=4,
                 batch_size=32, export_model=False, use_new_input=False):
        self.data_root=args.data_root
        # save options
        self.conv_dict = {}
        self.conv_feature = {}
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.preserve_ratio = preserve_ratio
        self.index = []
        self.pddata =pd.DataFrame()
        # options from args
        self.args = args
        self._init_data()
        self.args.rbound_dilation = self.T_len
        self.args.robound_out = 16
        self.args.rbound_kernel = 10
        self.getmodel(self.n_class)
        # build embedding (static part)
        self._build_state_embedding()
        self.reset()  # restore weight
        self.org_acc,model_ = self.train_agdata([],[],[],0)

        self.best_reward,self.best_loss = -math.inf,math.inf
        self.best_acc = -math.inf
        self.best_strategy = None
        self.best_strategy, self.best_classifier, self.best_model,self.best_channels_out,self.best_kernel_set,self.best_channels = [], self.classifier, [], [],[],[]
        self.org_w_size = sum(self.wsize_list) #params

    def sigmoid(x):
        s = 1/(1 + math.exp(-x))    
        return s

    def reward(self, loss, params):     
        acc = 1/(1 + math.exp(-loss)) 
        if params < 1:
            return -1000000
        gamma = 99
        return (acc / np.log(params)) * (100-gamma) + acc * gamma

    def _init_data(self):
        val_size = 5000
        self.train_loader, self.val_loader, self.channels, self.n_class, self.T_len, self.samples= get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        path = self.data_root,
                                                 
                                                                        shuffle=False)  
    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        for i in range(self.channels):
            for step_, data in enumerate(self.train_loader):
                x, Y_training = data
                ap, avg, ppv,maxf = self.get_feature_state(x[:,i].unsqueeze(1).cpu().detach().numpy(),self.conv1d_9) 
                ap2, avg2, ppv2,maxf2 = self.get_feature_state(x[:,i].unsqueeze(1).cpu().detach().numpy(),self.conv1d_7) 
            this_state = []
            this_state.append(i + 1)  
            this_state.append(ppv)  
            this_state.append(maxf)  
            this_state.append(ap)
            this_state.append(avg)  

            this_state.append(ppv2)  
            this_state.append(maxf2)  
            this_state.append(ap2) 
            this_state.append(avg2)  

            #  features need to be changed later
            this_state.append(0) 
            this_state.append(0)  
            this_state.append(0)
            this_state.append(0)  
            this_state.append(0)  
            this_state.append(0)  
            layer_embedding.append(np.array(this_state))
        # normalize the state
        layer_embedding = np.array(layer_embedding, np.float32)
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        self.layer_embedding = layer_embedding

    def reset(self):
        self.cur = 0
        obs = - np.ones([5, 15])
        self.es = 0
        self.cur_turn = []  
        self.channels_id = [] 
        self.cur_ind = 0 
        self.strategy = []  
        self.kernel_set = [] 
        self.wsize_list = []  
        self.channels_out = []
        self.model_list = [] 
        # reset layer embeddings
        self.layer_embedding[:, -1] = 0 
        self.layer_embedding[:, -2] = 0 
        self.layer_embedding[:, -3] = 0 
        self.layer_embedding[:, -4] = 0. 
        self.layer_embedding[:, -5] = 0. 
        self.layer_embedding[:, -6] = 0. 
        obs[-1] = self.layer_embedding[0].copy()  
        # for share index
        self.visited = [False] * ( self.channels * self.channels)
        self.index_buffer = {}
        
        return obs

    def step(self, action_, step_,obs_pre,last,episode):
        assert self.cur_ind <= int(self.channels)
        action = math.ceil (action_ [0][0] * self.args.rbound_dilation) 
        actionc = action_[0][1]
        actiono = math.ceil( action_[0][2] * (self.args.robound_out - 1)) + 1
        sizek = math.ceil(self.args.rbound_kernel*action_ [0][3]) 
        if action - sizek <= 0:
            sizek = action
            size = 1
        else: 
            size = math.ceil ((action -sizek )/(sizek-1+1e-6))

                  
        lc = self.getid(actionc)
        outc = int(actiono)
        if  outc:
            self.kernel_set.append([sizek,size]) 
            self.strategy.append(action_) 
            self.channels_id.append(lc)
            self.channels_out.append(outc)
            self.wsize_list.append(len(lc) * sizek * outc  ) 
            self.cur_turn.append(self.cur)
        else:
            self.kernel_set.append([0,0])
            self.strategy.append(action_)  
            self.channels_id.append([])
            self.model_list.append([])
            self.channels_out.append(0)
            self.wsize_list.append(0)  
            self.cur_turn.append(-1)
        info_set = None
        info_set_best  = None
        acc,val_acc,search_acc,classifier,model_list = 0,0, 0,[] ,[]
        reward = 0
        done = False
        if step_ >= 100:
            if sum(self.wsize_list) > 50000:
                self.kernel_set[-1] = [0,0]
                self.channels_id[-1] = []
                self.channels_out[-1] = 0
                self.wsize_list[-1] = 0
            if sum(self.wsize_list) == 0:
                loss, model_list = -10000000,[]
                reward = -10000000
            else:
                loss, model_list = self.train_agdata(self.kernel_set,  self.channels_id, self.channels_out,self.cur_turn)
                assert len(model_list) == len(self.kernel_set)
                assert len(model_list) == len(self.channels_id)
                assert len(model_list) == len(self.channels_out)
                reward = self.reward(loss, sum(self.wsize_list))

            if reward >self.best_reward:
                self.best_reward = reward  #float
                self.best_acc = acc
                self.best_strategy = self.strategy.copy() 
                self.best_classifier = classifier
                self.best_model = model_list.copy()  
                self.best_channels = self.channels_id.copy() 
                self.best_channels_out = self.channels_out 
                self.best_size = self.wsize_list.copy() 
                self.best_kernel_set = self.kernel_set.copy() 
                torch.save(self.best_model, self.args.output+'/'+self.args.dataset+'.pkl')
                torch.save(self.best_channels_out, self.args.output+'/'+self.args.dataset+'channelout.pkl')
                torch.save(self.best_channels, self.args.output+'/'+self.args.dataset+'channel.pkl')
                self.best_val_acc = 0
            if self.best_loss > loss:
                self.best_loss = loss
            if last:
                acc, val_acc,search_acc= self.val_forward(self.best_kernel_set, self.best_classifier, self.best_model, self.val_loader, self.best_channels, self.best_channels_out)
                self.best_val_acc = val_acc
            info_set = { 'train_loss': loss, 'kernel': self.kernel_set.copy(), 'strategy': self.strategy.copy(),'channel':self.channels_id.copy(), 'val_accuracy': val_acc, 'val_search':search_acc,'val_size': sum(self.wsize_list ),'channels_out': self.channels_out.copy()}
            info_set_best = { 'feature':sum(self.best_channels_out) * 2 * self.samples * 8,'reward':reward,'best_reward':self.best_reward,'train_accuracy': [acc,val_acc], 'val_search':search_acc,'now_size':sum(self.wsize_list ),'kernel': self.best_kernel_set.copy(), 'strategy': self.best_strategy.copy(),'channel':self.best_channels.copy(), 'val_accuracy': self.best_val_acc, 'best_size': sum(self.best_size ),'channels_out': self.best_channels_out.copy()}
            if sum(self.wsize_list) != 0:
                self.layer_embedding[self.cur_ind][-6] = (self.wsize_list[-1]) / sum(self.wsize_list) 
                self.layer_embedding[self.cur_ind][-5] = (self.channels_out[-1]) / sum(self.channels_out)            
            self.layer_embedding[self.cur_ind][-4] =action_ [0][0]  
            self.layer_embedding[self.cur_ind][-3] = action_ [0][1]  
            self.layer_embedding[self.cur_ind][-2] = action_ [0][2] 
            self.layer_embedding[self.cur_ind][-1] = action_ [0][3]
            obs = - np.ones([5, 15])
            obs[-1] = self.layer_embedding[self.cur_ind].copy()
            for i in range(3,-1,-1):
                obs[i] =  obs_pre[i+1]
            done = True
            return obs, reward, done, info_set,info_set_best   
        self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  
        self.cur_ind = self.cur_ind % self.channels
        if self.cur_ind == 0:
            self.cur = self.cur + 1
        if sum(self.wsize_list)!=0:
            self.layer_embedding[self.cur_ind][-6] = (self.wsize_list[-1]) / sum(self.wsize_list) 
            self.layer_embedding[self.cur_ind][-5] = (self.channels_out[-1]) / sum(self.channels_out)  
        self.layer_embedding[self.cur_ind][-4] =action_ [0][0] 
        self.layer_embedding[self.cur_ind][-3] = action_ [0][1]
        self.layer_embedding[self.cur_ind][-2] = action_ [0][2] 
        self.layer_embedding[self.cur_ind][-1] = action_ [0][3] 
        info_set = None
        info_set_best = None
        obs = - np.ones([5, 15])
        obs[-1] = self.layer_embedding[self.cur_ind].copy()
        for i in range(3,-1,-1):
            obs[i] =  obs_pre[i+1] 
        return obs, reward, done, info_set, info_set_best   

    def getid(self, ks):
        i = self.cur_ind
        ks *= (self.channels - 1)
        channel_step = int(np.clip(ks, 1, self.channels - self.cur_ind))
        channel_indices =[]
        while i < self.channels:
            channel_indices.append(i)
            i = i + channel_step
        return np.array(channel_indices)
    def getmodel(self, nf, stride=1, dilation=1, **kwargs):
        self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
        self.nf = nf
        self.stride = stride
        self.conv1d_9 = Conv1d_Adacket(1, 1, 9, 1, dilation=1).cuda()
        self.bn_ = nn.BatchNorm1d(1).cuda() 
        self.dilation = dilation
        self.conv1d_7 = Conv1d_Adacket(1, 1, 7, 1, dilation=1).cuda()

    def get_feature_state(self, x, conv1d_):
        self.padding_ = same_padding1d(x.shape[-1], conv1d_.kernel_size[0], dilation=1)
        self.relu = nn.ReLU().cuda()
        if len(x.shape) == 4:
            C =(self.bn_(conv1d_(nn.ConstantPad1d(self.padding_, value=0.)(torch.Tensor(x.squeeze(3)).cuda())))).cpu().detach().numpy()
         
        else:
            C = (self.bn_(conv1d_(nn.ConstantPad1d(self.padding_, value=0.)(torch.Tensor(x).cuda())))).cpu().detach().numpy()
        ap1 = np.array(behavior(torch.Tensor(C))[0].squeeze(3).squeeze(2).cpu())
        ap2 = np.array(behavior(torch.Tensor(C))[1].squeeze(3).squeeze(2).cpu())
        ap3 =  _PPV(C, 0).mean(axis=2)
        ap4 = np.array(nn.AdaptiveMaxPool1d(1)(torch.Tensor(C)).squeeze(2).cpu())
        return np.std(ap1), np.average(ap2), np.average(ap3), np.average(ap4)

    def get_params(self,ks, dila,outc,lc, cur):
        ni = len(lc)
        outc = int(outc)
        conv_id = str(cur)+str(lc)+str(outc)+"L"+str(ks)+"D"+str(dila)
        if conv_id in self.conv_dict:
            self.conv1d = self.conv_dict[conv_id][0]
            self.padding = self.conv_dict[conv_id][1]
            self.bn = self.conv_dict[conv_id][2]

        else:
            self.conv1d = Conv1d_Adacket(ni, outc, ks, stride=self.stride, dilation=dila).cuda()
            self.padding = same_padding1d(self.T_len, ks, dilation=dila)
            self.bn = nn.BatchNorm1d(ni,affine=False).cuda() # BN
            self.conv_dict[conv_id] = [self.conv1d,self.padding,self.bn]
        return [self.conv1d, self.padding, self.bn]



    def predict_feature(self, x, ks, conv, outc):
        #self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        outc = int(outc)
        ni = x.shape[1]
        self.conv1d = conv[0]
        self.padding = conv[1]
        self.bn = conv[2]
        self.relu = nn.ReLU().cuda()
        feature = []
        if len(x.shape) == 4:
             C_out = (self.conv1d(nn.ConstantPad1d(self.padding, value=0.)(self.bn(torch.Tensor(x.squeeze(3)).cuda())))).cpu().detach().numpy()
        else:
            C_out =((self.conv1d(nn.ConstantPad1d(self.padding, value=0.)(self.bn(torch.Tensor(x).cuda()))))).cpu().detach().numpy()
        for i in range(outc):
            C = C_out[:,i,:]
            ap = np.array(behavior(torch.Tensor(C)).squeeze(3).squeeze(2).cpu())
            feature.append(ap)
        return feature, [self.conv1d, self.padding, self.bn]
   
    def train_agdata(self, model,channels_id, channels_out,cur_turn):
        if len(model) == 0:
            return  0.0, []
        model_params = []
        if len(model) == 0:
            return  0.0, []
        for i in range(len(model)):
            if model[i][0] != 0:
                lc = channels_id [i]
                outc = int(channels_out[i])
                cur = cur_turn[i]
                conv_params = self.get_params(model[i][0], model[i][1],outc,lc,cur) #kernel,diala
                model_params.append(conv_params)
            else:
                model_params.append(0) 
        loss = data_compare(self.data_type,self.train_loader,self.n_class,channels_id,channels_out, model_params)
        return loss, model_params
        
    
    def val_forward(self, model, s, model_params, val_loader,channels_id, channels_out):
        acc,val_acc,search_acc = self.val_test( model_params, self.train_loader, channels_id, channels_out)
       
        return acc, val_acc,search_acc

    def val_test(self,  model_params, val_loader,channels_id, channels_out):
        for step_, data in enumerate(val_loader):
            if self.data_type == 'soil':
                x, Y_training = data['input'],  data['label']
            else:
                x, Y_training = data
            X_transform = []
            for i in range(len(model_params)):
                if model_params[i] == 0:
                    continue
                lc = channels_id [i]
                outc = int(channels_out[i])
                feature, conv_params = self.transform_feature(x[:,lc].cpu().detach().numpy(), model_params[i],outc)
                for i in feature:
                    X_transform.append(i)

            X_training_transform = np.array(X_transform).swapaxes(0,1)
            if  len(X_training_transform.shape) > 2:
                X_training_transform= X_training_transform.squeeze(2)
       # X_testing_transform = (X_testing_transform - X_testing_transform.min()) /(X_testing_transform.max() - X_testing_transform.min()) 
        clf = self.classifier.fit(X_training_transform, np.array(Y_training.cpu()))
        train_acc = clf.score(X_training_transform, np.array(Y_training.cpu()))
        

        for step_, data in enumerate(self.val_loader):
            if self.data_type == 'soil':
                x, Y_testing = data['input'],  data['label']
            else:
                x, Y_testing = data
            X_transform = []
            for i in range(len(model_params)):
                if model_params[i] == 0:
                    continue
                lc = channels_id [i]
                outc = int(channels_out[i])
                feature, conv_params = self.transform_feature(x[:,lc].cpu().detach().numpy(), model_params[i],outc)
                for i in feature:
                    X_transform.append(i)

            X_testing_transform = np.array(X_transform).swapaxes(0,1)
            if  len(X_testing_transform.shape) > 2:
                X_testing_transform= X_testing_transform.squeeze(2)
        val_acc = clf.score(X_testing_transform, np.array(Y_testing.cpu()))
        #print(train_acc,val_acc)
       # val_network(self.data_type,X_training_transform,np.array(Y_training.cpu()),X_testing_transform, np.array(Y_testing.cpu()))
        clf2,search_acc = eval_classification(X_training_transform,np.array(Y_training.cpu()),X_testing_transform, np.array(Y_testing.cpu()))
        s1 = pickle.dumps(clf)
        s2 = pickle.dumps(clf2)
        torch.save(s1, self.args.output+'/'+self.args.dataset+'class.pkl')
        torch.save(s2, self.args.output+'/'+self.args.dataset+'class_grid.pkl')
        return train_acc,val_acc, search_acc


    def val_test_net(self,  model_params, channels_id, channels_out):

        val_network_test(self.data_type,channels_id,channels_out, model_params )
        
        
        
        return 0


    def transform_feature(self, x, conv, outc):
        outc = int(outc)
        ni = x.shape[1]
        self.conv1d = conv[0]
        self.padding = conv[1]
        self.bn = conv[2]
        self.relu = nn.ReLU().cuda()
        feature = []
        if len(x.shape) == 4:
            C_out = (self.conv1d(nn.ConstantPad1d(self.padding, value=0.)(self.bn(torch.Tensor(x.squeeze(3)).cuda())))).cpu().detach().numpy()
        else:
            C_out = (self.conv1d(nn.ConstantPad1d(self.padding, value=0.)(self.bn(torch.Tensor(x).cuda())))).cpu().detach().numpy()
        for i in range(outc):
            C = C_out[:,i,:]
            ap = C.max(axis = 1).reshape(len(C),1)
            feature.append(ap)
            ap = _PPV(C, 0).mean(axis=1).reshape(len(C),1)
            feature.append(ap)
        return feature, [self.conv1d, self.padding, self.bn]
        
    def test_val(self):
        name = "AtrialFibrillation_2023-01-02-14.38-run1/"#"UWaveGestureLibrary_2023-01-01-21.58-run158/"#"BasicMotions_2023-01-02-14.38-run1/"#"ArticularyWordRecognition_2023-01-02-14.38-run180/"  #"UWaveGestureLibrary_2023-01-01-21.58-run158/"
        self.kernel_set = torch.load(self.args.output+'/'+name+self.args.dataset+'.pkl')
        self.channels_out = torch.load(self.args.output+'/'+name+self.args.dataset+'channelout.pkl')
        self.channels_id= torch.load(self.args.output+'/'+name+self.args.dataset+'channel.pkl')
        acc = self.val_test(self.kernel_set, self.train_loader, self.channels_id, self.channels_out)

    def test_val_model(self):
        self.kernel_set = torch.load(self.args.output+'/'+self.args.dataset+'.pkl')
        self.channels_out = torch.load(self.args.output+'/'+self.args.dataset+'channelout.pkl')
        self.channels_id= torch.load(self.args.output+'/'+self.args.dataset+'channel.pkl')
        if len(self.kernel_set) == 0:
            return 0,0,0
        else:
            train_acc,val_acc, search_acc = self.val_test(self.kernel_set, self.train_loader, self.channels_id, self.channels_out)
            return train_acc,val_acc, search_acc  


