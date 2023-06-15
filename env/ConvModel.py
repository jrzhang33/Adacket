
import torch.nn as nn
from tsai.all import *

class Net(Module):
    def __init__(self, channels_id, channels_out, model_params,c_out):
        self.conv_group = nn.ModuleList([])
        self.padding_group = []
        self.bn_group = nn.ModuleList([])
        self.channel_in = []
        self.channel_out = channels_out
        self.gap = GAP1d(1)
        self.c_out = c_out
        self.relu = nn.ReLU()
      #  self.fc = nn.Linear(sum(self.channel_out), self.c_out)
        for i in range(len(model_params)):
            if model_params[i] != 0:
                self.channel_in.append(channels_id[i])
                self.conv_group.append(model_params[i][0])
                #self.padding_group.append(model_params[i][1])
                self.bn_group.append(model_params[i][2])

    def forward(self, input):
        x_out = torch.Tensor([]).cuda()
        for i in range(len(self.conv_group)):
            lc = self.channel_in [i]
            conv = self.conv_group[i]
            bn = self.bn_group[i]
            x = input[:,lc].cpu().detach().numpy()
            padding = same_padding1d(x.shape[-1], conv.kernel_size[0], dilation=conv.dilation[0])
            if len(x.shape) == 4:
                C_out = self.relu((conv(nn.ConstantPad1d(padding, value=0.)(bn((torch.Tensor(x.squeeze(3)).cuda()))))))
            else:
                C_out = self.relu((conv(nn.ConstantPad1d(padding, value=0.)(bn((torch.Tensor(x).cuda()))))))

          #  C_out = self.gap(C_out)
            if len(x_out) == 0:
                x_out = C_out
            else:
                x_out=torch.cat([C_out,x_out],1)
        return x_out