import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm

class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):

        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')
        self.Wui = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')
        


        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')        
        self.Wuf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')
        self.Wuc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')
        self.Wuo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')       

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)        
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, u, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Wui(u) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Wuf(u) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Wuc(u) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Wuo(u) +  self.Who(h))
        ch = co * torch.tanh(cc)
        # ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        # cf = torch.sigmoid(self.Wxf(x)  + self.Whf(h))
        # cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        # co = torch.sigmoid(self.Wxo(x) +  self.Who(h))
        # ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())



class encoder_block(nn.Module):
    ''' encoder with CNN '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):
        
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='circular'))

        # self.act = nn.ReLU()
        self.act = nn.Tanh()
        # self.act = nn.SiLU()
        
        # self.Dropout = nn.Dropout()
        # nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        # x= self.Dropout(x)
        return x


class decoder_block(nn.Module):
    ''' decoder with CNN '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):
        
        super(decoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.deconv = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='circular'))

        # self.act = nn.ReLU()
        self.act = nn.Tanh()
        # self.act = nn.SiLU()
        # self.Dropout = nn.Dropout()
        nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        x = self.upsampling(x)
        x = self.deconv(x)
        x = self.act(x)
        # x = self.Dropout(x)
        return x


class source_encoder_block(nn.Module):
    ''' encoder source term with CNN '''
    def __init__(self, input_channels=64, hidden_channels=64, input_kernel_size=5, 
        input_stride=1, input_padding=2, downscale_factor=8):
        
        super(source_encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        # self.conv_source = weight_norm(nn.Conv2d(self.input_channels, 
        #     self.hidden_channels, self.input_kernel_size, self.input_stride, 
        #     self.input_padding, bias=True, padding_mode='circular'))
        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_source = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='circular'))

        # self.act = nn.ReLU()
        self.act = nn.Tanh()
        # self.act = nn.SiLU()
        # self.Dropout = nn.Dropout()
        nn.init.zeros_(self.conv_source.bias)

    def forward(self, source):
          x= self.pixelunshuffle(source)
          x = self.conv_source(x)
          x = self.act(x)
          # x= self.Dropout(x)
        # return self.act(self.conv_source(source))
          return x