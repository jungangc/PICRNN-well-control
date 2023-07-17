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

from PhyCRNet import PhyCRNet
from train_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)


BHPmat = scio.loadmat('BHP_full.mat')
BHP_vec = torch.tensor(BHPmat['BHP_full'], dtype=torch.float32).cuda()
# u_vec = torch.tensor([1500,1500], dtype=torch.float32).cuda()
# BHP_vec = u_vec.repeat(1, 200)

if __name__ == '__main__':
    print(os.getcwd())
    ######### download the ground truth data ############
#     data_dir = '/content/PhyCRNet-main-1phaseflow/Datasets/data/2dBurgers/burgers_1501x2x128x128.mat'    
    data_dir = '/scratch/user/jungangc/PICNN/PhyCRNet-main-1phaseflow-final-paramBHP-64by64-200by05/Datasets/data/onephaseflow/pressure_501x1x64x64.mat'  
    data = scio.loadmat(data_dir)
    # uv = data['uv'] # [t,c,h,w]  
    uv = data['Psim'] # [t,c,h,w]  

    # initial conidtion
    uv0 = uv[0:1,...] 
    inputs = torch.tensor(uv0, dtype=torch.float32).cuda() 
    # inputs = inputs/3000.0
    print("input size: ", inputs.size())
    # set initial states for convlstm
    num_convlstm = 1
#     (h0, c0) = (torch.randn(1, 128, 16, 16), torch.randn(1, 128, 16, 16))
    # (h0, c0) = (torch.randn(1, 64, 64, 64), torch.randn(1, 64, 64, 64))
    (h0, c0) = (torch.zeros(1, 64, 8, 8), torch.zeros(1, 64, 8, 8))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))
    

    # source_BHP[:,0, 9, 9] = torch.tensor([1500], dtype=torch.float32).repeat(200).cuda()
    # source_BHP[:,0, 54, 54] = torch.tensor([1500], dtype=torch.float32).repeat(200).cuda()
    # source_BHP = source_BHP/3000.0
    # print(source_BHP.size())
    # grid parameters
    time_steps = 300
    dt = 0.5
    dx = 1.0 / 128
    
    # source BHP 
    BHP = np.zeros((time_steps,1,64,64))
    source_BHP = torch.tensor(BHP, dtype=torch.float32).cuda() 
    source_BHP[:,0, 9, 9] = BHP_vec[0,:time_steps]
    source_BHP[:,0, 54, 54] = BHP_vec[1,:time_steps]

    ################# build the model #####################
    time_batch_size = 300
    steps = time_batch_size
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters_adam = 30000
    lr_adam = 2.3e-3 #1e-3 
    pre_model_save_path = '/scratch/user/jungangc/PICNN/PhyCRNet-main-1phaseflow-final-paramBHP-64by64-200by05/Codes/model/checkpoint1000.pt'
    model_save_path = '/scratch/user/jungangc/PICNN/PhyCRNet-main-1phaseflow-final-paramBHP-64by64-200by05/Codes/model/checkpoint2000.pt'
    fig_save_path = '/scratch/user/jungangc/PICNN/PhyCRNet-main-1phaseflow-final-paramBHP-64by64-200by05/Datasets/figures/'  

    model = PhyCRNet(
        input_channels = 1, 
  #     hidden_channels = [8, 32, 64, 64, 64, 32, 8], 
        hidden_channels = [16, 32, 64, 64, 64, 32, 16], 
        input_kernel_size = [4, 4, 4, 3, 3, 3, 3], 
        input_stride = [2, 2, 2, 1, 1, 1, 1], 
        input_padding = [1, 1, 1, 1, 1, 1, 1],  
        dt = dt,
        num_layers = [3, 1, 3, 3],
        upscale_factor = 8,
        step = steps, 
        effective_step = effective_step).cuda()

    start = time.time()
    train_loss = train(model, inputs, source_BHP, initial_state, n_iters_adam, time_batch_size, 
        lr_adam, dt, dx, model_save_path, pre_model_save_path, num_time_batch)
    end = time.time()
    
    np.save('/scratch/user/jungangc/PICNN/PhyCRNet-main-1phaseflow-final-paramBHP-64by64-200by05/Codes/model/train_loss', train_loss)  
    print('The training time is: ', (end-start))



















