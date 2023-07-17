import torch
import torch.nn as nn

import numpy as np
import scipy.io as scio
import os

# define the high-order finite difference kernels

Tmat = scio.loadmat('T_full.mat')
T = torch.tensor(Tmat['T_full'], dtype=torch.float32).cuda()

Accmat = scio.loadmat('Acc_full.mat')
Acc = torch.tensor(Accmat['Acc_full'], dtype=torch.float32).cuda()

Bmat = scio.loadmat('B_full.mat')
B = torch.tensor(Bmat['B_full'], dtype=torch.float32).cuda()

# u_vec = torch.tensor([1500,1500], dtype=torch.float32).cuda()
# U_vec = u_vec.repeat(202, 1, 1)
BHPmat = scio.loadmat('BHP_full.mat')
u_vec = torch.tensor(BHPmat['BHP_full'], dtype=torch.float32).cuda()
print(u_vec.size())
# u_vec1 = torch.cat((u_vec , u_vec[:, -2:-1]), dim=1)
u_vec1 = u_vec[:, :300]
u_vec2 = torch.unsqueeze(u_vec1, 0)
U_vec = u_vec2.permute(2, 0, 1)  # [t, c, n_control]

# output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
# U_vec = U_vec/3000
# print(U_vec.size())

Acc_inv = torch.linalg.inv(Acc)
M = torch.linalg.matmul(Acc_inv, T)
N = torch.linalg.matmul(Acc_inv, B)

P_mat = scio.loadmat('pressure_data.mat')
P_data = torch.tensor(P_mat['Pobs'], dtype=torch.float32).cuda()
# output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
# U_vec = U_vec/3000
print(P_data.size())

## ordering from the MATLAB matrix is difference than that in Pytorch 
# in MATLAB, the ordering is column first, row last; 
# in Pytorch, the ordering is row first, column last; 

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol

class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol
    
class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, delta_t , dx):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

#         # spatial derivative operator
#         self.laplace = Conv2dDerivative(
#             DerFilter = lapl_op,
#             resol = (dx**2),
#             kernel_size = 5,
#             name = 'laplace_operator').cuda()

#         self.dx = Conv2dDerivative(
#             DerFilter = partial_x,
#             resol = (dx*1),
#             kernel_size = 5,
#             name = 'dx_operator').cuda()

#         self.dy = Conv2dDerivative(
#             DerFilter = partial_y,
#             resol = (dx*1),
#             kernel_size = 5,
#             name = 'dy_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (delta_t *2),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, output, delta_t ):

#         # spatial derivatives
        # temporal derivative - p
        # print("output size: ", output.size())
        p_out = output[1:, 0:1, :, :]
        p_init = output[:-1, 0:1, :, :]
        p_diff = p_out-p_init
        p_diff_t = p_diff/delta_t
        lent = p_out.shape[0]
        lenx = p_out.shape[3]
        leny = p_out.shape[2]
        # make p_out and p_init to vector form   (lent, 1, lenx*leny)
        p_out1d = p_out.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        p_out_128by64 = p_out1d.transpose(0,1)
        p_out_vec = p_out_128by64.reshape(lenx*leny,1,lent)
        P_out_vec = p_out_vec.permute(2,1,0)

        p_init1d = p_init.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        p_init_128by64 = p_init1d.transpose(0,1)
        p_init_vec = p_init_128by64.reshape(lenx*leny,1,lent)
        P_init_vec = p_init_vec.permute(2,1,0)

        # temporal derivative - p_t
        p_conv1d = p_diff_t.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        # p_conv1d = p_conv1d.reshape(lenx*leny,1,lent)
        p_dt_128by64 = p_conv1d.transpose(0,1)
        p_dt_vec = p_dt_128by64.reshape(lenx*leny,1,lent)
        P_dt_vec = p_dt_vec.permute(2,1,0)


        p = output[1:, 0:1, :,:]  # [t, c, height(Y), width(X)]
        p_in = p.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        # p_conv1d = p_conv1d.reshape(lenx*leny,1,lent)
        p_128by64 = p_in.transpose(0,1)
        p_vec = p_128by64.reshape(lenx*leny,1,lent)
        P_vec = p_vec.permute(2,1,0)

        # 2D one phase flow eqn
        f_p = torch.einsum('ij, tcj->tci', -Acc, P_dt_vec)+\
              torch.einsum('ij, tcj->tci', T, P_vec)+\
              torch.einsum('ij, tcj->tci', B, U_vec)
        # # 2D one phase flow eqn
        # f_p = P_init_vec + \
        #       delta_t*(torch.einsum('ij, tcj->tci', M, P_out_vec)+\
        #               torch.einsum('ij, tcj->tci', N, U_vec))

        # residual = torch.cat((f_p[:20,:,:]*5, f_p[20:80,:,:], f_p[80:120,:,:]*3, f_p[120:,:,:]),dim=0)
        return f_p, P_out_vec

def compute_loss(output, loss_func, delta_t):
    ''' calculate the phycis loss '''

    # get physics loss
    # mse_loss = nn.MSELoss()
    mse_loss = nn.SmoothL1Loss(beta=50.0)
    # mse_loss = nn.SmoothL1Loss(beta=5.0)
    # mse_loss = nn.L1Loss()
    # output = output *3000    # to enforce the original CNN output is between 0 to 1
#     f_u, f_v = loss_func.get_phy_Loss(output)
    f_p, P_out_vec = loss_func.get_phy_Loss(output, delta_t)
#     loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 
    loss_p =  mse_loss(f_p, torch.zeros_like(f_p).cuda())
    # loss =  mse_loss(f_p, P_out_vec)
    
    # loss_d = compute_loss_data(output, P_data)
    # loss = loss_p + 0.1*loss_d
    loss = loss_p
    return loss

def compute_loss_data(output, P_data):
    p_pred = torch.cat((output[1:-1, 0:1, 9, 9], output[1:-1, 0:1, 54, 54]), dim=1)
    P_pred = p_pred.permute(1,0)
    
    P_data = P_data[:,1:]
    # mse_loss = nn.L1Loss()
    mse_loss = nn.SmoothL1Loss(beta=100.0)
    loss_data = mse_loss(P_pred, P_data).cuda()
    return loss_data

def compute_loss_adp(output, loss_func, delta_t, epoch):
    ''' calculate the phycis loss '''
    
    beta = [100, 10]
    # get physics loss
    # mse_loss = nn.MSELoss()
    if epoch < 10000: 
        mse_loss = nn.SmoothL1Loss(beta=beta[0])
    # elif epoch >=5000 and epoch < 15000:
    #     mse_loss = nn.SmoothL1Loss(beta=beta[1])
    else: 
        mse_loss = nn.SmoothL1Loss(beta=beta[1])
    # mse_loss = nn.L1Loss()
    # output = output *3000    # to enforce the original CNN output is between 0 to 1
#     f_u, f_v = loss_func.get_phy_Loss(output)
    f_p, P_out_vec = loss_func.get_phy_Loss(output, delta_t)
#     loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 
    loss =  mse_loss(f_p, torch.zeros_like(f_p).cuda())
    # loss =  mse_loss(f_p, P_out_vec)
    return loss
    
def compute_loss_w(output, loss_func, delta_t):
    ''' calculate the phycis loss '''
    
    # # Padding x axis due to periodic boundary condition
    # # shape: [t, c, h, w]
    # output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)

    # # Padding y axis due to periodic boundary condition
    # # shape: [t, c, h, w]
    # output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    # mse_loss = nn.MSELoss()
    mse_loss = nn.L1Loss(reduction='none')
    # output = output *3000    # to enforce the original CNN output is between 0 to 1
#     f_u, f_v = loss_func.get_phy_Loss(output)
    f_p, P_out_vec = loss_func.get_phy_Loss(output, delta_t)
#     loss =  mse_loss(f_u, torch.zeros_like(f_u).cuda()) + mse_loss(f_v, torch.zeros_like(f_v).cuda()) 
    loss =  mse_loss(f_p, torch.zeros_like(f_p).cuda()).detach()     # get the abs loss tensor [1000, 1, 8192]
    lmin, lmax = torch.min(loss.view(loss.shape[0], -1), dim=1)[0], torch.max(loss.view(loss.shape[0], -1), dim=1)[0]
    lmin, lmax = lmin.reshape(loss.shape[0], 1, 1).expand(loss.shape), \
                       lmax.reshape(loss.shape[0], 1, 1).expand(loss.shape)
    
    weights = 2.0 * (loss - lmin) / (lmax - lmin)    # w = a + b* (loss - min(loss))/(max(loss) - min(loss))
    
    w_loss = torch.mean(torch.abs(weights * (f_p - torch.zeros_like(f_p).cuda())))
    
    return w_loss