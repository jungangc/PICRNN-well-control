import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt

from loss import loss_generator, compute_loss, compute_loss_w, compute_loss_adp

def train(model, input, source_BHP, initial_state, n_iters, time_batch_size, learning_rate, 
          dt, dx, save_path, pre_model_save_path, num_time_batch):

    train_loss_list = []
    second_last_state = []
    prev_output = []

    batch_loss = 0.0
    best_loss = 1e9

    # load previous model
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    scheduler = StepLR(optimizer, step_size=100, gamma=0.995)  
    # optimizer = optim.LBFGS(model.parameters(), lr=learning_rate) 
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.98)  
    # model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, 
    #     pre_model_save_path)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    loss_func = loss_generator(dt, dx)
        
    for epoch in range(n_iters):
        # input: [t,b,c,h,w]
        optimizer.zero_grad()
        batch_loss = 0 
        
        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                hidden_state = initial_state
                u0 = input
            else:
                hidden_state = state_detached
                u0 = prev_output[-2:-1].detach() # second last output

            # output is a list
            output, second_last_state = model(hidden_state, u0, source_BHP)

            # [t, c, height (Y), width (X)]
            output = torch.cat(tuple(output), dim=0)  

            # # concatenate the initial state to the output for central diff
            # output = torch.cat((u0.cuda(), output), dim=0)

            # get loss
            if epoch <n_iters:
                loss = compute_loss(output, loss_func, dt)
                # loss = compute_loss_adp(output, loss_func, dt, epoch)
            else:
                loss = compute_loss_w(output, loss_func, dt)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            # update the state and output for next batch
            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                (h, c) = second_last_state[i]
                state_detached.append((h.detach(), c.detach())) # hidden state

        # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  #apply cilp gradient
        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.10f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0), 
            batch_loss))
        train_loss_list.append(batch_loss)

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss
    
        # if epoch % 600 ==0:
        #   # os.makedirs('/content/PhyCRNet-main-1phaseflow/Codes/model/checkpoint{i}.pt'.format(epoch))
        #   torch.save({
        #   'model_state_dict': model.state_dict(),
        #   'optimizer_state_dict': optimizer.state_dict(),
        #   'scheduler_state_dict': scheduler.state_dict()
        #     }, '/content/PhyCRNet-main-1phaseflow/Codes/model/checkpoint{i}.pt'.format(epoch))
    return train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
#     ''' 
#     axis_lim: [xmin, xmax, ymin, ymax]
#     uv_lim: [u_min, u_max, v_min, v_max]
#     num: Number of time step
#     '''

#     # get the limit 
#     xmin, xmax, ymin, ymax = axis_lim
#     u_min, u_max, v_min, v_max = uv_lim

#     # grid
#     x = np.linspace(xmin, xmax, 128+1)
#     x = x[:-1]
#     x_star, y_star = np.meshgrid(x, x)
    
#     u_star = true[num, 0, 1:-1, 1:-1]
#     u_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

#     v_star = true[num, 1, 1:-1, 1:-1]
#     v_pred = output[num, 1, 1:-1, 1:-1].detach().cpu().numpy()

#     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)

#     cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=0.9, edgecolors='none', 
#         cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
#     ax[0, 0].axis('square')
#     ax[0, 0].set_xlim([xmin, xmax])
#     ax[0, 0].set_ylim([ymin, ymax])
#     ax[0, 0].set_title('u-RCNN')
#     fig.colorbar(cf, ax=ax[0, 0])

#     cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', 
#         cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
#     ax[0, 1].axis('square')
#     ax[0, 1].set_xlim([xmin, xmax])
#     ax[0, 1].set_ylim([ymin, ymax])
#     ax[0, 1].set_title('u-Ref.')
#     fig.colorbar(cf, ax=ax[0, 1])

#     cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=0.9, edgecolors='none', 
#         cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
#     ax[1, 0].axis('square')
#     ax[1, 0].set_xlim([xmin, xmax])
#     ax[1, 0].set_ylim([ymin, ymax])
#     cf.cmap.set_under('whitesmoke')
#     cf.cmap.set_over('black')
#     ax[1, 0].set_title('v-RCNN')
#     fig.colorbar(cf, ax=ax[1, 0])

#     cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.9, edgecolors='none', 
#         cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
#     ax[1, 1].axis('square')
#     ax[1, 1].set_xlim([xmin, xmax])
#     ax[1, 1].set_ylim([ymin, ymax])
#     cf.cmap.set_under('whitesmoke')
#     cf.cmap.set_over('black')
#     ax[1, 1].set_title('v-Ref.')
#     fig.colorbar(cf, ax=ax[1, 1])

#     # plt.draw()
#     plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
#     plt.close('all')

#     return u_star, u_pred, v_star, v_pred


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))
