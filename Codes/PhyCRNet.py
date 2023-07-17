import torch
import torch.nn as nn
# from torch.autograd import Variable
import numpy as np

from network import ConvLSTMCell, encoder_block, decoder_block, source_encoder_block

# generalized version
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight.data)

    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class PhyCRNet(nn.Module):
   # ''' physics-informed convolutional-recurrent neural networks '''
    def __init__(self, input_channels, hidden_channels, 
        input_kernel_size, input_stride, input_padding, dt, 
        num_layers, upscale_factor, step=1, effective_step=[1]):

        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]
        self.num_decoder = num_layers[2]
        self.num_source_encoder = num_layers[3]

        # encoder - downsampling  
        for i in range(self.num_encoder):
            name = 'encoder{}'.format(i)
            cell = encoder_block(
                input_channels = self.input_channels[i], 
                hidden_channels = self.hidden_channels[i], 
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])

            setattr(self, name, cell)
            self._all_layers.append(cell)    

        # input_channels_source = [1, 8, 32, 64]
        # hidden_channels_source= [8, 32, 64]

        # source encoder - downsampling  
        for i in range(1):
            name = 'source_encoder{}'.format(i)
            cell = source_encoder_block(
                input_channels = 64, 
                hidden_channels = 64, 
                input_kernel_size = 5,
                input_stride = 1,
                input_padding = 2, downscale_factor=8)

            setattr(self, name, cell)
            # self._all_layers.append(cell)                 
            
        # ConvLSTM
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels = self.input_channels[i],
                hidden_channels = self.hidden_channels[i],
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])
        
            setattr(self, name, cell)
            self._all_layers.append(cell)  

        # decoder -- upsampling
        for i in range(self.num_encoder + self.num_convlstm, self.num_encoder + self.num_convlstm+self.num_decoder):
            name = 'decoder{}'.format(i)
            cell = decoder_block(
                input_channels = self.input_channels[i],
                hidden_channels = self.hidden_channels[i],
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])
        
            setattr(self, name, cell)
            self._all_layers.append(cell)  
        # output layer
        # self.output_layer = nn.Conv2d(2, 2, kernel_size = 5, stride = 1, 
        #                               padding=2, padding_mode='circular')
        self.output_layer1 = nn.Conv2d(16, 1, kernel_size = (5,5), stride = (1,1), 

                                      padding=(2,2), padding_mode='circular')
        # self.act = nn.ReLU() 
        # self.act = nn.Tanh() 

        # self.output_layer2 = nn.Conv2d(8, 1, kernel_size = (5,5), stride = (1,1), 

        #                               padding=(2,2), padding_mode='circular')
                                  
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)   

        # self.flatten = nn.Flatten()
        # self.dropout = nn.Dropout(0.5)
        # self.dense = nn.Linear(8192, 8192)
        # initialize weights
        self.apply(initialize_weights)
        # nn.init.zeros_(self.output_layer1.bias)
        nn.init.constant_(self.output_layer1.bias, 0)
        # nn.init.zeros_(self.output_layer2.bias)

    def forward(self, initial_state, x, source):
        
        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []
        outputs.append(x)
        for step in range(self.step):
            xt = x
            
            # if step==200 :
            #     xs = torch.unsqueeze(source[step-1,...], 0)
            # else:
            xs = torch.unsqueeze(source[step,...], 0)
            # print(xs.size())
            # encoder
            for i in range(self.num_encoder):
                name = 'encoder{}'.format(i)
                x = getattr(self, name)(x)

            # source encoder
            for i in range(1):
                name = 'source_encoder{}'.format(i)
                xs = getattr(self, name)(xs)
            # for i in range(self.num_encoder):
            #     name = 'encoder{}'.format(i)
            #     xs = getattr(self, name)(xs)

            # x = x + xs            # concatenate  
                
            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state = self.initial_state[i - self.num_encoder])  
                    internal_state.append((h,c))
                
                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                x, new_c = getattr(self, name)(x, xs, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)   
                 
            # decoder
            for i in range(self.num_encoder + self.num_convlstm, self.num_encoder + self.num_convlstm + self.num_decoder):
                name = 'decoder{}'.format(i)
                x = getattr(self, name)(x)
            # output
            # x = self.pixelshuffle(x)
            x = self.output_layer1(x)
            # x = self.act(x)
            # x = self.output_layer2(x)

            # x = self.flatten(x)
            # x = self.dropout(x)
            # x = self.dense(x)

            # residual connection
            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()
                
            if step in self.effective_step:
                outputs.append(x)

            # outputs = outputs*3000               

        return outputs, second_last_state

