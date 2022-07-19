import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function

class BinaryQuantize(Function):
    '''
        binary quantize function, from IR-Net
    ''' 
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k, t = k.cuda(), t.cuda() 
        grad_input = k * t * (1-torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class Maxout(nn.Module):
    '''
        Nonlinear function
    '''
    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super(Maxout, self).__init__()
        self.neg_scale = nn.Parameter(neg_init*torch.ones(1,channel,1,1), requires_grad=True)
        self.pos_scale = nn.Parameter(pos_init*torch.ones(1,channel,1,1), requires_grad=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Maxout
        x = self.pos_scale*self.relu(x) - self.neg_scale*self.relu(-x)
        return x

class BinaryActivation(nn.Module):
    '''
        learnable scale and zero point for activation
    '''
    def __init__(self):
        super(BinaryActivation, self).__init__() 
        self.scale = nn.Parameter(torch.tensor(1.0).cuda().float(), requires_grad=True)
        self.zero_point = nn.Parameter(torch.tensor(0.0).cuda().float(), requires_grad=True)
    
    def gradient_approx(self, x):
        '''
            from Bi-Real Net
        '''
        out_forward = torch.sign(x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out
        
    def forward(self, x): 
        x = (x-self.zero_point)/self.scale
        x = self.gradient_approx(x)
        return self.scale*(x + self.zero_point)

class LambdaLayer(nn.Module):
    '''
        for DownSample
    '''
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class AdaBin_Conv2d(nn.Conv2d):
    '''
        AdaBin Binary Neural Network
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, a_bit=1, w_bit=1):
        super(AdaBin_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.k = torch.tensor([10]).float().cpu()
        self.t = torch.tensor([0.1]).float().cpu() 
        self.binary_a = BinaryActivation()

    def forward(self, inputs):
        if self.a_bit==1:
            inputs = self.binary_a(inputs) 

        if self.w_bit==1:
            w = self.weight 
            offset = w.mean((1,2,3)).view(-1,1,1,1)
            scale = w.std((1,2,3)).view(-1,1,1,1)

            w = (w - offset)/scale 
            wb = BinaryQuantize().apply(w, self.k, self.t)
            weight = wb * scale + offset
        else: 
            weight = self.weight
        
        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output