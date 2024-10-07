# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from einops.layers.torch import Rearrange, Reduce

class Blockv1(nn.Module):
    def __init__(self, n_feat, conv_type='None'):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            )
    
    def forward(self, x):
        return self.body(x) 


class Blockv2(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            LocalAttention(n_feat, f=f),
            )
    
    def forward(self, x):
        return self.body(x) 


class Blockv3(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            LocalAttentionSpeed(n_feat, f=f),
            )
    
    def forward(self, x):
        return self.body(x) 

class Blockv4(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            LocalAttention(n_feat, f=f),
            )
    
    def forward(self, x):
        return self.body(x) 

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        # return self.avgpool(x)
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 
    

class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        f = f
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1])
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g #(w + g) #self.gate(x, w) 

class LocalAttentionSpeed(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        f = f
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, channels, 3, 1, 1),
            # SoftPooling2D(7, stride=3),
            # nn.Conv2d(f, channels, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(f, 1, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1])
        w = self.body(x)

        return x * g * w #(w + g) #self.gate(x, w)    
    
@ARCH_REGISTRY.register()
class PlainUSR_ultra_inference(nn.Module):
    def __init__(self, n_feat=32, im_feat=[32,16], attn_feat=4, scale=4):
        super(PlainUSR_ultra_inference, self).__init__()
        self.n_feat = n_feat
        self.scale = scale

        self.im_feat = im_feat

        self.head = nn.Conv2d(3,n_feat+3,3,1,1)

        self.block1 = Blockv1(self.im_feat[0],attn_feat)
        self.blockm = Blockv3(self.im_feat[1],attn_feat)
        self.block2 = Blockv1(self.im_feat[0],attn_feat)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feat+3, 3*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale))

        self.init_weights()

    def init_weights(self): 
        scale_squared = self.scale**2
        self.head.weight.data[-3, 0, 1, 1] += 1
        self.head.weight.data[-2, 1, 1, 1] += 1
        self.head.weight.data[-1, 2, 1, 1] += 1  

        self.tail[0].weight.data[               :  scale_squared, -3, 1, 1] += 1
        self.tail[0].weight.data[  scale_squared:2*scale_squared, -2, 1, 1] += 1     
        self.tail[0].weight.data[2*scale_squared:               , -1, 1, 1] += 1   

    def fast_forward(self,x):
        x = self.head(x)

        x[:,:self.im_feat[0]] = self.block1(x[:,:self.im_feat[0]]) 
        x[:,:self.im_feat[1]] = self.blockm(x[:,:self.im_feat[1]]) 
        x[:,:self.im_feat[0]] = self.block2(x[:,:self.im_feat[0]]) 

        x = self.tail(x)
        return x 
    
    def forward(self,x):
        return self.fast_forward(x)
        
@ARCH_REGISTRY.register()
class PlainUSR_general_inference(nn.Module):
    def __init__(self, n_feat=64, im_feat=[64,48,32], attn_feat=16, scale=4):
        super(PlainUSR_general_inference, self).__init__()
        self.n_feat = n_feat
        self.scale = scale

        self.im_feat = im_feat

        self.head = nn.Conv2d(3,n_feat+3,3,1,1)

        self.block1 = Blockv2(im_feat[0],attn_feat)
        self.block2 = Blockv2(im_feat[1],attn_feat)
        self.blockm = Blockv2(im_feat[2],attn_feat)
        self.block3 = Blockv2(im_feat[1],attn_feat)
        self.block4 = Blockv2(im_feat[0],attn_feat)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feat+3, 3*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale))

        self.init_weights()

    def init_weights(self): 
        scale_squared = self.scale**2
        self.head.weight.data[-3, 0, 1, 1] += 1
        self.head.weight.data[-2, 1, 1, 1] += 1
        self.head.weight.data[-1, 2, 1, 1] += 1  

        self.tail[0].weight.data[               :  scale_squared, -3, 1, 1] += 1
        self.tail[0].weight.data[  scale_squared:2*scale_squared, -2, 1, 1] += 1     
        self.tail[0].weight.data[2*scale_squared:               , -1, 1, 1] += 1   

    def fast_forward(self,x):
        x = self.head(x)

        x[:,:self.im_feat[0]] = self.block1(x[:,:self.im_feat[0]]) 
        x[:,:self.im_feat[1]] = self.block2(x[:,:self.im_feat[1]]) 
        x[:,:self.im_feat[2]] = self.blockm(x[:,:self.im_feat[2]]) 
        x[:,:self.im_feat[1]] = self.block3(x[:,:self.im_feat[1]]) 
        x[:,:self.im_feat[0]] = self.block4(x[:,:self.im_feat[0]]) 

        x = self.tail(x)
        return x 
    
    def forward(self,x):
        return self.fast_forward(x)


@ARCH_REGISTRY.register()
class PlainUSR_large_inference(nn.Module):
    def __init__(self, n_feat=80, im_feat=[80,64,48], attn_feat=16, scale=4):
        super(PlainUSR_large_inference, self).__init__()
        self.n_feat = n_feat
        self.scale = scale

        self.im_feat = im_feat

        self.head = nn.Conv2d(3,n_feat+3,3,1,1)

        self.block1 = Blockv4(im_feat[0],attn_feat)
        self.block2 = Blockv4(im_feat[1],attn_feat)
        self.blockm = Blockv4(im_feat[2],attn_feat)
        self.block3 = Blockv4(im_feat[1],attn_feat)
        self.block4 = Blockv4(im_feat[0],attn_feat)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feat+3, 3*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale))

        self.init_weights()

    def init_weights(self): 
        scale_squared = self.scale**2
        self.head.weight.data[-3, 0, 1, 1] += 1
        self.head.weight.data[-2, 1, 1, 1] += 1
        self.head.weight.data[-1, 2, 1, 1] += 1  

        self.tail[0].weight.data[               :  scale_squared, -3, 1, 1] += 1
        self.tail[0].weight.data[  scale_squared:2*scale_squared, -2, 1, 1] += 1     
        self.tail[0].weight.data[2*scale_squared:               , -1, 1, 1] += 1   

    def fast_forward(self,x):
        x = self.head(x)

        x[:,:self.im_feat[0]] = self.block1(x[:,:self.im_feat[0]]) 
        x[:,:self.im_feat[1]] = self.block2(x[:,:self.im_feat[1]]) 
        x[:,:self.im_feat[2]] = self.blockm(x[:,:self.im_feat[2]]) 
        x[:,:self.im_feat[1]] = self.block3(x[:,:self.im_feat[1]]) 
        x[:,:self.im_feat[0]] = self.block4(x[:,:self.im_feat[0]]) 

        x = self.tail(x)
        return x 
    
    def forward(self,x):
        return self.fast_forward(x)


if __name__ == '__main__':
 
    from torchsummaryX import summary

    net = PlainUSR_large_inference()
    summary(net,torch.randn((1,3,256,256)))

