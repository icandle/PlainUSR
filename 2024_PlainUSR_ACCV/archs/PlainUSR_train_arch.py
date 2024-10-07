# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from copy import deepcopy
from basicsr.utils.registry import ARCH_REGISTRY
from einops.layers.torch import Rearrange, Reduce

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

class MBConv(nn.Module):
    def __init__(self, n_feat, ratio=2):
        super().__init__()
        i_feat = n_feat*ratio
        self.expand_conv = nn.Conv2d(n_feat,i_feat,1,1,0)
        self.fea_conv = nn.Conv2d(i_feat,i_feat,3,1,0)
        self.reduce_conv = nn.Conv2d(i_feat,n_feat,1,1,0)
        self.se = ASR(i_feat)


    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)
        out = self.fea_conv(out) 
        out = self.se(out) + out_identity
        out = self.reduce_conv(out)
        out = out + x

        return out

    def switch_to_deploy(self):
        n_feat, _, _, _ = self.reduce_conv.weight.data.shape
        self.conv = nn.Conv2d(n_feat,n_feat,3,1,1)

        k0 = self.expand_conv.weight.data
        b0 = self.expand_conv.bias.data

        k1 = self.fea_conv.weight.data
        b1 = self.fea_conv.bias.data

        k2 = self.reduce_conv.weight.data
        b2 = self.reduce_conv.bias.data

        # first step: remove the ASR
        a = self.se.se(self.se.tensor)

        k1 = k1*(a.permute(1,0,2,3))
        b1 = b1*(a.view(-1))

        # second step: remove the middle identity
        for i in range(2*n_feat):
            k1[i,i,1,1] += 1.0 

        # third step: merge the first 1x1 convolution and the next 3x3 convolution
        merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merge_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, 2*n_feat, 3, 3) #.cuda()
        merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)       

        # third step: merge the remain 1x1 convolution
        merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
        merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

        # last step: remove the global identity
        for i in range(n_feat):
            merge_k0k1k2[i, i, 1, 1] += 1.0

        self.conv.weight.data = merge_k0k1k2.float()
        self.conv.bias.data = merge_b0b1b2.float()   

        for para in self.parameters():
            para.detach_()

        self.__delattr__('expand_conv')
        self.__delattr__('fea_conv')
        self.__delattr__('reduce_conv')
        self.__delattr__('se')


class ASR(nn.Module):
    def __init__(self, n_feat, ratio=2):
        super().__init__()
        self.n_feat = n_feat
        self.tensor = nn.Parameter(
            0.1*torch.ones((1, n_feat, 1, 1)),
            requires_grad=True
        )
        self.se = nn.Sequential(
            Reduce('b c 1 1 -> b c', 'mean'),
            nn.Linear(n_feat, n_feat//4, bias = False),
            nn.SiLU(),
            nn.Linear(n_feat//4, n_feat, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )
        self.init_weights()

    def init_weights(self): 
        # to make sure the inital [0.5,0.5,...,0.5]
        self.se[1].weight.data.fill_(1)    
        self.se[3].weight.data.fill_(1)
        
    def forward(self, x):
        attn = self.se(self.tensor)
        x = attn*x 
        return x

class Blockv1(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()

        self.body = nn.Sequential(
            MBConv(n_feat),
            nn.LeakyReLU(0.05,True),
            MBConv(n_feat),)

    def forward(self, x):
        return self.body(x) 
    
    def switch_to_deploy(self,prune):
        self.body[0].switch_to_deploy()
        self.body[2].switch_to_deploy()
        
        n_feat, _, _, _ = self.body[0].conv.weight.data.shape

        body = self.body
        self.__delattr__('body')     

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),) 
               
        self.body[0].weight.data = body[0].conv.weight.data
        self.body[0].bias.data = body[0].conv.bias.data
        self.body[2].weight.data = body[2].conv.weight.data
        self.body[2].bias.data = body[2].conv.bias.data

        if prune:
            x = self.body[0].weight.data
            self.body[0].weight.data = torch.where(x.abs()<1e-2, 0, x)
            x = self.body[2].weight.data
            self.body[2].weight.data = torch.where(x.abs()<1e-2, 0, x)

        for para in self.parameters():
            para.detach_()



class Blockv2(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()
        self.f = f
        self.body = nn.Sequential(
            # nn.Conv2d(n_feat,n_feat,3,1,1),
            # ASR(n_feat),
            MBConv(n_feat),
            nn.LeakyReLU(0.05,True),
            MBConv(n_feat),
            # nn.Conv2d(n_feat,n_feat,3,1,1),
            PConv(n_feat),
            LocalAttention(n_feat,f))

    def forward(self, x):
        return self.body(x) 
    
    def switch_to_deploy(self, prune=False):
        self.body[0].switch_to_deploy()
        self.body[2].switch_to_deploy()

        n_feat, _, _, _ = self.body[0].conv.weight.data.shape

        k3x3 = self.body[2].conv.weight.data
        b3x3 = self.body[2].conv.bias.data

        k1x1 = self.body[3].conv.weight.data
        b1x1 = self.body[3].conv.bias.data

        merge_w = F.conv2d(input=k3x3.permute(1, 0, 2, 3), weight=k1x1).permute(1, 0, 2, 3)
        merge_b = F.conv2d(input=b3x3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), weight=k1x1, bias=b1x1).view(-1)
        self.body[2].conv.weight.data[0:1,:,...] = merge_w.float()
        self.body[2].conv.bias.data[0:1] = merge_b.float()  

        body = self.body
        self.__delattr__('body')     

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            LocalAttention(n_feat,self.f)) 
               
        self.body[0].weight.data = body[0].conv.weight.data
        self.body[0].bias.data = body[0].conv.bias.data

        self.body[2].weight.data = body[2].conv.weight.data
        self.body[2].bias.data = body[2].conv.bias.data

        for i in [0,2,3]:
            self.body[3].body[i].weight.data = body[4].body[i].weight.data
            self.body[3].body[i].bias.data = body[4].body[i].bias.data

        if prune:
            x = self.body[0].weight.data
            self.body[0].weight.data = torch.where(x.abs()<1e-2, 0, x)
            x = self.body[2].weight.data
            self.body[2].weight.data = torch.where(x.abs()<1e-2, 0, x)

        for para in self.parameters():
            para.detach_()

class Blockv3(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()
        self.f = f
        self.body = nn.Sequential(
            # nn.Conv2d(n_feat,n_feat,3,1,1),
            # ASR(n_feat),
            MBConv(n_feat),
            nn.LeakyReLU(0.05,True),
            MBConv(n_feat),
            # nn.Conv2d(n_feat,n_feat,3,1,1),
            PConv(n_feat),
            LocalAttentionSpeed(n_feat,f))

    def forward(self, x):
        return self.body(x) 
    
    def switch_to_deploy(self, prune=False):
        self.body[0].switch_to_deploy()
        self.body[2].switch_to_deploy()

        n_feat, _, _, _ = self.body[0].conv.weight.data.shape

        k3x3 = self.body[2].conv.weight.data
        b3x3 = self.body[2].conv.bias.data

        k1x1 = self.body[3].conv.weight.data
        b1x1 = self.body[3].conv.bias.data

        merge_w = F.conv2d(input=k3x3.permute(1, 0, 2, 3), weight=k1x1).permute(1, 0, 2, 3)
        merge_b = F.conv2d(input=b3x3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), weight=k1x1, bias=b1x1).view(-1)
        self.body[2].conv.weight.data[0:1,:,...] = merge_w.float()
        self.body[2].conv.bias.data[0:1] = merge_b.float()  

        body = self.body
        self.__delattr__('body')     

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            LocalAttentionSpeed(n_feat,self.f)) 
               
        self.body[0].weight.data = body[0].conv.weight.data
        self.body[0].bias.data = body[0].conv.bias.data

        self.body[2].weight.data = body[2].conv.weight.data
        self.body[2].bias.data = body[2].conv.bias.data

        for i in [0]:
            self.body[3].body[i].weight.data = body[4].body[i].weight.data
            self.body[3].body[i].bias.data = body[4].body[i].bias.data

        if prune:
            x = self.body[0].weight.data
            self.body[0].weight.data = torch.where(x.abs()<1e-2, 0, x)
            x = self.body[2].weight.data
            self.body[2].weight.data = torch.where(x.abs()<1e-2, 0, x)

        for para in self.parameters():
            para.detach_()

class Blockv4(nn.Module):
    def __init__(self, n_feat, f=16):
        super().__init__()
        self.f = f
        self.body = nn.Sequential(
            # nn.Conv2d(n_feat,n_feat,3,1,1),
            # ASR(n_feat),
            MBConv(n_feat),
            nn.LeakyReLU(0.05,True),
            MBConv(n_feat),
            nn.LeakyReLU(0.05,True),
            MBConv(n_feat),
            # nn.Conv2d(n_feat,n_feat,3,1,1),
            PConv(n_feat),
            LocalAttention(n_feat,f))

    def forward(self, x):
        return self.body(x) 
    
    def switch_to_deploy(self, prune=False):
        self.body[0].switch_to_deploy()
        self.body[2].switch_to_deploy()
        self.body[4].switch_to_deploy()

        n_feat, _, _, _ = self.body[0].conv.weight.data.shape

        k3x3 = self.body[4].conv.weight.data
        b3x3 = self.body[4].conv.bias.data

        k1x1 = self.body[5].conv.weight.data
        b1x1 = self.body[5].conv.bias.data

        merge_w = F.conv2d(input=k3x3.permute(1, 0, 2, 3), weight=k1x1).permute(1, 0, 2, 3)
        merge_b = F.conv2d(input=b3x3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), weight=k1x1, bias=b1x1).view(-1)
        self.body[4].conv.weight.data[0:1,:,...] = merge_w.float()
        self.body[4].conv.bias.data[0:1] = merge_b.float()  

        body = self.body
        self.__delattr__('body')     

        self.body = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            nn.LeakyReLU(0.05,True),
            nn.Conv2d(n_feat,n_feat,3,1,1),
            LocalAttention(n_feat,self.f)) 
               
        self.body[0].weight.data = body[0].conv.weight.data
        self.body[0].bias.data = body[0].conv.bias.data

        self.body[2].weight.data = body[2].conv.weight.data
        self.body[2].bias.data = body[2].conv.bias.data

        self.body[4].weight.data = body[4].conv.weight.data
        self.body[4].bias.data = body[4].conv.bias.data

        for i in [0,2,3]:
            self.body[5].body[i].weight.data = body[6].body[i].weight.data
            self.body[5].body[i].bias.data = body[6].body[i].bias.data

        if prune:
            x = self.body[0].weight.data
            self.body[0].weight.data = torch.where(x.abs()<1e-2, 0, x)
            x = self.body[2].weight.data
            self.body[2].weight.data = torch.where(x.abs()<1e-2, 0, x)
            x = self.body[4].weight.data
            self.body[4].weight.data = torch.where(x.abs()<1e-2, 0, x)

        for para in self.parameters():
            para.detach_()


class PConv(nn.Module):
    def __init__(self, n_feat, conv_type='None'):
        super().__init__()
        self.n_feat = n_feat
        self.conv = nn.Conv2d(n_feat,1,1,1,0)
        self.init_weights()
    def forward(self,x):
        x1 = self.conv(x)
        x2 = x[:,1:].clone()
        return torch.cat([x1,x2],dim=1)
    def init_weights(self): 
        self.conv.weight.data.fill_(1/self.n_feat)
        self.conv.bias.data.fill_(0)


class PartialConv(nn.Module):
    def __init__(self, n_feat, conv_type='None'):
        super().__init__()
        self.conv = nn.Conv2d(32,32,3,1,1)
        self.init_weights()
    def forward(self,x):
        if self.eval():
            x[:,:32] = self.conv(x[:,:32])
            return x
        x1 = self.conv(x[:,:32].clone())
        x2 = x[:,32:].clone()
        return torch.cat([x1,x2],dim=1)

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 
    
    
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
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
        g = self.gate(x[:,:1].clone())
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
class PlainUSR_ultra(nn.Module):
    def __init__(self, n_feat=32, im_feat=[32,16], attn_feat=4, scale=4):
        super(PlainUSR_ultra, self).__init__()
        self.n_feat = n_feat
        self.scale = scale

        self.im_feat = im_feat

        self.head = nn.Conv2d(3,n_feat+3,3,1,1)

        self.block1 = Blockv1(im_feat[0],attn_feat)
        self.blockm = Blockv3(im_feat[1],attn_feat)
        self.block2 = Blockv1(im_feat[0],attn_feat)

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
        if self.eval():
            return self.fast_forward(x)
        
        x = self.head(x)

        x,pic = x.split([self.im_feat[0],3],1)
        x = self.block1(x) 
        x1,x2 = x.split([self.im_feat[1],self.im_feat[0]-self.im_feat[1]],1)
        x11 = self.blockm(x11)
        x1 = torch.cat([x11,x2],1)
        x1 = self.block2(x1) 

        x = torch.cat([x,pic],1)
        x = self.tail(x)

        return x 
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

@ARCH_REGISTRY.register()
class PlainUSR_general(nn.Module):
    def __init__(self, n_feat=64, im_feat=[64,48,32], attn_feat=16, scale=4):
        super(PlainUSR_general, self).__init__()
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
        if self.eval():
            return self.fast_forward(x)
        
        x = self.head(x)

        x,pic = x.split([self.im_feat[0],3],1)
        x = self.block1(x) 
        x1,x2 = x.split([self.im_feat[1],self.im_feat[0]-self.im_feat[1]],1)
        x1 = self.block2(x1) 
        x11,x12 = x1.split([self.im_feat[2],self.im_feat[1]-self.im_feat[2]],1)
        x11 = self.blockm(x11)
        x1 = torch.cat([x11,x12],1)
        x1 = self.block3(x1) 
        x = torch.cat([x1,x2],1)
        x = self.block4(x) 

        x = torch.cat([x,pic],1)
        x = self.tail(x)

        return x 
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


@ARCH_REGISTRY.register()
class PlainUSR_large(nn.Module):
    def __init__(self, n_feat=80, im_feat=[80,64,48], attn_feat=16, scale=4):
        super(PlainUSR_large, self).__init__()
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
        if self.eval():
            return self.fast_forward(x)
        
        x = self.head(x)

        x,pic = x.split([self.im_feat[0],3],1)
        x = self.block1(x) 
        x1,x2 = x.split([self.im_feat[1],self.im_feat[0]-self.im_feat[1]],1)
        x1 = self.block2(x1) 
        x11,x12 = x1.split([self.im_feat[2],self.im_feat[1]-self.im_feat[2]],1)
        x11 = self.blockm(x11)
        x1 = torch.cat([x11,x12],1)
        x1 = self.block3(x1) 
        x = torch.cat([x1,x2],1)
        x = self.block4(x) 

        x = torch.cat([x,pic],1)
        x = self.tail(x)

        return x 
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def model_convert(model:torch.nn.Module, save_path=None, prune=False, do_copy=True):
    if do_copy:
        model = deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy(prune)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == '__main__':
 
    from torchsummaryX import summary
    from PIL import Image
    import torchvision.transforms as transforms

    # net = PlainUSR_general(n_feat=32,im_feat=[32,16,8],attn_feat=4,scale=2)
    net = PlainUSR_ultra(n_feat=16,im_feat=[16,8],attn_feat=1,scale=2)
    # net = PlainUSR_large(n_feat=80,im_feat=[80,64,48],attn_feat=16,scale=2)


    net.eval()
    net.load_state_dict(torch.load('model_zoo/PlainUSRU_Ori_x2.pth')['params_ema'],True)

    # img = Image.open(r'C:\Z_Document\dataset\ClassicSR\LR\LRBI\Set14\x4\comic_LRBI_x4.png').convert('RGB')
    img = Image.open(r'C:\Z_Document\dataset\ClassicSR\LR\LRBI\Urban100\x2\img_092_LRBI_x2.png').convert('RGB')
    x = transforms.ToTensor()(img)    
    x = x.unsqueeze(0)

    f = nn.L1Loss()

    net_rep = model_convert(net,'model_zoo/PlainUSRU_Rep_x2.pth',False)
    # net_rep = model_convert(net)   
    y1 = net(x)
    y2 = net_rep(x)

    print(net_rep)
    # summary(net_rep,torch.randn((1,3,256,256)))
    print(f(y1,y2))

    
    img1 = transforms.ToPILImage()(y1.squeeze(0))
    # img1.show()
    img2 = transforms.ToPILImage()(y2.squeeze(0))
    img2.show()    
