# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F



class RRRB(nn.Module):
    def __init__(self, n_feats):
        super(RRRB, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out


class Block(nn.Module):
    def __init__(self,
                  channels):
        super(Block, self).__init__()
        
        self.attn = ESA(channels, nn.Conv2d)
        #inter_channels = 48
        self.conv1 = RRRB(channels) 
        self.conv2 = RRRB(channels) 
        self.conv3 = RRRB(32)
        self.relu = nn.LeakyReLU(0.05,True) 
        
        
    def forward(self, x):
        x = self.attn(x) 
        res = x
        x = self.relu(self.conv1(x))
        x[:,:32] = self.relu(self.conv3(x[:,:32]))
        x = self.relu(self.conv2(x))
        x = x + res
        
        return x 

    
class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = 16 #n_feats //4 #
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                            mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

class Net(nn.Module):
    def __init__(self,
                  scale=4,
                  in_channels=3,
                  out_channels=3,
                  feature_channels=48,
                  reduction=1/3,
                  deployed = True,
                  upscale=4):
        super(Net, self).__init__()
        
        self.conv_first = nn.Conv2d(in_channels, feature_channels, 3, 1, 1)
        
        self.block1 = Block(48)
        self.block2 = Block(48)
        self.block3 = Block(48)
        self.block4 = Block(48)

        self.conv_last = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)

        self.tail = nn.Sequential(
            nn.Conv2d(feature_channels, out_channels*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale))
        
    
    def forward(self, x):
        x = self.conv_first(x)
        res = x

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.conv_last(x) + res
        
        x = self.tail(x)
        return  x
    
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
                
    