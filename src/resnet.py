import sys, os
import numpy as np
import torch
import torch.optim as optim
import pickle as pk
import numpy as np
import pickle as pk
import matplotlib
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import math
from cbam_v2 import *

class ResidualBlock(nn.Module):
    """
    Residual block for 3D CNN
    """

    def __init__(self, nf_inp, nf_out, ksize, padding=None, act='tanh'):
        super().__init__()
        self.ksize = ksize
        self.conv1 = nn.Conv3d(in_channels=nf_inp, out_channels=nf_out, kernel_size=ksize, padding=padding).bfloat16()
        if act == 'tanh':
            self.act1 = nn.Tanh()
        elif act == 'lrelu':
            self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv3d(in_channels=nf_out, out_channels=nf_out, kernel_size=ksize, padding=padding).bfloat16()
        if act == 'tanh':
            self.act2 = nn.Tanh()
        elif act == 'lrelu':
            self.act2 = nn.LeakyReLU(0.2)
        
        if nf_out != nf_inp:
            self.linear = nn.Linear(nf_inp, nf_out, bias=False).bfloat16()
        else:
            self.linear = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        x_to_add = x[..., (self.ksize + 1) // 2:-(self.ksize + 1) // 2,
                                  (self.ksize + 1) // 2:-(self.ksize + 1) // 2,
                                  (self.ksize + 1) // 2:-(self.ksize + 1) // 2]
        if self.linear is not None:
            x_to_add = torch.moveaxis(self.linear(torch.moveaxis(x_to_add,1,4)),4,1)
        return self.act2(out) + x_to_add

class CNN3D_stackout(nn.Module):
    """
    3D CNN with multiple output channels. Moreover, can convolve with filters of different sizes.
    """

    def __init__(
            self,
            ksize,
            nside_in,
            nside_out,
            ninp,
            nfeature,
            layers_types=['res', 'res', 'res', 'res'],
            # layers_types=['res', 'res'],            
            act='tanh',
            padding='valid'
        ):
        super().__init__()
        self.ksize = ksize
        self.nside_in = nside_in
        self.nside_out = nside_out
        # self.nbatch = nbatch
        self.nfeature = nfeature
        # self.nout = nout
        self.ninp = ninp
        # Define the convolutional layers
        self.n_cnn_tot = 0

        layers_j_all = []
        for j in range(len(layers_types)):
            if j == 0:
                ninp_j = self.ninp
                # nout_j = self.nfeature // 4
                nout_j = self.nfeature
            # elif j == 1:
            #     ninp_j = self.nfeature // 4
            #     nout_j = self.nfeature // 2
            # elif j == 2:
            #     ninp_j = self.nfeature // 2
            #     nout_j = self.nfeature                
            else:
                ninp_j = self.nfeature
                nout_j = self.nfeature
            if layers_types[j] == 'cnn':
                layers_j_all.append(nn.Conv3d(
                    ninp_j,
                    nout_j,
                    kernel_size=ksize,
                    padding=padding,
                    ))
                if act == 'tanh':
                    layers_j_all.append(nn.Tanh())
                elif act == 'lrelu':
                    layers_j_all.append(nn.LeakyReLU(0.2))
                self.n_cnn_tot += 1
            elif layers_types[j] == 'res':
                layers_j_all.append(ResidualBlock(
                    ninp_j,
                    nout_j,
                    ksize,
                    padding=padding,
                    act=act,
                    ))
                self.n_cnn_tot += 2
            elif layers_types[j] == 'res_cbam':
                layers_j_all.append(ResBlock3D_with_CBAM(
                    ninp_j,
                    nout_j,
                    kernel_size=ksize
                    # act=act,
                    ))
                self.n_cnn_tot += 2
            else:
                raise ValueError('Invalid layer type')
        self.layers_all = nn.Sequential(*layers_j_all)

    def forward(self, cond_mat, pool_type='mean', act='tanh'):
        """
        cond_mat: (nsim, ninp, dim_in+padding, dim_in+padding, dim_in+padding)
        Here dim_in is the number of voxels per side, obtained by dividing nside_in by nbatch
        """
        nsim = cond_mat.shape[0]
        # dim_out = self.nside_out // self.nbatch
        # dim_in = self.nside_in // self.nbatch
        dim_out = self.nside_out
        # dim_in = self.nside_in


        # every convolution reduces the size by ksize - 1, so check the input size
        # padded_dim = dim_in + self.n_cnn_tot * (self.ksize - 1)
        # if cond_mat.shape[2] != padded_dim:
            # raise ValueError('Invalid input size')
        cond_cnn = self.layers_all(cond_mat)
        # print(cond_cnn.shape)
        # The input density can be at higher resolution. In this case, we need to downsample it
        npools = int(np.log2(cond_cnn.shape[2] // dim_out))
        if npools > 0:
            for ji in range(npools):
                if pool_type == 'mean':
                    cond_cnn = nn.AvgPool3d(2)(cond_cnn)
                elif pool_type == 'max':
                    cond_cnn = nn.MaxPool3d(2)(cond_cnn)
                else:
                    raise ValueError('Invalid pooling type')
                if act == 'tanh':
                    cond_cnn = nn.Tanh()(cond_cnn)
                elif act == 'lrelu':
                    cond_cnn = nn.LeakyReLU(0.2)(cond_cnn)
                else:
                    raise ValueError('Invalid activation type')
        # first shift the nout dimension to last axis:
        cond_cnn = cond_cnn.permute(0, 2, 3, 4, 1)
        cond_out_all = (cond_cnn).reshape(nsim, dim_out**3, self.nfeature)
        # cond_out_all = cond_cnn
        return cond_out_all

