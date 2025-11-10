import sys, os
import numpy as np
import torch
import torch.optim as optim
import pickle as pk
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import pickle as pk
import matplotlib
# from xformers.components.attention import ScaledDotProduct
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import math
from resnet import *
# from vit_cbam import *
from cbam import *
# from vit import *
from vit_wpos_embed import *
import math


# class ResidualBlock(nn.Module):
#     """
#     Residual block for 3D CNN
#     """

#     def __init__(self, nf_inp, nf_out, ksize, padding=None, act='tanh'):
#         super().__init__()
#         self.ksize = ksize
#         self.conv1 = nn.Conv3d(in_channels=nf_inp, out_channels=nf_out, kernel_size=ksize, padding=padding).bfloat16()
#         if act == 'tanh':
#             self.act1 = nn.Tanh()
#         elif act == 'lrelu':
#             self.act1 = nn.LeakyReLU(0.2)
#         self.conv2 = nn.Conv3d(in_channels=nf_out, out_channels=nf_out, kernel_size=ksize, padding=padding).bfloat16()
#         if act == 'tanh':
#             self.act2 = nn.Tanh()
#         elif act == 'lrelu':
#             self.act2 = nn.LeakyReLU(0.2)
        
#         if nf_out != nf_inp:
#             self.linear = nn.Linear(nf_inp, nf_out, bias=False).bfloat16()
#         else:
#             self.linear = None

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.act1(out)
#         out = self.conv2(out)
#         x_to_add = x[..., (self.ksize + 1) // 2:-(self.ksize + 1) // 2,
#                                   (self.ksize + 1) // 2:-(self.ksize + 1) // 2,
#                                   (self.ksize + 1) // 2:-(self.ksize + 1) // 2]
#         if self.linear is not None:
#             x_to_add = torch.moveaxis(self.linear(torch.moveaxis(x_to_add,1,4)),4,1)
#         return self.act2(out) + x_to_add

# class CNN3D_stackout(nn.Module):
#     """
#     3D CNN with multiple output channels. Moreover, can convolve with filters of different sizes.
#     """

#     def __init__(
#             self,
#             ksize,
#             nside_in,
#             nside_out,
#             ninp,
#             nfeature,
#             layers_types=['res', 'res', 'res', 'res'],
#             # layers_types=['res', 'res'],            
#             act='tanh',
#             padding='valid'
#         ):
#         super().__init__()
#         self.ksize = ksize
#         self.nside_in = nside_in
#         self.nside_out = nside_out
#         # self.nbatch = nbatch
#         self.nfeature = nfeature
#         # self.nout = nout
#         self.ninp = ninp
#         # Define the convolutional layers
#         self.n_cnn_tot = 0

#         layers_j_all = []
#         for j in range(len(layers_types)):
#             if j == 0:
#                 ninp_j = self.ninp
#                 nout_j = self.nfeature // 4
#             elif j == 1:
#                 ninp_j = self.nfeature // 4
#                 nout_j = self.nfeature // 2
#             elif j == 2:
#                 ninp_j = self.nfeature // 2
#                 nout_j = self.nfeature                
#             else:
#                 ninp_j = self.nfeature
#                 nout_j = self.nfeature
#             if layers_types[j] == 'cnn':
#                 layers_j_all.append(nn.Conv3d(
#                     ninp_j,
#                     nout_j,
#                     kernel_size=ksize,
#                     padding=padding,
#                     ))
#                 if act == 'tanh':
#                     layers_j_all.append(nn.Tanh())
#                 elif act == 'lrelu':
#                     layers_j_all.append(nn.LeakyReLU(0.2))
#                 self.n_cnn_tot += 1
#             elif layers_types[j] == 'res':
#                 layers_j_all.append(ResidualBlock(
#                     ninp_j,
#                     nout_j,
#                     ksize,
#                     padding=padding,
#                     act=act,
#                     ))
#                 self.n_cnn_tot += 2
#             else:
#                 raise ValueError('Invalid layer type')
#         self.layers_all = nn.Sequential(*layers_j_all)

#     def forward(self, cond_mat, pool_type='mean', act='tanh'):
#         """
#         cond_mat: (nsim, ninp, dim_in+padding, dim_in+padding, dim_in+padding)
#         Here dim_in is the number of voxels per side, obtained by dividing nside_in by nbatch
#         """
#         nsim = cond_mat.shape[0]
#         # dim_out = self.nside_out // self.nbatch
#         # dim_in = self.nside_in // self.nbatch
#         dim_out = self.nside_out
#         # dim_in = self.nside_in


#         # every convolution reduces the size by ksize - 1, so check the input size
#         # padded_dim = dim_in + self.n_cnn_tot * (self.ksize - 1)
#         # if cond_mat.shape[2] != padded_dim:
#             # raise ValueError('Invalid input size')
#         cond_cnn = self.layers_all(cond_mat)
#         # print(cond_cnn.shape)
#         # The input density can be at higher resolution. In this case, we need to downsample it
#         npools = int(np.log2(cond_cnn.shape[2] // dim_out))
#         if npools > 0:
#             for ji in range(npools):
#                 if pool_type == 'mean':
#                     cond_cnn = nn.AvgPool3d(2)(cond_cnn)
#                 elif pool_type == 'max':
#                     cond_cnn = nn.MaxPool3d(2)(cond_cnn)
#                 else:
#                     raise ValueError('Invalid pooling type')
#                 if act == 'tanh':
#                     cond_cnn = nn.Tanh()(cond_cnn)
#                 elif act == 'lrelu':
#                     cond_cnn = nn.LeakyReLU(0.2)(cond_cnn)
#                 else:
#                     raise ValueError('Invalid activation type')
#         # first shift the nout dimension to last axis:
#         cond_cnn = cond_cnn.permute(0, 2, 3, 4, 1)
#         cond_out_all = (cond_cnn).reshape(nsim, dim_out**3, self.nfeature)
#         # cond_out_all = cond_cnn
#         return cond_out_all





    
# class Attention(nn.Module):

#     def __init__(self, n_head, n_embd_kv, n_embd_q, dropout, Td, flash=True, attn_bias=False):
#         super().__init__()
#         # assert config.n_embd % config.n_head == 0
#         # key, query, value projections for all heads, but in a batch
#         self.q_attn = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)
#         self.k_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
#         self.v_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
#         # output projection
#         # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#         self.c_proj = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)        
#         # regularization
#         # self.attn_dropout = nn.Dropout(config.dropout)
#         self.resid_dropout = nn.Dropout(dropout)
#         self.n_head = n_head
#         self.n_embd_q = n_embd_q
#         self.dropout = dropout
#         # self.is_causal = is_causal
#         self.flash = flash
#         if not self.flash:
#             self.attn_dropout = nn.Dropout(self.dropout)
#             print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
#             # causal mask to ensure that attention is only applied to the left in the input sequence
#             self.register_buffer("bias", torch.tril(torch.ones(Td, Td))
#                                         .view(1, 1, Td, Td))
#         else:
#             attn_causal_mask = torch.tril(torch.ones(Td, Td)).view(1, 1, Td, Td)
#             attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 0, float('-inf'))
#             attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 1., 0.0)
#             # self.register_buffer("bias", torch.tril(torch.ones(Td, Td))
#                                         # .view(1, 1, Td, Td))
#             self.register_buffer("attn_causal_mask", attn_causal_mask)            

#             # self.scaled_dot_product_ = ScaledDotProduct(dropout=dropout, causal=True)

#     def forward(self, xd, xe=None, maskd=None, maske=None, return_kv=False, use_cache=False, q=None, k=None, v=None, batch_size=None):
#         if use_cache:
#             # If using cache, q should be the query for the new token and k,v should be cached
#             assert k is not None and v is not None
            
#             # Compute attention with the provided k,v
#             y = torch.nn.functional.scaled_dot_product_attention(
#                 q, k, v, 
#                 attn_mask=None, 
#                 dropout_p=self.dropout if self.training else 0, 
#                 is_causal=False
#             )
            
#             # Reshape and project
#             y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd_q)
#             y = self.resid_dropout(self.c_proj(y))
            
#             return y

#         B, Td, C = xd.size() # batch size, sequence length, embedding dimensionality (n_embd)
        

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         # q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
#         if q is None:
#             q = self.q_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         if xe is not None:        
#             _, Te, _ = xe.size() # batch size, sequence length, embedding dimensionality (n_embd)
#             k = self.k_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
#             v = self.v_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
#             if self.flash:
#                 y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=maske, dropout_p=self.dropout if self.training else 0, is_causal=False)       
#             else:
#                 # manual implementation of attention
#                 att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#                 # att = att.masked_fill(self.bias[:,:,:Td,:Td] == 0, float('-inf'))
#                 if maske is not None:
#                     att += maske                
#                 att = F.softmax(att, dim=-1)
#                 att = self.attn_dropout(att)
#                 y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)                        
#         else:
#             k = self.k_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
#             v = self.v_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
#             if maskd is not None:
#                 maskd = maskd.view(B, 1, 1, Td).expand(B, self.n_head, Td, Td) # specifies new size
#             if self.flash:
#                 mask_causal = self.attn_causal_mask[:,:,:Td,:Td].expand(B, self.n_head, Td, Td)
#                 if maskd is None:
#                     attn_mask = mask_causal  
#                 else:
#                     attn_mask = mask_causal + maskd
#                 y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
#             else:
#                 # manual implementation of attention
#                 att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#                 att = att.masked_fill(self.bias[:,:,:Td,:Td] == 0, float('-inf'))
#                 if maskd is not None:
#                     att += maskd
#                 att = F.softmax(att, dim=-1)
#                 att = self.attn_dropout(att)
#                 y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)                


#         y = y.transpose(1, 2).contiguous().view(B, Td, C) # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_dropout(self.c_proj(y))
#         if return_kv:
#             return y, k, v
#         return y


# class RotaryPositionEmbedding(nn.Module):
#     def __init__(self, dim, max_seq_len=2048):
#         super().__init__()
#         self.dim = dim
#         self.max_seq_len = max_seq_len
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)

#     def forward(self, seq_len, device):
#         t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
#         sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
#         sin = torch.sin(sinusoid_inp)
#         cos = torch.cos(sinusoid_inp)
#         return sin, cos

#     def rotate_half(self, x):
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat((-x2, x1), dim=-1)

#     def apply_rotary_pos_emb(self, x, sin, cos):
#         print(sin.shape, cos.shape, x.shape)
#         sin = sin[:, None, :, :].expand_as(x)
#         cos = cos[:, None, :, :].expand_as(x)
#         x_rot = self.rotate_half(x)
#         x_out = (x * cos) + (x_rot * sin)
#         return x_out

# class RotaryPositionEmbedding(nn.Module):
#     def __init__(self, dim, max_seq_len=2048):
#         super().__init__()
#         self.dim = dim
#         self.max_seq_len = max_seq_len
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)

#     def forward(self, seq_len, device):
#         t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
#         sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
#         sin = torch.sin(sinusoid_inp)
#         cos = torch.cos(sinusoid_inp)
#         # Reshape to [seq_len, dim//2] for easier usage
#         return sin, cos

#     def rotate_half(self, x):
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat((-x2, x1), dim=-1)

#     def apply_rotary_pos_emb(self, x, sin, cos):
#         # x is [batch, heads, seq_len, head_dim]
#         # sin and cos are [seq_len, dim//2]
        
#         # Reshape sin and cos to match dimensions needed for broadcasting
#         sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
#         cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        
#         # Expand to match batch and heads dimensions
#         sin = sin.expand(x.shape[0], x.shape[1], -1, -1)  # [batch, heads, seq_len, dim//2]
#         cos = cos.expand(x.shape[0], x.shape[1], -1, -1)  # [batch, heads, seq_len, dim//2]
        
#         # Apply rotation
#         x_rot = self.rotate_half(x)
#         x_out = (x * cos) + (x_rot * sin)
#         return x_out


def earth_mover_distance_loss(logits: torch.Tensor, true_labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Calculates the Earth Mover's Distance (EMD) loss for ordinal classification.

    This loss assumes classes have a natural ordering. It measures the L1 distance
    between the Cumulative Distribution Function (CDF) of the predicted probabilities
    and the CDF of the true labels.

    Args:
        logits: Raw output scores from the model (before softmax).
                Shape: (batch_size, num_classes)
        true_labels: Ground truth class indices.
                     Shape: (batch_size,)
                     Values should be integers from 0 to num_classes - 1.
        ignore_index: Specifies a target value that is ignored and does not contribute
                      to the loss. Default: -100.

    Returns:
        torch.Tensor: The mean EMD loss over the batch.
    """
    num_classes = logits.shape[1]
    batch_size = logits.shape[0]

    # Create a mask to ignore specified indices
    mask = (true_labels != ignore_index)
    # If all labels are ignored, return 0 loss
    if not mask.any():
        # Return a tensor with requires_grad=True if logits requires grad
        return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

    # 1. Convert logits to probabilities
    pred_probs = F.softmax(logits, dim=1)

    # 2. Calculate predicted CDF
    # cumsum computes the cumulative sum along a given dimension
    pred_cdf = torch.cumsum(pred_probs, dim=1)
    # Shape: (batch_size, num_classes)

    # 3. Create true label distribution (one-hot encoding)
    # Ensure true_labels are long type for one_hot
    true_labels_long = true_labels.long()
    # Check for out-of-bounds labels
    if torch.any(true_labels_long < 0) or torch.any(true_labels_long >= num_classes):
        raise ValueError(f"true_labels contain values out of range [0, {num_classes-1}]")

    true_dist = F.one_hot(true_labels_long, num_classes=num_classes).float()
    # Shape: (batch_size, num_classes)

    # 4. Calculate true label CDF
    true_cdf = torch.cumsum(true_dist, dim=1)
    # Shape: (batch_size, num_classes)

    # 5. Calculate EMD (L1 distance between CDFs)
    # Sum the absolute differences along the class dimension
    # emd = torch.sum(torch.abs(pred_cdf - true_cdf), dim=1)
    emd = torch.sum((pred_cdf - true_cdf)**2, dim=1)
    # Shape: (batch_size,)
    
    # 6. Apply the mask and average the loss over the valid samples
    masked_emd = emd[mask]
    mean_emd = torch.mean(masked_emd)
    # import pdb; pdb.set_trace()

    return mean_emd

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be divisible by 2"
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        sin = torch.sin(sinusoid_inp)  # [seq_len, dim//2]
        cos = torch.cos(sinusoid_inp)  # [seq_len, dim//2]
        return sin, cos

    def apply_rotary_pos_emb(self, x, sin, cos):
        # x is [batch, heads, seq_len, head_dim]
        # sin and cos are [seq_len, dim//2]
        
        # Get dimensions
        batch, heads, seq_len, head_dim = x.shape
        half_dim = head_dim // 2
        
        # Reshape sin and cos for easier indexing
        sin = sin[:, :half_dim]  # [seq_len, half_dim]
        cos = cos[:, :half_dim]  # [seq_len, half_dim]
        
        # Expand dimensions for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_dim]
        
        # Expand to match batch and head dimensions
        sin = sin.expand(batch, heads, -1, -1)  # [batch, heads, seq_len, half_dim]
        cos = cos.expand(batch, heads, -1, -1)  # [batch, heads, seq_len, half_dim]
        
        # Split x into first and second half along embedding dimension
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotary embeddings
        x_out = torch.cat([
            x1 * cos - x2 * sin,  # Real part
            x2 * cos + x1 * sin   # Imaginary part
        ], dim=-1)
        
        return x_out

class Attention(nn.Module):
    def __init__(self, n_head, n_embd_kv, n_embd_q, dropout, Td, flash=True, attn_bias=False):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.q_attn = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)
        self.k_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
        self.v_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
        # output projection
        self.c_proj = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)        
        # regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd_q = n_embd_q
        self.dropout = dropout
        self.flash = flash
        
        # Add RoPE module
        self.rope = RotaryPositionEmbedding(n_embd_q // n_head, max_seq_len=Td)
        
        if not self.flash:
            self.attn_dropout = nn.Dropout(self.dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(Td, Td))
                                        .view(1, 1, Td, Td))
        else:
            attn_causal_mask = torch.tril(torch.ones(Td, Td)).view(1, 1, Td, Td)
            attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 0, float('-inf'))
            attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 1., 0.0)
            self.register_buffer("attn_causal_mask", attn_causal_mask)            

    def forward(self, xd, xe=None, maskd=None, maske=None, return_kv=False, use_cache=False, q=None, k=None, v=None, batch_size=None):
        if use_cache:
            # If using cache, q should be the query for the new token and k,v should be cached
            assert k is not None and v is not None
            
            # Get current sequence position for RoPE
            seq_len = 1  # For single token
            sin, cos = self.rope(seq_len, q.device)
            
            # Apply RoPE to query
            q = self.rope.apply_rotary_pos_emb(q, sin, cos)
            
            # Compute attention with the provided k,v
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=False
            )
            
            # Reshape and project
            y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd_q)
            y = self.resid_dropout(self.c_proj(y))
            
            return y

        B, Td, C = xd.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Get rotary embeddings
        sin, cos = self.rope(Td, xd.device)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if q is None:
            q = self.q_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            # Apply RoPE to query
            q = self.rope.apply_rotary_pos_emb(q, sin, cos)
            
        if xe is not None:        
            _, Te, _ = xe.size()
            k = self.k_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            
            # Apply RoPE to keys (if xe and xd have same dimensions)
            if Te == Td:
                k = self.rope.apply_rotary_pos_emb(k, sin, cos)
            else:
                # Handle different sequence lengths
                sin_e, cos_e = self.rope(Te, xe.device)
                k = self.rope.apply_rotary_pos_emb(k, sin_e, cos_e)
                
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=maske, dropout_p=self.dropout if self.training else 0, is_causal=False)       
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                if maske is not None:
                    att += maske                
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            k = self.k_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
            
            # Apply RoPE to keys
            k = self.rope.apply_rotary_pos_emb(k, sin, cos)
            
            if maskd is not None:
                maskd = maskd.view(B, 1, 1, Td).expand(B, self.n_head, Td, Td)
                
            if self.flash:
                mask_causal = self.attn_causal_mask[:,:,:Td,:Td].expand(B, self.n_head, Td, Td)
                if maskd is None:
                    attn_mask = mask_causal
                else:
                    attn_mask = mask_causal + maskd
                # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:Td,:Td] == 0, float('-inf'))
                if maskd is not None:
                    att += maskd
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, Td, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if return_kv:
            return y, k, v
        return y

class MLP(nn.Module):

    def __init__(self, n_embd_q, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd_q, 4 * n_embd_q)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd_q, n_embd_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
    
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x = x / rms * self.weight
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if hasattr(config, 'flash'):
            flash = config.flash
        else:
            flash = False
        print('Using flash: ', flash)
        # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_1 = RMSNorm(config.n_embd)
        self.selfattn = Attention(config.n_head, config.n_embd, config.n_embd, config.dropout, config.block_size, flash=flash, attn_bias=False)
        # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)   
        self.ln_2 = RMSNorm(config.n_embd)     
        self.crossattn = Attention(config.n_head, config.n_embd, config.n_embd, config.dropout, config.block_size, flash=flash, attn_bias=False)        
        # self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = RMSNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x, xe=None, maskd=None, maske=None, return_kv=False, use_kv_cache=False, k_cache=None, v_cache=None):
        if use_kv_cache and k_cache is not None and v_cache is not None:
            batch_size = x.size(0)
            x_new = self.ln_1(x)
            # import pdb; pdb.set_trace()
            head_size = self.selfattn.n_embd_q // self.selfattn.n_head
            q = self.selfattn.q_attn(x_new).view(batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)
            
            # Update k,v with the new token
            k_new = self.selfattn.k_attn(x_new).view(batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)
            v_new = self.selfattn.v_attn(x_new).view(batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)
            
            # Concatenate new k,v with cached k,v
            k_updated = torch.cat([k_cache, k_new], dim=2)
            v_updated = torch.cat([v_cache, v_new], dim=2)
            
            # Calculate attention
            attn_output = self.selfattn(x_new,
                q=q, k=k_updated, v=v_updated, 
                use_cache=True, 
                batch_size=batch_size
            )
            
            x = x + attn_output
            k_self, v_self = k_updated, v_updated  # Update cache
            if xe is not None:
                x_cross = self.crossattn(self.ln_2(x), xe=xe, maskd=maskd, maske=maske)
                x = x + x_cross
            
            # MLP
            x = x + self.mlp(self.ln_3(x))

        else:
            x = x + self.selfattn(self.ln_1(x), xe=None, maskd=maskd, maske=None)
            x = x + self.crossattn(self.ln_2(x), xe=xe, maskd=maskd, maske=None)        
            x = x + self.mlp(self.ln_3(x))

        if return_kv:
            return x, k_self, v_self            
        return x        


class HaloDecoderModel(nn.Module):

    def __init__(self, config_dict):
        super().__init__()
        from dataclasses import dataclass, make_dataclass
        fields = [(key, type(value)) for key, value in config_dict.items()]
        DynamicDataClass = make_dataclass("DynamicDataClass", fields)
        config = DynamicDataClass(**config_dict)
        self.config = config
        dmo_cond_embed_type = self.config.dmo_cond_embed_type
        if dmo_cond_embed_type == 'resnet':
            self.cnn3D = CNN3D_stackout(config.ksize,
                        config.density_grid_in,
                        config.density_grid_out,
                        config.ninp_density,
                        config.n_embd - config.nparams,
                        layers_types=config.layers_types
                                    )        
        elif dmo_cond_embed_type == 'vit':
            self.cnn3D = Vision3DTransformer(
                in_channels=config.ninp_density,
                patch_size=config.patch_size,
                embed_dim=config.n_embd,
                depth=config.n_layers_vit,
                num_heads=config.n_heads_vit,
                dropout=config.dropout,
                cross_attn_dim=config.n_embd - config.nparams,
                layers_types=config.layers_types
            )
                
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Linear(config.vocab_size, config.n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))  
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)      
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
            # n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, density_all, params=None, maskd=None, targets=None, loss_type='cross_entropy'):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        xe = self.cnn3D(density_all)
        params_to_concat = params[:, None, :].expand(-1, xe.shape[1], -1)
        xe = torch.cat((xe, params_to_concat), dim=-1)

        tok_emb = self.transformer.wte(idx.long()) # token embeddings of shape (b, t, n_embd)

        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # pos_emb = self.transformer.wpe(pos.long()) # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, xe=xe, maskd=maskd)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            if loss_type == 'cross_entropy':
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.pad_token)
            elif loss_type == 'EMD':
                loss = earth_mover_distance_loss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.pad_token)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            # loss = F.cross_entropy(logits, targets, ignore_index=1)            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def generate(self, density_all, params=None, max_new_tokens=100, temperature=1.0, top_k=None, start_token=1, end_token=None, pad_token=None):
        """
        Generate sequences from the model using the provided density field and parameters.
        
        Args:
            density_all: Tensor of shape [batch_size, channels, x, y, z] - the density field input
            params: Tensor of shape [batch_size, n_params] - cosmological parameters
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: If specified, restricts sampling to the top k most likely tokens
            start_token: Token ID to start the generation with
            end_token: Token ID indicating the end of a sequence. If None, uses config.end_token.
            pad_token: Token ID for padding, also treated as end of sequence. If None, uses config.pad_token.
            
        Returns:
            Generated sequences as a tensor of shape [batch_size, seq_length]
        """
        device = density_all.device
        batch_size = density_all.shape[0]

        # Use end_token and pad_token from config if not provided
        end_token = end_token if end_token is not None else getattr(self.config, 'end_token', None)
        pad_token = pad_token if pad_token is not None else getattr(self.config, 'pad_token', None)

        xe = self.cnn3D(density_all)
        # Expand params and concat with CNN embeddings
        params_to_concat = params[:, None, :].expand(-1, xe.shape[1], -1)
        xe = torch.cat((xe, params_to_concat), dim=-1)
        
        # Start with just the start token
        idx = torch.ones((batch_size, 1), dtype=torch.long, device=device) * start_token

        # Initialize KV cache
        n_layer, n_head = self.config.n_layer, self.config.n_head
        head_size = self.config.n_embd // n_head
        k_cache = [torch.zeros(batch_size, n_head, 0, head_size, device=device, dtype=torch.bfloat16) 
                for _ in range(n_layer)]
        v_cache = [torch.zeros(batch_size, n_head, 0, head_size, device=device, dtype=torch.bfloat16) 
                for _ in range(n_layer)]
        
        from tqdm import tqdm
        # Keep track of sequences that are still actively generating
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Generate tokens one at a time
        # Loop generates max_new_tokens - 1 additional tokens
        for i in tqdm(range(max_new_tokens - 1)):
            pos = torch.tensor([i], dtype=torch.long, device=device)
            # Only process the most recent token with KV cache
            tok_emb = self.transformer.wte(idx[:, -1:].long())  # (batch_size, 1, n_embd)
            pos_emb = self.transformer.wpe(pos.long())  # (1, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            
            # Process through transformer blocks with KV cache
            for j, block in enumerate(self.transformer.h):
                x, k_cache[j], v_cache[j] = block(
                    x, 
                    xe=xe, 
                    return_kv=True, 
                    use_kv_cache=True,
                    k_cache=k_cache[j],
                    v_cache=v_cache[j]
                )
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)  # (batch_size, 1, vocab_size)
            
            # Apply temperature and optional top-k filtering
            logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Append the sampled token
            idx = torch.cat([idx, next_token], dim=1)
            
            # Check for finished sequences (using end_token or pad_token)
            is_finished = torch.zeros_like(active_sequences)
            if end_token is not None:
                is_finished = is_finished | (next_token == end_token).squeeze(-1)
            if pad_token is not None:
                 is_finished = is_finished | (next_token == pad_token).squeeze(-1)

            # Update the mask of sequences that are still active
            # Only update sequences that were previously active
            newly_finished = active_sequences & is_finished
            active_sequences = active_sequences & (~is_finished)

            # Break if no sequences are active
            if not active_sequences.any():
                break
        return idx

    # def generate(self, density_all, params=None, max_new_tokens=100, temperature=1.0, top_k=None, start_token=1):
    #     """
    #     Generate sequences from the model using the provided density field and parameters.
        
    #     Args:
    #         density_all: Tensor of shape [batch_size, channels, x, y, z] - the density field input
    #         params: Tensor of shape [batch_size, n_params] - cosmological parameters
    #         max_new_tokens: Maximum number of tokens to generate
    #         temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
    #         top_k: If specified, restricts sampling to the top k most likely tokens
    #         start_token: Token ID to start the generation with
            
    #     Returns:
    #         Generated sequences as a tensor of shape [batch_size, seq_length]
    #     """
    #     device = density_all.device
    #     batch_size = density_all.shape[0]

    #     new_samples_jb = np.ones((batch_size, max_new_tokens))
    #     ind_jb = np.arange(batch_size)
        
    #     # Pre-compute the CNN embeddings for the density field
    #     xe = self.cnn3D(density_all)
    #     # Expand params and concat with CNN embeddings
    #     params_to_concat = params[:, None, :].expand(-1, xe.shape[1], -1)
    #     xe = torch.cat((xe, params_to_concat), dim=-1)
        
    #     # Start with just the start token
    #     idx = torch.ones((batch_size, 1), dtype=torch.long, device=device) * start_token

    #     new_samples_jb[:, 0] = idx[:,0].cpu().detach().numpy()
        
    #     # Initialize KV cache
    #     n_layer, n_head = self.config.n_layer, self.config.n_head
    #     head_size = self.config.n_embd // n_head
    #     k_cache = [torch.zeros(batch_size, n_head, 0, head_size, device=device, dtype=torch.bfloat16) 
    #             for _ in range(n_layer)]
    #     v_cache = [torch.zeros(batch_size, n_head, 0, head_size, device=device, dtype=torch.bfloat16) 
    #             for _ in range(n_layer)]
        
    #     from tqdm import tqdm
    #     # Generate tokens one at a time
    #     for i in tqdm(range(0, max_new_tokens)):
    #         # if len(ind_jb) > 0:
    #             # Get position for the new token
    #             # idx_cond = torch.tensor(new_samples_jb[ind_jb, :jt], dtype=torch.long, device=device)

    #         pos = torch.tensor([i], dtype=torch.long, device=device)
            
    #         # Only process the most recent token with KV cache
    #         tok_emb = self.transformer.wte(idx[:, -1:].long())  # (batch_size, 1, n_embd)
    #         pos_emb = self.transformer.wpe(pos.long())  # (1, n_embd)
    #         x = self.transformer.drop(tok_emb + pos_emb)
            
    #         # Process through transformer blocks with KV cache
    #         for j, block in enumerate(self.transformer.h):
    #             x, k_cache[j], v_cache[j] = block(
    #                 x, 
    #                 xe=xe, 
    #                 return_kv=True, 
    #                 use_kv_cache=True,
    #                 k_cache=k_cache[j],
    #                 v_cache=v_cache[j]
    #             )
            
    #         x = self.transformer.ln_f(x)
    #         logits = self.lm_head(x)  # (batch_size, 1, vocab_size)
            
    #         # Apply temperature and optional top-k filtering
    #         logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
            
    #         # Sample from the distribution
    #         probs = F.softmax(logits, dim=-1)
    #         next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
    #         # Append the sampled token
    #         idx = torch.cat([idx, next_token], dim=1)
            
    #         # Check if all sequences have reached the end token
    #         if hasattr(self.config, 'pad_token') and (next_token == self.config.pad_token).all():
    #             break
    
    #     return idx