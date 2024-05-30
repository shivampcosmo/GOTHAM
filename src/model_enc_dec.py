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
from xformers.components.attention import ScaledDotProduct
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import math

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
                nout_j = self.nfeature // 4
            elif j == 1:
                ninp_j = self.nfeature // 4
                nout_j = self.nfeature // 2
            elif j == 2:
                ninp_j = self.nfeature // 2
                nout_j = self.nfeature                
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



import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class Attention(nn.Module):

    def __init__(self, n_head, n_embd_kv, n_embd_q, dropout, Td, flash=True, attn_bias=False):
        super().__init__()
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_attn = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)
        self.k_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
        self.v_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
        # output projection
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)        
        # regularization
        # self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        # self.n_embd = config.n_embd
        self.dropout = dropout
        # self.is_causal = is_causal
        self.flash = flash
        if not self.flash:
            self.attn_dropout = nn.Dropout(self.dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(Td, Td))
                                        .view(1, 1, Td, Td))
        else:
            attn_causal_mask = torch.tril(torch.ones(Td, Td)).view(1, 1, Td, Td)
            attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 0, float('-inf'))
            attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 1., 0.0)
            # self.register_buffer("bias", torch.tril(torch.ones(Td, Td))
                                        # .view(1, 1, Td, Td))
            self.register_buffer("attn_causal_mask", attn_causal_mask)            

            # self.scaled_dot_product_ = ScaledDotProduct(dropout=dropout, causal=True)

    def forward(self, xd, xe=None, maskd=None, maske=None):
        B, Td, C = xd.size() # batch size, sequence length, embedding dimensionality (n_embd)
        

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = self.q_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if xe is not None:        
            _, Te, _ = xe.size() # batch size, sequence length, embedding dimensionality (n_embd)
            k = self.k_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=maske, dropout_p=self.dropout if self.training else 0, is_causal=False)       
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                # att = att.masked_fill(self.bias[:,:,:Td,:Td] == 0, float('-inf'))
                if maske is not None:
                    att += maske                
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)                        
        else:
            k = self.k_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
            if maskd is not None:
                maskd = maskd.view(B, 1, 1, Td).expand(B, self.n_head, Td, Td) # specifies new size
            if self.flash:
                mask_causal = self.attn_causal_mask[:,:,:Td,:Td].expand(B, self.n_head, Td, Td)
                if maskd is None:
                    attn_mask = mask_causal  
                else:
                    attn_mask = mask_causal + maskd
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:Td,:Td] == 0, float('-inf'))
                if maskd is not None:
                    att += maskd
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)                


        y = y.transpose(1, 2).contiguous().view(B, Td, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
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


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if hasattr(config, 'flash'):
            flash = config.flash
        else:
            flash = False
        print('Using flash: ', flash)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.selfattn = Attention(config.n_head, config.n_embd, config.n_embd, config.dropout, config.block_size, flash=flash, attn_bias=False)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)        
        self.crossattn = Attention(config.n_head, config.n_embd, config.n_embd, config.dropout, config.block_size, flash=flash, attn_bias=False)        
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x, xe=None, maskd=None, maske=None):
        x = x + self.selfattn(self.ln_1(x), xe=None, maskd=maskd, maske=None)
        x = x + self.crossattn(self.ln_2(x), xe=xe, maskd=maskd, maske=None)        
        x = x + self.mlp(self.ln_3(x))
        return x        



import math
class HaloDecoderModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnn3D = CNN3D_stackout(config.ksize,
                    config.density_grid_in,
                    config.density_grid_out,
                    config.ninp_density,
                    config.n_embd)        
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Linear(config.vocab_size, config.n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
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
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, density_all, maskd=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        xe = self.cnn3D(density_all)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, xe=xe, maskd=maskd)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.pad_token)
            # loss = F.cross_entropy(logits, targets, ignore_index=1)            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


    def generate(self, idx_inp, density_inp, max_new_tokens, temperature=1.0, nvox_samp=32):
        # idx is (B, T) array of indices in the current context
        idx_all = []
        for jv in range(nvox_samp):
            idx = idx_inp
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.config.block_size:]
                # get the predictions
                logits, _ = self(idx_cond, density_inp[jv,...].unsqueeze(0))
                # focus only on the last time step
                logits = logits[:, -1, :] / temperature # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                if idx_next == self.config.end_token:
                    break
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            idx_all.append(idx)
        return idx_all





