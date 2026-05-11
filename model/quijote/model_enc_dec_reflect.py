import sys, os
import numpy as np
import torch
import torch.optim as optim
import pickle as pk
# from xformers.components.attention import ScaledDotProduct
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
from resnet import *
from vit_wpos_embed_v3 import *
from torch.utils.checkpoint import checkpoint
import torch.nn.init as init

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
        # fself.rope = RotaryPositionEmbedding(n_embd_q // n_head, max_seq_len=Td)
        
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

    def forward(self, xd, xe=None, maskd=None, maske=None, return_kv=False, use_cache=False, q=None, k=None, v=None, batch_size=None, ptdtype=torch.float16):
        if use_cache:
            # If using cache, q should be the query for the new token and k,v should be cached
            assert k is not None and v is not None
            
            # Get current sequence position for RoPE
            # seq_len = 1  # For single token
            # sin, cos = self.rope(seq_len, q.device)
            
            # Apply RoPE to query
            # q = self.rope.apply_rotary_pos_emb(q, sin, cos)
            
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
        # sin, cos = self.rope(Td, xd.device)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if q is None:
            q = self.q_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            # Apply RoPE to query
            # q = self.rope.apply_rotary_pos_emb(q, sin, cos)
            
        if xe is not None:        
            _, Te, _ = xe.size()
            k = self.k_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            
            # Apply RoPE to keys (if xe and xd have same dimensions)
            # if Te == Td:
                # k = self.rope.apply_rotary_pos_emb(k, sin, cos)
            # else:
                # Handle different sequence lengths
                # sin_e, cos_e = self.rope(Te, xe.device)
                # k = self.rope.apply_rotary_pos_emb(k, sin_e, cos_e)
                
            if self.flash:
                
                # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                #     y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=maske, dropout_p=self.dropout if self.training else 0, is_causal=False)       
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=maske, dropout_p=self.dropout if self.training else 0, is_causal=False)       
                    y = torch.nn.functional.scaled_dot_product_attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), attn_mask=maske, dropout_p=self.dropout if self.training else 0, is_causal=False)       


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
            # k = self.rope.apply_rotary_pos_emb(k, sin, cos)
            
            if maskd is not None:
                maskd = maskd.view(B, 1, 1, Td).expand(B, self.n_head, Td, Td)
                
            if self.flash:
                mask_causal = self.attn_causal_mask[:,:,:Td,:Td].expand(B, self.n_head, Td, Td)
                if maskd is None:
                    attn_mask = mask_causal
                else:
                    attn_mask = mask_causal + maskd

                attn_mask = (attn_mask == 0.0)
                attn_mask = attn_mask.to(k.device, dtype=torch.bool)

                # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
                # import torch.utils.benchmark as benchmark

                # def check_flash_attention_compatible(q, k, v, attn_mask):
                #     try:
                #         with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                #             torch.nn.functional.scaled_dot_product_attention(
                #                 q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
                #         return True
                #     except RuntimeError as e:
                #         print(f"Flash Attention not compatible: {e}")
                #         return False

                # # Add inside your forward method before the attention calculation:
                # is_compatible = check_flash_attention_compatible(q.to(ptdtype), k.to(ptdtype), v.to(ptdtype), attn_mask)
                # print(f"Flash Attention compatible: {is_compatible}")
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                    # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
                    y = torch.nn.functional.scaled_dot_product_attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)                    

                # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                #     y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
                # def attention_fn(q, k, v, mask):
                #     return torch.nn.functional.scaled_dot_product_attention(
                #         q, k, v, 
                #         attn_mask=mask, 
                #         dropout_p=self.dropout if self.training else 0, 
                #         is_causal=False
                #     )
                # y = checkpoint(attention_fn, q, k, v, attn_mask)
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

    def forward(self, x, xe=None, maskd=None, maske=None, return_kv=False, use_kv_cache=False, k_cache=None, v_cache=None, cross_k_cache=None, cross_v_cache=None, cache_pos=None):
        if use_kv_cache and k_cache is not None and v_cache is not None:
            batch_size = x.size(0)
            x_new = self.ln_1(x)
            # import pdb; pdb.set_trace()
            head_size = self.selfattn.n_embd_q // self.selfattn.n_head
            q = self.selfattn.q_attn(x_new).view(batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)
            
            # Update k,v with the new token
            k_new = self.selfattn.k_attn(x_new).view(batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2) #[batch, n_head, seq_len_new, head_dim]
            v_new = self.selfattn.v_attn(x_new).view(batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)

            seq_len_new = k_new.size(2)
            if cache_pos is not None:
                # Static KV cache: write in-place (avoids O(n^2) torch.cat allocations)
                k_cache[:, :, cache_pos:cache_pos+seq_len_new, :] = k_new
                v_cache[:, :, cache_pos:cache_pos+seq_len_new, :] = v_new
                k_updated = k_cache[:, :, :cache_pos+seq_len_new, :]
                v_updated = v_cache[:, :, :cache_pos+seq_len_new, :]
            else:
                # Dynamic KV cache (fallback for backward compatibility)
                k_updated = torch.cat([k_cache, k_new], dim=2)
                v_updated = torch.cat([v_cache, v_new], dim=2)
            
            # Calculate attention
            attn_output = self.selfattn(x_new,
                q=q, k=k_updated, v=v_updated, 
                use_cache=True, 
                batch_size=batch_size
            )
            
            x = x + attn_output
            # When using static cache, return full buffer (already updated in-place)
            # When using dynamic cache, return the concatenated result
            k_self = k_cache if cache_pos is not None else k_updated
            v_self = v_cache if cache_pos is not None else v_updated
            if xe is not None:
                if cross_k_cache is not None and cross_v_cache is not None:
                    # Use pre-computed cross-attention K,V (avoids recomputing from xe every step)
                    x_ln2 = self.ln_2(x)
                    q_cross = self.crossattn.q_attn(x_ln2).view(batch_size, -1, self.crossattn.n_head, head_size).transpose(1, 2)
                    x_cross = self.crossattn(x_ln2, use_cache=True, q=q_cross, k=cross_k_cache, v=cross_v_cache, batch_size=batch_size)
                else:
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


# from CHARM all_models.py
class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, activation="tanh"):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        return self.network(x)


# from CHARM all_models.py
class SumGaussModel(nn.Module):
    """
    This function is for the quantization of the halo field. That is it models the probability of 
    observing number of halos in a given voxel as a sum of gausians.
    """

    def __init__(
            self,
            dim=1,
            hidden_dim=8,
            base_network=FCNN,
            num_cond=0,
            ngauss=1,
            mu_all=None,
            sig_all=None,
            device=None
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.num_cond = num_cond
        self.ngauss = ngauss

        self.mu_all = torch.tensor(mu_all, device=device)
        self.sig_all = torch.tensor(sig_all, device=device)
        self.var_all = torch.tensor(sig_all**2, device=device)

        self.layer_init = base_network(self.num_cond, self.ngauss, hidden_dim)

        if self.num_cond == 0:
            self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, cond_inp=None):
        dev = x.device
        
        mu_all, var_all = self.mu_all.to(dev), self.var_all.to(dev)
        pw_all_inp = self.layer_init(cond_inp)
            

        pw_all = nn.Softmax(dim=1)(pw_all_inp)
            # pw_all = nn.Softmax(dim=1)(torch.log(pw_all_inp))
        Li_all = torch.zeros(x.shape[0])
        Li_all = Li_all.to(dev)
        gauss = (1.0 / torch.sqrt(2 * np.pi * var_all[None, :]))*torch.exp(-0.5 * (x[:, None]-mu_all[None, :])**2/var_all[None, :])
        Li_all = torch.sum(pw_all * gauss, dim=1)
        neglogP = -torch.log(Li_all + 1e-30)
        return neglogP
        

    def inverse(self, cond_inp=None):
        device = cond_inp.device

        pw_all_inp = self.layer_init(cond_inp)
        pw_all = nn.Softmax(dim=1)(pw_all_inp)
            
            # pw_all = nn.Softmax(dim=1)(torch.log(pw_all_inp))

        var_all = self.var_all
        mu_all = self.mu_all

        idx = torch.multinomial(pw_all, num_samples=1).squeeze(-1)
            # import pdb; pdb.set_trace()
            # loop over gaussians
            # z = torch.empty(0, device=counts.device)
        z_out = mu_all[idx] + torch.randn(pw_all.shape[0], device=device) * torch.sqrt(var_all[idx])
        z_out = torch.round(z_out).clamp(min=0, max=68).long()          
        return z_out

    def sample(self, cond_inp=None):
        x = self.inverse(cond_inp)
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
                        config.n_embd, # - config.nparams,
                        layers_types=config.layers_types,
                                    )        
        elif dmo_cond_embed_type == 'vit':
            self.cnn3D = Vision3DTransformer(
                in_channels=config.ninp_density,
                patch_size=config.patch_size,
                embed_dim=config.n_embd,
                depth=config.n_layers_vit,
                num_heads=config.n_heads_vit,
                dropout=config.dropout,
                cross_attn_dim=config.n_embd, # - config.nparams,
                layers_types=config.layers_types,
                cosmo_bins=config.vocab_size,
            )
        
        if config.loss_type == 'SumGauss':
            self.sum_gauss_model = SumGaussModel(
                dim=1,
                hidden_dim=64,
                base_network=FCNN,
                num_cond=config.n_embd,
                ngauss=config.vocab_size,
                mu_all=np.arange(0, config.vocab_size),
                sig_all=np.ones(config.vocab_size)*config.gauss_delta,
                device=config.device
            )
                
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Linear(config.vocab_size, config.n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            whe = nn.Embedding(config.max_nhalo, config.n_embd),
            wprope = nn.Embedding(config.nprops, config.n_embd),
            wce = nn.Embedding(6, config.n_embd),
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

    @torch.cuda.amp.autocast()
    def forward(self, idx, density_all, params=None, maskd=None, targets=None):
        #print(">>> entering forward", flush=True)
        #print("idx shape:", idx.shape, flush=True)
        #print("density_all shape:", density_all.shape, flush=True)
        #print("params shape:", params.shape, flush=True)
        #print("maskd shape:", maskd.shape, flush=True)
        #print("targets shape:", targets.shape, flush=True)
        device = idx.device
        b, t = idx.size()   # b: batch size, t: token length
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        xe = self.cnn3D(density_all, idx[:,1:6]) # (b, N_patches, cross_attn_dim or embed_dim)
        #print("xe: ", xe[0:3,:2,-10:], flush=True)

        #params_to_concat = params[:, None, :].expand(-1, xe.shape[1], -1)
        #xe = torch.cat((xe, params_to_concat), dim=-1) # [b, N_patch, d_embed + n_params]

        #print("params: ", params[0:3], flush=True)
        #print("params_to_concat: ", params_to_concat[0:3,:2,-10:], flush=True)
        #print("xe after concatenate: ", xe[0:3,:2,-10:], flush=True)

        tok_emb = self.transformer.wte(idx.long()) # token embeddings of shape (b, t, n_embd)

        end_token_index = (targets == self.config.end_token).nonzero(as_tuple=True)[1] + 1 #(b,)
        n_halo_actual = (end_token_index - 7) // self.config.nprops  # shape: (b,)
        halo_ids_full = torch.arange(self.config.max_nhalo, device=device).repeat_interleave(self.config.nprops)
        prop_ids_full = torch.arange(self.config.max_nhalo * self.config.nprops, device=device) % self.config.nprops
        haloid_emb_full = self.transformer.whe(halo_ids_full.long())   # (max_nhalo*nprops, n_embd)
        prop_emb_full   = self.transformer.wprope(prop_ids_full.long()) # (max_nhalo*nprops, n_embd)
        halo_mask = torch.arange(self.config.max_nhalo, device=device).repeat_interleave(self.config.nprops).unsqueeze(0)  # (1, max_nhalo*nprops)
        halo_mask = halo_mask < n_halo_actual.unsqueeze(1)  
        halo_end_max = 7 + self.config.max_nhalo * self.config.nprops
        tok_emb[:,7:halo_end_max,:] += (haloid_emb_full.unsqueeze(0).expand(b,-1,-1)+prop_emb_full.unsqueeze(0).expand(b,-1,-1)) * halo_mask.unsqueeze(2)
        cosmo_emb = self.transformer.wce(torch.arange(0,6, dtype=torch.long, device=device))  # (6, n_embd)
        tok_emb[:,1:7,:] += cosmo_emb.unsqueeze(0).expand(b,-1,-1)

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, xe=xe, maskd=maskd)    # x: galaxy tokens, xe: features of density field
        x = self.transformer.ln_f(x)  # (512, 129, 64)
        if targets is not None:
            #loss_array = torch.zeros(5, device=device)
            #edges = [0,10,20,40,80,290]
            # if we are given some desired targets also calculate the loss
            if self.config.loss_type == 'cross_entropy':
                logits = self.lm_head(x) # from n_embd to vocab_size
                logits = torch.nan_to_num(logits, nan=-1e2, posinf=-1e2, neginf=-1e2)
                logits = torch.clamp(logits, min=-1e2, max=1e2)  # clamp logits to avoid NaNs
                '''
                if self.training == True:
                    for i in range(5):
                        loss_array[i]=F.cross_entropy(logits[:, edges[i]:edges[i+1], :].reshape(-1, logits.size(-1)),
                                                    targets[:, edges[i]:edges[i+1]].reshape(-1),
                                                    ignore_index=self.config.pad_token)
                '''    
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),targets.reshape(-1),ignore_index=self.config.pad_token)
            elif self.config.loss_type == 'SumGauss':
                loss = self.sum_gauss_model(targets.reshape(-1),cond_inp=x.reshape(-1, x.size(-1)))
                # Mask out the padding tokens
                mask = (targets.reshape(-1) != self.config.pad_token).float()
                loss = (loss * mask).sum() / mask.sum()
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
            # loss = F.cross_entropy(logits, targets, ignore_index=1)            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            #logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return loss

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
        Om_min, Om_max = 0.1, 0.5
        Ob_min, Ob_max = 0.03, 0.07
        h0_min, h0_max = 0.5, 0.9
        ns_min, ns_max = 0.8, 1.2
        sigma8_min, sigma8_max = 0.6, 1.0
        nbin = 131
        Om = params[:,0]
        Ob = params[:,1]
        h0 = params[:,2]
        ns = params[:,3]
        sigma8 = params[:,4]
        Om_token = torch.round((Om - Om_min) / (Om_max - Om_min) * nbin).clamp(0, nbin).long()
        Ob_token = torch.round((Ob - Ob_min)/(Ob_max - Ob_min) * nbin).clamp(0, nbin).long()
        h0_token = torch.round((h0 - h0_min)/(h0_max - h0_min) * nbin).clamp(0, nbin).long()
        ns_token = torch.round((ns - ns_min)/(ns_max - ns_min) * nbin).clamp(0, nbin).long()
        sigma8_token = torch.round((sigma8 - sigma8_min)/(sigma8_max - sigma8_min) * nbin).clamp(0, nbin).long()
        # Expand params and concat with CNN embeddings
        #params_to_concat = params[:, None, :].expand(-1, xe.shape[1], -1)
        #xe = torch.cat((xe, params_to_concat), dim=-1)
        
        # Start with just the start token
        idx = torch.ones((batch_size, 6), dtype=torch.long, device=device) * start_token
        idx[:,1] = Om_token  # set second token to Om token
        idx[:,2] = sigma8_token
        idx[:,3] = Ob_token
        idx[:,4] = h0_token
        idx[:,5] = ns_token

        xe = self.cnn3D(density_all, idx[:,1:6])

        # Pre-compute cross-attention K,V (constant across all decode steps)
        cross_k_cache = []
        cross_v_cache = []
        for block in self.transformer.h:
            head_size_cross = block.crossattn.n_embd_q // block.crossattn.n_head
            k_cross = block.crossattn.k_attn(xe).view(batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
            v_cross = block.crossattn.v_attn(xe).view(batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
            cross_k_cache.append(k_cross)
            cross_v_cache.append(v_cross)

        # Initialize static KV cache (pre-allocated to avoid O(n^2) torch.cat)
        n_layer, n_head = self.config.n_layer, self.config.n_head
        head_size = self.config.n_embd // n_head
        max_cache_len = 8 + max_new_tokens  # prefix tokens + generation tokens
        k_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size, device=device, dtype=torch.bfloat16)
                for _ in range(n_layer)]
        v_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size, device=device, dtype=torch.bfloat16)
                for _ in range(n_layer)]
        cache_pos = 0
        
        # Keep track of sequences that are still actively generating
        # Generate the number of halos
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        tok_emb = self.transformer.wte(idx.long()) # token embeddings of shape (b, 6, n_embd)
        cosmo_emb = self.transformer.wce(torch.arange(0,5, dtype=torch.long, device=device))  # (5, n_embd)
        tok_emb[:,1:6,:] += cosmo_emb.unsqueeze(0).expand(batch_size,-1,-1)
        x = tok_emb
        input_len = x.size(1)
        for j, block in enumerate(self.transformer.h):
            x, k_cache[j], v_cache[j] = block(
                x,
                xe=xe,
                return_kv=True,
                use_kv_cache=True,
                k_cache=k_cache[j],
                v_cache=v_cache[j],
                cross_k_cache=cross_k_cache[j],
                cross_v_cache=cross_v_cache[j],
                cache_pos=cache_pos
            )
        cache_pos += input_len
        x = self.transformer.ln_f(x[:, -1:, :])
        if self.config.loss_type == 'cross_entropy':
            logits = self.lm_head(x)  # (batch_size, 1, vocab_size)
            # Apply temperature and optional top-k filtering
            logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')      
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        elif self.config.loss_type == 'SumGauss':
            next_token = self.sum_gauss_model.inverse(cond_inp=x.view(-1, x.size(-1))).view(-1, 1)
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

        # Generate the first halo token
        tok_emb = self.transformer.wte(idx[:, -1:].long()) # token embeddings of shape (b, 1, n_embd)
        cosmo_emb = self.transformer.wce(torch.arange(5,6, dtype=torch.long, device=device))  # (1, n_embd)
        tok_emb += cosmo_emb.unsqueeze(0).expand(batch_size,-1,-1)
        x = tok_emb
        for j, block in enumerate(self.transformer.h):
            x, k_cache[j], v_cache[j] = block(
                x,
                xe=xe,
                return_kv=True,
                use_kv_cache=True,
                k_cache=k_cache[j],
                v_cache=v_cache[j],
                cross_k_cache=cross_k_cache[j],
                cross_v_cache=cross_v_cache[j],
                cache_pos=cache_pos
            )
        cache_pos += 1
        x = self.transformer.ln_f(x)
        if self.config.loss_type == 'cross_entropy':
            logits = self.lm_head(x)  # (batch_size, 1, vocab_size)
            # Apply temperature and optional top-k filtering
            logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')      
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        elif self.config.loss_type == 'SumGauss':
            next_token = self.sum_gauss_model.inverse(cond_inp=x.view(-1, x.size(-1))).view(-1, 1)
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
        # Generate tokens one at a time
        # Loop generates max_new_tokens - 1 additional tokens

        for i in range(max_new_tokens - 1):
            tok_emb = self.transformer.wte(idx[:, -1:].long())  # (batch_size, 1, n_embd)
            haloid = torch.tensor([i//self.config.nprops], dtype=torch.long, device=device)
            propid = torch.tensor([i%self.config.nprops], dtype=torch.long, device=device)
            # Only process the most recent token with KV cache
            haloid_emb = self.transformer.whe(haloid)  # (1, n_embd)
            propid_emb = self.transformer.wprope(propid)  # (1, n_embd)
            tok_emb += haloid_emb.unsqueeze(0).expand(batch_size,-1,-1) + propid_emb.unsqueeze(0).expand(batch_size,-1,-1)
            x = tok_emb
            # Process through transformer blocks with KV cache
            for j, block in enumerate(self.transformer.h):
                x, k_cache[j], v_cache[j] = block(
                    x,
                    xe=xe,
                    return_kv=True,
                    use_kv_cache=True,
                    k_cache=k_cache[j],
                    v_cache=v_cache[j],
                    cross_k_cache=cross_k_cache[j],
                    cross_v_cache=cross_v_cache[j],
                    cache_pos=cache_pos
                )
            cache_pos += 1

            x = self.transformer.ln_f(x)

            if self.config.loss_type == 'cross_entropy':
                logits = self.lm_head(x)  # (batch_size, 1, vocab_size)
                # Apply temperature and optional top-k filtering
                logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')      
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            elif self.config.loss_type == 'SumGauss':
                next_token = self.sum_gauss_model.inverse(cond_inp=x.view(-1, x.size(-1))).view(-1, 1)
            
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
    

    # =========================================================================
    # Reflection-based Reynolds operator (4 configs: identity, reflect-x/y/z)
    # =========================================================================

    def _build_reflected_density(self, density_all, axis, vel_channel_indices):
        """Flip spatial axis and negate velocity channels for that axis.

        Args:
            density_all: (B, C, Gx, Gy, Gz)
            axis: 0=x, 1=y, 2=z  ->  flips dim 2, 3, 4 respectively
            vel_channel_indices: list of 3 lists; vel_channel_indices[axis] = channel indices
        Returns:
            reflected density field (same shape, cloned)
        """
        d = density_all.flip(axis + 2).clone()
        for ch in vel_channel_indices[axis]:
            d[:, ch] *= -1
        return d

    def _unreflect_logits(self, logits, flip_range, do_flip):
        """If do_flip, reverse logits[0:flip_range]; indices >= flip_range untouched."""
        if not do_flip:
            return logits
        out = logits.clone()
        out[:, :flip_range] = logits[:, :flip_range].flip(dims=[1])
        return out

    def generate_with_reflection_reynolds(self, density_all, params=None, max_new_tokens=100,
                                          temperature=1.0, top_k=None, start_token=1,
                                          end_token=None, pad_token=None):
        """
        Generate halo catalog using a 4-config reflection Reynolds operator.

        Symmetry configs:
          0 - identity
          1 - reflect x  (flip dim-2 of density, negate vx channels)
          2 - reflect y  (flip dim-3 of density, negate vy channels)
          3 - reflect z  (flip dim-4 of density, negate vz channels)

        All 4 configs generate tokens in the SAME order (no look-ahead).
        Before averaging, logits from a reflected config are un-reflected so
        they live in the original frame.

        Token vocabulary:
          P   = pos_nvocab = 40   position tokens 0..P-1  (P-1-t reflection)
          V   = nvocab     = 131  property tokens 0..V    (V-t reflection, max token=V)
          end_token = 135
        """
        device = density_all.device
        batch_size = density_all.shape[0]

        end_token = end_token if end_token is not None else getattr(self.config, 'end_token', None)
        pad_token = pad_token if pad_token is not None else getattr(self.config, 'pad_token', None)

        P = 40   # 40: position token range 0..P-1
        V = 131       # 131: max velocity/scalar token index is V (nvocab)
        nprops = self.config.nprops  # 8: [x, y, z, mass, vx, vy, vz, conc]

        # Density field channel layout (18 channels = 2 + 8 + 8):
        #   snap0: ch 0-1  (dens, log)
        #   snap1: ch 2-9  (dens, log, pad1, log_pad1, pad2, vx, vy, vz)
        #   snap2: ch 10-17 (same structure as snap1)
        VEL_CHANNEL_INDICES = [
            [7, 15],   # vx
            [8, 16],   # vy
            [9, 17],   # vz
        ]

        # Build 4 density fields
        density_list = [density_all]
        for ax in range(3):
            density_list.append(self._build_reflected_density(density_all, ax, VEL_CHANNEL_INDICES))

        # Tokenize cosmological parameters
        Om_min, Om_max = 0.1, 0.5
        Ob_min, Ob_max = 0.03, 0.07
        h0_min, h0_max = 0.5, 0.9
        ns_min, ns_max = 0.8, 1.2
        sigma8_min, sigma8_max = 0.6, 1.0
        nbin = 131

        Om_token      = torch.round((params[:, 0] - Om_min)      / (Om_max      - Om_min)      * nbin).clamp(0, nbin).long()
        Ob_token      = torch.round((params[:, 1] - Ob_min)      / (Ob_max      - Ob_min)      * nbin).clamp(0, nbin).long()
        h0_token      = torch.round((params[:, 2] - h0_min)      / (h0_max      - h0_min)      * nbin).clamp(0, nbin).long()
        ns_token      = torch.round((params[:, 3] - ns_min)      / (ns_max      - ns_min)      * nbin).clamp(0, nbin).long()
        sigma8_token  = torch.round((params[:, 4] - sigma8_min)  / (sigma8_max  - sigma8_min)  * nbin).clamp(0, nbin).long()

        # KV-cache state per config
        class ReflState:
            def __init__(self, batch_size, n_layer, n_head, head_size, max_cache_len, device):
                self.idx = None
                self.xe = None
                self.cross_k_cache = []
                self.cross_v_cache = []
                self.k_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size,
                                            device=device, dtype=torch.bfloat16) for _ in range(n_layer)]
                self.v_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size,
                                            device=device, dtype=torch.bfloat16) for _ in range(n_layer)]
                self.cache_pos = 0

        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_size = self.config.n_embd // n_head
        max_cache_len = 8 + max_new_tokens

        states = [ReflState(batch_size, n_layer, n_head, head_size, max_cache_len, device)
                  for _ in range(4)]

        # Initialize idx (prefix) and encode density fields
        for i, state in enumerate(states):
            state.idx = torch.ones((batch_size, 6), dtype=torch.long, device=device) * start_token
            state.idx[:, 1] = Om_token
            state.idx[:, 2] = sigma8_token
            state.idx[:, 3] = Ob_token
            state.idx[:, 4] = h0_token
            state.idx[:, 5] = ns_token

            state.xe = self.cnn3D(density_list[i], state.idx[:, 1:6])

            for block in self.transformer.h:
                head_size_cross = block.crossattn.n_embd_q // block.crossattn.n_head
                k_cross = block.crossattn.k_attn(state.xe).view(
                    batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
                v_cross = block.crossattn.v_attn(state.xe).view(
                    batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
                state.cross_k_cache.append(k_cross)
                state.cross_v_cache.append(v_cross)

        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        output_idx = torch.ones((batch_size, 6), dtype=torch.long, device=device) * start_token
        output_idx[:, 1] = Om_token
        output_idx[:, 2] = sigma8_token
        output_idx[:, 3] = Ob_token
        output_idx[:, 4] = h0_token
        output_idx[:, 5] = ns_token

        # ---- helpers --------------------------------------------------------

        def forward_through_transformer(state, tok_emb):
            x = tok_emb
            input_len = x.size(1)
            for j, block in enumerate(self.transformer.h):
                x, state.k_cache[j], state.v_cache[j] = block(
                    x, xe=state.xe, return_kv=True, use_kv_cache=True,
                    k_cache=state.k_cache[j], v_cache=state.v_cache[j],
                    cross_k_cache=state.cross_k_cache[j], cross_v_cache=state.cross_v_cache[j],
                    cache_pos=state.cache_pos
                )
            state.cache_pos += input_len
            return self.transformer.ln_f(x[:, -1:, :])

        def get_logits(x):
            return self.lm_head(x)[:, -1, :] / temperature

        def sample_from_logits(logits_avg):
            if self.config.loss_type == 'cross_entropy':
                if top_k is not None:
                    v, _ = torch.topk(logits_avg, min(top_k, logits_avg.size(-1)))
                    logits_avg = logits_avg.clone()
                    logits_avg[logits_avg < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_avg, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            else:
                return self.sum_gauss_model.inverse(
                    cond_inp=logits_avg.view(-1, logits_avg.size(-1))).view(-1, 1)

        def check_finished(token):
            is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if token.dim() == 1:
                if end_token is not None:
                    is_finished = is_finished | (token == end_token)
                if pad_token is not None:
                    is_finished = is_finished | (token == pad_token)
            else:
                if end_token is not None:
                    is_finished = is_finished | (token == end_token).any(dim=1)
                if pad_token is not None:
                    is_finished = is_finished | (token == pad_token).any(dim=1)
            return is_finished

        # REFLECT_AXIS[c] = which spatial axis config c reflects (None = identity)
        REFLECT_AXIS = [None, 0, 1, 2]

        def get_pos_embedding(halo_idx, prop_idx):
            """Positional embedding for the LAST token when generating (halo_idx, prop_idx).

            Rule: whe/wprope encode the position of the *last generated* token, not
            the token being predicted.

              last token              -> embedding used
              N_halo  (h=0, p=0)     -> wce(5)
              conc of halo h-1       -> whe(h-1) + wprope(nprops-1)
              prop (p-1) of halo h   -> whe(h)   + wprope(p-1)
            """
            if prop_idx == 0 and halo_idx == 0:
                emb = self.transformer.wce(
                    torch.tensor([5], dtype=torch.long, device=device))  # (1, n_embd)
                return emb.unsqueeze(0).expand(batch_size, 1, -1)        # (B, 1, n_embd)
            elif prop_idx == 0:
                haloid_emb = self.transformer.whe(
                    torch.tensor([halo_idx - 1], dtype=torch.long, device=device))
                propid_emb = self.transformer.wprope(
                    torch.tensor([nprops - 1], dtype=torch.long, device=device))
                return (haloid_emb + propid_emb).unsqueeze(0).expand(batch_size, 1, -1)
            else:
                haloid_emb = self.transformer.whe(
                    torch.tensor([halo_idx], dtype=torch.long, device=device))
                propid_emb = self.transformer.wprope(
                    torch.tensor([prop_idx - 1], dtype=torch.long, device=device))
                return (haloid_emb + propid_emb).unsqueeze(0).expand(batch_size, 1, -1)

        def unreflect_logits(logits, prop_idx, config_idx):
            """Un-reflect logits from config's frame back to original frame.

            Position axis (prop_idx == axis):    flip logits[0:P]  only
              - indices >= P (including end_token=135) are left untouched so
                <END> probability is correctly preserved.
            Velocity axis (prop_idx == axis+4):  flip logits[0:V+1]
              - velocity tokens span 0..V (=nvocab), so V+1 values total.
            Scalars / non-reflected props:        no change.
            """
            axis = REFLECT_AXIS[config_idx]
            if axis is None:
                return logits
            if prop_idx == axis:              # position
                return self._unreflect_logits(logits, P, True)
            if prop_idx == axis + 4:          # velocity (vx=4, vy=5, vz=6)
                return self._unreflect_logits(logits, V + 1, True)
            return logits

        def reflect_token(t, prop_idx, config_idx):
            """Transform sampled token t (original frame) into config's reflected frame.

            t: (B, 1) long tensor.  end_token is passed through unchanged.
            """
            axis = REFLECT_AXIS[config_idx]
            if axis is None:
                return t
               # (B, 1) bool tensor: only apply reflection to valid position tokens; leave end_token and any OOB tokens unchanged
            
            if prop_idx == axis:              # position: P-1-t
                mask_valid = (t < P)
                '''
                results = torch.where(mask_not_end, P - 1 - t, t)
                index = np.where(results.cpu().numpy() < 0)[0]
                print("index: ", index)
                print(t[index], "P-1-t:", P - 1 - t[index])
                if (results<0).any() or (results>135).any():
                    print("Warning: position token out of range for reflection. t:", t, "P-1-t:", P - 1 - t)
                    print(results)
                    exit(1)
                '''
                return torch.where(mask_valid, P - 1 - t, t)

            if prop_idx == axis + 4:          # velocity: V-t
                mask_valid = (t <= V)
                '''
                if (results<0).any() or (results>135).any():
                    print("Warning: velocity token out of range for reflection. t:", t, "V-t:", V - t)
                    exit(1)
                '''
                return torch.where(mask_valid, V - t, t)
            return t

        # ---- cosmology prefix pass ------------------------------------------
        x_list = []
        weight = [3.,1.,1.,1.]
        for state in states:
            tok_emb = self.transformer.wte(state.idx.long())
            cosmo_emb = self.transformer.wce(
                torch.arange(0, 5, dtype=torch.long, device=device))
            tok_emb[:, 1:6, :] += cosmo_emb.unsqueeze(0).expand(batch_size, -1, -1)
            x = forward_through_transformer(state, tok_emb)
            x_list.append(x)

        # ---- N_halo token (scalar: average all 4 configs directly) ----------
        #logits_avg = sum(get_logits(x) for x in x_list) / 4.0
        logits_list = [get_logits(x) for x in x_list]
        logits_avg = sum(w * logits for w, logits in zip(weight, logits_list)) / sum(weight)
        nhalo_token = sample_from_logits(logits_avg)
        for state in states:
            state.idx = torch.cat([state.idx, nhalo_token], dim=1)
        output_idx = torch.cat([output_idx, nhalo_token], dim=1)

        is_finished = check_finished(nhalo_token)
        active_sequences = active_sequences & (~is_finished)

        # ---- main generation loop -------------------------------------------
        # prop_idx: 0=x, 1=y, 2=z, 3=mass, 4=vx, 5=vy, 6=vz, 7=conc
        for halo_idx in range(self.config.max_nhalo):
            if not active_sequences.any():
                break

            done_this_halo = False
            for prop_idx in range(nprops):
                pos_emb = get_pos_embedding(halo_idx, prop_idx)  # (B, 1, n_embd)
                '''
                # for different prop_idx, choose different reflection configs to average over:
                if prop_idx in [0, 4]:   # x or vx -> reflect-x config is the "odd one out"
                    config_indices = [0, 2, 3]  # average over identity, reflect-y, reflect-z
                elif prop_idx in [1, 5]: # y or vy -> reflect-y config is the "odd one out"
                    config_indices = [0, 1, 3]  # average over identity, reflect-x, reflect-z
                elif prop_idx in [2, 6]: # z or vz -> reflect-z config is the "odd one out"
                    config_indices = [0, 1, 2]  # average over identity, reflect-x, reflect-y
                else:                    # scalar props -> average over identity only
                    config_indices = [0]
                '''
                # Forward each config
                x_list = []
                for state in states:
                    tok_emb = (self.transformer.wte(state.idx[:, -1:].long()) + pos_emb)
                    x = forward_through_transformer(state, tok_emb)
                    x_list.append(x)

                # Un-reflect logits -> average -> sample
                logits_list = [unreflect_logits(get_logits(x), prop_idx, c)
                               for c, x in enumerate(x_list)]
                #logits_avg = sum(logits_list) / 4.0
                #logits_avg = sum(logits_list[c] for c in config_indices) / len(config_indices)
                logits_avg = sum(w * logits for w, logits in zip(weight, logits_list)) / sum(weight)

                next_token = sample_from_logits(logits_avg)  # (B, 1)

                # Feed reflected token to each config's state
                for c, state in enumerate(states):
                    t_c = reflect_token(next_token, prop_idx, c)
                    state.idx = torch.cat([state.idx, t_c], dim=1)

                output_idx = torch.cat([output_idx, next_token], dim=1)

                # <END> is expected at prop_idx==0, but may appear anywhere during
                # generation; check at every step to avoid corrupting the sequence.
                is_end = check_finished(next_token)
                active_sequences = active_sequences & (~is_end)
                if not active_sequences.any():
                    done_this_halo = True
                    break

            if done_this_halo:
                break

        return output_idx

    def generate_with_reflection_reynolds_8config(self, density_all, params=None, max_new_tokens=100,
                                                   temperature=1.0, top_k=None, start_token=1,
                                                   end_token=None, pad_token=None):
        """
        Generate halo catalog using an 8-config reflection Reynolds operator.

        All 8 sign combinations (sx, sy, sz):
          0 - ( 1, 1, 1)  identity
          1 - (-1, 1, 1)  reflect x
          2 - ( 1,-1, 1)  reflect y
          3 - ( 1, 1,-1)  reflect z
          4 - (-1,-1, 1)  reflect x and y
          5 - (-1, 1,-1)  reflect x and z
          6 - ( 1,-1,-1)  reflect y and z
          7 - (-1,-1,-1)  reflect x, y, and z

        Token vocabulary:
          P   = pos_nvocab = 40   position tokens 0..P-1  (P-1-t reflection)
          V   = nvocab     = 131  property tokens 0..V    (V-t   reflection)
          end_token = 135
        """
        device = density_all.device
        batch_size = density_all.shape[0]

        end_token = end_token if end_token is not None else getattr(self.config, 'end_token', None)
        pad_token = pad_token if pad_token is not None else getattr(self.config, 'pad_token', None)

        P = 40
        V = 131
        nprops = self.config.nprops  # 8: [x, y, z, mass, vx, vy, vz, conc]

        VEL_CHANNEL_INDICES = [
            [7, 15],   # vx channels
            [8, 16],   # vy channels
            [9, 17],   # vz channels
        ]

        # 8 sign configurations: (sx, sy, sz)
        REFLECT_SIGNS = [
            ( 1,  1,  1),
            (-1,  1,  1),
            ( 1, -1,  1),
            ( 1,  1, -1),
            (-1, -1,  1),
            (-1,  1, -1),
            ( 1, -1, -1),
            (-1, -1, -1),
        ]
        N_CONFIGS = len(REFLECT_SIGNS)

        def build_density_for_config(signs):
            """Flip each axis where sign == -1 and negate its velocity channels."""
            d = density_all.clone()
            for ax, s in enumerate(signs):
                if s == -1:
                    d = d.flip(ax + 2).clone()
                    for ch in VEL_CHANNEL_INDICES[ax]:
                        d[:, ch] *= -1
            return d

        density_list = [build_density_for_config(signs) for signs in REFLECT_SIGNS]

        # Tokenize cosmological parameters
        Om_min, Om_max = 0.1, 0.5
        Ob_min, Ob_max = 0.03, 0.07
        h0_min, h0_max = 0.5, 0.9
        ns_min, ns_max = 0.8, 1.2
        sigma8_min, sigma8_max = 0.6, 1.0
        nbin = 131

        Om_token     = torch.round((params[:, 0] - Om_min)     / (Om_max     - Om_min)     * nbin).clamp(0, nbin).long()
        Ob_token     = torch.round((params[:, 1] - Ob_min)     / (Ob_max     - Ob_min)     * nbin).clamp(0, nbin).long()
        h0_token     = torch.round((params[:, 2] - h0_min)     / (h0_max     - h0_min)     * nbin).clamp(0, nbin).long()
        ns_token     = torch.round((params[:, 3] - ns_min)     / (ns_max     - ns_min)     * nbin).clamp(0, nbin).long()
        sigma8_token = torch.round((params[:, 4] - sigma8_min) / (sigma8_max - sigma8_min) * nbin).clamp(0, nbin).long()

        # KV-cache state per config
        class ReflState:
            def __init__(self, batch_size, n_layer, n_head, head_size, max_cache_len, device):
                self.idx = None
                self.xe = None
                self.cross_k_cache = []
                self.cross_v_cache = []
                self.k_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size,
                                            device=device, dtype=torch.bfloat16) for _ in range(n_layer)]
                self.v_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size,
                                            device=device, dtype=torch.bfloat16) for _ in range(n_layer)]
                self.cache_pos = 0

        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_size = self.config.n_embd // n_head
        max_cache_len = 8 + max_new_tokens

        states = [ReflState(batch_size, n_layer, n_head, head_size, max_cache_len, device)
                  for _ in range(N_CONFIGS)]

        # Build prefix idx and encode density fields
        for i, state in enumerate(states):
            state.idx = torch.ones((batch_size, 6), dtype=torch.long, device=device) * start_token
            state.idx[:, 1] = Om_token
            state.idx[:, 2] = sigma8_token
            state.idx[:, 3] = Ob_token
            state.idx[:, 4] = h0_token
            state.idx[:, 5] = ns_token

            state.xe = self.cnn3D(density_list[i], state.idx[:, 1:6])

            for block in self.transformer.h:
                head_size_cross = block.crossattn.n_embd_q // block.crossattn.n_head
                k_cross = block.crossattn.k_attn(state.xe).view(
                    batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
                v_cross = block.crossattn.v_attn(state.xe).view(
                    batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
                state.cross_k_cache.append(k_cross)
                state.cross_v_cache.append(v_cross)

        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        output_idx = torch.ones((batch_size, 6), dtype=torch.long, device=device) * start_token
        output_idx[:, 1] = Om_token
        output_idx[:, 2] = sigma8_token
        output_idx[:, 3] = Ob_token
        output_idx[:, 4] = h0_token
        output_idx[:, 5] = ns_token

        # ---- helpers --------------------------------------------------------

        def forward_through_transformer(state, tok_emb):
            x = tok_emb
            input_len = x.size(1)
            for j, block in enumerate(self.transformer.h):
                x, state.k_cache[j], state.v_cache[j] = block(
                    x, xe=state.xe, return_kv=True, use_kv_cache=True,
                    k_cache=state.k_cache[j], v_cache=state.v_cache[j],
                    cross_k_cache=state.cross_k_cache[j], cross_v_cache=state.cross_v_cache[j],
                    cache_pos=state.cache_pos
                )
            state.cache_pos += input_len
            return self.transformer.ln_f(x[:, -1:, :])

        def get_logits(x):
            return self.lm_head(x)[:, -1, :] / temperature

        def sample_from_logits(logits_avg):
            if self.config.loss_type == 'cross_entropy':
                if top_k is not None:
                    v, _ = torch.topk(logits_avg, min(top_k, logits_avg.size(-1)))
                    logits_avg = logits_avg.clone()
                    logits_avg[logits_avg < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_avg, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            else:
                return self.sum_gauss_model.inverse(
                    cond_inp=logits_avg.view(-1, logits_avg.size(-1))).view(-1, 1)

        def check_finished(token):
            is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if token.dim() == 1:
                if end_token is not None:
                    is_finished = is_finished | (token == end_token)
                if pad_token is not None:
                    is_finished = is_finished | (token == pad_token)
            else:
                if end_token is not None:
                    is_finished = is_finished | (token == end_token).any(dim=1)
                if pad_token is not None:
                    is_finished = is_finished | (token == pad_token).any(dim=1)
            return is_finished

        # prop_idx -> which axis (0=x, 1=y, 2=z) it is odd under, or None
        PROP_AXIS = {0: 0, 1: 1, 2: 2, 4: 0, 5: 1, 6: 2}

        def get_pos_embedding(halo_idx, prop_idx):
            if prop_idx == 0 and halo_idx == 0:
                emb = self.transformer.wce(
                    torch.tensor([5], dtype=torch.long, device=device))
                return emb.unsqueeze(0).expand(batch_size, 1, -1)
            elif prop_idx == 0:
                haloid_emb = self.transformer.whe(
                    torch.tensor([halo_idx - 1], dtype=torch.long, device=device))
                propid_emb = self.transformer.wprope(
                    torch.tensor([nprops - 1], dtype=torch.long, device=device))
                return (haloid_emb + propid_emb).unsqueeze(0).expand(batch_size, 1, -1)
            else:
                haloid_emb = self.transformer.whe(
                    torch.tensor([halo_idx], dtype=torch.long, device=device))
                propid_emb = self.transformer.wprope(
                    torch.tensor([prop_idx - 1], dtype=torch.long, device=device))
                return (haloid_emb + propid_emb).unsqueeze(0).expand(batch_size, 1, -1)

        def unreflect_logits_8(logits, prop_idx, config_idx):
            """Un-reflect logits for each axis independently if that axis is flipped."""
            signs = REFLECT_SIGNS[config_idx]
            axis = PROP_AXIS.get(prop_idx, None)
            if axis is None or signs[axis] == 1:
                return logits
            if prop_idx in (0, 1, 2):       # position token
                return self._unreflect_logits(logits, P, True)
            else:                            # velocity token (prop_idx in 4,5,6)
                return self._unreflect_logits(logits, V + 1, True)

        def reflect_token_8(t, prop_idx, config_idx):
            """Transform sampled token (original frame) into the config's reflected frame."""
            signs = REFLECT_SIGNS[config_idx]
            axis = PROP_AXIS.get(prop_idx, None)
            if axis is None or signs[axis] == 1:
                return t
            if prop_idx in (0, 1, 2):       # position
                mask_valid = (t < P)
                return torch.where(mask_valid, P - 1 - t, t)
            else:                            # velocity (prop_idx in 4,5,6)
                mask_valid = (t <= V)
                return torch.where(mask_valid, V - t, t)

        def select_config_indices(prop_idx):
            """Exclude configs where the relevant axis has sign == -1."""
            axis = PROP_AXIS.get(prop_idx, None)
            if axis is None:
                return list(range(N_CONFIGS))
            return [c for c, s in enumerate(REFLECT_SIGNS) if s[axis] == 1]

        # ---- cosmology prefix pass ------------------------------------------
        x_list = []
        for state in states:
            tok_emb = self.transformer.wte(state.idx.long())
            cosmo_emb = self.transformer.wce(
                torch.arange(0, 5, dtype=torch.long, device=device))
            tok_emb[:, 1:6, :] += cosmo_emb.unsqueeze(0).expand(batch_size, -1, -1)
            x = forward_through_transformer(state, tok_emb)
            x_list.append(x)

        # ---- N_halo token (average all 8 configs) ---------------------------
        logits_avg = sum(get_logits(x) for x in x_list) / float(N_CONFIGS)
        nhalo_token = sample_from_logits(logits_avg)
        for state in states:
            state.idx = torch.cat([state.idx, nhalo_token], dim=1)
        output_idx = torch.cat([output_idx, nhalo_token], dim=1)

        is_finished = check_finished(nhalo_token)
        active_sequences = active_sequences & (~is_finished)

        # ---- main generation loop -------------------------------------------
        for halo_idx in range(self.config.max_nhalo):
            if not active_sequences.any():
                break

            done_this_halo = False
            for prop_idx in range(nprops):
                pos_emb = get_pos_embedding(halo_idx, prop_idx)

                #config_indices = select_config_indices(prop_idx)

                x_list = []
                for state in states:
                    tok_emb = (self.transformer.wte(state.idx[:, -1:].long()) + pos_emb)
                    x = forward_through_transformer(state, tok_emb)
                    x_list.append(x)

                #logits_list = [unreflect_logits_8(get_logits(x_list[c]), prop_idx, c)
                #               for c in config_indices]
                logits_list = [unreflect_logits_8(get_logits(x_list[c]), prop_idx, c)
                               for c in range(N_CONFIGS)]
                logits_avg = sum(logits_list) / len(logits_list)

                next_token = sample_from_logits(logits_avg)

                for c, state in enumerate(states):
                    t_c = reflect_token_8(next_token, prop_idx, c)
                    state.idx = torch.cat([state.idx, t_c], dim=1)

                output_idx = torch.cat([output_idx, next_token], dim=1)

                is_end = check_finished(next_token)
                active_sequences = active_sequences & (~is_end)
                if not active_sequences.any():
                    done_this_halo = True
                    break

            if done_this_halo:
                break

        return output_idx