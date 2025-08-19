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
from cbam_v2 import *
from vit_wpos_embed_v2 import *
import math
from torch.utils.checkpoint import checkpoint

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

    def forward(self, xd, xe=None, maskd=None, maske=None, return_kv=False, use_cache=False, q=None, k=None, v=None, batch_size=None, ptdtype=torch.float16):
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
            k = self.rope.apply_rotary_pos_emb(k, sin, cos)
            
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
            wpe = nn.Embedding(7, config.n_embd),
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
    def forward(self, idx, density_all, params=None, maskd=None, targets=None, loss_type='cross_entropy'):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        xe = self.cnn3D(density_all)
        params_to_concat = params[:, None, :].expand(-1, xe.shape[1], -1)
        xe = torch.cat((xe, params_to_concat), dim=-1)

        tok_emb = self.transformer.wte(idx.long()) # token embeddings of shape (b, t, n_embd)

        pos0 = torch.arange(0, 1, dtype=torch.long, device=device) # shape (t)
        pos_emb0 = self.transformer.wpe(pos0.long()) # position embeddings of shape (1, n_embd)
        pos1 = torch.remainder(torch.arange(0, t-1, dtype=torch.long, device=device), 6) + 1
        pos_emb1 = self.transformer.wpe(pos1.long()) # position embeddings of shape (t-1, n_embd)
        pos_emb = torch.cat((pos_emb0, pos_emb1), dim=0) # shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, xe=xe, maskd=maskd)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = torch.nan_to_num(logits, nan=-1e2, posinf=-1e2, neginf=-1e2)
            logits = torch.clamp(logits, min=-1e2, max=1e2)  # clamp logits to avoid NaNs
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