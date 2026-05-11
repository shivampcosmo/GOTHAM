"""
Property-grouped token layout variant of model_enc_dec_cos_fast.py

Old layout:  START, cosmo×6, N_halos, [x1,y1,z1,m1,vx1,vy1,vz1,c1], [x2,...], ..., END, PAD
New layout:  START, cosmo×6, N_halos, [x1,...,xN], [y1,...,yN], ..., [c1,...,cN], END, PAD

Token at halo-section position j (0-indexed):
    prop_id = j // max_nhalo     (which property: 0=x,1=y,2=z,3=mass,4=vx,5=vy,6=vz,7=conc)
    halo_id = j % max_nhalo      (which halo)
"""

import sys, os
import numpy as np
import torch
import torch.optim as optim
import pickle as pk
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
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        return sin, cos

    def apply_rotary_pos_emb(self, x, sin, cos):
        batch, heads, seq_len, head_dim = x.shape
        half_dim = head_dim // 2
        sin = sin[:, :half_dim].unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)
        cos = cos[:, :half_dim].unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, n_head, n_embd_kv, n_embd_q, dropout, Td, flash=True, attn_bias=False):
        super().__init__()
        self.q_attn = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)
        self.k_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
        self.v_attn = nn.Linear(n_embd_kv, n_embd_kv, bias=attn_bias)
        self.c_proj = nn.Linear(n_embd_q, n_embd_q, bias=attn_bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd_q = n_embd_q
        self.dropout = dropout
        self.flash = flash

        if not self.flash:
            self.attn_dropout = nn.Dropout(self.dropout)
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(Td, Td)).view(1, 1, Td, Td))
        else:
            attn_causal_mask = torch.tril(torch.ones(Td, Td)).view(1, 1, Td, Td)
            attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 0, float('-inf'))
            attn_causal_mask = attn_causal_mask.masked_fill(attn_causal_mask == 1., 0.0)
            self.register_buffer("attn_causal_mask", attn_causal_mask)

    def forward(self, xd, xe=None, maskd=None, maske=None, return_kv=False,
                use_cache=False, q=None, k=None, v=None, batch_size=None,
                ptdtype=torch.float16):
        if use_cache:
            assert k is not None and v is not None
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
            y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd_q)
            y = self.resid_dropout(self.c_proj(y))
            return y

        B, Td, C = xd.size()

        if q is None:
            q = self.q_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)

        if xe is not None:
            _, Te, _ = xe.size()
            k = self.k_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xe).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)

            if self.flash:
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                        attn_mask=maske,
                        dropout_p=self.dropout if self.training else 0,
                        is_causal=False)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                if maske is not None:
                    att += maske
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            k = self.k_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.v_attn(xd).view(B, Td, self.n_head, C // self.n_head).transpose(1, 2)

            if maskd is not None:
                maskd = maskd.view(B, 1, 1, Td).expand(B, self.n_head, Td, Td)

            if self.flash:
                mask_causal = self.attn_causal_mask[:, :, :Td, :Td].expand(B, self.n_head, Td, Td)
                if maskd is None:
                    attn_mask = mask_causal
                else:
                    attn_mask = mask_causal + maskd
                attn_mask = (attn_mask == 0.0).to(k.device, dtype=torch.bool)
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                        attn_mask=attn_mask,
                        dropout_p=self.dropout if self.training else 0,
                        is_causal=False)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :Td, :Td] == 0, float('-inf'))
                if maskd is not None:
                    att += maskd
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, Td, C)
        y = self.resid_dropout(self.c_proj(y))
        if return_kv:
            return y, k, v
        return y


class MLP(nn.Module):
    def __init__(self, n_embd_q, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd_q, 4 * n_embd_q)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd_q, n_embd_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        flash = config.flash if hasattr(config, 'flash') else False
        print('Using flash: ', flash)
        self.ln_1 = RMSNorm(config.n_embd)
        self.selfattn = Attention(config.n_head, config.n_embd, config.n_embd,
                                  config.dropout, config.block_size, flash=flash, attn_bias=False)
        self.ln_2 = RMSNorm(config.n_embd)
        self.crossattn = Attention(config.n_head, config.n_embd, config.n_embd,
                                   config.dropout, config.block_size, flash=flash, attn_bias=False)
        self.ln_3 = RMSNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x, xe=None, maskd=None, maske=None, return_kv=False,
                use_kv_cache=False, k_cache=None, v_cache=None,
                cross_k_cache=None, cross_v_cache=None, cache_pos=None):
        if use_kv_cache and k_cache is not None and v_cache is not None:
            batch_size = x.size(0)
            x_new = self.ln_1(x)
            head_size = self.selfattn.n_embd_q // self.selfattn.n_head
            q = self.selfattn.q_attn(x_new).view(
                batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)
            k_new = self.selfattn.k_attn(x_new).view(
                batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)
            v_new = self.selfattn.v_attn(x_new).view(
                batch_size, -1, self.selfattn.n_head, head_size).transpose(1, 2)

            seq_len_new = k_new.size(2)
            if cache_pos is not None:
                k_cache[:, :, cache_pos:cache_pos + seq_len_new, :] = k_new
                v_cache[:, :, cache_pos:cache_pos + seq_len_new, :] = v_new
                k_updated = k_cache[:, :, :cache_pos + seq_len_new, :]
                v_updated = v_cache[:, :, :cache_pos + seq_len_new, :]
            else:
                k_updated = torch.cat([k_cache, k_new], dim=2)
                v_updated = torch.cat([v_cache, v_new], dim=2)

            attn_output = self.selfattn(x_new, q=q, k=k_updated, v=v_updated,
                                        use_cache=True, batch_size=batch_size)
            x = x + attn_output
            k_self = k_cache if cache_pos is not None else k_updated
            v_self = v_cache if cache_pos is not None else v_updated

            if xe is not None:
                if cross_k_cache is not None and cross_v_cache is not None:
                    x_ln2 = self.ln_2(x)
                    q_cross = self.crossattn.q_attn(x_ln2).view(
                        batch_size, -1, self.crossattn.n_head, head_size).transpose(1, 2)
                    x_cross = self.crossattn(x_ln2, use_cache=True, q=q_cross,
                                             k=cross_k_cache, v=cross_v_cache, batch_size=batch_size)
                else:
                    x_cross = self.crossattn(self.ln_2(x), xe=xe, maskd=maskd, maske=maske)
                x = x + x_cross

            x = x + self.mlp(self.ln_3(x))
        else:
            x = x + self.selfattn(self.ln_1(x), xe=None, maskd=maskd, maske=None)
            x = x + self.crossattn(self.ln_2(x), xe=xe, maskd=maskd, maske=None)
            x = x + self.mlp(self.ln_3(x))

        if return_kv:
            return x, k_self, v_self
        return x


class FCNN(nn.Module):
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


class SumGaussModel(nn.Module):
    def __init__(self, dim=1, hidden_dim=8, base_network=FCNN, num_cond=0,
                 ngauss=1, mu_all=None, sig_all=None, device=None):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.num_cond = num_cond
        self.ngauss = ngauss
        self.mu_all = torch.tensor(mu_all, device=device)
        self.sig_all = torch.tensor(sig_all, device=device)
        self.var_all = torch.tensor(sig_all ** 2, device=device)
        self.layer_init = base_network(self.num_cond, self.ngauss, hidden_dim)
        if self.num_cond == 0:
            self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, cond_inp=None):
        dev = x.device
        mu_all, var_all = self.mu_all.to(dev), self.var_all.to(dev)
        pw_all = nn.Softmax(dim=1)(self.layer_init(cond_inp))
        Li_all = torch.zeros(x.shape[0], device=dev)
        gauss = (1.0 / torch.sqrt(2 * np.pi * var_all[None, :])) * \
                torch.exp(-0.5 * (x[:, None] - mu_all[None, :]) ** 2 / var_all[None, :])
        Li_all = torch.sum(pw_all * gauss, dim=1)
        return -torch.log(Li_all + 1e-30)

    def inverse(self, cond_inp=None):
        device = cond_inp.device
        pw_all = nn.Softmax(dim=1)(self.layer_init(cond_inp))
        idx = torch.multinomial(pw_all, num_samples=1).squeeze(-1)
        z_out = self.mu_all[idx] + torch.randn(pw_all.shape[0], device=device) * \
                torch.sqrt(self.var_all[idx])
        return torch.round(z_out).clamp(min=0, max=68).long()

    def sample(self, cond_inp=None):
        return self.inverse(cond_inp)


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
                                        config.n_embd,
                                        layers_types=config.layers_types)
        elif dmo_cond_embed_type == 'vit':
            self.cnn3D = Vision3DTransformer(
                in_channels=config.ninp_density,
                patch_size=config.patch_size,
                embed_dim=config.n_embd,
                depth=config.n_layers_vit,
                num_heads=config.n_heads_vit,
                dropout=config.dropout,
                cross_attn_dim=config.n_embd,
                layers_types=config.layers_types,
                cosmo_bins=config.vocab_size,
            )

        if config.loss_type == 'SumGauss':
            self.sum_gauss_model = SumGaussModel(
                dim=1, hidden_dim=64, base_network=FCNN,
                num_cond=config.n_embd, ngauss=config.vocab_size,
                mu_all=np.arange(0, config.vocab_size),
                sig_all=np.ones(config.vocab_size) * config.gauss_delta,
                device=config.device
            )

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            whe=nn.Embedding(config.max_nhalo, config.n_embd),
            wprope=nn.Embedding(config.nprops, config.n_embd),
            wce=nn.Embedding(6, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.cuda.amp.autocast()
    def forward(self, idx, density_all, params=None, maskd=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        xe = self.cnn3D(density_all, idx[:, 1:6])

        tok_emb = self.transformer.wte(idx.long())  # (b, t, n_embd)

        # Derive actual halo count from END token position
        n_halo_actual = idx[:,6]  # (b,)

        # --- Property-grouped halo embeddings (compact layout) ---
        # In compact layout, halo section for sample b has N[b]*nprops tokens.
        # At position j: prop_id = j // N[b],  halo_id = j % N[b]  (both sample-specific).
        max_nhalo = self.config.max_nhalo
        nprops = self.config.nprops
        total_halo_tokens = max_nhalo * nprops
        n_halo_actual = n_halo_actual.unsqueeze(1).repeat(1, total_halo_tokens)  # (B, max_nhalo*nprops)
        j = torch.arange(total_halo_tokens, device=device).unsqueeze(0)  # (1, T)
        safe_n = n_halo_actual.clamp(min=1)
        halo_id = j % safe_n
        prop_id = j // safe_n
        valid = (halo_id < n_halo_actual) & (prop_id < nprops)  # (B, max_nhalo*nprops)              # (b, T)
        prop_id = prop_id.clamp(0, nprops - 1)

        haloid_emb_all = self.transformer.whe(halo_id.long())            # (b, max_nhalo*nprops, n_embd)
        prop_emb_all = self.transformer.wprope(prop_id.long())           # (b, max_nhalo*nprops, n_embd)
        combined = (haloid_emb_all + prop_emb_all) * valid.unsqueeze(-1).float()

        halo_end_max = 7 + total_halo_tokens
        tok_emb[:, 7:halo_end_max, :] += combined

        # Cosmology token positional embeddings (unchanged)
        cosmo_emb = self.transformer.wce(torch.arange(0, 6, dtype=torch.long, device=device))  # (6, n_embd)
        tok_emb[:, 1:7, :] += cosmo_emb.unsqueeze(0).expand(b, -1, -1)

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, xe=xe, maskd=maskd)
        x = self.transformer.ln_f(x)

        if targets is not None:
            if self.config.loss_type == 'cross_entropy':
                logits = self.lm_head(x)
                logits = torch.nan_to_num(logits, nan=-1e2, posinf=-1e2, neginf=-1e2)
                logits = torch.clamp(logits, min=-1e2, max=1e2)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1),
                                       ignore_index=self.config.pad_token)
            elif self.config.loss_type == 'SumGauss':
                loss = self.sum_gauss_model(targets.reshape(-1),
                                            cond_inp=x.reshape(-1, x.size(-1)))
                mask = (targets.reshape(-1) != self.config.pad_token).float()
                loss = (loss * mask).sum() / mask.sum()
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        else:
            loss = None

        return loss

    def generate(self, density_all, params=None, max_new_tokens=100, temperature=1.0,
                 top_k=None, start_token=1, end_token=None, pad_token=None):
        """
        Generate sequences in property-grouped token order.

        Token ordering after the 7 prefix tokens:
            [x_0, x_1, ..., x_{N-1},
             y_0, y_1, ..., y_{N-1},
             ...
             c_0, c_1, ..., c_{N-1},
             END, PAD, ...]

        At generation step i (0-indexed within the halo section, x_0 is step 0):
            halo_id = i % n_halo   (n_halo = max_new_tokens // nprops)
            prop_id = i // n_halo
        """
        device = density_all.device
        batch_size = density_all.shape[0]

        end_token = end_token if end_token is not None else getattr(self.config, 'end_token', None)
        pad_token = pad_token if pad_token is not None else getattr(self.config, 'pad_token', None)

        Om_min, Om_max = 0.1, 0.5
        Ob_min, Ob_max = 0.03, 0.07
        h0_min, h0_max = 0.5, 0.9
        ns_min, ns_max = 0.8, 1.2
        sigma8_min, sigma8_max = 0.6, 1.0
        nbin = 131

        Om_token = torch.round((params[:, 0] - Om_min) / (Om_max - Om_min) * nbin).clamp(0, nbin).long()
        Ob_token = torch.round((params[:, 1] - Ob_min) / (Ob_max - Ob_min) * nbin).clamp(0, nbin).long()
        h0_token = torch.round((params[:, 2] - h0_min) / (h0_max - h0_min) * nbin).clamp(0, nbin).long()
        ns_token = torch.round((params[:, 3] - ns_min) / (ns_max - ns_min) * nbin).clamp(0, nbin).long()
        sigma8_token = torch.round(
            (params[:, 4] - sigma8_min) / (sigma8_max - sigma8_min) * nbin).clamp(0, nbin).long()

        # Build prefix: [START, Om, sigma8, Ob, h0, ns]
        idx = torch.ones((batch_size, 6), dtype=torch.long, device=device) * start_token
        idx[:, 1] = Om_token
        idx[:, 2] = sigma8_token
        idx[:, 3] = Ob_token
        idx[:, 4] = h0_token
        idx[:, 5] = ns_token

        xe = self.cnn3D(density_all, idx[:, 1:6])

        # Pre-compute cross-attention K, V (constant across decode steps)
        cross_k_cache, cross_v_cache = [], []
        for block in self.transformer.h:
            head_size_cross = block.crossattn.n_embd_q // block.crossattn.n_head
            k_cross = block.crossattn.k_attn(xe).view(
                batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
            v_cross = block.crossattn.v_attn(xe).view(
                batch_size, -1, block.crossattn.n_head, head_size_cross).transpose(1, 2)
            cross_k_cache.append(k_cross)
            cross_v_cache.append(v_cross)

        n_layer, n_head = self.config.n_layer, self.config.n_head
        head_size = self.config.n_embd // n_head
        max_cache_len = 8 + max_new_tokens
        k_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size,
                               device=device, dtype=torch.bfloat16) for _ in range(n_layer)]
        v_cache = [torch.zeros(batch_size, n_head, max_cache_len, head_size,
                               device=device, dtype=torch.bfloat16) for _ in range(n_layer)]
        cache_pos = 0
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # ---- Step 1: forward pass on prefix [START, Om, sigma8, Ob, h0, ns] ----
        # Predict N_halos token
        tok_emb = self.transformer.wte(idx.long())  # (b, 6, n_embd)
        cosmo_emb = self.transformer.wce(
            torch.arange(0, 5, dtype=torch.long, device=device))  # (5, n_embd) for cosmo tokens 1..5
        tok_emb[:, 1:6, :] += cosmo_emb.unsqueeze(0).expand(batch_size, -1, -1)
        x = tok_emb
        input_len = x.size(1)
        for j, block in enumerate(self.transformer.h):
            x, k_cache[j], v_cache[j] = block(
                x, xe=xe, return_kv=True, use_kv_cache=True,
                k_cache=k_cache[j], v_cache=v_cache[j],
                cross_k_cache=cross_k_cache[j], cross_v_cache=cross_v_cache[j],
                cache_pos=cache_pos)
        cache_pos += input_len
        x = self.transformer.ln_f(x[:, -1:, :])
        next_token = self._sample_token(x, temperature, top_k)
        n_halo_token = next_token
        idx = torch.cat([idx, next_token], dim=1)  # append N_halos token
        active_sequences = active_sequences & ~self._is_done(next_token, end_token, pad_token)

        # ---- Step 2: embed N_halos token with cosmo_emb[5], predict x_0 ----
        tok_emb = self.transformer.wte(idx[:, -1:].long())  # (b, 1, n_embd)
        cosmo_emb_5 = self.transformer.wce(
            torch.arange(5, 6, dtype=torch.long, device=device))  # (1, n_embd)
        tok_emb += cosmo_emb_5.unsqueeze(0).expand(batch_size, -1, -1)
        x = tok_emb
        for j, block in enumerate(self.transformer.h):
            x, k_cache[j], v_cache[j] = block(
                x, xe=xe, return_kv=True, use_kv_cache=True,
                k_cache=k_cache[j], v_cache=v_cache[j],
                cross_k_cache=cross_k_cache[j], cross_v_cache=cross_v_cache[j],
                cache_pos=cache_pos)
        cache_pos += 1
        x = self.transformer.ln_f(x)
        next_token = self._sample_token(x, temperature, top_k)
        idx = torch.cat([idx, next_token], dim=1)  # append x_0
        active_sequences = active_sequences & ~self._is_done(next_token, end_token, pad_token)

        # ---- Main loop: generate halo property tokens one at a time ----
        # Compact layout: halo section = [x1..xN, y1..yN, ..., c1..cN].
        # At halo-section index i: halo_id = i % n_halo,  prop_id = i // n_halo.
        nprops = self.config.nprops
        safe_n = n_halo_token.clamp(min=1, max=self.config.max_nhalo)  # (B,1) target halos
        for i in range(max_new_tokens - 1):
            halo_id = i % safe_n # (B,1)
            prop_id = i // safe_n # (B,1)
            prop_id = prop_id.clamp(0, nprops - 1)

            tok_emb = self.transformer.wte(idx[:, -1:].long())  # (b, 1, n_embd)
            haloid_emb = self.transformer.whe(
                torch.tensor(halo_id, dtype=torch.long, device=device))  # (b, 1, n_embd)
            propid_emb = self.transformer.wprope(
                torch.tensor(prop_id, dtype=torch.long, device=device))  # (b, 1, n_embd)
            tok_emb += haloid_emb + propid_emb

            x = tok_emb
            for j, block in enumerate(self.transformer.h):
                x, k_cache[j], v_cache[j] = block(
                    x, xe=xe, return_kv=True, use_kv_cache=True,
                    k_cache=k_cache[j], v_cache=v_cache[j],
                    cross_k_cache=cross_k_cache[j], cross_v_cache=cross_v_cache[j],
                    cache_pos=cache_pos)
            cache_pos += 1
            x = self.transformer.ln_f(x)
            next_token = self._sample_token(x, temperature, top_k)
            idx = torch.cat([idx, next_token], dim=1)
            active_sequences = active_sequences & ~self._is_done(next_token, end_token, pad_token)

            if not active_sequences.any():
                break

        return idx

    def _sample_token(self, x, temperature, top_k):
        """Sample next token from transformer output x of shape (B, 1, n_embd)."""
        if self.config.loss_type == 'cross_entropy':
            logits = self.lm_head(x)[:, -1, :] / temperature  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)  # (B, 1)
        elif self.config.loss_type == 'SumGauss':
            return self.sum_gauss_model.inverse(
                cond_inp=x.view(-1, x.size(-1))).view(-1, 1)

    def _is_done(self, token, end_token, pad_token):
        """Return bool tensor (B,) indicating which sequences just ended."""
        done = torch.zeros(token.size(0), dtype=torch.bool, device=token.device)
        if end_token is not None:
            done = done | (token.squeeze(-1) == end_token)
        if pad_token is not None:
            done = done | (token.squeeze(-1) == pad_token)
        return done