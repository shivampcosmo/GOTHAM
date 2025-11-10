import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from cbam import *


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, layers_types=['res_cbam']):
        super().__init__()
        if 'cnn' in layers_types[0]:
            self.proj = nn.Conv3d(in_channels, embed_dim, 
                                kernel_size=3, stride=patch_size, bias=False)
        if 'cbam' in layers_types[0]:
            self.proj = ResBlock3D_with_CBAM(in_channels, embed_dim, 
                                kernel_size=3, stride=patch_size)

        self.bn1 = nn.RMSNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, D1, D2, D3)
        x = self.proj(x)  # (B, embed_dim, D1//patch_size, D2//patch_size, D3//patch_size)
        x = self.bn1(rearrange(x, 'b c d1 d2 d3 -> b d1 d2 d3 c'))  # (B, N_patches, embed_dim)
        x = rearrange(x, 'b d1 d2 d3 c -> b (d1 d2 d3) c')  # (B, N_patches, embed_dim)        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Use scaled_dot_product_attention (Flash Attention)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            # Rearrange for flash attention input format
            q = q.transpose(1, 2)  # (B, N, num_heads, head_dim)
            k = k.transpose(1, 2)  # (B, N, num_heads, head_dim)
            v = v.transpose(1, 2)  # (B, N, num_heads, head_dim)
            
            # Apply flash attention
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # (B, N, num_heads, head_dim)
            
            x = x.reshape(B, N, C)  # (B, N, C)
        
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.5, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=False)
        # self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.attn = FlashMultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Vision3DTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=1.5,
        dropout=0.0,
        cross_attn_dim=None,  # Dimension for cross-attention with LLM
        layers_types=['res_cbam']
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size, layers_types=layers_types)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, embed_dim))  # Max 1000 patches as placeholder
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, bias=False)
        
        # Project to cross-attention dimension if needed
        self.cross_attn_dim = cross_attn_dim
        if cross_attn_dim is not None and cross_attn_dim != embed_dim:
            self.cross_attn_proj = nn.Linear(embed_dim, cross_attn_dim)
        else:
            self.cross_attn_proj = nn.Identity()
            
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.001)
    
    def forward(self, x):
        # x: (B, C, D1, D2, D3)
        B = x.shape[0]
        
        # Patch embeddings
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)
        N_patches = x.shape[1]
        
        # Add position embeddings
        pos_embed = self.pos_embed[:, :N_patches, :]
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to cross-attention dimension if needed
        x = self.cross_attn_proj(x)
        
        return x  # (B, N_patches, cross_attn_dim or embed_dim)
