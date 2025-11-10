"""
Hybrid Transformer + Neural Spline Flow with Modern Architecture Choices - OPTIMIZED VERSION
- Vectorized operations for significant speedup
- RMSNorm for efficiency and stability
- Pre-norm architecture for better gradient flow
- Bias-free projections and attention layers
- GroupNorm in flow layers for batch-size independence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from torch.cuda.amp import GradScaler, autocast
from vit_wpos_embed import *
from torch.distributions import HalfNormal

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in LLaMA and Gemma.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class Attention(nn.Module):
    """
    Custom Multi-Head Attention module with KV Caching support.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for attention with optional KV cache.

        Args:
            query: Query tensor [batch_size, query_len, d_model]
            key: Key tensor [batch_size, kv_len, d_model]
            value: Value tensor [batch_size, kv_len, d_model]
            attn_mask: Additional attention mask [batch_size, n_heads, query_len, kv_len] or broadcastable
            key_padding_mask: Padding mask for keys [batch_size, kv_len] where True = padded
            past_key_value: Tuple of (past_key, past_value)

        Returns:
            Tuple of (attention_output, present_key_value)
        """
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key.shape

        # 1. Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape for multi-head attention
        q = q.view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Handle KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v)

        # Update kv_len to include cached length
        cached_kv_len = k.shape[2]
        
        # 4. Create attention mask
        # Corrected logic to handle causal and padding masks simultaneously
        
        # Determine if this pass should be causal (i.e., for self-attention during training)
        is_causal_pass = query_len > 1 and past_key_value is None

        final_mask = None
        use_is_causal_flag = False
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:,:,0]
        if is_causal_pass:
            if key_padding_mask is not None:
                # Case 1: Causal + Padding. Must create a combined mask manually.
                causal_mask = torch.tril(torch.ones(query_len, cached_kv_len, dtype=torch.bool, device=q.device))
                padding_mask = (~key_padding_mask).view(batch_size, 1, 1, cached_kv_len)
                final_mask = causal_mask & padding_mask
            else:
                # Case 2: Causal only. Can use the efficient built-in flag.
                use_is_causal_flag = True
        elif key_padding_mask is not None:
            # Case 3: Padding only.
            final_mask = (~key_padding_mask).view(batch_size, 1, 1, cached_kv_len)

        # Also account for an explicit attn_mask if provided
        if attn_mask is not None:
            if final_mask is not None:
                 final_mask = final_mask & attn_mask
            else:
                 final_mask = attn_mask

        # 5. Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=final_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_is_causal_flag # This is now correctly set
        )

        # 6. Reshape and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        output = self.out_proj(attn_output)

        return output, present_key_value


def searchsorted(bin_locations, inputs, eps=1e-6):
    """
    Searches for which bin each input falls into.
    """
    bin_locations = bin_locations[..., :-1] + eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def rational_quadratic_spline(
    inputs,
    widths,
    heights,
    derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3
):
    """
    Rational-quadratic spline transformation.

    Args:
        inputs: Input tensor
        widths: Bin widths
        heights: Bin heights
        derivatives: Derivatives at knots
        inverse: If True, compute inverse transform
        left, right, bottom, top: Bounds for transformation
        min_bin_width, min_bin_height, min_derivative: Minimum values for numerical stability

    Returns:
        outputs: Transformed values
        log_abs_det: Log absolute determinant of Jacobian
    """
    num_bins = widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('min_bin_width * num_bins must be less than 1')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('min_bin_height * num_bins must be less than 1')

    # Normalize widths and heights
    # widths = F.softmax(widths, dim=-1)
    log_widths = F.log_softmax(widths, dim=-1)
    widths = torch.exp(log_widths)

    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Corrected: Ensure positive derivatives for interior knots, then pad with 1s for the boundaries
    derivatives = F.softplus(derivatives) + min_derivative
    derivatives = F.pad(derivatives, (1, 1), mode='constant', value=1.0)

    if inverse:
        # Inverse transform
        bin_idx = searchsorted(cumheights, inputs)
    else:
        # Forward transform
        bin_idx = searchsorted(cumwidths, inputs)

    # Clamp bin_idx to be within the valid range [0, num_bins-1].
    # This handles inputs that fall outside the defined tail_bounds.
    bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)

    # Gather parameters for the relevant bins
    input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    input_derivatives = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives_plus_one = derivatives.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    if inverse:
        # Compute inverse
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2
        ) + input_bin_heights * input_derivatives
        b = input_bin_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2
        )
        c = -input_derivatives * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        # assert (discriminant >= 0).all(), "Discriminant must be non-negative"
        discriminant = (discriminant + 1e-8).clamp(min=0)

        xi = (2 * c) / (-b - torch.sqrt(discriminant + 1e-6))
        outputs = xi * input_bin_widths + input_cumwidths

        # Compute derivative for log determinant
        derivative_numerator = input_derivatives.pow(2) * (
            input_derivatives_plus_one * xi.pow(2) + 2 * input_derivatives * xi * (1 - xi) +
            input_derivatives_plus_one * (1 - xi).pow(2)
        )
        derivative_denominator = (
            input_derivatives + (input_derivatives_plus_one - input_derivatives) * xi
        ).pow(2)
        log_abs_det = -torch.log(derivative_numerator / derivative_denominator + 1e-8)

    else:
        # Compute forward transform
        xi = (inputs - input_cumwidths) / input_bin_widths

        numerator = input_bin_heights * (
            input_derivatives * xi.pow(2) + input_derivatives_plus_one * (1 - xi).pow(2)
        )
        denominator = input_derivatives + (
            input_derivatives_plus_one - input_derivatives
        ) * xi
        outputs = input_cumheights + numerator / (denominator + 1e-10)

        # Compute derivative for log determinant
        derivative_numerator = input_derivatives.pow(2) * (
            input_derivatives_plus_one * xi.pow(2) + 2 * input_derivatives * xi * (1 - xi) +
            input_derivatives_plus_one * (1 - xi).pow(2)
        )
        derivative_denominator = (
            input_derivatives + (input_derivatives_plus_one - input_derivatives) * xi
        ).pow(2)
        log_abs_det = torch.log(derivative_numerator / derivative_denominator + 1e-8)

    return outputs, log_abs_det

class NeuralSplineFlowLayer(nn.Module):
    """
    Neural spline flow layer with conditional transformations.
    Uses GroupNorm for batch-size independence.
    """

    def __init__(self, dim: int = 6, hidden_dim: int = 128, context_dim: int = 768,
                 n_bins: int = 8, tail_bound_min: float = -3.0, tail_bound_max: float = 3.0, n_groups: int = 8):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.tail_bound_min = tail_bound_min
        self.tail_bound_max = tail_bound_max


        # Network to compute spline parameters from context
        # Corrected: We need n_bins widths, n_bins heights, and n_bins-1 interior derivatives
        params_per_dim = 3 * n_bins - 1

        # Bias-free linear layers with GroupNorm
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim, bias=False),
            nn.GroupNorm(n_groups, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GroupNorm(n_groups, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * params_per_dim, bias=False)
        )

        # Initialize last layer with small weights
        nn.init.normal_(self.context_net[-1].weight, 0, 0.001)

    def forward(self, x: torch.Tensor, context: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or reverse pass through neural spline flow layer.

        Args:
            x: Input tensor [batch_size, seq_len, 6] or [n_elements, 6]
            context: Context from transformer [batch_size, seq_len, context_dim] or [n_elements, context_dim]
            reverse: If True, perform inverse transformation

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        # Handle both batched and flattened inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x = x.reshape(-1, dim)
            context = context.reshape(-1, context.shape[-1])
        else:
            batch_size = x.shape[0]
            dim = x.shape[1]

        if torch.isnan(x).any():
            print("NaN in flow input!")
        if torch.isnan(context).any():
            print("NaN in context!")


        # Compute spline parameters
        params = self.context_net(context)
        if torch.isnan(params).any():
            print("NaN in context_net output!")
            print(f"Params stats: min={params.min():.3f}, max={params.max():.3f}")

        # Corrected: Reshape to the correct number of parameters
        params = params.reshape(-1, dim, 3 * self.n_bins - 1)

        # Split parameters
        widths = params[..., :self.n_bins]
        heights = params[..., self.n_bins:2*self.n_bins]
        derivatives = params[..., 2*self.n_bins:] # This will now correctly have n_bins-1 elements

        if torch.isnan(widths).any() or torch.isnan(heights).any() or torch.isnan(derivatives).any():
            print("NaN in width/height/derivative parameters!")

        # Apply spline transformation to each dimension
        outputs = []
        log_dets = []

        for d in range(dim):
            out, log_det = rational_quadratic_spline(
                x[:, d],
                widths[:, d, :],
                heights[:, d, :],
                derivatives[:, d, :],
                inverse=reverse,
                left=self.tail_bound_min,
                right=self.tail_bound_max,
                bottom=self.tail_bound_min,
                top=self.tail_bound_max
            )
            outputs.append(out)
            log_dets.append(log_det)

        # Stack outputs
        output = torch.stack(outputs, dim=-1)
        log_det_total = torch.stack(log_dets, dim=-1).sum(dim=-1)

        # Reshape back to original shape
        if len(original_shape) == 3:
            output = output.reshape(original_shape)
            log_det_total = log_det_total.reshape(batch_size, -1)

        return output, log_det_total


class ConditionalNormalizingFlow(nn.Module):
    """
    Stack of neural spline flow layers for full expressivity.
    """

    def __init__(self, dim: int = 6, n_layers: int = 4, hidden_dim: int = 128,
                 context_dim: int = 768, n_bins: int = 8, tail_bound_min: float = -3.0,
                 tail_bound_max: float = 3.0, n_groups: int = 8):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        # Create neural spline flow layers with GroupNorm
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                NeuralSplineFlowLayer(dim, hidden_dim, context_dim, n_bins, tail_bound_min, 
                tail_bound_max, n_groups)
            )

        # Learnable permutation parameters
        self.register_buffer('permute_indices', torch.arange(dim))
        
    def get_permutation(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation and inverse permutation for a given layer."""
        if layer_idx % 2 == 0:
            return self.permute_indices, torch.argsort(self.permute_indices)
        else:
            # Simple alternating pattern: reverse for odd layers
            perm = torch.arange(self.dim - 1, -1, -1, device=self.permute_indices.device)
            return perm, perm  # Reverse is its own inverse

    def forward(self, x: torch.Tensor, context: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward or reverse pass through the flow.

        Args:
            x: Input tensor [batch_size, seq_len, 6] or [n_elements, 6]
            context: Context from transformer [batch_size, seq_len, context_dim] or [n_elements, context_dim]
            reverse: If True, perform inverse transformation

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        # Initialize log determinant
        if x.dim() == 3:
            log_det_total = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        else:
            log_det_total = torch.zeros(x.shape[0], device=x.device)

        if not reverse:
            # Forward pass through flow
            for i, layer in enumerate(self.layers):
                # Get permutation for this layer
                perm, inv_perm = self.get_permutation(i)
                
                # Apply permutation
                x = x[..., perm]
                
                # Apply flow layer
                x, log_det = layer(x, context, reverse=False)
                log_det_total = log_det_total + log_det

                if torch.isnan(log_det).any():
                    print(f"NaN in log_det at layer {i}")
                    log_det = log_det.clamp(min=-20, max=20)
                
                # Apply inverse permutation
                x = x[..., inv_perm]
        else:
            # Reverse pass through flow (for generation)
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                
                # Get permutation for this layer
                perm, inv_perm = self.get_permutation(i)
                
                # Apply permutation
                x = x[..., perm]
                
                # Apply flow layer
                x, log_det = layer(x, context, reverse=True)
                log_det_total = log_det_total + log_det
                
                # Apply inverse permutation
                x = x[..., inv_perm]

        return x, log_det_total


class ContinuousTransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture and bias-free attention.
    Uses RMSNorm and follows modern LLM design patterns.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()

        # Pre-norm for self-attention
        self.norm1 = RMSNorm(d_model)
        self.self_attention = Attention(d_model, n_heads, dropout=dropout, bias=False)

        # Pre-norm for cross-attention
        self.norm2 = RMSNorm(d_model)
        self.cross_attention = Attention(d_model, n_heads, dropout=dropout, bias=False)

        # Pre-norm for feed-forward
        self.norm3 = RMSNorm(d_model)

        # Bias-free feed-forward network with SwiGLU activation (used in LLaMA)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        x: torch.Tensor,
        field_embeddings: Optional[torch.Tensor] = None,
        self_padding_mask: Optional[torch.Tensor] = None,
        field_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block with pre-norm and optional KV cache.
        """
        # --- Start of Correction ---
        # The cache is only for the self-attention layer.
        present_self_attn_cache = None

        # 1. Pre-norm self-attention with residual and KV cache
        normed = self.norm1(x)
        attn_out, present_self_attn_cache = self.self_attention(
            query=normed, key=normed, value=normed,
            key_padding_mask=self_padding_mask,
            past_key_value=past_key_value
        )
        x = x + attn_out

        # 2. Pre-norm cross-attention with residual (if field embeddings provided)
        if field_embeddings is not None:
            normed = self.norm2(x)
            
            # For cross-attention, K and V are static and come from field_embeddings.
            # NO KV-cache is used here.
            cross_attn_out, _ = self.cross_attention(
                query=normed, key=field_embeddings, value=field_embeddings,
                key_padding_mask=field_key_padding_mask,
                past_key_value=None # Explicitly disable caching for cross-attention
            )
            x = x + cross_attn_out

        # 3. Pre-norm feed-forward with SwiGLU and residual
        normed = self.norm3(x)
        ff_out = self.w3(F.silu(self.w1(normed)) * self.w2(normed))
        x = x + self.dropout(ff_out)

        # Return the cache for the self-attention layer ONLY.
        return x, present_self_attn_cache if use_cache else None


class HybridTransformerFlow(nn.Module):
    """
    Main model combining continuous transformer with neural spline flows.
    Modern architecture with RMSNorm, pre-norm, bias-free layers, and GroupNorm.

    Key features:
    - Pre-norm transformer blocks with RMSNorm
    - Bias-free attention and projections
    - GroupNorm in flow layers for batch independence
    - SwiGLU activation in feed-forward layers
    - Matrix input format with -100 padding
    - OPTIMIZED with vectorized operations
    """

    def __init__(
        self,
        config_dict
    ):
        super().__init__()
        # extract all the config dict keys as self attributes
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.max_seq_len = self.max_blocks * self.input_dim  # Max sequence length in elements
        # Special learned embeddings for control tokens
        self.start_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        # Padding embedding as a fixed zero tensor (not learnable)
        self.register_buffer('pad_embedding', torch.zeros(1, self.d_model))

        # Bias-free linear projections
        self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)
        self.output_projection = nn.Linear(self.d_model, self.input_dim, bias=False)


        # Learned positional encoding
        # self.pos_encoding = nn.Parameter(torch.randn(1, self.max_seq_len) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.max_blocks) * 0.02)        

        # Transformer blocks with modern architecture
        self.transformer_blocks = nn.ModuleList([
            ContinuousTransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout_val)
            for _ in range(self.n_layers)
        ])

        # Final RMSNorm before output heads
        self.final_norm = RMSNorm(self.d_model)

        # Neural spline flow with GroupNorm
        self.flow = ConditionalNormalizingFlow(
            self.input_dim, self.flow_layers, self.flow_hidden, self.d_model, self.n_bins, 
            self.tail_bound_min, self.tail_bound_max, self.n_groups
        )

        self.vit3D = Vision3DTransformer(
            in_channels=self.ninp_density,
            patch_size=self.patch_size,
            embed_dim=self.d_model,
            depth=self.n_layers_vit,
            num_heads=self.n_heads_vit,
            dropout=self.dropout_val,
            cross_attn_dim=self.d_model - self.nparams,
            layers_types=self.layers_types
        )

        # Binary decision head (bias-free)
        self.decision_head = nn.Linear(self.d_model, 2, bias=False)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_val)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following modern practices."""
        # Initialize linear layers
        def init_linear(module):
            if isinstance(module, nn.Linear):
                # Use scaled initialization (GPT-2 style)
                std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)

        self.apply(init_linear)

        # Special initialization for output projections (smaller)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        nn.init.normal_(self.decision_head.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def create_padding_mask(self, seq_lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        """
        OPTIMIZED: Vectorized creation of padding mask.
        
        Args:
            seq_lengths: Tensor of sequence lengths [batch_size]
            max_len: Maximum sequence length
            device: Device to create mask on
            
        Returns:
            Padding mask [batch_size, max_len] where True indicates padded positions
        """
        batch_size = seq_lengths.shape[0]
        position_indices = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = position_indices >= seq_lengths.unsqueeze(1)
        return mask

    def matrix_to_sequences(self, data_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        OPTIMIZED: Vectorized conversion from matrix format to sequences.
        
        Args:
            data_matrix: Input matrix [batch_size, max_blocks * input_dim] with -100 padding
            
        Returns:
            valid_data: Tensor of shape [batch_size, max_blocks, input_dim]
            n_blocks: Tensor of shape [batch_size] with number of blocks per sequence
            block_valid_mask: Boolean mask of shape [batch_size, max_blocks] indicating valid blocks
        """
        batch_size, total_dim = data_matrix.shape
        max_blocks = total_dim // self.input_dim
        
        # Reshape to blocks
        data_reshaped = data_matrix.reshape(batch_size, max_blocks, self.input_dim)
        
        # Find valid blocks (those that don't contain pad_value)
        is_pad = (data_reshaped == self.pad_value)
        block_is_invalid = is_pad.any(dim=2)  # [batch_size, max_blocks]
        
        # Find first invalid block per sequence using cumsum trick
        first_invalid_flag = (block_is_invalid.float().cumsum(dim=1) == 1) & block_is_invalid
        has_invalid = block_is_invalid.any(dim=1)
        
        # Get index of first invalid block
        first_invalid_idx = torch.where(
            has_invalid,
            first_invalid_flag.float().argmax(dim=1),
            torch.tensor(max_blocks, device=data_matrix.device)
        )
        
        # Create mask for valid blocks
        block_indices = torch.arange(max_blocks, device=data_matrix.device).unsqueeze(0)
        block_valid_mask = block_indices < first_invalid_idx.unsqueeze(1)
        
        # Count valid blocks per sequence
        n_blocks = block_valid_mask.sum(dim=1)
        
        # Apply mask to get valid data (keep structure, invalid blocks become zeros)
        valid_data = data_reshaped * block_valid_mask.unsqueeze(2)
        return valid_data, n_blocks, block_valid_mask

    def encode_sequence_from_matrix(self, data_matrix: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        OPTIMIZED: Vectorized encoding of sequences from matrix format.

        Args:
            data_matrix: Input matrix [batch_size, max_blocks * input_dim] with -100 padding
            device: Device to create tensors on

        Returns:
            embeddings: Encoded sequence [batch_size, seq_len, d_model]
            values: Original values [batch_size, seq_len, input_dim]
            decision_mask: Positions where decisions are made [batch_size, seq_len]
            seq_lengths: Tensor of actual sequence lengths [batch_size]
        """
        batch_size = data_matrix.shape[0]
        
        # Get valid data and metadata
        values, n_blocks, block_valid_mask = self.matrix_to_sequences(data_matrix)
        
        # Create embeddings with START token
        embeddings = self.input_projection(values[:,:-1,:])
        embeddings = torch.cat((self.start_embedding.expand(batch_size, -1, -1), embeddings), dim=1)
        decision_mask = torch.zeros(batch_size, self.max_blocks, dtype=torch.bool, device=device)
                
        n_elements = n_blocks * self.input_dim
        max_n_blocks = n_blocks.max().item()
        pad_mask = torch.ones_like(embeddings, dtype=torch.bool, device=device)
        pad_mask[:,0,:] = False  # START token is not padding
        if max_n_blocks > 0:
            for b in range(batch_size):
                n_blocks_b = n_blocks[b].item()
                if n_blocks_b > 0:
                    embeddings[b, n_blocks_b:,:] = self.pad_embedding[None,:]
                    pad_mask[b, 1:n_blocks_b+1,:] = False
                    decision_mask[b, :n_blocks_b] = True
        return embeddings, values, decision_mask, pad_mask

    def sequences_to_matrix(self, sequences: List[torch.Tensor], max_blocks: int) -> torch.Tensor:
        """
        OPTIMIZED: Batch conversion of sequences to matrix format.
        
        Args:
            sequences: List of generated sequences
            max_blocks: Maximum number of blocks
            
        Returns:
            Matrix of shape [n_samples, max_blocks * input_dim] with -100 padding
        """
        n_samples = len(sequences)
        device = sequences[0].device if sequences and sequences[0].numel() > 0 else 'cpu'
        
        # Initialize matrix with padding
        matrix = torch.full((n_samples, max_blocks * self.input_dim), 
                           self.pad_value, device=device, dtype=torch.float32)
        
        # Vectorized assignment for all sequences at once
        for i, seq in enumerate(sequences):
            if seq.numel() > 0:
                n_elements = min(seq.numel(), matrix.shape[1])
                matrix[i, :n_elements] = seq.reshape(-1)[:n_elements]
        
        return matrix

    def forward(self, data_matrix: torch.Tensor,
                vit_fields: Optional[torch.Tensor] = None,
                params: Optional[torch.Tensor] = None,
                dist_type: str = 'half_gaussian',
                use_cache: bool = False,
                past_key_values: Optional[List[Tuple]] = None,
                flow_weight = None
               ) -> Dict[str, torch.Tensor]:
        """
        OPTIMIZED: Forward pass with vectorized operations.

        Args:
            data_matrix: Input matrix [batch_size, max_blocks * input_dim] with -100 padding
            vit_fields: Field embeddings from vision transformer [batch_size, n_field_tokens, d_model]

        Returns:
            Dictionary containing:
                - flow_log_prob: Log probability from normalizing flow
                - decision_loss: Binary cross-entropy for continuation decisions
                - total_loss: Combined loss
        """
        device = next(self.parameters()).device
        
        # Ensure data is on correct device
        if data_matrix.device != device:
            data_matrix = data_matrix.to(device)

        # Use optimized encoding
        embeddings, values, decision_mask, padding_mask = self.encode_sequence_from_matrix(data_matrix, device)
        batch_size, seq_len, _ = embeddings.shape


        # Add positional encoding to sequence
        embeddings = embeddings + self.pos_encoding[:, :seq_len, None].expand(-1, -1, self.d_model)
        embeddings = self.dropout(embeddings)

        # Create padding mask for field tokens if needed
        field_key_padding_mask = None
        if vit_fields is not None:
            if vit_fields.device != device:
                vit_fields = vit_fields.to(device)
            vit_fields = self.vit3D(vit_fields)
            if params.device != device:
                params = params.to(device)
            params_to_concat = params[:, None, :].expand(-1, vit_fields.shape[1], -1)
            vit_fields = torch.cat((vit_fields, params_to_concat), dim=-1)

        # Pass through transformer blocks
        hidden = embeddings
        next_cache = [] if use_cache else None
        for i, block in enumerate(self.transformer_blocks):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden, cache = block(
                hidden,
                field_embeddings=vit_fields,
                self_padding_mask=padding_mask,
                field_key_padding_mask=field_key_padding_mask,
                use_cache=use_cache,
                past_key_value=layer_past_key_value
            )
            if use_cache:
                next_cache.append(cache)

        # Apply final norm
        hidden = self.final_norm(hidden)
        # OPTIMIZED: Decision loss computation
        if decision_mask.any():
            decision_logits = self.decision_head(hidden)
            
            # Vectorized target creation
            _, n_blocks, _ = self.matrix_to_sequences(data_matrix)
            
            decision_targets = decision_mask.to(torch.long)
            
            decision_loss = F.cross_entropy(
                decision_logits[decision_mask],
                decision_targets[decision_mask],
                reduction='mean'
            )
        else:
            decision_loss = torch.tensor(0.0, device=device)

        _, n_blocks, _ = self.matrix_to_sequences(data_matrix)
        
        n_elements = n_blocks * self.input_dim
        data_mask = decision_mask

        if data_mask.any():
            if torch.max(n_blocks) > 0:
                # Forward through flow
                latent, log_det = self.flow(
                    values,
                    hidden,
                    reverse=False
                )

                # Log probability under Gaussian prior
                if dist_type == 'gaussian':
                    log_prob_prior = -0.5 * (latent ** 2).sum(dim=-1) - 0.5 * self.input_dim * math.log(2 * math.pi)
                elif dist_type == 'half_gaussian':
                    hf = HalfNormal(1)
                    log_prob_prior = hf.log_prob(latent)
                    log_prob_prior = log_prob_prior.sum(dim=-1)  # Sum over input_dim

                log_prob = log_prob_prior + log_det
                log_prob = log_prob[data_mask]
                flow_log_prob = log_prob.mean()
            else:
                flow_log_prob = torch.tensor(0.0, device=device)
        else:
            flow_log_prob = torch.tensor(0.0, device=device)

        # Combined loss
        if flow_weight is None:
            flow_weight = 0  # Start with lower weight for flow
        if flow_weight == 0:
            total_loss = decision_loss
        else:
            total_loss = decision_loss + flow_weight * (-flow_log_prob)        

        results = {
            'flow_log_prob': flow_log_prob,
            'decision_loss': decision_loss,
            'total_loss': total_loss,
        }
        
        if use_cache:
            results['past_key_values'] = next_cache
            
        return results

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        vit_fields: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
        dist_type: str = 'half_gaussian',
        temperature: float = 1.0,
        max_blocks: Optional[int] = None,
        truncation: float = 2.0
    ) -> torch.Tensor:
        """
        Generate sequences using autoregressive transformer with flow sampling.
        
        Args:
            n_samples: Number of samples to generate
            vit_fields: Field embeddings from vision transformer [n_samples, H, W, D, C]
            params: Parameters [n_samples, nparams]
            temperature: Temperature for decision sampling
            max_blocks: Maximum blocks to generate (defaults to self.max_blocks)
            truncation: Truncate Gaussian samples to [-truncation, truncation]
            
        Returns:
            Generated matrix [n_samples, max_blocks * input_dim] with self.pad_value padding
        """
        device = next(self.parameters()).device
        if max_blocks is None:
            max_blocks = self.max_blocks
        
        # Process vit fields if provided
        field_embeddings = None
        if vit_fields is not None:
            if vit_fields.device != device:
                vit_fields = vit_fields.to(device)
            field_embeddings = self.vit3D(vit_fields)
            if params is not None:
                if params.device != device:
                    params = params.to(device)
                params_expanded = params[:, None, :].expand(-1, field_embeddings.shape[1], -1)
                field_embeddings = torch.cat((field_embeddings, params_expanded), dim=-1)
        
        # Initialize with START token
        hidden = self.start_embedding.expand(n_samples, -1, -1)
        hidden = hidden + self.pos_encoding[:, :1, None].expand(-1, -1, self.d_model)
        hidden = self.dropout(hidden)
        
        # Initialize output tensor with padding
        output = torch.full((n_samples, max_blocks, self.input_dim), 
                            self.pad_value, device=device, dtype=torch.float32)
        
        # Initialize KV cache
        past_key_values = None
        
        # Track which sequences are still generating
        active_mask = torch.ones(n_samples, dtype=torch.bool, device=device)
        
        for block_idx in range(max_blocks):
            if not active_mask.any():
                break
                
            # Pass through transformer blocks with KV cache
            current_hidden = hidden
            next_cache = []
            
            for layer_idx, block in enumerate(self.transformer_blocks):
                layer_past = past_key_values[layer_idx] if past_key_values is not None else None
                
                current_hidden, cache = block(
                    current_hidden,
                    field_embeddings=field_embeddings,
                    self_padding_mask=None,
                    field_key_padding_mask=None,
                    use_cache=True,
                    past_key_value=layer_past
                )
                next_cache.append(cache)
            
            past_key_values = next_cache
            
            # Apply final norm
            current_hidden = self.final_norm(current_hidden)
            
            # Get embedding for current position (last token)
            current_embedding = current_hidden[:, -1:, :]
            
            # Sample from truncated Gaussian prior
            if dist_type == 'gaussian':
                z = torch.randn(n_samples, 1, self.input_dim, device=device) * temperature
            elif dist_type == 'half_gaussian':
                hf = HalfNormal(1)
                z = hf.sample((n_samples, 1, self.input_dim)).to(device) * temperature

            # z = torch.clamp(z, min=-truncation, max=truncation)
            
            # Transform through flow (reverse direction)
            with torch.no_grad():
                generated_value, _ = self.flow(z, current_embedding, reverse=True)
            
            # Clamp output to valid data range
            generated_value = torch.clamp(generated_value, min=self.tail_bound_min, max=self.tail_bound_max)
            generated_value = generated_value.squeeze(1)  # [n_samples, input_dim]
            
            # Store generated block only for active sequences
            output[active_mask, block_idx] = generated_value[active_mask]
            
            # Decide whether to continue
            decision_logits = self.decision_head(current_embedding)
            continue_prob = torch.softmax(decision_logits / temperature, dim=-1)[:, :, 1]
            continue_generating = torch.bernoulli(continue_prob).bool().squeeze(1)
            
            # Update active mask
            active_mask = active_mask & continue_generating
            
            # Prepare next hidden state for next iteration
            if block_idx < max_blocks - 1 and active_mask.any():
                # Project generated value to embedding space
                next_embedding = self.input_projection(generated_value.unsqueeze(1))
                
                # Add positional encoding
                pos_idx = block_idx + 1  # +1 because of START token
                next_embedding = next_embedding + self.pos_encoding[:, pos_idx:pos_idx+1, None].expand(-1, -1, self.d_model)
                next_embedding = self.dropout(next_embedding)
                
                # Concatenate to hidden sequence
                hidden = torch.cat([hidden, next_embedding], dim=1)
        
        # Reshape to matrix format [n_samples, max_blocks * input_dim]
        output_matrix = output.reshape(n_samples, max_blocks * self.input_dim)
        
        return output_matrix

