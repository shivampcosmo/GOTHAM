"""
Hybrid Transformer + Neural Spline Flow with Modern Architecture Choices
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
        # For SDPA, True means "attend" and False means "mask out"
        combined_mask = None
        
        # Handle padding mask
        if key_padding_mask is not None:
            # key_padding_mask is [B, S], True for padded tokens
            # Convert to attention mask format [B, 1, 1, S] where True = attend
            padding_attn_mask = (~key_padding_mask).view(batch_size, 1, 1, cached_kv_len)
            combined_mask = padding_attn_mask.to(q.dtype)
        
        # During training with full sequences, use causal mask
        is_causal = query_len > 1 and past_key_value is None
        
        # If we have an additional attention mask, combine it
        if attn_mask is not None and combined_mask is not None:
            combined_mask = combined_mask * attn_mask
        elif attn_mask is not None:
            combined_mask = attn_mask

        # 5. Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
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
    widths = F.softmax(widths, dim=-1)
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

    # Ensure positive derivatives
    derivatives = F.softplus(derivatives) + min_derivative
    derivatives = F.pad(derivatives, (1, 1), mode='constant', value=1.0)

    if inverse:
        # Inverse transform
        bin_idx = searchsorted(cumheights, inputs)
    else:
        # Forward transform
        bin_idx = searchsorted(cumwidths, inputs)

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
        assert (discriminant >= 0).all(), "Discriminant must be non-negative"

        xi = (2 * c) / (-b - torch.sqrt(discriminant + 1e-8))
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
        outputs = input_cumheights + numerator / denominator

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
                 n_bins: int = 8, tail_bound: float = 3.0, n_groups: int = 8):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.tail_bound = tail_bound

        # Network to compute spline parameters from context
        # For each dimension, we need: n_bins widths, n_bins heights, n_bins+1 derivatives
        params_per_dim = 3 * n_bins + 1

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
        nn.init.normal_(self.context_net[-1].weight, 0, 0.01)

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

        # Compute spline parameters
        params = self.context_net(context)
        params = params.reshape(-1, dim, 3 * self.n_bins + 1)

        # Split parameters
        widths = params[..., :self.n_bins]
        heights = params[..., self.n_bins:2*self.n_bins]
        derivatives = params[..., 2*self.n_bins:]

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
                left=-self.tail_bound,
                right=self.tail_bound,
                bottom=-self.tail_bound,
                top=self.tail_bound
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
                 context_dim: int = 768, n_bins: int = 8, tail_bound: float = 3.0,
                 n_groups: int = 8):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        # Create neural spline flow layers with GroupNorm
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                NeuralSplineFlowLayer(dim, hidden_dim, context_dim, n_bins, tail_bound, n_groups)
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
        past_key_value: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block with pre-norm and optional KV cache.
        """
        present_key_values = []

        # 1. Pre-norm self-attention with residual and KV cache
        normed = self.norm1(x)
        self_attn_cache = past_key_value[0] if past_key_value is not None else None

        attn_out, present_self_attn_cache = self.self_attention(
            query=normed, key=normed, value=normed,
            key_padding_mask=self_padding_mask,
            past_key_value=self_attn_cache
        )
        if use_cache:
            present_key_values.append(present_self_attn_cache)
        x = x + attn_out

        # 2. Pre-norm cross-attention with residual (if field embeddings provided)
        if field_embeddings is not None:
            normed = self.norm2(x)
            cross_attn_cache = past_key_value[1] if past_key_value is not None and len(past_key_value) > 1 else None

            # For cross-attention, K and V come from field_embeddings
            cross_attn_out, present_cross_attn_cache = self.cross_attention(
                query=normed, key=field_embeddings, value=field_embeddings,
                key_padding_mask=field_key_padding_mask,
                past_key_value=cross_attn_cache
            )
            if use_cache:
                present_key_values.append(present_cross_attn_cache)
            x = x + cross_attn_out

        # 3. Pre-norm feed-forward with SwiGLU and residual
        normed = self.norm3(x)
        ff_out = self.w3(F.silu(self.w1(normed)) * self.w2(normed))
        x = x + self.dropout(ff_out)

        return x, tuple(present_key_values) if use_cache else None


class HybridTransformerFlow(nn.Module):
    """
    Main model combining continuous transformer with neural spline flows.
    Modern architecture with RMSNorm, pre-norm, bias-free layers, and GroupNorm.

    Key features:
    - Pre-norm transformer blocks with RMSNorm
    - Bias-free attention and projections
    - GroupNorm in flow layers for batch independence
    - SwiGLU activation in feed-forward layers
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 6,
        d_ff: int = 3072,
        flow_layers: int = 4,
        flow_hidden: int = 128,
        n_bins: int = 8,
        tail_bound: float = 3.0,
        max_seq_len: int = 121,  # Maximum: 1 START + 20*6 data = 121
        dropout: float = 0.1,
        n_groups: int = 8  # Groups for GroupNorm
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers  # Store for initialization

        # Special learned embeddings for control tokens
        self.start_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Padding embedding as a fixed zero tensor (not learnable)
        self.register_buffer('pad_embedding', torch.zeros(1, 1, d_model))

        # Bias-free linear projections
        self.input_projection = nn.Linear(input_dim, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, input_dim, bias=False)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer blocks with modern architecture
        self.transformer_blocks = nn.ModuleList([
            ContinuousTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final RMSNorm before output heads
        self.final_norm = RMSNorm(d_model)

        # Neural spline flow with GroupNorm
        self.flow = ConditionalNormalizingFlow(
            input_dim, flow_layers, flow_hidden, d_model, n_bins, tail_bound, n_groups
        )

        # Binary decision head (bias-free)
        self.decision_head = nn.Linear(d_model, 2, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

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

    def create_padding_mask(self, seq_lengths: List[int], max_len: int, device: torch.device) -> torch.Tensor:
        """
        Create padding mask where True indicates padded positions.
        
        Args:
            seq_lengths: List of actual sequence lengths
            max_len: Maximum sequence length in the batch
            device: Device to create mask on
            
        Returns:
            Padding mask [batch_size, max_len]
        """
        batch_size = len(seq_lengths)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = False
        return mask

    def encode_sequence(self, data: List[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Encode a variable-length sequence into model format.

        Args:
            data: List of tensors, each [6] or [n_blocks, 6]
            device: Device to create tensors on

        Returns:
            embeddings: Encoded sequence [batch_size, seq_len, d_model]
            values: Original values [batch_size, seq_len, 6]
            decision_mask: Positions where decisions are made [batch_size, seq_len]
            seq_lengths: List of actual sequence lengths (before padding)
        """
        batch_size = len(data)
        sequences = []
        values_list = []
        decision_masks = []
        seq_lengths = []

        for batch_idx in range(batch_size):
            seq = []
            vals = []
            dec_mask = []

            # Start token (decision point: empty sequence or not)
            seq.append(self.start_embedding.squeeze(0))
            vals.append(torch.zeros(1, self.input_dim, device=device))
            dec_mask.append(True)  # Decision needed after START

            if data[batch_idx].numel() > 0:  # Non-empty sequence
                blocks = data[batch_idx].reshape(-1, self.input_dim)

                for elem_idx, element in enumerate(blocks):
                    # Add data element
                    projected = self.input_projection(element.unsqueeze(0))
                    seq.append(projected.squeeze(0))
                    vals.append(element.unsqueeze(0))

                    # Decision needed after each element (every position can decide to stop)
                    dec_mask.append(True)

            sequences.append(torch.stack(seq))
            values_list.append(torch.cat(vals))
            decision_masks.append(torch.tensor(dec_mask, device=device))
            seq_lengths.append(len(seq))

        # Pad sequences to same length
        max_len = max(seq_lengths)
        max_len = min(max_len, self.max_seq_len)

        padded_sequences = []
        padded_values = []
        padded_decisions = []

        for seq, vals, dec in zip(sequences, values_list, decision_masks):
            cur_len = seq.shape[0]
            pad_len = max_len - cur_len
            if pad_len > 0:
                pad_emb = self.pad_embedding.squeeze(0).expand(pad_len, -1)
                seq = torch.cat([seq, pad_emb])

                pad_vals = torch.zeros(pad_len, self.input_dim, device=device)
                vals = torch.cat([vals, pad_vals])

                pad_dec = torch.zeros(pad_len, dtype=torch.bool, device=device)
                dec = torch.cat([dec, pad_dec])

            padded_sequences.append(seq)
            padded_values.append(vals)
            padded_decisions.append(dec)

        embeddings = torch.stack(padded_sequences)
        values = torch.stack(padded_values)
        decision_mask = torch.stack(padded_decisions)

        return embeddings, values, decision_mask, seq_lengths

    def forward(self, data: List[torch.Tensor],
                vit_fields: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_values: Optional[List[Tuple]] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with optional field conditioning.

        Args:
            data: List of variable-length sequences, each [n_blocks, 6]
            vit_fields: Field embeddings from vision transformer [batch_size, n_field_tokens, d_model]

        Returns:
            Dictionary containing:
                - flow_log_prob: Log probability from normalizing flow
                - decision_loss: Binary cross-entropy for continuation decisions
                - total_loss: Combined loss
        """
        device = next(self.parameters()).device

        # Encode sequences
        embeddings, values, decision_mask, seq_lengths = self.encode_sequence(data, device)
        batch_size, seq_len, _ = embeddings.shape

        # Create padding mask
        padding_mask = self.create_padding_mask(seq_lengths, seq_len, device)

        # Add positional encoding to sequence
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :]
        embeddings = self.dropout(embeddings)

        # Create padding mask for field tokens if needed
        field_key_padding_mask = None
        if vit_fields is not None:
            field_key_padding_mask = (vit_fields.abs().sum(dim=-1) == 0)

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

        # Decision loss (only at decision points)
        if decision_mask.any():
            decision_logits = self.decision_head(hidden)

            # Create targets: 1 if there's more data, 0 if sequence ends
            decision_targets = torch.zeros_like(decision_mask, dtype=torch.long)
            for b in range(batch_size):
                dec_positions = torch.where(decision_mask[b])[0]
                for i, pos in enumerate(dec_positions[:-1]):
                    # Check if there's valid data at the next position
                    next_pos = pos + 1
                    if next_pos < seq_lengths[b]:
                        decision_targets[b, pos] = 1

            decision_loss = F.cross_entropy(
                decision_logits[decision_mask],
                decision_targets[decision_mask],
                reduction='mean'
            )
        else:
            decision_loss = torch.tensor(0.0, device=device)

        # Flow-based density estimation (only on actual data positions)
        # Create mask for actual data positions (excluding START token and padding)
        data_mask = torch.zeros_like(decision_mask)
        for b in range(batch_size):
            if data[b].numel() > 0:
                n_elements = data[b].reshape(-1, self.input_dim).shape[0]
                # Data positions start at index 1 (after START token)
                data_mask[b, 1:n_elements+1] = True

        if data_mask.any():
            # Extract only the data positions for flow computation
            data_hidden = hidden[data_mask]
            data_values = values[data_mask]

            # Reshape for flow processing
            n_data = data_hidden.shape[0]
            if n_data > 0:
                # Forward through flow
                latent, log_det = self.flow(
                    data_values.reshape(n_data, self.input_dim),
                    data_hidden.reshape(n_data, self.d_model),
                    reverse=False
                )

                # Log probability under Gaussian prior
                log_prob_prior = -0.5 * (latent ** 2).sum(dim=-1) - 0.5 * self.input_dim * math.log(2 * math.pi)
                log_prob = log_prob_prior + log_det
                flow_log_prob = log_prob.mean()
            else:
                flow_log_prob = torch.tensor(0.0, device=device)
        else:
            flow_log_prob = torch.tensor(0.0, device=device)

        # Combined loss
        total_loss = decision_loss - flow_log_prob

        results = {
            'flow_log_prob': flow_log_prob,
            'decision_loss': decision_loss,
            'total_loss': total_loss,
        }
        
        if use_cache:
            results['past_key_values'] = next_cache
            
        return results

    @torch.no_grad()
    def generate(self, n_samples: int = 1, vit_fields: Optional[torch.Tensor] = None,
                 temperature: float = 1.0, max_blocks: int = 20) -> List[torch.Tensor]:
        """
        Efficiently generate new sequences using KV Caching.
        """
        self.eval()
        device = next(self.parameters()).device

        # Initialize lists to hold the generated sequences for each sample in the batch
        generated_sequences = [[] for _ in range(n_samples)]

        # All samples start with the START token
        input_embeddings = self.start_embedding.repeat(n_samples, 1, 1)

        # Initialize KV cache
        past_key_values = None

        # Process field embeddings
        field_key_padding_mask = None
        if vit_fields is not None:
            # Expand vit_fields if n_samples > batch_size of fields
            if vit_fields.shape[0] < n_samples:
                vit_fields = vit_fields.repeat_interleave(n_samples // vit_fields.shape[0], dim=0)
            field_key_padding_mask = (vit_fields.abs().sum(dim=-1) == 0)

        # Keep track of which sequences are still being generated
        unfinished_sequences = torch.ones(n_samples, dtype=torch.bool, device=device)

        current_pos = 0
        max_elements = max_blocks * self.input_dim
        
        for step in range(max_elements + 1):
            # If all sequences are finished, stop
            if not unfinished_sequences.any():
                break

            # Add positional encoding for the current step
            pos_emb = self.pos_encoding[:, current_pos:current_pos+1, :]
            model_input = input_embeddings + pos_emb

            # Transformer forward pass (one step)
            hidden = model_input
            next_past_key_values = []
            
            for i, block in enumerate(self.transformer_blocks):
                layer_past_kv = past_key_values[i] if past_key_values is not None else None
                hidden, cache = block(
                    hidden,
                    field_embeddings=vit_fields,
                    self_padding_mask=None,  # No padding during generation
                    field_key_padding_mask=field_key_padding_mask,
                    past_key_value=layer_past_kv,
                    use_cache=True
                )
                next_past_key_values.append(cache)
            
            past_key_values = next_past_key_values
            hidden = self.final_norm(hidden)

            # Decision: continue or stop?
            decision_logits = self.decision_head(hidden)
            decision_probs = F.softmax(decision_logits / temperature, dim=-1).squeeze(1)
            decisions = torch.multinomial(decision_probs, 1).squeeze(1)

            # Update unfinished sequences
            # Only sequences that were already unfinished can become finished
            newly_finished = unfinished_sequences & (decisions == 0)
            unfinished_sequences = unfinished_sequences & (decisions == 1)

            # If this was just the START token and we decided to stop, don't generate
            if step == 0 and newly_finished.any():
                # Empty sequences for those that stopped at START
                continue

            # Only generate if we're continuing (not at a decision point that decided to stop)
            if not should_decide or decisions.any():
                # Generate next data element for sequences that are continuing
                if unfinished_sequences.any():
                    # Sample from Gaussian prior
                    z = torch.randn(n_samples, 1, self.input_dim, device=device)
                    
                    # Transform through flow (reverse direction)
                    data_value, _ = self.flow(
                        z.squeeze(1),
                        hidden.squeeze(1),
                        reverse=True
                    )
                    data_value = data_value.unsqueeze(1)

                    # Store generated values for unfinished sequences
                    for i in range(n_samples):
                        if unfinished_sequences[i]:
                            generated_sequences[i].append(data_value[i].squeeze())

                    # Project the generated data for next step
                    # Only update embeddings for unfinished sequences
                    new_embeddings = self.input_projection(data_value)
                    input_embeddings = torch.where(
                        unfinished_sequences.unsqueeze(-1).unsqueeze(-1),
                        new_embeddings,
                        input_embeddings  # Keep old embeddings for finished sequences
                    )
                
                current_pos += 1

        # Format output
        output_list = []
        for seq_tensors in generated_sequences:
            if seq_tensors:
                output_list.append(torch.stack(seq_tensors))
            else:
                output_list.append(torch.zeros(0, self.input_dim, device=device))

        return output_list


# Keep the same dataset and training utilities
class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for variable-length sequences of 6D vectors with optional field conditioning.
    """

    def __init__(self, data: List[np.ndarray], fields: Optional[List[np.ndarray]] = None):
        """
        Args:
            data: List of numpy arrays, each of shape [n_blocks * 6] or [n_blocks, 6]
            fields: Optional list of field embeddings, each [n_field_tokens, d_model]
        """
        self.data = []
        for item in data:
            if isinstance(item, np.ndarray):
                tensor = torch.FloatTensor(item)
                if tensor.numel() > 0:
                    tensor = tensor.reshape(-1, 6)
                self.data.append(tensor)
            else:
                self.data.append(torch.FloatTensor(item))

        self.fields = None
        if fields is not None:
            self.fields = [torch.FloatTensor(f) for f in fields]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.fields is not None:
            return self.data[idx], self.fields[idx]
        return self.data[idx], None


def collate_fn(batch: List) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """Custom collate function that preserves variable-length sequences and stacks fields."""
    if len(batch[0]) == 2:
        sequences = [item[0] for item in batch]
        fields = [item[1] for item in batch if item[1] is not None]

        if fields:
            max_field_len = max(f.shape[0] for f in fields)
            padded_fields = []
            for f in fields:
                if f.shape[0] < max_field_len:
                    pad_len = max_field_len - f.shape[0]
                    padding = torch.zeros(pad_len, f.shape[1])
                    f = torch.cat([f, padding], dim=0)
                padded_fields.append(f)
            fields_tensor = torch.stack(padded_fields)
            return sequences, fields_tensor
        return sequences, None
    else:
        return batch, None


def train_model(
    model: HybridTransformerFlow,
    train_data: List[np.ndarray],
    train_fields: Optional[List[np.ndarray]] = None,
    val_data: Optional[List[np.ndarray]] = None,
    val_fields: Optional[List[np.ndarray]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,  # Reduced from 0.1
    gradient_accumulation_steps: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Training loop for the hybrid model with optional field conditioning.
    """
    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    dataset = SequenceDataset(train_data, train_fields)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # AdamW optimizer with modern hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)  # Modern beta2 value
    )

    # Cosine annealing with warmup
    warmup_steps = min(500, len(dataloader) * 2)
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Mixed precision training
    scaler = GradScaler()
    
    # Training loop
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        total_flow_loss = 0
        total_decision_loss = 0
        n_batches = 0

        for batch_idx, (batch_data, batch_fields) in enumerate(dataloader):
            # Move fields to device if present
            if batch_fields is not None:
                batch_fields = batch_fields.to(device)

            # Forward pass with mixed precision
            with autocast():
                outputs = model(batch_data, vit_fields=batch_fields)
                loss = outputs['total_loss'] / gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Track losses
            total_loss += outputs['total_loss'].item()
            total_flow_loss += outputs['flow_log_prob'].item()
            total_decision_loss += outputs['decision_loss'].item()
            n_batches += 1

        # Print progress
        avg_loss = total_loss / n_batches
        avg_flow = total_flow_loss / n_batches
        avg_decision = total_decision_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Flow Log Prob: {avg_flow:.4f}")
            print(f"  Decision Loss: {avg_decision:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Validation if provided
            if val_data is not None:
                model.eval()
                with torch.no_grad():
                    val_batch_data = val_data[:batch_size]
                    val_batch_fields = None
                    if val_fields is not None:
                        val_batch_fields = torch.stack([torch.FloatTensor(f) for f in val_fields[:batch_size]]).to(device)

                    val_outputs = model(val_batch_data, vit_fields=val_batch_fields)
                    val_loss = val_outputs['total_loss'].item()
                    print(f"  Validation Loss: {val_loss:.4f}")
                model.train()

    return model


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic data for demonstration
    def generate_synthetic_data(n_samples: int = 1000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate synthetic correlated data and field embeddings for testing."""
        data = []
        fields = []

        for _ in range(n_samples):
            # Random number of blocks (0-20)
            n_blocks = np.random.randint(0, 21)

            if n_blocks == 0:
                data.append(np.array([]).reshape(0, 6))
            else:
                # Generate correlated 6D vectors
                mean = np.random.randn(6) * 0.5
                A = np.random.randn(6, 6)
                cov = A @ A.T * 0.1 + np.eye(6) * 0.1

                blocks = np.random.multivariate_normal(mean, cov, n_blocks)
                data.append(blocks.astype(np.float32))

            # Generate synthetic field embeddings
            n_field_tokens = 196  # 14x14 spatial tokens
            d_model = 768
            field_embedding = np.random.randn(n_field_tokens, d_model).astype(np.float32) * 0.1
            fields.append(field_embedding)

        return data, fields

    # Generate training data
    print("Generating synthetic training data with field embeddings...")
    train_data, train_fields = generate_synthetic_data(1000)
    val_data, val_fields = generate_synthetic_data(100)

    # Initialize model with modern architecture
    print("Initializing model with modern architecture (RMSNorm, bias-free, GroupNorm)...")
    model = HybridTransformerFlow(
        input_dim=6,
        d_model=768,
        n_heads=12,
        n_layers=6,
        d_ff=3072,
        flow_layers=4,
        flow_hidden=256,
        n_bins=16,
        tail_bound=4.0,
        dropout=0.1,
        n_groups=8
    )

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\nStarting training with modern optimization...")
    trained_model = train_model(
        model,
        train_data,
        train_fields=train_fields,
        val_data=val_data,
        val_fields=val_fields,
        epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=2
    )

    # Generate samples
    print("\nGenerating samples conditioned on fields...")
    trained_model.eval()

    test_fields = torch.stack([torch.FloatTensor(f) for f in val_fields[:5]])
    generated_samples = trained_model.generate(
        n_samples=5,
        vit_fields=test_fields,
        temperature=0.8
    )

    for i, sample in enumerate(generated_samples):
        print(f"\nGenerated sample {i+1} (conditioned on field):")
        n_blocks = sample.shape[0] // 6 if sample.numel() > 0 else 0
        print(f"  Number of blocks: {n_blocks}")
        print(f"  Total elements: {sample.shape[0]}")
        if sample.numel() > 0:
            print(f"  First element: {sample[0].cpu().numpy()}")
            print(f"  Mean per dim: {sample.mean(dim=0).cpu().numpy()}")
            print(f"  Std per dim: {sample.std(dim=0).cpu().numpy()}")