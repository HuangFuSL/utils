from typing import TYPE_CHECKING

import torch

from .module import Module

class RotaryTemporalEmbedding(Module):
    '''
    Implements rotary positional embedding proposed in "RoFormer: Enhanced Transformer with Rotary Position Embedding" (https://arxiv.org/abs/2104.09864).

    .. math::
        \\begin{aligned}
            \\boldsymbol R &= \\text{diag}(\\boldsymbol R_1, \\ldots, \\boldsymbol R_{\\lfloor n / 2\\rfloor}) \\\\
            \\boldsymbol R_i &= \\begin{bmatrix}
                \\cos (t\\theta_i) & -\\sin (t\\theta_i) \\\\
                \\sin (t\\theta_i) & \\cos (t\\theta_i)
            \\end{bmatrix} \\\\
            \\theta_i &= \\frac{1}{10000^{2(i - 1)/d_{model}}}
        \\end{aligned}

    Args:
        embedding_dim (int): The dimension of the embedding space. Must be even.
        denom (float): The denominator (10000.0) for the positional encoding.

    Shapes:
        * Input shape: x (\\*, embedding_dim), t (\\*)
        * Output shape: (\\*, embedding_dim)
    '''
    def __init__(self, embedding_dim: int, denom: float = 10000.0):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")

        self.embedding_dim = embedding_dim
        expon = torch.arange(0, embedding_dim, 2).float() / embedding_dim
        scale = torch.exp(-torch.log(torch.tensor(denom)) * expon)
        ones = torch.ones_like(scale)

        if TYPE_CHECKING:
            self.scale = scale
            self.ones = ones

        self.register_buffer('scale', scale)
        self.register_buffer('ones', ones)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the rotary temporal embedding.

        Args:
            t (torch.Tensor): Time record tensor of shape (\\*).
            x (torch.Tensor): Input tensor of shape (\\*, embedding_dim).

        Returns:
            torch.Tensor: Embedding of shape (\\*, embedding_dim).
        '''
        if x.dim() < 2 or x.shape[-1] != self.embedding_dim:
            raise ValueError(f'Input tensor must have last dimension of size {self.embedding_dim}, but got {x.shape[-1]}.')
        t_shape = x.shape[:-1]
        if t.shape != t_shape:
            raise ValueError(f'Time tensor must have shape {t_shape}, but got {t.shape}.')
        # Calculate the rotary embedding using complex arithmetic
        cos = torch.cos(torch.einsum('...,k->...k', t, self.scale)).expand(*t_shape, -1)
        sin = torch.sin(torch.einsum('...,k->...k', t, self.scale)).expand(*t_shape, -1)
        if x.dtype in (torch.float32, torch.float64):
            x_complex = torch.view_as_complex(x.reshape(*t_shape, -1, 2).contiguous())
            x_rotated = x_complex * (cos + 1j * sin)
            return torch.view_as_real(x_rotated).reshape(*x.shape)

        # Fallback to real arithmetic
        x_half = x.reshape(*t_shape, -1, 2)
        interleave = torch.stack([-self.ones, self.ones], dim=-1).to(dtype=x.dtype).expand(*t_shape, -1, 2)
        x_rotated = (
            x_half * cos.unsqueeze(-1) +
            x_half.flip(-1) * sin.unsqueeze(-1) * interleave
        ).reshape(*t_shape, -1)
        return x_rotated

class SinusoidalTemporalEmbedding(Module):
    '''
    Implements sinusoidal positional embedding proposed in "Attention is All You Need".

    .. math::

        PE_{(batch, pos, i)} = \\left\\{\\begin{aligned}
        &\\sin\\left(\\frac{pos}{10000^{2k/d_{model}}}\\right) &\\text{if } i = 2k \\\\
        &\\cos\\left(\\frac{pos}{10000^{2k/d_{model}}}\\right) &\\text{if } i = 2k + 1
        \\end{aligned}\\right.

    Args:
        embedding_dim (int): The dimension of the embedding space.
        denom (float): The denominator (10000.0) for the positional encoding.

    Shapes:

        * Input shape: (\\*, embedding_dim)
        * Output shape: (\\*, embedding_dim)
    '''
    def __init__(self, embedding_dim: int, denom: float = 10000.0):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")

        self.embedding_dim = embedding_dim
        expon = torch.arange(0, embedding_dim, 2).float() / embedding_dim
        self.register_buffer('scale', torch.exp(-torch.log(torch.tensor(denom)) * expon))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the circular temporal embedding.

        Args:
            t (torch.Tensor): Time record tensor of shape (\\*).

        Returns:
            torch.Tensor: Embedding of shape (\\*, embedding_dim).
        '''
        return torch.stack([
            torch.sin(torch.einsum('...,k->...k', t, self.scale)),
            torch.cos(torch.einsum('...,k->...k', t, self.scale)),
        ], dim=-1).view(*t.shape, self.embedding_dim)

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    '''
    A Transformer encoder layer with additional functionality to get attention maps.
    '''
    def get_attention_map(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None, is_causal: bool = False
    ):
        '''
        Get the attention map from the encoder layer.

        Args:
            src (torch.Tensor):
            src_mask (torch.Tensor | None):
            src_key_padding_mask (torch.Tensor | None):
            is_causal (bool):

        Returns:
            torch.Tensor: The attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        '''
        if not isinstance(src, torch.Tensor):
            raise TypeError('src must be a torch.Tensor.')
        if src.dim() < 2:
            raise ValueError('src must have at least 2 dimensions.')

        if self.norm_first:
            # Apply normalization before the attention layer
            src = self.norm1(src)

        # Forward pass to get attention weights
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            is_causal=is_causal
        )
        # Fill nan with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        return attn_weights

class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def get_self_attention_map(
        self, tgt: torch.Tensor, memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ):
        '''
        Get the self-attention map from the decoder layer.

        Args:
            tgt (torch.Tensor):
            tgt_mask (torch.Tensor | None):
            memory_mask (torch.Tensor | None):
            tgt_key_padding_mask (torch.Tensor | None):
            memory_key_padding_mask (torch.Tensor | None):
            tgt_is_causal (bool):
            memory_is_causal (bool):

        Returns:
            torch.Tensor: The attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        '''
        if not isinstance(tgt, torch.Tensor):
            raise TypeError('tgt must be a torch.Tensor.')
        if tgt.dim() < 2:
            raise ValueError('tgt must have at least 2 dimensions.')

        if self.norm_first:
            # Apply normalization before the attention layer
            tgt = self.norm1(tgt)

        # Forward pass to get attention weights
        attn_output, attn_weights = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            is_causal=tgt_is_causal
        )
        # Fill nan with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        return attn_weights

    def get_cross_attention_map(
        self, tgt: torch.Tensor, memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ):
        '''
        Get the cross-attention map from the decoder layer.

        Args:
            tgt (torch.Tensor):
            memory (torch.Tensor):
            tgt_mask (torch.Tensor | None):
            memory_mask (torch.Tensor | None):
            tgt_key_padding_mask (torch.Tensor | None):
            memory_key_padding_mask (torch.Tensor | None):
            tgt_is_causal (bool):
            memory_is_causal (bool):

        Returns:
            torch.Tensor: The attention weights of shape (batch_size, num_heads, tgt_seq_len, memory_seq_len).
        '''
        if not isinstance(tgt, torch.Tensor):
            raise TypeError('tgt must be a torch.Tensor.')
        if tgt.dim() < 2:
            raise ValueError('tgt must have at least 2 dimensions.')

        if self.norm_first:
            # Apply normalization before the attention layer
            tgt = self.norm1(tgt)

        tgt = tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        tgt = self.norm2(tgt)
        # Second pass cross-attention
        attn_output, attn_weights = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            is_causal=memory_is_causal
        )
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        return attn_weights
