'''
nn.py - Utilities Modules for PyTorch tensors

Originally in ctorch.py
'''


import torch
import warnings

from . import functional as local_F

class Module(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._numel = 0
        self._device_tracker = torch.nn.Parameter(torch.tensor(0.0, device='cpu'))

    @property
    def device(self):
        '''
        Returns the device of the module
        '''
        return self._device_tracker.device

    @property
    def num_parameters(self):
        '''
        Returns the number of parameters in the module
        '''
        if self._numel == 0:
            self._numel = sum(
                p.numel()
                for name, p in self.named_parameters(recurse=True)
                if '_device_tracker' not in name
            )
        return self._numel

class Activation(Module):
    ''' Arbitrary activation function module. '''
    def __init__(self, name: str, *args, **kwargs):
        super().__init__()
        activation = {
            'relu': torch.nn.ReLU,
            'sigmoid': torch.nn.Sigmoid,
            'tanh': torch.nn.Tanh,
            'softmax': torch.nn.Softmax,
            'softplus': torch.nn.Softplus,
            'leaky_relu': torch.nn.LeakyReLU,
            'leakyrelu': torch.nn.LeakyReLU,
            'elu': torch.nn.ELU,
            'selu': torch.nn.SELU,
            'gelu': torch.nn.GELU,
            'swish': torch.nn.SiLU,
            'mish': torch.nn.Mish
        }.get(name.lower(), None)
        if activation is None:
            raise ValueError(f'Unknown activation function: {name}')
        self.activation = activation(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the activation function.
        '''
        return self.activation(x)


class GradientReversalLayer(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the gradient reversal layer.
        '''
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')
        return local_F.gradient_reversal(x, self.alpha)

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def get_attention_map(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None, is_causal: bool = False
    ):
        '''
        Get the attention map from the encoder layer.
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