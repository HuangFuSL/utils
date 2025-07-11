'''
ctorch.py

Author: HuangFuSL
Date: 2025-06-26

This module provides utility functions for handling tensors in PyTorch.
'''
import dataclasses
import time
from typing import Any, Dict, List, Tuple

import torch


@dataclasses.dataclass
class GpuStat:
    idx: int
    name: str
    avg_util: float
    avg_free_gb: float
    total_gb: float

try:
    import pynvml

    def _sample_once(handles) -> List[Tuple[float, float, float]]:
        stats = []
        for h in handles:
            u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            stats.append((u, mem.free, mem.total))
        return stats

    def _collect_stats(samples: int, interval: float) -> List[GpuStat]:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

        util_hist = [[0.0] * samples for _ in range(n)]
        free_hist = [[0.0] * samples for _ in range(n)]
        total = [0.0] * n

        for s in range(samples):
            time.sleep(interval) if s else None
            snap = _sample_once(handles)
            for i, (u, f, t) in enumerate(snap):
                util_hist[i][s] = u
                free_hist[i][s] = f
                if s == 0:
                    total[i] = t

        gpu_stats = []
        for i, h in enumerate(handles):
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            total_gb = total[i] / 1024 ** 3
            avg_util = sum(util_hist[i]) / samples
            avg_free_gb = (sum(free_hist[i]) / samples) // 1024 ** 3
            gpu_stats.append(GpuStat(i, name, avg_util, avg_free_gb, total_gb))

        pynvml.nvmlShutdown()
        return gpu_stats
except ImportError:
    def _sample_once(handles) -> List[Tuple[float, float, float]]:
        raise ImportError(
            'pynvml is required to collect GPU statistics. '
            'Please install it with `pip install pynvml`.'
        )
    def _collect_stats(samples: int, interval: float) -> List[GpuStat]:
        raise ImportError(
            'pynvml is required to collect GPU statistics. '
            'Please install it with `pip install pynvml`.'
        )


def get_best_device(
    window_sec: float = 3.0,
    interval_sec: float = 0.5
) -> str:
    if torch.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        if pynvml is None:
            raise ImportError(
                'pynvml is required to detect CUDA devices. Please install it with `pip install pynvml`.'
            )
        pynvml.nvmlInit()
        stats = _collect_stats(
            samples=int(window_sec / interval_sec),
            interval=interval_sec
        )
        stats.sort(key=lambda s: (s.avg_util, -s.avg_free_gb, -s.total_gb))
        pynvml.nvmlShutdown()
        return f'cuda:{stats[0].idx}'
    # CPU fallback
    return 'cpu'

class Module(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device_tracker = torch.nn.Parameter(torch.tensor(0.0, device='cpu'))

    @property
    def device(self):
        '''
        Returns the device of the module
        '''
        return self._device_tracker.device


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

def _pad_rtl(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    '''
    Alter the sequence from left-aligned to right-aligned.
    '''
    if sequence.device.type != 'cpu':
        raise ValueError('Currently only CPU tensors are supported.')
    if sequence.dim() < 2:
        raise ValueError('Input sequence must have at least 2 dimensions.')
    B, T = sequence.shape[:2]   # (B, T, ...)
    max_len = sequence.size(1)

    # Build torch.gather index
    arange = torch.arange(max_len)
    shift = max_len - lengths
    gather_idx = arange.unsqueeze(0) - shift.unsqueeze(1)
    sentinel_row = max_len
    gather_idx[gather_idx < 0] = sentinel_row
    gather_idx = gather_idx.long()

    # Add a pivot row to the left
    pad_row = torch.full_like(sequence[:, :1], 0)
    seq_ext = torch.cat([sequence, pad_row], dim=1)

    # Gather the right-aligned sequences
    if sequence.dim() == 2:
        idx = gather_idx
    else:
        idx = gather_idx.unsqueeze(-1).expand(-1, -1, *sequence.shape[2:])

    right = torch.gather(seq_ext, 1, idx)
    return right.contiguous()

def _pad_ltr(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    '''
    Alter the sequence from right-aligned to left-aligned.
    '''
    if sequence.device.type != 'cpu':
        raise ValueError('Currently only CPU tensors are supported.')
    if sequence.dim() < 2:
        raise ValueError('Input sequence must have at least 2 dimensions.')
    B, T = sequence.shape[:2]
    if torch.max(lengths).item() != T:
        raise ValueError(
            f'Lengths {lengths} must match the second dimension of the sequence {T}.'
        )

    # Build index matrix
    arange = torch.arange(T)
    shift = (T - lengths).unsqueeze(1)
    gather_idx = arange + shift

    sentinel = T
    gather_idx[gather_idx >= T] = sentinel
    gather_idx = gather_idx.long()

    # Append pivot row
    pad_row = torch.full_like(sequence[:, :1], 0)
    seq_ext = torch.cat([sequence, pad_row], dim=1)

    # Apply the gather index
    if sequence.dim() == 2:
        left = torch.gather(seq_ext, 1, gather_idx)
    else:
        left = torch.gather(
            seq_ext, 1,
            gather_idx.unsqueeze(-1).expand(*gather_idx.shape, *sequence.shape[2:])
        )

    # Cut off the extra row and restore the shape
    left = left[:, :T]
    return left.contiguous()

def pad_packed_sequence_right(
    sequence: torch.nn.utils.rnn.PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
):
    '''
    Like `torch.nn.utils.rnn.pad_packed_sequence` but right-aligns the sequences.
    '''
    # First, pad the packed sequence to the left
    # Shape: B, T, ... or T, B, ... depending on batch_first
    left, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        sequence,
        batch_first=True,
        padding_value=padding_value,
        total_length=total_length,
    )

    right = _pad_rtl(left, lengths)
    right = right.to(sequence.data.device)
    # Match shape to `batch_first`
    if not batch_first:
        return right.transpose(0, 1).contiguous(), lengths

    return right, lengths

def pack_padded_sequence_right(
    input: torch.Tensor,
    lengths: torch.Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = False
) -> torch.nn.utils.rnn.PackedSequence:
    '''
    Like `torch.nn.utils.rnn.pack_padded_sequence` but accepts right-aligned sequences.
    '''

    if batch_first:
        seq = input
    else:
        seq = input.transpose(0, 1)

    seq = _pad_ltr(seq, lengths)
    if not batch_first:
        seq = seq.transpose(0, 1)

    return torch.nn.utils.rnn.pack_padded_sequence(
        seq,
        lengths.cpu(),
        batch_first=batch_first,
        enforce_sorted=enforce_sorted,
    ).to(input.device)


def unpad_sequence_right(
    input: torch.Tensor,
    lengths: torch.Tensor,
    batch_first: bool = False,
) -> List[torch.Tensor]:
    '''
    Like `torch.nn.utils.rnn.unpad_sequence` but accepts right-aligned sequences.
    '''
    if not batch_first:
        input = input.transpose(0, 1)

    input = _pad_ltr(input, lengths)
    input = input.to(lengths.device)
    if not batch_first:
        input = input.transpose(0, 1)

    return torch.nn.utils.rnn.unpad_sequence(input, lengths.cpu(), batch_first=batch_first)


def get_key_padding_mask_left(lengths: torch.Tensor):
    '''
    Create a key padding mask for sequences based on their lengths, with sequences left-aligned.
    '''
    # lengths should be a 1D long tensor
    if lengths.dim() != 1:
        raise ValueError('Lengths must be a 1D tensor.')
    ls = lengths.to(torch.long).tolist()
    return torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l) for l in ls],
        batch_first=True, padding_value=0, padding_side='right'
    ).to(dtype=torch.bool)

def get_key_padding_mask_right(lengths: torch.Tensor):
    '''
    Create a key padding mask for sequences based on their lengths, with sequences right-aligned.
    '''
    # lengths should be a 1D long tensor
    if lengths.dim() != 1:
        raise ValueError('Lengths must be a 1D tensor.')
    ls = lengths.to(torch.long).tolist()
    return torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l) for l in ls],
        batch_first=True, padding_value=0, padding_side='left'
    ).to(dtype=torch.bool)

def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    '''
    Get the memory size of a tensor in bytes.
    '''
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input must be a torch.Tensor.')
    return tensor.element_size() * tensor.numel()

def get_model_memory_size(model: torch.nn.Module) -> int:
    '''
    Get the total memory size of a model in bytes.
    '''
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Input must be a torch.nn.Module.')
    return sum(get_tensor_memory_size(param) for param in model.parameters())