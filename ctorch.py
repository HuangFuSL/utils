'''
ctorch.py

Author: HuangFuSL
Date: 2025-06-26

This module provides utility functions for handling tensors in PyTorch.
'''
from typing import List

import torch


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