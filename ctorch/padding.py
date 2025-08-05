'''
padding.py - Utilities for handling PackedSequences

Originally in ctorch.py
Author: HuangFuSL
Date: 2025-06-26
'''

from typing import Callable, List, TypeVar

import torch

PackedSequence = torch.nn.utils.rnn.PackedSequence
PackedOrTensor = TypeVar('PackedOrTensor', torch.Tensor, PackedSequence)


def packed_unary_op(
    func: Callable[[torch.Tensor], torch.Tensor], x: PackedOrTensor
) -> PackedOrTensor:
    '''
    Apply an unary element-wise function to a PackedSequence or a regular tensor.

    Args:
        func (Callable): An element-wise function to apply.
        x (PackedSequence | torch.Tensor): The input data, either a PackedSequence or a regular tensor.

    Returns:
        y (PackedSequence | torch.Tensor): The output after applying the function. If the input is a PackedSequence, the output will also be a PackedSequence, otherwise it will be a regular tensor.
    '''
    if isinstance(x, torch.Tensor):
        return func(x)

    ret = func(x.data)
    if ret.shape != x.data.shape:
        raise ValueError(
            f"Function {func.__name__} changed the shape of the data from {x.data.shape} to {ret.shape}."
        )
    y = x._replace(data=ret)

    return y

def packed_binary_op(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: PackedOrTensor,
    b: PackedOrTensor,
) -> PackedOrTensor:
    '''
    Apply a binary element-wise operation to a PackedSequence or a regular tensor.

    Args:
        op (Callable): A binary operation to apply.
        a (PackedSequence | torch.Tensor): The first input data, either a PackedSequence or a regular tensor.
        b (PackedSequence | torch.Tensor): The second input data, either a PackedSequence or a regular tensor.

    Returns:
        out (PackedSequence | torch.Tensor): The output after applying the operation.
    '''
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return op(a, b)
    if isinstance(a, PackedSequence) and isinstance(b, PackedSequence):
        for fieldname in ['batch_sizes', 'sorted_indices', 'unsorted_indices']:
            if not (getattr(a, fieldname) is not None and getattr(b, fieldname) is not None) \
                or (not torch.equal(getattr(a, fieldname), getattr(b, fieldname))):
                    raise ValueError(
                        f"PackedSequences must have identical {fieldname}."
                    )

        if a.data.shape != b.data.shape:
            raise ValueError(
                f"PackedSequences must have identical data shapes: {a.data.shape} vs {b.data.shape}."
            )
        ret = op(a.data, b.data)
        if ret.shape != a.data.shape:
            raise ValueError(
                f"Operation {op.__name__} changed the shape of the data from {a.data.shape} to {ret.shape}."
            )
        return a._replace(data=ret)
    else:
        raise TypeError(
            'Both inputs must be either PackedSequence or torch.Tensor, but got '
            f'{type(a)} and {type(b)}.'
        )


def packed_forward(
    module: torch.nn.Module, packed_input: PackedOrTensor
) -> PackedOrTensor:
    '''
    Forward pass for a module with packed input.

    Args:
        module (torch.nn.Module): The neural network to apply.
        packed_input (PackedSequence | torch.Tensor): The packed input data.

    Returns:
        out (PackedSequence | torch.Tensor): The output after applying the module. If the input is a PackedSequence, the output will also be a PackedSequence, otherwise it will be a regular tensor.
    '''
    return packed_unary_op(module.forward, packed_input)

def packed_concat(
    packed_seq: List[PackedSequence], dim: int = -1
):
    '''
    Concatenate a list of PackedSequence objects along a specified dimension. Notice that the length of the packed sequences must be the same.

    Args:
        packed_seq (List[PackedSequence]): List of PackedSequence objects to concatenate.
        dim (int): Dimension along which to concatenate. Default is -1 (last dimension). The dimension must not be 0 (the packed time dimension). The sequence length dimension (dimension 1 of the padded tensor where batch_size is True) is omitted.

    Returns:
        out (PackedSequence): A new PackedSequence object containing the concatenated data.
    '''
    if not packed_seq:
        raise ValueError('packed_seq must not be empty.')

    # Reference metadata from the first sequence
    ref = packed_seq[0]
    batch_sizes = ref.batch_sizes
    sorted_indices = ref.sorted_indices
    unsorted_indices = ref.unsorted_indices

    # Normalise dim (allow negative values)
    data_dim = ref.data.dim()
    if dim < 0:
        dim += data_dim
    if dim == 0:
        raise ValueError("Concatenation along dim=0 (packed time dimension) is invalid.")
    if not (0 < dim and dim < data_dim):
        raise IndexError(f"dim must be in range [-{data_dim}, {data_dim-1}]")

    # Sanity checks: all meta data must match
    for p in packed_seq[1:]:
        if not torch.equal(p.batch_sizes, batch_sizes):
            raise ValueError("All PackedSequence objects must have identical batch_sizes.")
        if (sorted_indices is None) ^ (p.sorted_indices is None) or \
           (sorted_indices is not None and not torch.equal(p.sorted_indices, sorted_indices)):
            raise ValueError("All packed sequences must share the same sorted_indices.")
        if (unsorted_indices is None) ^ (p.unsorted_indices is None) or \
           (unsorted_indices is not None and not torch.equal(p.unsorted_indices, unsorted_indices)):
            raise ValueError("All packed sequences must share the same unsorted_indices.")

    # Actual concatenation of data tensors
    data_cat = torch.cat([p.data for p in packed_seq], dim=dim)

    # Return a new PackedSequence with the shared metadata
    return PackedSequence(
        data_cat,
        batch_sizes,
        sorted_indices,
        unsorted_indices
    )


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
            gather_idx.unsqueeze(-1).expand(*
                                            gather_idx.shape, *sequence.shape[2:])
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
