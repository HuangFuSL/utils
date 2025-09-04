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

def masked_select(values_input: PackedOrTensor, mask_input: PackedOrTensor) -> PackedSequence:
    '''
    Perform masked selection of variable length on a PackedSequence or a regular tensor.

    Args:
        values_input (PackedOrTensor): The input values to select from.
        mask_input (PackedOrTensor): The mask indicating which values to select.

    Returns:
        PackedSequence: A PackedSequence containing only the selected values.

    Example:

        .. code-block:: python

            from torch.nn.utils.rnn import pad_packed_sequence
            values_list = [
                [[1., 10.], [2., 20.], [3., 30.]],
                [[4., 40.], [5., 50.], [0.,  0.]],
            ]
            mask_list = [
                [True, False, True],
                [False, True, False],
            ]

            values = torch.tensor(values_list, dtype=torch.float32)
            mask = torch.tensor(mask_list, dtype=torch.bool)

            out = masked_select(values, mask)
            out_padded, out_len = pad_packed_sequence(out, batch_first=True)

            assert out_padded.tolist() == [
                [[1.0, 10.0], [3.0, 30.0]],
                [[5.0, 50.0], [0.0,  0.0]],
            ]
            assert out_len.tolist() == [2, 1]
    '''
    # Sanity checks
    # Pad value sequences and check shape
    if isinstance(values_input, PackedSequence):
        values_padded, values_length = torch.nn.utils.rnn.pad_packed_sequence(
            values_input, batch_first=True
        )
    else:
        values_padded = values_input
        values_length = torch.full((values_input.size(0),), values_input.size(1), dtype=torch.long)
    if values_padded.dim() < 2:
        raise ValueError('Values must be at least 2D tensor.')
    B, L, *H = values_padded.shape

    # Convert, pad mask sequences and check shape
    mask_input = mask_input.to(torch.bool)
    if isinstance(mask_input, PackedSequence):
        mask_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            mask_input, batch_first=True, padding_value=False
        )
    else:
        mask_padded = mask_input
    if mask_padded.dim() != 2:
        raise ValueError('Mask must be a 2D tensor.')
    B_mask, L_mask = mask_padded.shape

    # Batch size and device should match
    if B_mask != B:
        raise ValueError(f'Batch size of values ({B}) and mask ({B_mask}) must match.')
    if mask_padded.device != values_padded.device:
        raise ValueError('Values and mask must be on the same device.')

    # Handle inconsistent sequence length
    if L_mask > L:
        mask_padded = mask_padded[:, :L]
    elif L_mask < L:
        mask_padded = torch.cat([mask_padded, mask_padded.new_zeros((B_mask, L - L_mask))], dim=1)
    mask_padded &= (torch.arange(L) < values_length[:, None]).to(mask_padded.device)

    # Handle empty sequences
    selected_len = mask_padded.long().sum(dim=1)
    if torch.any(selected_len <= 0):
        raise ValueError('Some sequences have zero length after masking.')

    # Allocate output tensors
    out_len = int(selected_len.max().item())
    out_padded = values_padded.new_zeros((B, out_len, *H))

    new_positions = (mask_padded.long().cumsum(dim=1) - 1).clamp_min(0)
    batch_idx, in_padded_idx = torch.nonzero(mask_padded, as_tuple=True)
    out_padded_idx = new_positions[batch_idx, in_padded_idx]
    out_padded[batch_idx, out_padded_idx] = values_padded[batch_idx, in_padded_idx]

    out_packed = torch.nn.utils.rnn.pack_padded_sequence(
        out_padded, lengths=selected_len.cpu(),
        batch_first=True, enforce_sorted=False
    )
    return out_packed

def packed_unary_op(
    func: Callable[[torch.Tensor], torch.Tensor], x: PackedOrTensor
) -> PackedOrTensor:
    '''
    Apply an unary sample-wise function to a PackedSequence or a regular tensor.

    Args:
        func (Callable): An sample-wise function to apply.
        x (PackedSequence | torch.Tensor): The input data, either a PackedSequence or a regular tensor.

    Returns:
        PackedSequence | torch.Tensor: The output after applying the function. If the input is a PackedSequence, the output will also be a PackedSequence, otherwise it will be a regular tensor.
    '''
    if isinstance(x, torch.Tensor):
        return func(x)

    ret = func(x.data)
    if ret.shape[0] != x.data.shape[0]:
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
        PackedSequence | torch.Tensor: The output after applying the operation.
    '''
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return op(a, b)
    if isinstance(a, PackedSequence) and isinstance(b, PackedSequence):
        for fieldname in ['batch_sizes', 'sorted_indices', 'unsorted_indices']:
            if (getattr(a, fieldname) is not None and getattr(b, fieldname) is not None \
                and not torch.equal(getattr(a, fieldname), getattr(b, fieldname))) \
                or (getattr(a, fieldname) is None) ^ (getattr(b, fieldname) is None):
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
        PackedSequence | torch.Tensor: The output after applying the module. If the input is a PackedSequence, the output will also be a PackedSequence, otherwise it will be a regular tensor.
    '''
    return packed_unary_op(module, packed_input)

def packed_concat(
    packed_seq: List[PackedSequence], dim: int = -1
):
    '''
    Concatenate a list of PackedSequence objects along a specified dimension. Notice that the length of the packed sequences must be the same.

    Args:
        packed_seq (List[PackedSequence]): List of PackedSequence objects to concatenate.
        dim (int): Dimension along which to concatenate. Default is -1 (last dimension). The dimension must not be 0 (the packed time dimension). The sequence length dimension (dimension 1 of the padded tensor where batch_size is True) is omitted.

    Returns:
        PackedSequence: A new PackedSequence object containing the concatenated data.
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

    Args:
        sequence (torch.nn.utils.rnn.PackedSequence): The packed sequence to pad.
        batch_first (bool): If True, the output will be of shape (batch_size, seq_len, ...). If False, the output will be of shape (seq_len, batch_size, ...).
        padding_value (float): The value to use for padding.
        total_length (int | None): If specified, the output will be padded to this length. If None, the output will be padded to the maximum length of the sequences in the packed sequence

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The padded tensor and the lengths of the original sequences.
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

    Args:
        input (torch.Tensor): The input tensor, which should be right-aligned.
        lengths (torch.Tensor): A 1D tensor containing the lengths of each sequence.
        batch_first (bool): If True, the input is expected to be of shape (batch_size, seq_len, ...). Otherwise, the input is expected to be of shape (seq_len, batch_size, ...).
        enforce_sorted (bool): If True, the input sequences must be sorted by length in descending order. If False, the input sequences can be in any order.

    Returns:
        torch.nn.utils.rnn.PackedSequence: A packed sequence object containing the right-aligned sequences.
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

    Args:
        input (torch.Tensor): The input tensor, which should be right-aligned.
        lengths (torch.Tensor): A 1D tensor containing the lengths of each sequence.
        batch_first (bool): If True, the input is expected to be of shape (batch_size, seq_len, ...). Otherwise, the input is expected to be of shape (seq_len, batch_size, ...).

    Returns:
        List[torch.Tensor]: A list of tensors, each representing a sequence with padding removed.
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

    Args:
        lengths (torch.Tensor): A 1D tensor containing the lengths of each sequence.

    Returns:
        torch.Tensor: A boolean mask tensor indicating the padding positions.
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

    Args:
        lengths (torch.Tensor): A 1D tensor containing the lengths of each sequence.

    Returns:
        torch.Tensor: A boolean mask tensor indicating the padding positions.
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

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        int: The memory size of the tensor in bytes.
    '''
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input must be a torch.Tensor.')
    return tensor.element_size() * tensor.numel()


def get_model_memory_size(model: torch.nn.Module) -> int:
    '''
    Get the total memory size of a model in bytes.

    Args:
        model (torch.nn.Module): The input model.

    Returns:
        int: The total memory size of the model in bytes.
    '''
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Input must be a torch.nn.Module.')
    return sum(get_tensor_memory_size(param) for param in model.parameters())
