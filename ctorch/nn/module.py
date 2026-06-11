import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Mapping, Sequence, Tuple

import torch

PackedSequence = torch.nn.utils.rnn.PackedSequence

TENSOR_TYPE = (torch.Tensor, PackedSequence)
SEQUENCE_TYPE = (list, tuple)
T_NAME = lambda x: type(x).__name__

def _shape_of(obj: Any) -> Any:
    '''
    Helper function to get the shape of a tensor or a PackedSequence or a nested structure of them. For PackedSequence, the shape is inferred as if it were padded in batch-first format.

    Args:
        obj (Any): The object to get the shape of.
    '''
    if isinstance(obj, torch.Size):
        return None # torch.Size should not appear here
    elif isinstance(obj, torch.Tensor):
        return obj.shape
    elif isinstance(obj, PackedSequence):
        batch_sizes = obj.batch_sizes

        if batch_sizes.numel() == 0:
            return torch.Size([0, 0, *obj.data.shape[1:]])

        max_time = int(batch_sizes.numel())
        batch_size = int(batch_sizes[0].item())
        feature_shape = obj.data.shape[1:]

        return torch.Size([batch_size, max_time, *feature_shape])
    elif isinstance(obj, Mapping):
        return {k: _shape_of(v) for k, v in obj.items()}
    elif isinstance(obj, SEQUENCE_TYPE):
        return [_shape_of(item) for item in obj]
    else:
        return None

def check_shape(obj, shape) -> Generator[Tuple[str, str], None, None]:
    if shape is None:
        return
    elif isinstance(shape, torch.Size):
        obj_shape = _shape_of(obj)
        if isinstance(obj, TENSOR_TYPE):
            if obj_shape != shape:
                yield '', f'Shape mismatch. Expected {shape}, got {obj_shape}'
        else:
            yield '', f'Type mismatch. Expected a tensor, got {T_NAME(obj)}'
    elif isinstance(shape, Mapping):
        if not isinstance(obj, Mapping):
            yield '', f'Type mismatch. Expected a mapping, got {T_NAME(obj)}'
        elif shape.keys() == obj.keys():
            for k in shape:
                result = check_shape(obj[k], shape[k])
                for level, msg in result:
                    yield f'[{k!r}]{level}', msg
        else:
            yield '', f'Keys do not match. Expected {list(shape.keys())}, got {list(obj.keys())}'
    elif isinstance(shape, SEQUENCE_TYPE):
        if isinstance(obj, TENSOR_TYPE) or not isinstance(obj, SEQUENCE_TYPE):
            yield '', f'Type mismatch. Expected a sequence, got {T_NAME(obj)}'
            return
        elif len(shape) == len(obj):
            for i, (o, s) in enumerate(zip(obj, shape)):
                for level, msg in check_shape(o, s):
                    yield f'[{i}]{level}', msg
        else:
            yield '', f'Length mismatch. Expected length {len(shape)}, got {len(obj)}'
    else:
        yield '', f'Type error. Unexpected type for shape check: {T_NAME(shape)}'

class Module(torch.nn.Module):
    '''
    A base class for all modules in ctorch. Supports device tracking and parameter counting.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._numel = -1
        self.register_buffer('_device_tracker', torch.tensor(0))
        self._debug = False

        if TYPE_CHECKING:
            self._device_tracker: torch.Tensor

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for method_name in [
            'output_shape', 'guard_input_shape', 'guard_output_shape'
        ]:
            method = cls.__dict__.get(method_name, None)
            if method is not None:
                setattr(cls, method_name, torch.jit.unused(method))

    @torch.jit.unused
    def output_shape(
        self, *args: torch.Size | None, **kwargs: torch.Size | None
    ) -> torch.Size | Sequence[torch.Size] | Dict[Any, torch.Size] | None:
        '''
        Get the output shape of the ``module.forward()`` given input shapes. Should be implemented by subclasses if output shape checking is desired. For `PackedSequence` inputs, the shapes will be assumed to be padded in batch-first format.

        Args:
            *args (torch.Size | None): Shapes of the input tensors. None if corresponding input is not a tensor.
            **kwargs (torch.Size | None): Shapes of any additional input tensors. None if corresponding input is not a tensor.

        Returns:
            torch.Size | Sequence[torch.Size] | Dict[Any, torch.Size] | None: The shape of the output tensor.
        '''
        return None # None if not perform shape checking


    @torch.jit.unused
    def guard_input_shape(self, *args: Any, **kwargs: Any):
        '''
        A hook to check the input shapes of the ``module.forward()``. Should be implemented by subclasses if input shape checking is desired.

        Args:
            *args (torch.Tensor): Input tensors.
            **kwargs (torch.Tensor): Any additional input tensors.
        '''
        return None # No check by default


    @torch.jit.unused
    def guard_output_shape(
        self, output_obj: Any, *input_args: Any, **input_kwargs: Any
    ):
        '''
        A hook to check the output shape of the ``module.forward()``.

        Args:
            output_obj (torch.Tensor | Sequence[torch.Tensor] | Dict[Any, torch.Tensor]): The output object from the forward pass.
            *input_args (torch.Tensor): Input tensors.
            **input_kwargs (torch.Tensor): Any additional input tensors.
        '''
        output_shape = self.output_shape(
            *[_shape_of(arg) for arg in input_args],
            **{k: _shape_of(v) for k, v in input_kwargs.items()}
        )

        err_msg = '\n'.join([
            f'- output{result}: {msg}'
            for result, msg in check_shape(output_obj, output_shape)
        ])
        if err_msg:
            raise ValueError(f'Output shape check failed on output of {self.__class__.__name__}:\n{err_msg}')

    @torch.jit.unused
    def summary(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        '''
        Get a summary of the module, including the number of parameters, and the output shape given input shapes.

        Args:
            *args (torch.Tensor): Input tensors.
            **kwargs (torch.Tensor): Any additional input tensors.

        Returns:
            List[Dict[str, Any]]: A list of records for each module in the hierarchy, in [{
                name, type, num_parameters, input_args_shape, input_kwargs_shape, output_shape
            }]
        '''
        records = []
        def pre_hook(module, input_args, input_kwargs):
            record = {
                'id': len(records),
                'type': module.__class__.__name__,
                'num_parameters': sum(
                    p.numel()
                    for p in module.parameters(recurse=False)
                ),
                'num_trainable_parameters': sum(
                    p.numel()
                    for p in module.parameters(recurse=False) if p.requires_grad
                ),
                'input_args_shape': [_shape_of(arg) for arg in input_args],
                'input_kwargs_shape': {k: _shape_of(v) for k, v in input_kwargs.items()},
            }

            # Special handling for MultiheadAttention to include out_proj parameters in the count
            if isinstance(module, torch.nn.MultiheadAttention):
                out_proj = getattr(module, 'out_proj', None)
                if out_proj is not None:
                    record['num_parameters'] += sum(
                        p.numel() for p in out_proj.parameters()
                    )
                    record['num_trainable_parameters'] += sum(
                        p.numel()
                        for p in out_proj.parameters() if p.requires_grad
                    )

            records.append(record)
            if hasattr(module, '_stack'):
                module._stack.append(records[-1]['id']) # type: ignore
            else:
                module._stack = [records[-1]['id']]  # type: ignore
            if hasattr(module, '_summary'):
                module._summary[records[-1]['id']] = records[-1] # type: ignore
            else:
                module._summary = {records[-1]['id']: records[-1]}  # type: ignore
        def post_hook(module, input_args, input_kwargs, output_obj):
            if hasattr(module, '_stack'):
                id_ = module._stack.pop()
                record = module._summary.get(id_)
                if record is not None:
                    record['output_shape'] = _shape_of(output_obj)
        handles = []
        for module in self.modules():
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(post_hook, with_kwargs=True))
        try:
            self(*args, **kwargs)
        except Exception as e:
            warnings.warn(f'Error occurred during forward pass: {e}', RuntimeWarning, stacklevel=2)
        for name, module in self.named_modules():
            if not hasattr(module, '_summary'):
                continue
            for k, v in module._summary.items(): # type: ignore
                records[k]['name'] = name
            del module._summary
            if hasattr(module, '_stack'):
                del module._stack
        for handle in handles:
            handle.remove()
        return sorted(records, key=lambda r: r['id'])

    @torch.jit.unused
    def debug(self):
        '''
        Enable debug mode (only for custom ``ctorch.nn.Modules``).
        '''
        if self._debug:
            return
        self._debug = True
        for module in self.modules():
            if isinstance(module, Module):
                module.debug()
        # Attach shape check hook
        def pre_hook(module, input_args, input_kwargs):
            module.guard_input_shape(*input_args, **input_kwargs)
        self.register_forward_pre_hook(pre_hook, with_kwargs=True)
        def post_hook(module, input_args, input_kwargs, output_obj):
            module.guard_output_shape(output_obj, *input_args, **input_kwargs)
        self.register_forward_hook(post_hook, with_kwargs=True)

    @torch.jit.unused
    def _check_non_negative(self, *args: torch.Tensor):
        '''
        Check that the given tensors contain only non-negative values.

        Args:
            *args (torch.Tensor): Tensors to check.
        '''
        if not self._debug:
            return
        for i, tensor in enumerate(args, start=1):
            self._check_tensor(
                tensor, check_nan=True, check_inf=True, check_extreme_value=None
            )
            if (tensor < 0).any().item():
                raise ValueError(f'Negative values detected in tensor #{i}.')

    @torch.jit.unused
    def _check_zero_to_one(self, *args: torch.Tensor):
        '''
        Check that the given tensors contain only values in the range [0, 1].

        Args:
            *args (torch.Tensor): Tensors to check.
        '''
        if not self._debug:
            return
        for i, tensor in enumerate(args, start=1):
            self._check_tensor(
                tensor, check_nan=True, check_inf=True, check_extreme_value=None
            )
            self._check_non_negative(tensor)
            if (tensor > 1).any().item():
                raise ValueError(f'Values outside [0, 1] detected in tensor #{i}.')

    @torch.jit.unused
    def _check_tensor(
        self, *args: torch.Tensor,
        check_nan: bool = True, check_inf: bool = True,
        check_extreme_value: int | float | None = 50,
        strict_extreme_value: bool = False
    ):
        '''
        Check for NaN, Inf, or large values in the given tensors.

        Args:
            *args (torch.Tensor): Tensors to check.
            check_nan (bool): Whether to check for NaN values.
            check_inf (bool): Whether to check for Inf values.
            check_extreme_value (int | float | None): Threshold for large values. If None, no check is performed.
            strict_extreme_value (bool): If True, raise an error on large values; otherwise, issue a warning.
        '''
        if not self._debug:
            return
        for i, tensor in enumerate(args, start=1):
            if check_nan and torch.isnan(tensor).any():
                raise ValueError(f'NaN values detected in tensor #{i}.')
            if check_inf and torch.isinf(tensor).any():
                raise ValueError(f'Inf values detected in tensor #{i}.')
            if check_extreme_value is not None and \
                torch.any(tensor.abs() > check_extreme_value):

                max_extreme = tensor.max().item()
                min_extreme = tensor.min().item()

                prompt = f'Extreme values [{min_extreme}, {max_extreme}] detected in tensor #{i}, which may lead to numerical instability.'
                if strict_extreme_value:
                    raise ValueError(prompt)
                else:
                    warnings.warn(prompt, RuntimeWarning, stacklevel=2)

    @torch.jit.unused
    def _check_gradients(
        self,
        check_nan: bool = True, check_inf: bool = True,
        check_extreme_value: int | float | None = 50
    ):
        '''
        Check for NaN or Inf values in the gradients of the module's parameters.

        Args:
            check_nan (bool): Whether to check for NaN values.
            check_inf (bool): Whether to check for Inf values.
            check_extreme_value (int | float | None): Threshold for large values. If None, no check is performed.
        '''
        if not self._debug:
            return
        failed_params = []
        for name, param in self.named_parameters(recurse=True):
            if param.grad is not None:
                try:
                    self._check_tensor(
                        param.grad,
                        check_nan=check_nan,
                        check_inf=check_inf,
                        check_extreme_value=check_extreme_value, strict_extreme_value=True
                    )
                except ValueError as e:
                    failed_params.append((name, str(e)))
        if failed_params:
            raise ValueError(
                'Gradient check failed for the following parameters:\n' +
                '\n'.join([f'- {name}: {error}' for name, error in failed_params])
            )

    @property
    def device(self):
        '''
        Get the device of the module.

        Returns:
            torch.device: The device on which the module's parameters are located.
        '''
        return self._device_tracker.device

    @property
    @torch.jit.unused
    def num_parameters(self):
        '''
        Get the number of parameters in the module.

        Returns:
            int: The total number of parameters in the module.
        '''
        if self._numel == -1:
            self._numel = sum(
                p.numel()
                for _, p in self.named_parameters(recurse=True)
            )
        return self._numel
