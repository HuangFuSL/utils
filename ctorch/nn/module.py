import warnings
from typing import TYPE_CHECKING

import torch


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

    def debug(self):
        '''
        Enable debug mode (only for custom ``ctorch.nn.Modules``).
        '''
        self._debug = True
        for module in self.children():
            if isinstance(module, Module):
                module.debug()

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
