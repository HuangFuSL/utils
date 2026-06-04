import warnings
import math
from typing import TYPE_CHECKING, Callable, Literal, Tuple

import torch

from .linear import Activation
from .module import Module


def _to_pair(x) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


class ConvBlock(Module):
    '''
    A convolutional block consisting of a convolutional layer, batch normalization, activation function, dropout, and pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | Tuple[int, int]): Size of the convolutional kernel.
        conv_kwargs (Dict[str, Any] | None): Additional keyword arguments for the convolutional layer. The layer should not be lazy.
        conv_cls (Callable[..., torch.nn.Module]): The convolutional layer class to use (e.g., torch.nn.Conv2d).
        activation (str | Activation | None): Activation function to apply after the convolution. If None, no activation is applied.
        batchnorm (bool): Whether to apply batch normalization after the convolution.
        dropout (float | None): Dropout rate to apply after the activation. If None, no dropout is applied.
        pool_type (Literal['max', 'avg'] | Tuple[Literal['max', 'avg'], int] | torch.nn.Module | None): Type of pooling to apply after dropout. Can be:

            * ``'max'``: Apply max pooling with kernel size 2.
            * ``'avg'``: Apply average pooling with kernel size 2.
            * ``('max', k)``: Apply max pooling with kernel size k.
            * ``('avg', k)``: Apply average pooling with kernel size k.
            * A custom pooling layer instance (e.g., torch.nn.AdaptiveAvgPool2d).
            * If None, no pooling is applied.
    '''
    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int | Tuple[int, int], *,
        conv_cls: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
        activation: str | Activation | None = 'relu',
        batchnorm: bool = False,
        dropout: float | None = None,
        pool_type: Literal['max', 'avg'] |
            Tuple[Literal['max', 'avg'], int] |
            torch.nn.Module | None = None,
        residual: bool = False,
        **conv_kwargs,
    ):
        super().__init__()
        if TYPE_CHECKING:
            self._conv_kernel_size: Tuple[int, int]
            self._conv_stride: Tuple[int, int]
            self._conv_padding: Tuple[int, int]
            self._conv_dilation: Tuple[int, int]
            self._pool_kernel_size: Tuple[int, int] | None
            self._pool_stride: Tuple[int, int] | None
        # Special handling for bias kwargs
        if batchnorm:
            if 'bias' not in conv_kwargs:
                conv_kwargs['bias'] = False
            elif conv_kwargs['bias']:
                warnings.warn(
                    'Redundant bias in convolutional layer when using batch normalization.',
                    RuntimeWarning
                )
        # Check for redundant kwargs
        if {'in_channels', 'out_channels', 'kernel_size'}.intersection(conv_kwargs.keys()):
            raise ValueError('in_channels, out_channels, and kernel_size should not be included in conv_kwargs.')

        conv = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **conv_kwargs
        )
        self._conv_kernel_size = _to_pair(conv.kernel_size)
        self._conv_stride = _to_pair(conv.stride)
        self._conv_padding = _to_pair(conv.padding)
        self._conv_dilation = _to_pair(conv.dilation)
        match activation:
            case str() as name:
                act = Activation(name)
            case act if isinstance(act, torch.nn.Module):
                act = act
            case None:
                act = torch.nn.Identity()

        if batchnorm:
            bn = torch.nn.BatchNorm2d(out_channels)
        else:
            bn = torch.nn.Identity()

        match dropout:
            case float() as dropout if 0.0 <= dropout and dropout <= 1.0:
                dp = torch.nn.Dropout2d(dropout)
            case None:
                dp = torch.nn.Identity()
            case _:
                raise ValueError(f'Invalid dropout value: {dropout}')

        match pool_type:
            case 'max':
                ds = torch.nn.MaxPool2d(2)
            case 'avg':
                ds = torch.nn.AvgPool2d(2)
            case ('max', k) if isinstance(k, int) and k > 0:
                ds = torch.nn.MaxPool2d(k)
            case ('avg', k) if isinstance(k, int) and k > 0:
                ds = torch.nn.AvgPool2d(k)
            case _ if isinstance(pool_type, torch.nn.Module):
                ds = pool_type
            case None:
                ds = torch.nn.Identity()
            case _:
                raise ValueError(f'Invalid pool_type: {pool_type}')

        # Store pooling parameters for output shape computation
        if isinstance(ds, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            self._pool_kernel_size = _to_pair(ds.kernel_size)
            self._pool_stride = _to_pair(ds.stride)
        else:
            self._pool_kernel_size = None
            self._pool_stride = None

        if residual and in_channels != out_channels:
            self.res = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.res = torch.nn.Identity()
        self.residual = residual

        model_seq = [conv, bn, act, dp, ds]
        self.model = torch.nn.Sequential(*model_seq)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (\\*, out_channels, H', W').
        '''
        y = self.model(x)
        if self.residual:
            y = y + self.res(x)
        return y

    def output_shape(
        self, *args: torch.Size | None, **kwargs: torch.Size | None
    ) -> torch.Size | None:
        x_shape = args[0]
        if x_shape is None:
            return None
        *batch, _, h, w = x_shape

        def _conv_out(size: int, K: int, S: int, P: int, D: int) -> int:
            return math.floor((size + 2 * P - D * (K - 1) - 1) / S) + 1

        h = _conv_out(h, self._conv_kernel_size[0], self._conv_stride[0],
                       self._conv_padding[0], self._conv_dilation[0])
        w = _conv_out(w, self._conv_kernel_size[1], self._conv_stride[1],
                       self._conv_padding[1], self._conv_dilation[1])

        pk = self._pool_kernel_size
        if pk is not None:
            assert pk is not None
            ps = self._pool_stride
            assert ps is not None
            h = _conv_out(h, pk[0], ps[0], 0, 1)
            w = _conv_out(w, pk[1], ps[1], 0, 1)

        return torch.Size(tuple(batch) + (self.out_channels, h, w))

    def guard_input_shape(self, *args, **kwargs):
        x = args[0]
        if x.dim() < 3:
            raise ValueError(
                f'{self.__class__.__name__}: expected >=3D input (..., C, H, W), '
                f'got {x.dim()}D'
            )
        if x.shape[-3] != self.in_channels:
            raise ValueError(
                f'{self.__class__.__name__}: expected {self.in_channels} input channels, '
                f'got {x.shape[-3]}'
            )
