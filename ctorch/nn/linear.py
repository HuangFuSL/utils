from typing import Callable

import torch

from .. import functional as local_F
from .module import Module


class GradientReversalLayer(Module):
    '''
    A layer that reverses the gradient during backpropagation.

    Args:
        alpha (float): The scaling factor for the gradient reversal. Default is 1.0.
    '''

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the gradient reversal layer.

        Args:
            x (torch.Tensor): Input tensor.
        '''
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')
        return local_F.gradient_reversal(x, self.alpha)


class FuncWrapper(Module):
    '''
    A module that wraps an arbitrary function.

    Args:
        func (Callable): The function to wrap. It should take a torch.Tensor as input and return a torch.Tensor as output.
    '''
    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        if not callable(func):
            raise TypeError('func must be callable.')
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the function wrapper.

        Args:
            x (torch.Tensor): Input tensor.
        '''
        return self.func(x)


class Activation(Module):
    '''
    Arbitrary activation function module.

    Args:
        name (str): The name of the activation function.
        *args: Positional arguments for the activation function.
        **kwargs: Keyword arguments for the activation function.
    '''
    def __init__(self, name: str, *args, **kwargs):
        super().__init__()
        activation = {
            'relu': torch.nn.ReLU,
            'sigmoid': torch.nn.Sigmoid,
            'tanh': torch.nn.Tanh,
            'softmax': torch.nn.Softmax,
            'logsoftmax': torch.nn.LogSoftmax,
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

class DNN(Module):
    '''
    A Deep Neural Network (DNN) or Multi-Layer Perceptron (MLP) module.

    Args:
        layer_dims (\\*int): The dimensions of each layer in the network, including input and output dimensions.
        layer_type (Type[torch.nn.Module]): The type of layer to use (e.g., Linear).
        flip_gradient (bool): Whether to apply a gradient reversal layer at the beginning.
        batchnorm (bool): Whether to apply batch normalization after each linear layer.
        batchnorm_class (Type[torch.nn.Module]): The batch normalization class to use, default is BatchNorm1d. Can be set to BatchNorm2d or BatchNorm3d for convolutional layers, or LayerNorm and InstanceNorm for other use cases. For GroupNorms, pass it via a lambda function.
        bias (bool): Whether to include a bias term in the linear layers. When batch normalization is used, bias is often redundant and can be set to False.
        dropout (float | None): Dropout rate to apply after each layer. If None, no dropout is applied.
        activation (str | Activation | None): Activation function to apply after each layer.
        residual (bool): Whether to add a residual connection from input to output, requiring input and output dimensions to match.
        bare_last_layer (bool): Whether to remove activation and dropout after the output of the last layer, defaults to False.

    Shapes:

        * Input shape: (\\*, layer_dims[0])
        * Output shape: (\\*, layer_dims[-1])
    '''
    def __init__(
        self, *layer_dims: int,
        layer_type: Callable[[int, int, bool], torch.nn.Module] = torch.nn.Linear,
        flip_gradient: bool = False,
        batchnorm: bool = False,
        batchnorm_class: Callable[[int], torch.nn.Module] = torch.nn.BatchNorm1d,
        bias: bool = True,
        dropout: float | None = None,
        activation: str | Activation | None = 'relu',
        residual: bool = False,
        bare_last_layer: bool = False
    ):
        super(DNN, self).__init__()

        # Sanity checks
        if len(layer_dims) < 2:
            raise ValueError('DNN must have at least one layer.')
        if any(dim <= 0 for dim in layer_dims):
            raise ValueError('All layer dimensions must be positive integers.')
        if dropout is not None and (dropout < 0.0 or dropout > 1.0):
            raise ValueError('Dropout must be between 0.0 and 1.0.')

        in_dims, out_dims = layer_dims[:-1], layer_dims[1:]
        num_layers = len(in_dims)
        layers = []

        if flip_gradient:
            self.rev = GradientReversalLayer()
        else:
            self.rev = torch.nn.Identity()
        if residual and layer_dims[0] != layer_dims[-1]:
            self.res = torch.nn.Linear(layer_dims[0], layer_dims[-1], bias=False)
        else:
            self.res = torch.nn.Identity()

        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            current_layer = []
            current_layer.append(layer_type(in_dim, out_dim, bias))
            if batchnorm:
                current_layer.append(batchnorm_class(out_dim))
            if not bare_last_layer or i < num_layers - 1:
                if isinstance(activation, str):
                    current_layer.append(Activation(activation))
                elif isinstance(activation, Activation):
                    current_layer.append(activation)
                if dropout is not None:
                    current_layer.append(torch.nn.Dropout(dropout))

            layers.append(torch.nn.Sequential(*current_layer))

        self.residual = residual
        self.batchnorm = batchnorm
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward pass for the DNN module.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, layer_dims[0]).

        Returns:
            torch.Tensor: Output tensor of shape (\\*, layer_dims[-1]).
        '''
        x = self.rev(x)
        y = self.seq(x)

        if self.residual:
            y = y + self.res(x)
        return y

class MonotonicLinear(Module):
    '''
    Implements a monotonic linear layer. The monotonicity is enforced by applying a non-negative activation function to the weights.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        non_neg_func (str | Callable): Element-wise non-negative activation function to use. Should be one of:

            * ``relu``: :math:`f(x) = \\max(0, x)`
            * ``softplus``: :math:`f(x) = \\log(1 + \\exp(x))`
            * ``sigmoid``: :math:`f(x) = \\frac{1}{1 + \\exp(-x)}`
            * ``elu``: :math:`f(x) = ELU(x) + 1`
            * ``abs``: :math:`f(x) = |x|`
            * ``square``: :math:`f(x) = x^2`
            * ``exp``: :math:`f(x) = e^x`

    To keep the monotonicity, non-monotonic activations including ``softmax``, ``GELU``, ``SiLU`` and ``Mish`` should not be used. Normalization techniques such as layer normalization or batch normalization should also be avoided.

    Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_features)
    '''
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        non_neg_func: str | Callable[[torch.Tensor], torch.Tensor] = 'softplus'
    ):
        super().__init__()
        if not isinstance(non_neg_func, str):
            self.non_neg_act = non_neg_func
        elif non_neg_func in { 'relu', 'softplus', 'sigmoid' }:
            self.non_neg_act = getattr(torch.nn.functional, non_neg_func)
        elif non_neg_func == 'elu':
            self.non_neg_act = lambda x: torch.nn.functional.elu(x) + 1
        elif non_neg_func in { 'abs', 'square', 'exp' }:
            self.non_neg_act = getattr(torch, non_neg_func)
        else:
            raise ValueError(f'A non-negative activation should be used to clip weights, got {non_neg_func}.')

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=False)

        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the monotonic linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (\\*, out_features).
        '''
        return torch.einsum('...i,ji->...j', x, self.non_neg_act(self.weight)) + self.bias

class IndependentNoisyLinear(Module):
    '''
    Implements a noisy linear layer according to https://arxiv.org/abs/1706.10295

    The layer works the same way as a standard linear layer, but with added noise during training.

    .. math::
        \\begin{aligned}
            w &= w_\\mu + w_\\sigma \\odot \\varepsilon \\\\
            b &= b_\\mu + b_\\sigma \\odot \\varepsilon
        \\end{aligned}

    The parameters are initialized as:

    .. math::
        \\begin{aligned}
            w_\\mu, b_\\mu &\\sim \\mathcal U(-1 / \\sqrt{d_{\\text{in}}}), 1 / \\sqrt{d_{\\text{int}}}) \\\\
            w_\\sigma, b_\\sigma &= \\sigma_{\\text{init}}
        \\end{aligned}

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        init_sigma: (float): The initial sigma coefficient :math:`\\sigma_{\\text{init}}`, default is 0.017.

    Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_features)
    '''
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        init_sigma: float = 0.017
    ):
        super().__init__()
        sigma = init_sigma
        bound = 1 / in_features ** 0.5

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.full((out_features, in_features), sigma))
        torch.nn.init.uniform_(self.weight_mu, -bound, bound)

        if self.bias:
            self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
            self.bias_sigma = torch.nn.Parameter(torch.full((out_features,), sigma))
            torch.nn.init.uniform_(self.bias_mu, -bound, bound)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the noisy linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (\\*, out_features).
        '''
        if self.training:
            weight_z = torch.randn_like(self.weight_sigma)
            weight = self.weight_mu + weight_z * self.weight_sigma
            if self.bias:
                bias_z = torch.randn_like(self.bias_sigma)
                bias = self.bias_mu + self.bias_sigma * bias_z
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class FactorizedNoisyLinear(Module):
    '''
    Implements a noisy linear layer according to https://arxiv.org/abs/1706.10295

    The layer works the same way as a standard linear layer, but with added noise during training.

    .. math::
        \\begin{aligned}
            z_\\text{in} &\\sim \\mathcal N(0, I_{d_{\\text{in}}}) \\\\
            z_\\text{out} &\\sim \\mathcal N(0, I_{d_{\\text{out}}}) \\\\
            f(x) &= \\text{sign}(x) \\odot \\sqrt{|x|} \\\\
            w &= w_\\mu + w_\\sigma \\odot (f(z_\\text{in}) f(z_\\text{out})^\\top) \\\\
            b &= b_\\mu + b_\\sigma \\odot f(z_\\text{out})
        \\end{aligned}

    The parameters are initialized as:

    .. math::
        \\begin{aligned}
            w_\\mu, b_\\mu &\\sim \\mathcal U(-1 / \\sqrt{d_{\\text{in}}}), 1 / \\sqrt{d_{\\text{int}}}) \\\\
            w_\\sigma, b_\\sigma &= \\sigma_{\\text{init}} / \\sqrt{d_{\\text{in}}}
        \\end{aligned}

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        init_sigma: (float): The initial sigma coefficient :math:`\\sigma_{\\text{init}}`, default is 0.5.

    Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_features)
    '''
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        init_sigma: float = 0.5
    ):
        super().__init__()
        sigma = init_sigma * in_features ** -0.5
        bound = 1 / in_features ** 0.5

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.full((out_features, in_features), sigma))
        torch.nn.init.uniform_(self.weight_mu, -bound, bound)

        if self.bias:
            self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
            self.bias_sigma = torch.nn.Parameter(torch.full((out_features,), sigma))
            torch.nn.init.zeros_(self.bias_mu)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

    @staticmethod
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the noisy linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (\\*, out_features).
        '''
        if self.training:
            out_z = self.f(self.weight_mu.new_empty(self.out_features).normal_())
            in_z = self.f(out_z.new_empty(self.in_features).normal_())
            weight_z = torch.outer(out_z, in_z)
            weight = self.weight_mu + weight_z * self.weight_sigma
            if self.bias:
                bias = self.bias_mu + self.bias_sigma * out_z
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

class CholeskyTrilLinear(Module):
    '''
    Implements a linear layer that returns a lower-triangular matrix that is positive definite.

    Args:
        in_features (int): Number of input features.
        out_dim (int): The output matrix dimension.
        bias (bool): Whether to include a bias term.
        eps (float): The small value added to the main diagonal of the matrix
        scale (float): The maximum scale of the matrix elements.
        non_neg_func (str | Callable): Element-wise non-negative activation function on the diagonal elements.

            * ``softplus``: :math:`f(x) = \\log(1 + \\exp(x))`
            * ``elu``: :math:`f(x) = ELU(x) + 1`
            * ``sigmoid``: :math:`f(x) = 1 / (1 + e^{-x})`
            * ``square``: :math:`f(x) = x^2`
            * ``exp``: :math:`f(x) = e^x`

    Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_dim, out_dim)
    '''
    def __init__(
        self, in_features: int, out_dim: int,
        bias: bool = True, eps: float = 1e-4, scale: float | None = None,
        non_neg_func: str | Callable[[torch.Tensor], torch.Tensor] = 'softplus'
    ):
        super().__init__()

        self.in_features = in_features
        self.out_lower = out_dim * (out_dim - 1) // 2
        self.out_dim = out_dim
        self.scale = scale

        if not isinstance(non_neg_func, str):
            self.non_neg_act = non_neg_func
        elif non_neg_func in {'softplus', 'sigmoid'}:
            self.non_neg_act = getattr(torch.nn.functional, non_neg_func)
        elif non_neg_func == 'elu':
            self.non_neg_act = lambda x: torch.nn.functional.elu(x) + 1
        elif non_neg_func in {'square', 'exp'}:
            self.non_neg_act = getattr(torch, non_neg_func)
        else:
            raise ValueError(
                f'A non-negative activation should be used to clip diagonals, got {non_neg_func}.'
            )
        self.diag_layer = torch.nn.Linear(in_features, self.out_dim, bias)
        if out_dim > 1:
            self.lower_layer = torch.nn.Linear(in_features, self.out_lower, bias)
        else:
            self.lower_layer = None

        self.indices: torch.Tensor
        self.register_buffer('indices', torch.tril_indices(out_dim, out_dim, -1))
        self.eps: torch.Tensor
        self.register_buffer('eps', torch.tensor(eps))

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Return the main diagonal of the resulting matrix.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_dim)
        '''
        diag = self.diag_layer(x)
        ret = self.non_neg_act(diag) + self.eps
        if self.scale is not None:
            ret = torch.tanh(ret / self.scale) * self.scale
        return ret

    def pd(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Return the positive definite :math:`LL^\\top`.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_dim, out_dim)
        '''
        L = self(x)
        return L @ L.transpose(-1, -2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the positive definite linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Returns:
            torch.Tensor: Tril tensor of shape (\\*, out_dim, out_dim).
        '''
        *B, _ = x.shape
        out = torch.diag_embed(self.diag(x))
        if self.lower_layer is not None:
            ll = self.lower_layer(x)
            if self.scale is not None:
                ll = torch.tanh(ll / self.scale) * self.scale
            out[..., self.indices[0], self.indices[1]] = ll
        return out
