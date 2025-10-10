'''
nn.py - Utilities Modules for PyTorch tensors

Originally in ctorch.py
'''


from typing import TYPE_CHECKING, Any, Callable, List, Protocol
import torch
import warnings

from . import functional as local_F

class Module(torch.nn.Module):
    '''
    A base class for all modules in ctorch. Supports device tracking and parameter counting.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._numel = 0
        self._device_tracker = torch.nn.Parameter(torch.tensor(0.0, device='cpu'))

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
        if self._numel == 0:
            self._numel = sum(
                p.numel()
                for name, p in self.named_parameters(recurse=True)
                if '_device_tracker' not in name
            )
        return self._numel

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

class DNN(Module):
    '''
    A Deep Neural Network (DNN) or Multi-Layer Perceptron (MLP) module.

    Args:
        layer_dims (\\*int): The dimensions of each layer in the network, including input and output dimensions.
        layer_type (Type[torch.nn.Module]): The type of layer to use (e.g., Linear).
        flip_gradient (bool): Whether to apply a gradient reversal layer at the beginning.
        batchnorm (bool): Whether to apply batch normalization after each linear layer.
        bias (bool): Whether to include a bias term in the linear layers.
        dropout (float | None): Dropout rate to apply after each layer. If None, no dropout is applied.
        activation (str | Activation | None): Activation function to apply after each layer.
        residual (bool): Whether to add a residual connection from input to output, requiring input and output dimensions to match.

    Shapes:

        * Input shape: (\\*, layer_dims[0])
        * Output shape: (\\*, layer_dims[-1])
    '''
    def __init__(
        self, *layer_dims: int,
        layer_type: Callable[[int, int, bool], torch.nn.Module] = torch.nn.Linear,
        flip_gradient: bool = False,
        batchnorm: bool = False,
        bias: bool = True,
        dropout: float | None = None,
        activation: str | Activation | None = 'relu',
        residual: bool = False
    ):
        super(DNN, self).__init__()

        # Sanity checks
        if len(layer_dims) < 2:
            raise ValueError('DNN must have at least one layer.')
        if any(dim <= 0 for dim in layer_dims):
            raise ValueError('All layer dimensions must be positive integers.')
        if residual and layer_dims[0] != layer_dims[-1]:
            raise ValueError('Residual connections require input and output dimensions to match.')
        if dropout is not None and (dropout < 0.0 or dropout > 1.0):
            raise ValueError('Dropout must be between 0.0 and 1.0.')
        # Warning conditions
        # 1. batchnorm and bias are both True
        if batchnorm and bias:
            warnings.warn(
                'Redundant bias in linear layers when using batch normalization.',
                RuntimeWarning
            )

        in_dims, out_dims = layer_dims[:-1], layer_dims[1:]
        layers = []

        if flip_gradient:
            self.rev = GradientReversalLayer()
        else:
            self.rev = torch.nn.Identity()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            current_layer = []
            current_layer.append(layer_type(in_dim, out_dim, bias))
            if batchnorm:
                current_layer.append(torch.nn.BatchNorm1d(out_dim))
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
            y = y + x
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


class MultivariateNormalClass(Protocol):
    '''
    Internal protocol for ``GaussianLinear`` module. Any valid ``gaussian_type`` parameter should follow the following protocol.

    .. code-block:: python

        def __call__(
            self, loc: torch.Tensor, covariance_matrix: torch.Tensor | None = None,
            precision_matrix: torch.Tensor | None = None,
            scale_tril: torch.Tensor | None = None,
            validate_args: Any = None
        ) -> torch.distributions.Distribution:
            ...
    '''
    def __call__(
        self, loc: torch.Tensor, covariance_matrix: torch.Tensor | None = None,
        precision_matrix: torch.Tensor | None = None,
        scale_tril: torch.Tensor | None = None,
        validate_args: Any = None
    ) -> torch.distributions.Distribution:
        ...


class GaussianLinear(Module):
    '''
    Implements a linear layer that returns a multivariate Gaussian distribution

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
        mu_scale (float | None): If not None, the mean value is scaled by a tanh function with the given scale.
        cov_scale (float): The boundary value for covariance matrix elements.
        eps (float): Minimum value added to the diagonal of the covariance matrix.
        non_neg_func (str | Callable): Non-negative mapping of the diagonal elements of the covariance matrix.
        gaussian_type (MultivariateNormalClass): The constructor of target distribution.

    Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_features)
    '''
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        mu_scale: float | None = None, Sigma_scale: float | None = None,
        eps: float = 1e-4,
        non_neg_func: str | Callable[[torch.Tensor], torch.Tensor] = 'softplus',
        gaussian_type: MultivariateNormalClass =
            torch.distributions.MultivariateNormal,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gaussian_type = gaussian_type
        self.mu_scale = mu_scale
        self.linear_mean = torch.nn.Linear(in_features, out_features, bias)
        self.linear_cov = CholeskyTrilLinear(
            in_features, out_features, bias, eps, Sigma_scale, non_neg_func
        )

        torch.nn.init.xavier_normal_(self.linear_mean.weight, 0.1)
        torch.nn.init.zeros_(self.linear_mean.bias)
        torch.nn.init.xavier_normal_(self.linear_cov.diag_layer.weight, 0.1)
        torch.nn.init.constant_(self.linear_cov.diag_layer.bias, -1)
        if self.linear_cov.lower_layer is not None:
            torch.nn.init.xavier_normal_(self.linear_cov.lower_layer.weight, 0.1)
            torch.nn.init.zeros_(self.linear_cov.lower_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        '''
        Forward pass for the positive definite linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Returns:
            torch.distributions.Distribution: The target distribution.
        '''
        mean = self.mean(x)
        cov_tril = self.linear_cov(x)
        return self.gaussian_type(mean, scale_tril=cov_tril)

    def cov(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Return the covariance matrix of the target distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_features, out_features)
        '''
        return self.linear_cov.pd(x)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Return the mean value of the target distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, in_features).

        Shapes:

        * Input shape: (\\*, in_features)
        * Output shape: (\\*, out_features)
        '''
        if self.mu_scale is None:
            return self.linear_mean(x)
        scale = self.mu_scale
        return torch.tanh(self.linear_mean(x) / scale) * scale


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


class DeEmbedding(Module):
    def __init__(self, embedding: torch.nn.Embedding):
        '''
        De-embedding layer that maps from the embedding space back to the multinomial distribution over the vocabulary.

        Args:
            embedding (torch.nn.Embedding): The embedding layer to de-embed from.

        Shapes:

            * Input shape: (\\*, embedding.embedding_dim)
            * Output shape: (\\*, embedding.num_embeddings)
        '''
        super().__init__()
        self.embedding = embedding
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the de-embedding layer.

        Args:
            x (torch.Tensor): Tensor of shape (\\*, D), where D is the embedding dimension.

        Returns:
            torch.Tensor: Tensor of shape (\\*, num_embeddings), where num_embeddings is the size of the embedding.
        '''
        if x.dim() < 2:
            raise ValueError('Input tensor must have at least 2 dimensions.')
        if x.shape[-1] != self.embedding.embedding_dim:
            raise ValueError(f'Input tensor must have last dimension of size {self.embedding.embedding_dim}, but got {x.shape[-1]}.')

        ret = torch.einsum('...d,ed->...e', x, self.embedding.weight)
        return self.softmax(ret)

class FeatureEmbedding(Module):
    '''
    An embedding layer for encoding N multiple categorical features.

    Args:
        num_features (List[int]): The number of unique values for each categorical feature.
        embedding_size (List[int] | int): The size of the embedding for each feature. If a single integer is provided, it will be used for all features.
        padding_idx (int | None):
        max_norm (float | None):
        norm_type (float):
        scale_grad_by_freq (bool):
        sparse (bool):

    Shapes:

        * Input shape: (\\*, num_features)
        * Output shape: (\\*, sum(embedding_size))
    '''
    def __init__(
        self, num_features: List[int], embedding_size: List[int] | int,
        padding_idx: int | None = None, max_norm: float | None = None,
        norm_type: float = 2.0, scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super().__init__()
        # Type conversion and sanity checks
        if isinstance(embedding_size, int):
            embedding_size = [embedding_size] * len(num_features)
        if len(num_features) != len(embedding_size):
            raise ValueError('num_features and embedding_size must have the same length.')

        # Normalization should be done separately
        self.num_features = num_features
        num_features_tensor = torch.tensor(num_features, dtype=torch.long).unsqueeze(0)
        if TYPE_CHECKING:
            self.num_features_tensor = num_features_tensor
        self.register_buffer('num_features_tensor', num_features_tensor)
        self.embedding_size = embedding_size

        _total_embeddings = sum(num_features + [0])
        _max_embedding_size = max(embedding_size + [0])
        self.embedding = torch.nn.Embedding(
            num_embeddings=_total_embeddings,
            embedding_dim=_max_embedding_size,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            max_norm=max_norm,
            norm_type=norm_type,
            sparse=sparse
        )

        # If any embedding size is different from the max, we need to slice the output
        self.need_slice = any(_ != _max_embedding_size for _ in embedding_size)

        # Offset is used to calculate the start index of each feature's embedding
        offset = torch.cumsum(torch.tensor([0, *self.num_features[:-1]]), dim=0, dtype=torch.long)
        if TYPE_CHECKING:
            self.offset = offset
        self.register_buffer('offset', offset)

    @property
    def total_embedding_size(self) -> int:
        '''
        Gets the total embedding size, which is the sum of all individual embedding sizes.

        Returns:
            int: Total embedding size.
        '''
        return sum(self.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the embedding layer.

        Args:
            x: Tensor of shape (\\*, num_features)

        Returns:
            torch.Tensor: Tensor of shape (\\*, sum(embedding_size))
        '''
        # Sanity checks
        if x.dim() < 2:
            raise ValueError('Input tensor must have at least 2 dimensions.')
        if x.shape[-1] != len(self.num_features):
            raise ValueError(f'Input tensor must have last dimension of size {len(self.num_features)}, but got {x.shape[-1]}.')
        if x.dtype not in (torch.int, torch.long):
            raise TypeError('Input tensor must be of type torch.int/long.')
        if not torch.all((x >= 0).all(dim=-1) & (x < self.num_features_tensor).all(dim=-1)):
            raise ValueError('Input tensor contains out-of-bound indices.')

        # Flatten the last dim
        *shape, num_features = x.shape
        x = x.view(*shape, -1)
        x = x + self.offset.expand(*shape, -1)  # Add offset to each feature's index

        # Get embeddings
        embeddings = self.embedding(x)

        # Return sliced embeddings if needed
        if self.need_slice:
            embeddings = torch.cat([
                embeddings[..., i:i + size]
                for i, size in zip(self.offset, self.embedding_size)
            ], dim=-1)

        return embeddings.view(*shape, -1)
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