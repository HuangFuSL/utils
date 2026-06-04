import abc
from typing import Any, Callable, Protocol

import torch

from .linear import CholeskyTrilLinear
from .module import Module


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

    def guard_input_shape(self, *args, **kwargs):
        x = args[0]
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f'{self.__class__.__name__}: expected input dim {self.in_features}, '
                f'got {x.shape[-1]}'
            )

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

class DDPM(Module, abc.ABC):
    '''
    Implements the Denoising Diffusion Probabilistic Models (DDPM) framework proposed in "Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2006.11239).

    Args:
        n_dim (int): The dimension of the input data.
        n_condition (int): The dimension of the condition vector. Can be 0 for categorical condition.
        num_steps (int): The number of diffusion steps.
        betas (torch.Tensor | None): The noise schedule for the diffusion process. If None, a linear schedule from 1e-4 to 2e-2 will be used.
        tilde_sigma (bool): If True, use :math:`\\tilde\\beta_t` instead of :math:`\\beta_t` for :math:`\\sigma_t`.
    '''
    def __init__(
        self, n_dim: int, n_condition: int,
        num_steps: int = 1000, betas: torch.Tensor | None = None,
        tilde_sigma: bool = False
    ):
        super().__init__()
        if n_dim <= 0:
            raise ValueError('n_dim must be positive.')
        if n_condition < 0:
            raise ValueError('n_condition must be a non-negative integer.')
        if num_steps <= 0:
            raise ValueError('num_steps must be positive.')
        if betas is None:
            betas = torch.linspace(1e-4, 2e-2, num_steps)
        if not torch.all((betas > 0) & (betas < 1)):
            raise ValueError('All beta values must be in the range (0, 1).')
        if betas.shape != (num_steps,):
            raise ValueError(f'betas must have shape ({num_steps},), but got {betas.shape}.')

        self.n_dim = n_dim
        self.n_condition = n_condition
        self.num_steps = num_steps
        self.tilde_sigma = tilde_sigma

        self.beta: torch.Tensor
        self.alpha: torch.Tensor
        self.bar_alpha: torch.Tensor

        self.register_buffer('beta', betas)
        self.register_buffer('alpha', 1 - betas)
        self.register_buffer('bar_alpha', torch.cumprod(self.alpha, dim=0))

    def output_shape(
        self, *args: torch.Size | None, **kwargs: torch.Size | None
    ) -> torch.Size | None:
        return args[0]

    def guard_input_shape(self, *args, **kwargs):
        x = args[0]
        if x.shape[-1] != self.n_dim:
            raise ValueError(
                f'{self.__class__.__name__}: expected input dim {self.n_dim}, '
                f'got {x.shape[-1]}'
            )

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        '''
        Predict the noise added to the input tensor at a given step.

        Args:
            x (torch.Tensor): The input tensor of shape (\\*, D).
            c (torch.Tensor): The condition tensor of shape (\\*, C) or (\\*) if C = 0.
            t (torch.Tensor): The diffusion step tensor of shape (\\*).

        Returns:
            torch.Tensor: The predicted noise tensor of shape (\\*, D).
        '''
        pass

    @torch.no_grad()
    def predict(
        self, c: torch.Tensor, x_T: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Predict the original input tensor from the noisy tensor at step T.

        Args:
            c (torch.Tensor): The condition tensor of shape (\\*, C) or (\\*) if C = 0.
            x_T (torch.Tensor | None): The noised tensor at step T of shape (\\*, D), if provided. If None, it will be initialized as standard Gaussian noise.

        Returns:
            torch.Tensor: The predicted original tensor of shape (\\*, D).
        '''
        if self.n_condition:
            B = c.shape[:-1]
        else:
            B = c.shape

        if x_T is None:
            x_t = torch.randn(*B, self.n_dim, device=c.device)
        else:
            if x_T.shape != (*B, self.n_dim):
                raise ValueError(f'x_T must have shape ({[*B, self.n_dim]}), but got {x_T.shape}.')
            x_t = x_T

        for step in reversed(range(self.num_steps)):
            z = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
            t = torch.full(B, step, device=c.device, dtype=torch.long)

            noise = self(x_t, c, t)
            alpha_t = self.alpha[t].unsqueeze(-1)
            bar_alpha_t = self.bar_alpha[t].unsqueeze(-1)

            if step == 0:
                var = torch.zeros_like(bar_alpha_t)
            else:
                var = self.beta[t].unsqueeze(-1)

            if self.tilde_sigma:
                t_1 = torch.clamp(t - 1, min=0)
                bar_alpha_t_1 = self.bar_alpha[t_1].unsqueeze(-1)
                var *= (1 - bar_alpha_t_1) / (1 - bar_alpha_t)

            x_next = (x_t - noise * (1 - alpha_t) / torch.sqrt(1 - bar_alpha_t)) / torch.sqrt(alpha_t) + var.sqrt() * z
            x_t = x_next
        return x_t

    def predict_noise(
        self, c: torch.Tensor, x: torch.Tensor,
        z: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''
        Predict the noise added to the input tensor.

        Args:
            c (torch.Tensor): The condition tensor of shape (\\*, C) or (\\*) if C = 0.
            x (torch.Tensor): The raw input tensor (without noise) of shape (\\*, D).
            z (torch.Tensor): The noise tensor of shape (\\*, D).
            t (torch.Tensor | None): The diffusion step tensor of shape (\\*). If None, it will be randomly initialized.

        Returns:
            torch.Tensor: The predicted noise tensor of shape (\\*, D).
        '''
        B = x.shape[:-1]
        if (self.n_condition and c.shape[:-1] != B) \
            or (not self.n_condition and c.shape != B):
            raise ValueError(f'Condition tensor shape {c.shape} is not compatible with input tensor shape {x.shape}.')

        if t is None:
            t = torch.randint(0, self.num_steps, B, device=x.device)
        bar_alpha = self.bar_alpha[t].unsqueeze(-1)
        noisy_x = torch.sqrt(bar_alpha) * x + torch.sqrt(1 - bar_alpha) * z
        return self(noisy_x, c, t)

    def loss(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the loss for the diffusion process. Override to implement different loss functions.

        Args:
            c (torch.Tensor): The condition tensor of shape (\\*, C) or (\\*) if C = 0.
            x (torch.Tensor): The input tensor of shape (\\*, D).

        Returns:
            torch.Tensor: The loss tensor of shape (\\*).
        '''
        z = torch.randn_like(x)
        return torch.nn.functional.mse_loss(
            self.predict_noise(c, x, z), z, reduction='none'
        ).mean(dim=-1)
