import abc
from typing import TYPE_CHECKING, Any, Callable, Protocol, Tuple

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


class ZeroInflatedLogNormal(Module):
    '''
    Implements a zero-inflated log-normal distribution layer. A zero-inflated log-normal distribution is a mixture of a point mass at zero and a log-normal distribution.

    .. math::
        \\begin{aligned}
            P(X = 0) &= \\sigma(-\\text{logit}) \\\\
            P(X > 0) &= 1 - \\sigma(-\\text{logit}) \\\\
            \\log X | X > 0 &\\sim \\mathcal{N}(\\mu, \\sigma^2) \\\\
        \\end{aligned}

    Args:
        zero (float): The minimum value for the output.
        eps (float): A small value to avoid numerical issues.

    Shapes:
        * Input shape: (\\*, 2) or (\\*, 3), where the last dimension represents (logit, mu) or (logit, mu, log_sigma).
        * Output shape: (\\*)
    '''
    def __init__(self, zero: float = 0.0, eps: float = 1e-8):
        super(ZeroInflatedLogNormal, self).__init__()
        self.register_buffer('zero', torch.tensor(zero))
        self.register_buffer('const', 0.5 * torch.log(torch.tensor(2 * torch.pi)))
        self.register_buffer('eps', torch.tensor(eps))
        if TYPE_CHECKING:
            self.zero: torch.Tensor
            self.const: torch.Tensor
            self.eps: torch.Tensor

    def check_input_shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        '''
        Check if the input tensor has sigma dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, 2) or (\\*, 3).

        Returns:
            Tuple[torch.Tensor, ...]: The (logit, mu, log_sigma) tensors.
        '''

        if x.dim() <= 1:
            raise ValueError('Input tensor must have at least 2 dimensions.')
        if x.shape[-1] not in {2, 3}:
            raise ValueError('The last dimension of input tensor must be 2 or 3.')
        if x.shape[-1] == 3:
            logit, mu, log_sigma = x.chunk(3, dim=-1)
        else:
            logit, mu = x.chunk(2, dim=-1)
            log_sigma = torch.zeros_like(mu)
        return tuple(t.squeeze(-1) for t in (logit, mu, log_sigma))

    def forward(self, x: torch.Tensor):
        '''
        Forward pass for the zero-inflated log-normal distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, 2) or (\\*, 3).

        Returns:
            torch.Tensor: Output tensor of shape (\\*).
        '''
        logit, mu, log_sigma = self.check_input_shape(x)

        prob = torch.sigmoid(logit)
        return (
            prob * torch.exp(mu + 0.5 * torch.exp(log_sigma * 2))
            + self.zero
        )

    def loss(self, x, target):
        '''
        Compute the negative log-likelihood loss for the zero-inflated log-normal distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (\\*, 2) or (\\*, 3).
            target (torch.Tensor): Target tensor of shape (\\*).
        Returns:
            torch.Tensor: Loss tensor of shape (\\*).
        '''
        logit, mu, log_sigma = self.check_input_shape(x)
        target = (target - self.zero).clamp(min=0.0)

        nonzero_mask = (target > self.eps).float()
        inflation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, nonzero_mask, reduction='none'
        )
        lognormal_loss = 0.5 * ((mu - torch.log(target + self.eps)) ** 2) * torch.exp(-2 * log_sigma)
        jacobian_loss = torch.log(target + self.eps) + log_sigma + self.const
        return (
            inflation_loss +
            lognormal_loss * nonzero_mask +
            jacobian_loss * nonzero_mask
        )

class NegativeBinomial(Module):
    '''
    Implements the negative log-likelihood loss for the Negative Binomial distribution.

    Let :math:`g_\\mu` and :math:`g_\\alpha` be the outputs of the network. The mean :math:`\\mu` and dispersion :math:`\\alpha` are obtained via ``activation`` function to ensure positivity.

    The log-likelihood loss is computed as:

    .. math::
        \\begin{aligned}
            \\mathcal{L}(x; \\mu, \\alpha) &= \\log \\Gamma(x + \\alpha) - \\log \\Gamma(\\alpha) - \\log \\Gamma(x + 1) \\\\
            &+ \\alpha (\\log \\alpha - \\log (\\mu + \\alpha)) + x (\\log \\mu - \\log (\\mu + \\alpha))
        \\end{aligned}

    Args:
        alpha_param (bool): If True, use a learnable parameter for alpha. Default is False.
        activation (str): The activation function to ensure positivity of mu and alpha. Should be either 'softplus' or 'exp'. Default is 'softplus'.

    Shapes:
        * Input shape: (\\*, 2) if alpha_param is False, else (\\*).
        * Output shape: (\\*)
    '''
    def __init__(
        self, alpha_param: bool = False, activation: str = 'softplus'
    ):
        super().__init__()
        if activation not in {'softplus', 'exp'}:
            raise ValueError(f'Unsupported activation function: {activation}')
        self.activation = activation
        self.alpha_param = alpha_param
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))

    def _check_input_shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.alpha_param:
            if x.size(-1) != 2:
                raise ValueError('The last dimension of input tensor must be 2 when alpha_param is False.')
            log_mu, log_alpha = x.chunk(2, dim=-1)
            log_mu, log_alpha = log_mu.squeeze(-1), log_alpha.squeeze(-1)
        else:
            log_mu = x
            log_alpha = self.log_alpha.expand_as(x)
        if self.activation == 'softplus':
            log_alpha = (torch.nn.functional.softplus(log_alpha) + 1e-8).log()
            log_mu = (torch.nn.functional.softplus(log_mu) + 1e-8).log()
        return log_mu, log_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_mu, _ = self._check_input_shape(x)
        return log_mu.exp()

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_mu, log_alpha = self._check_input_shape(x)
        log_mu_alpha = torch.logaddexp(log_mu, log_alpha)
        alpha_exp = log_alpha.exp()
        target = target.float()

        return -(
            + torch.lgamma(target + alpha_exp)
            - torch.lgamma(alpha_exp)
            - torch.lgamma(target + 1) \
            + alpha_exp * (log_alpha - log_mu_alpha)
            + target * (log_mu - log_mu_alpha)
        )

class StackedTruncatedNormal(Module):
    '''
    Implements the truncated normal distribution. The truncated probability density function is **stacked to the span boundaries**.

    Args:
        lb (float): The lower bound of the truncation, default is -inf.
        ub (float): The upper bound of the truncation, default is inf.
        eps (float): A small value to avoid numerical issues.
        sigma_activation (str): The activation function to ensure positivity of sigma. Should be either 'softplus' or 'exp'.
    '''

    def __init__(
        self, lb: float = float('-inf'), ub: float = float('inf'),
        eps: float = 1e-6, sigma_activation: str = 'softplus'
    ):
        super().__init__()
        if lb >= ub:
            raise ValueError(f'Require lb < ub, got lb={lb}, ub={ub}')
        if sigma_activation not in {'softplus', 'exp'}:
            raise ValueError(f'Unsupported sigma activation: {sigma_activation}')
        self.register_buffer('lb', torch.tensor(lb))
        self.register_buffer('ub', torch.tensor(ub))
        self.normal = torch.distributions.Normal(0.0, 1.0)
        self.sigma_activation = sigma_activation
        self.eps = eps

        self.sigma_scale = torch.nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )

        if TYPE_CHECKING:
            self.lb: torch.Tensor
            self.ub: torch.Tensor

        self.has_lb = torch.isfinite(self.lb).item()
        self.has_ub = torch.isfinite(self.ub).item()


    def _check_input_shape(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != 2:
            raise ValueError('The last dimension of input tensor must be 2.')
        mu, raw_sigma = x.chunk(2, dim=-1)
        mu, raw_sigma = mu.squeeze(-1), raw_sigma.squeeze(-1)
        if self.sigma_activation == 'softplus':
            sigma = torch.nn.functional.softplus(raw_sigma) + self.eps
        else:
            sigma = torch.exp(raw_sigma) + self.eps

        return mu, sigma * torch.nn.functional.softplus(self.sigma_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._check_input_shape(x)

        z_l, z_u = (self.lb - mu) / sigma, (self.ub - mu) / sigma
        Phi_l, Phi_u = torch.special.ndtr(z_l), torch.special.ndtr(z_u)
        phi_l, phi_u = self.normal.log_prob(z_l).exp(), self.normal.log_prob(z_u).exp()

        ret = mu * (Phi_u - Phi_l) + sigma * (phi_l - phi_u)
        if self.has_lb:
            left_mean = (self.lb * Phi_l).nan_to_num(0.0)
            ret = ret + left_mean
        if self.has_ub:
            right_mean = (self.ub * (1 - Phi_u)).nan_to_num(0.0)
            ret = ret + right_mean

        return ret

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._check_input_shape(x)

        is_left = target <= self.lb
        is_right = target >= self.ub
        z_target = (target - mu) / sigma
        z_l, z_u = (self.lb - mu) / sigma, (self.ub - mu) / sigma
        log_Phi_l = torch.special.log_ndtr(z_l)
        log_Phi_u = torch.special.log_ndtr(-z_u)

        nll = -self.normal.log_prob(z_target) + torch.log(sigma)
        if self.has_lb:
            nll = torch.where(is_left, -log_Phi_l, nll)
        if self.has_ub:
            nll = torch.where(is_right, -log_Phi_u, nll)

        return nll

class LogStackedTruncatedNormal(StackedTruncatedNormal):
    '''
    Implements the truncated log-normal distribution. The truncated probability density function is **stacked to the span boundaries**. The ``lb`` and ``ub`` are defined in the log-space.

    Args:
        lb (float): The lower bound of the truncation in log-space, default is -inf.
        ub (float): The upper bound of the truncation in log-space, default is inf.
        eps (float): A small value to avoid numerical issues.
        sigma_activation (str): The activation function to ensure positivity of sigma. Should be either 'softplus' or 'exp'.
    '''
    def __init__(
        self, lb: float = float('-inf'), ub: float = float('inf'),
        eps: float = 1e-6, sigma_activation: str = 'softplus',
    ):
        super().__init__(lb, ub, eps, sigma_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._check_input_shape(x)

        z_l, z_u = (self.lb - mu) / sigma, (self.ub - mu) / sigma
        Phi_l, Phi_u = torch.special.ndtr(z_l), torch.special.ndtr(z_u)

        offset = mu + sigma.pow(2)
        z_l_shift = (self.lb - offset) / sigma
        z_u_shift = (self.ub - offset) / sigma
        Phi_l_shift = torch.special.ndtr(z_l_shift)
        Phi_u_shift = torch.special.ndtr(z_u_shift)

        ret = torch.exp(mu + 0.5 * sigma.pow(2)) * (Phi_u_shift - Phi_l_shift)
        if self.has_lb:
            left_mean = (torch.exp(self.lb) * Phi_l).nan_to_num(0.0)
            ret = ret + left_mean
        if self.has_ub:
            right_mean = (torch.exp(self.ub) * (1 - Phi_u)).nan_to_num(0.0)
            ret = ret + right_mean

        return ret

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_tgt = target.log()
        jacobian_mask = (log_tgt > self.lb) & (log_tgt < self.ub)
        return super().loss(x, log_tgt) + log_tgt * jacobian_mask.float()

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
