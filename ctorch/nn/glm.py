import abc
import enum
from typing import TYPE_CHECKING, Callable, Generic, List, NamedTuple, Type, TypeVar

import torch

from .linear import Activation, FuncWrapper
from .module import Module

T = TypeVar('T', bound=NamedTuple)
Transform = torch.nn.Module | Callable[[torch.Tensor], torch.Tensor] | str
OptionalTransform = Transform | None

def _ensure_shape(predictors: T) -> T:
    # Expand all predictors to the same shape if they have different shapes
    max_shape = torch.broadcast_shapes(*[p.shape for p in predictors])
    return predictors._replace(
        **{k: p.expand(max_shape) for k, p in predictors._asdict().items()}
    )

class PredictorMode(enum.Enum):
    GLOBAL_LEARNABLE = enum.auto()
    GLOBAL_FIXED = enum.auto()
    SAMPLEWISE = enum.auto()

class Predictor(NamedTuple):
    '''
    A predictor for a GLM, which can be either a global learnable parameter, a global fixed parameter, or a sample-wise predictor.

    Args:
        name (str): The name of the predictor, which should be unique across all predictors in the model.
        mode (PredictorMode): The mode of the predictor, which determines how it is registered and used in the model.
        init_or_value (float | int | None): The initial value for the predictor if it is a global learnable parameter, or the fixed value if it is a global fixed parameter. Should be None for sample-wise predictors.
        raw_transform (OptionalTransform): An optional transform to apply to the raw predictor value before using it in the model. This can be a string name of an activation function (e.g., 'relu', 'softplus'), a torch.nn.Module instance, or a callable function. If None, no transform is applied.
    '''
    mode: PredictorMode
    init_or_value: float | int | None = None
    raw_transform: OptionalTransform = None

class BaseGLM(Module, abc.ABC, Generic[T]):
    '''
    Base class for Generalized Linear Models (GLMs). This class defines the common interface and functionality for GLM implementations. The forward method computes the predicted mean of the response variable given the input features, while the loss method computes the sample-wise negative log-likelihood loss for the given input and target tensors.
    '''
    predictor_tuple_type: Type[T]

    def __init__(self):
        super().__init__()
        self.global_predictors = []
        self.samplewise_predictors = []
        self.transforms = torch.nn.ModuleDict()
        self._fc = None

    def _guard_predictors(self, eta: T) -> None:
        # Raise an error if any predictor falls outside the expected range.
        pass

    def _finalize_register_predictors(self):
        if set(self.predictor_tuple_type._fields) != set(self.global_predictors + self.samplewise_predictors):
            raise ValueError('The registered predictors do not match the predictor tuple type fields.')
        self._fc = torch.nn.LazyLinear(out_features=self.num_predictors)

    def _register_predictor(self, name: str, predictor: Predictor):
        if name in self.global_predictors or \
            name in self.samplewise_predictors:
            raise ValueError(f'Predictor {name} has already been registered.')
        match predictor:
            case Predictor(
                PredictorMode.GLOBAL_LEARNABLE, float() | int() as init, _
            ):
                self.register_parameter(name, torch.nn.Parameter(torch.tensor(init)))
                self.global_predictors.append(name)
            case Predictor(
                PredictorMode.GLOBAL_FIXED, float() | int() as value, None
            ):
                self.register_buffer(name, torch.tensor(value))
                self.global_predictors.append(name)
            case Predictor(
                PredictorMode.SAMPLEWISE, None, _
            ):
                self.samplewise_predictors.append(name)
            case _:
                raise ValueError(f'Invalid predictor registration: {predictor}')
        match predictor.raw_transform:
            case str() as activation:
                self.transforms[name] = Activation(activation)
            case torch.nn.Module() as module:
                self.transforms[name] = module
            case func if callable(func):
                self.transforms[name] = FuncWrapper(func)
            case None:
                pass
            case _:
                raise ValueError(f'Invalid raw_transform for predictor {name}: {predictor.raw_transform}')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        eta_list = list(torch.unbind(self.fc(input), dim=-1))
        return self.eta_forward(self.to_predictors(eta_list))

    def loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eta_list = list(torch.unbind(self.fc(input), dim=-1))
        return self.eta_loss(self.to_predictors(eta_list), target)

    def to_predictors(self, eta_list: List[torch.Tensor]) -> T:
        samplewise_map = {
            name: eta_list[i]
            for i, name in enumerate(self.samplewise_predictors)
        }

        predictors_dict = {}
        for name in self.predictor_tuple_type._fields:
            if name in samplewise_map:
                value = samplewise_map[name]
            else:
                value = getattr(self, name)
            if name in self.transforms:
                predictors_dict[name] = self.transforms[name](value)
            else:
                predictors_dict[name] = value

        ret = _ensure_shape(self.predictor_tuple_type(**predictors_dict))
        if self._debug:
            self._guard_predictors(ret)
        return ret

    @property
    def num_predictors(self) -> int:
        return len(self.samplewise_predictors)

    @property
    def fc(self) -> torch.nn.Module:
        if self._fc is None:
            raise RuntimeError('Predictors have not been registered. Please call _finalize_register_predictors after registering all predictors.')
        return self._fc

    @abc.abstractmethod
    def eta_forward(self, input: T) -> torch.Tensor:
        '''
        Forward pass to compute the linear predictors (eta) before applying the link function. This is useful for computing the distribution parameters in the loss function.

        Args:
            input (torch.Tensor): Input tensor of shape (\\*, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (\\*, num_predictors).
        '''
        pass

    @abc.abstractmethod
    def eta_loss(self, eta: T, target: torch.Tensor) -> torch.Tensor:
        pass


class NegativeBinomial(BaseGLM):
    '''
    Implements the negative log-likelihood loss for the Negative Binomial distribution.

    Let :math:`g_\\mu` and :math:`g_\\alpha` be the outputs of the network. The mean :math:`\\mu` and dispersion :math:`\\alpha` are obtained via ``activation`` function to ensure positivity.

    The log-likelihood is computed as:

    .. math::
        \\begin{aligned}
            \\mathcal{L}(x; \\mu, \\alpha) &= \\log \\Gamma(x + \\alpha) - \\log \\Gamma(\\alpha) - \\log \\Gamma(x + 1) \\\\
            &+ \\alpha (\\log \\alpha - \\log (\\mu + \\alpha)) + x (\\log \\mu - \\log (\\mu + \\alpha))
        \\end{aligned}

    Args:
        mu (Predictor): The predictor for the mean parameter :math:`\\mu`.
        alpha (Predictor): The predictor for the dispersion parameter :math:`\\alpha`.

    Shapes:
        * Input shape: (\\*, in_features).
        * Output shape: (\\*)
    '''
    predictor_tuple_type = NamedTuple('NegativeBinomialPredictors', [
        ('mu', torch.Tensor),
        ('alpha', torch.Tensor),
    ])

    def __init__(self, mu: Predictor, alpha: Predictor):
        super().__init__()
        self._register_predictor('mu', mu)
        self._register_predictor('alpha', alpha)
        self._finalize_register_predictors()

    def _guard_predictors(self, eta) -> None:
        if torch.any(eta.mu <= 0):
            raise ValueError('Predictor mu must be positive.')
        if torch.any(eta.alpha <= 0):
            raise ValueError('Predictor alpha must be positive.')

    def eta_forward(self, eta) -> torch.Tensor:
        return eta.mu

    def eta_loss(self, eta, target) -> torch.Tensor:
        self._check_non_negative(target)
        log_mu, log_alpha = eta.mu.log(), eta.alpha.log()
        log_mu_alpha = torch.logaddexp(log_mu, log_alpha)
        alpha_exp = eta.alpha
        target = target.float()

        return -(
            + torch.lgamma(target + alpha_exp)
            - torch.lgamma(alpha_exp)
            - torch.lgamma(target + 1)
            + alpha_exp * (log_alpha - log_mu_alpha)
            + target * (log_mu - log_mu_alpha)
        )

class StackedTruncatedNormal(BaseGLM):
    '''
    Implements the truncated normal distribution. The truncated probability density function is **stacked to the span boundaries**.

    Args:
        sigma (Predictor): The predictor for the standard deviation parameter :math:`\\sigma`.
        lb (float): The lower bound of the truncation, default is -inf.
        ub (float): The upper bound of the truncation, default is inf.

    Shapes:
        * Input shape: (\\*, in_features).
        * Output shape: (\\*)
    '''
    predictor_tuple_type = NamedTuple('StackedTruncatedNormalPredictors', [
        ('mu', torch.Tensor),
        ('sigma', torch.Tensor),
    ])

    def __init__(
        self, sigma: Predictor, lb: float = float('-inf'), ub: float = float('inf'),
    ):
        super().__init__()
        if lb >= ub:
            raise ValueError(f'Require lb < ub, got lb={lb}, ub={ub}')
        self.register_buffer('lb', torch.tensor(lb))
        self.register_buffer('ub', torch.tensor(ub))
        self._register_predictor('mu', Predictor(PredictorMode.SAMPLEWISE))
        self._register_predictor('sigma', sigma)
        self.normal = torch.distributions.Normal(0.0, 1.0)

        if TYPE_CHECKING:
            self.lb: torch.Tensor
            self.ub: torch.Tensor

        self.has_lb = torch.isfinite(self.lb).item()
        self.has_ub = torch.isfinite(self.ub).item()
        self._finalize_register_predictors()

    def _guard_predictors(self, eta) -> None:
        if torch.any(eta.sigma <= 0):
            raise ValueError('Predictor sigma must be positive.')

    def eta_forward(self, eta) -> torch.Tensor:
        mu, sigma = eta.mu, eta.sigma

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

    def eta_loss(self, eta, target: torch.Tensor) -> torch.Tensor:
        mu, sigma = eta.mu, eta.sigma

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
        sigma (Predictor): The predictor for the standard deviation parameter :math:`\\sigma`.
        lb (float): The lower bound of the truncation in log-space, default is -inf.
        ub (float): The upper bound of the truncation in log-space, default is inf.

    Shapes:
        * Input shape: (\\*, in_features).
        * Output shape: (\\*)
    '''
    def __init__(
        self, sigma: Predictor, lb: float = float('-inf'), ub: float = float('inf')
    ):
        super().__init__(sigma, lb, ub)

    def _guard_predictors(self, eta) -> None:
        if torch.any(eta.sigma <= 0):
            raise ValueError('Predictor sigma must be positive.')

    def eta_forward(self, eta) -> torch.Tensor:
        mu, sigma = eta.mu, eta.sigma

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

    def eta_loss(self, eta, target: torch.Tensor) -> torch.Tensor:
        log_tgt = target.log()
        jacobian_mask = (log_tgt > self.lb) & (log_tgt < self.ub)
        return super().eta_loss(eta, log_tgt) + log_tgt * jacobian_mask.float()

class Tweedie(BaseGLM):
    '''
    Implements Tweedie mean prediction and sample-wise Tweedie loss.

    This module assumes the most common Tweedie regime:

    .. math::
        1 < p < 2

    with log link or softplus link for the mean function and variance function:

    .. math::
        \\mathrm{Var}(Y) = \\phi \\mu^p

    Notes:
        - ``forward`` returns the predicted mean :math:`\\mu`.
        - ``loss`` returns the sample-wise Tweedie loss without reduction.
        - The implemented loss is the standard Tweedie quasi-likelihood,
          equivalently half the unit deviance divided by :math:`\\phi`:

          .. math::
              \\mathcal{L}(y, \\mu)
              =
              \\frac{1}{\\phi}
              \\left[
                  \\frac{y^{2-p}}{(1-p)(2-p)}
                  -
                  \\frac{y \\mu^{1-p}}{1-p}
                  +
                  \\frac{\\mu^{2-p}}{2-p}
              \\right]

          This is the standard objective used in practice for Tweedie regression.
          It is not the exact log-density, whose normalizing term has no simple
          closed form for :math:`1 < p < 2`.

    Args:
        phi (Predictor): The dispersion parameter.
        power (Predictor): The power parameter. Should be in the range (1, 2).

    Shapes:
        * Input shape: (\\*, in_features).
        * Output shape: (\\*)
    '''
    predictor_tuple_type = NamedTuple('TweediePredictors', [
        ('mu', torch.Tensor),
        ('phi', torch.Tensor),
        ('power', torch.Tensor),
    ])

    def __init__(self, mu_transform: Transform, phi: Predictor, power: Predictor):
        super().__init__()
        self._register_predictor('mu', Predictor(
            PredictorMode.SAMPLEWISE, None, mu_transform
        ))
        self._register_predictor('phi', phi)
        self._register_predictor('power', power)
        self._finalize_register_predictors()

    def _guard_predictors(self, eta) -> None:
        if torch.any(eta.mu <= 0):
            raise ValueError('Predictor mu must be positive.')
        if torch.any(eta.phi <= 0):
            raise ValueError('Predictor phi must be positive.')
        if torch.any((eta.power <= 1) | (eta.power >= 2)):
            raise ValueError('Predictor power must be in the range (1, 2).')

    def eta_forward(self, eta) -> torch.Tensor:
        return eta.mu

    def eta_loss(self, eta, target: torch.Tensor) -> torch.Tensor:
        mu, phi, p = eta.mu, eta.phi, eta.power
        self._check_non_negative(target)

        nll = (
            target.pow(2.0 - p) / ((1.0 - p) * (2.0 - p))
            - target * mu.pow(1.0 - p) / (1.0 - p)
            + mu.pow(2.0 - p) / (2.0 - p)
        ) / phi

        return nll

class ZeroInflatedLogNormal(BaseGLM):
    '''
    Implements a zero-inflated log-normal distribution layer. A zero-inflated log-normal distribution is a mixture of a point mass at zero and a log-normal distribution.

    .. math::
        \\begin{aligned}
            P(X = 0) &= \\sigma(-\\text{logit}) \\\\
            P(X > 0) &= 1 - \\sigma(-\\text{logit}) \\\\
            \\log X | X > 0 &\\sim \\mathcal{N}(\\mu, \\sigma^2) \\\\
        \\end{aligned}

    Args:
        sigma (Predictor | None): The predictor for the standard deviation parameter :math:`\\sigma`. If None, a global fixed predictor with value 1 is used.

    Shapes:
        * Input shape: (\\*, in_features).
        * Output shape: (\\*)
    '''
    predictor_tuple_type = NamedTuple('ZeroInflatedLogNormalPredictors', [
        ('logit', torch.Tensor),
        ('mu', torch.Tensor),
        ('sigma', torch.Tensor),
    ])
    def __init__(self, sigma: Predictor | None = None):
        super(ZeroInflatedLogNormal, self).__init__()
        self._register_predictor('logit', Predictor(PredictorMode.SAMPLEWISE))
        self._register_predictor('mu', Predictor(PredictorMode.SAMPLEWISE))
        if sigma is None:
            sigma = Predictor(PredictorMode.GLOBAL_FIXED, 1)
        self._register_predictor('sigma', sigma)

        self._finalize_register_predictors()
        self.register_buffer('const', 0.5 * torch.log(torch.tensor(2 * torch.pi)))
        if TYPE_CHECKING:
            self.const: torch.Tensor

    def _guard_predictors(self, eta) -> None:
        if torch.any(eta.sigma <= 0):
            raise ValueError('Predictor sigma must be positive.')

    def eta_forward(self, eta) -> torch.Tensor:
        logit, mu, sigma = eta.logit, eta.mu, eta.sigma

        prob = torch.sigmoid(logit)
        return prob * torch.exp(mu + 0.5 * sigma.pow(2))

    def eta_loss(self, eta, target):
        logit, mu, sigma = eta.logit, eta.mu, eta.sigma

        nonzero_mask = target > 0
        safe_target = torch.where(nonzero_mask, target, torch.ones_like(target))
        inflation_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, nonzero_mask.float(), reduction='none'
        )
        lognormal_loss = 0.5 * ((mu - torch.log(safe_target)) ** 2) / sigma.pow(2)
        jacobian_loss = torch.log(safe_target) + torch.log(sigma) + self.const
        return (
            inflation_loss +
            torch.where(nonzero_mask, lognormal_loss, torch.zeros_like(lognormal_loss)) +
            torch.where(nonzero_mask, jacobian_loss, torch.zeros_like(jacobian_loss))
        )
