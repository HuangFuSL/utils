import abc
import enum
import collections
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

    Any subclasses should:

    1. register predictors using the ``_register_predictor`` method in the ``__init__`` method.
    2. call ``_finalize_register_predictors`` after all predictors have been registered.
    3. implement the ``inverse_link`` method to compute the predicted mean from the transformed predictors.
    4. implement the ``negative_log_likelihood`` method to compute the negative log-likelihood loss given the transformed predictors and the target tensor.
    5. (optional) implement the ``_guard_predictors`` method to check if the predictors are within the expected range, which will be called in debug mode.
    '''
    predictor_tuple_type: Type[T]

    def __init__(self):
        super().__init__()
        self.global_predictors = []
        self.samplewise_predictors = []
        self.transforms = torch.nn.ModuleDict()
        self._fc = None

    def _guard_predictors(self, predictors: T) -> str | None:
        # Return an error message if any predictor falls outside the expected range.
        return

    def _guard_target(self, target: torch.Tensor) -> str | None:
        # Return an error message if the target tensor falls outside the expected range (e.g., negative values for count data).
        return

    def _finalize_register_predictors(self):
        if set(self.predictor_tuple_type._fields) != set(self.global_predictors + self.samplewise_predictors):
            raise ValueError('The registered predictors do not match the predictor tuple type fields.')
        self._fc = torch.nn.LazyLinear(out_features=self.num_samplewise_predictors)

    def _register_predictor(self, name: str, predictor: Predictor):
        if name in self.global_predictors or \
            name in self.samplewise_predictors:
            raise ValueError(f'Predictor {name} has already been registered.')
        match predictor:
            case Predictor(
                PredictorMode.GLOBAL_LEARNABLE, float() | int() as init, _
            ):
                self.register_parameter(name, torch.nn.Parameter(
                    torch.tensor(init).float()
                ))
                self.global_predictors.append(name)
            case Predictor(
                PredictorMode.GLOBAL_FIXED, float() | int() as value, None
            ):
                self.register_buffer(name, torch.tensor(value).float())
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
        predictor_list = list(torch.unbind(self.fc(input), dim=-1))
        return self.inverse_link(self.to_predictors(predictor_list))

    def sample_forward(self, input: torch.Tensor) -> torch.Tensor:
        predictor_list = list(torch.unbind(self.fc(input), dim=-1))
        return self.sample(self.to_predictors(predictor_list))

    def loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._debug:
            msg = self._guard_target(target)
            if msg is not None:
                raise ValueError(f'Target guard failed: {msg}')
        predictor_list = list(torch.unbind(self.fc(input), dim=-1))
        return self.negative_log_likelihood(self.to_predictors(predictor_list), target)

    def to_predictors(self, predictor_list: List[torch.Tensor]) -> T:
        samplewise_map = {
            name: predictor_list[i]
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
            msg = self._guard_predictors(ret)
            if msg is not None:
                raise ValueError(f'Predictor guard failed: {msg}')
        return ret

    @property
    def num_samplewise_predictors(self) -> int:
        return len(self.samplewise_predictors)

    @property
    def fc(self) -> torch.nn.Module:
        if self._fc is None:
            raise RuntimeError('Predictors have not been registered. Please call _finalize_register_predictors after registering all predictors.')
        return self._fc

    @abc.abstractmethod
    def inverse_link(self, input: T) -> torch.Tensor:
        '''
        Forward pass to compute the predicted mean of the response variable given the transformed predictors.

        Args:
            input (T): A named tuple containing the transformed predictors.

        Returns:
            torch.Tensor: Output tensor of shape (\\*, num_predictors).
        '''
        pass

    def sample(self, input: T) -> torch.Tensor:
        '''
        Forward pass to compute a sample from the predicted distribution given the transformed predictors.

        Args:
            input (T): A named tuple containing the transformed predictors.

        Returns:
            torch.Tensor: A sample drawn from the predicted distribution, of shape (\\*,).
        '''
        raise NotImplementedError('Sampling is not implemented for this GLM.')

    @abc.abstractmethod
    def negative_log_likelihood(self, predictors: T, target: torch.Tensor) -> torch.Tensor:
        '''
        Compute the sample-wise negative log-likelihood loss for the given transformed predictors and target tensor.

        Args:
            predictors (T): A named tuple containing the transformed predictors.
            target (torch.Tensor): The target tensor of shape (\\*,).

        Returns:
            torch.Tensor: The sample-wise negative log-likelihood loss tensor of shape (\\*,).
        '''
        pass


class LinearRegression(BaseGLM):
    '''
    Linear regression model with Gaussian likelihood and identity link function. This serves as a prediction head with MSE loss.
    '''
    predictor_tuple_type = NamedTuple('LinearRegressionPredictors', [
        ('mu', torch.Tensor),
        ('sigma', torch.Tensor),
    ])

    def __init__(self, sigma: int | float | Predictor = 1):
        super().__init__()
        self._register_predictor('mu', Predictor(PredictorMode.SAMPLEWISE))
        if not isinstance(sigma, Predictor):
            sigma = Predictor(PredictorMode.GLOBAL_FIXED, float(sigma))
        self._register_predictor('sigma', sigma)
        self._finalize_register_predictors()

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.sigma <= 0):
            return 'Predictor sigma must be positive.'

    def inverse_link(self, predictors) -> torch.Tensor:
        return predictors.mu

    def sample(self, predictors) -> torch.Tensor:
        return predictors.mu + \
            predictors.sigma * torch.randn_like(
                predictors.mu, device=predictors.mu.device
            )

    def negative_log_likelihood(self, predictors, target) -> torch.Tensor:
        return 0.5 * (
            torch.log(predictors.sigma.pow(2)) +
            (target - predictors.mu).pow(2) / predictors.sigma.pow(2)
        )


class LogisticRegression(BaseGLM):
    '''
    Logistic regression model with Bernoulli likelihood and logit link function. This serves as a prediction head with binary cross-entropy loss.

    Note that the output of the model is the predicted probability of the positive class, obtained by applying the sigmoid function to the linear predictor (logit).
    '''
    predictor_tuple_type = NamedTuple('LogisticRegressionPredictors', [
        ('logit', torch.Tensor),
    ])

    def __init__(self):
        super().__init__()
        self._register_predictor('logit', Predictor(PredictorMode.SAMPLEWISE))
        self._finalize_register_predictors()

    def _guard_target(self, target: torch.Tensor):
        if torch.any((target != 0) & (target != 1)):
            return 'Target tensor for LogisticRegression must be binary (0 or 1).'

    def inverse_link(self, predictors) -> torch.Tensor:
        return torch.sigmoid(predictors.logit)

    def sample(self, predictors) -> torch.Tensor:
        return (torch.rand_like(
            predictors.logit, device=predictors.logit.device
        ) < torch.sigmoid(predictors.logit)).float()

    def negative_log_likelihood(self, predictors, target) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            predictors.logit, target.float(), reduction='none'
        )


def DirichletRegression(
    K: int, transform: Transform = torch.nn.functional.softplus
) -> BaseGLM:
    '''
    Dirichlet regression model for simplex regression.

    Args:
        K (int): The number of classes.

    Shapes:

        * Input shape: (\\*, in_features).
        * Output shape: (\\*, K) - the predicted class weight for each of the K classes.
        * Target shape: (\\*, K)
    '''
    if K < 2:
        raise ValueError(f'Number of classes K must be at least 2, got K={K}.')

    t = collections.namedtuple(
        'DirichletRegressionPredictors', [f'alpha_{i}' for i in range(K)]
    )
    class DirichletRegressionImpl(BaseGLM):
        predictor_tuple_type = t

        def __init__(self):
            super().__init__()
            for i in range(K):
                self._register_predictor(f'alpha_{i}', Predictor(
                    PredictorMode.SAMPLEWISE, raw_transform=transform
                ))

            self._finalize_register_predictors()

        def _guard_predictors(self, predictors: t) -> str | None:
            if torch.any(torch.stack(predictors, dim=-1) <= 0):
                return 'Predictors for DirichletRegression must be positive.'

        def _guard_target(self, target: torch.Tensor) -> str | None:
            if not target.is_floating_point():
                return 'Target tensor for DirichletRegression must be a floating point tensor.'
            if target.shape[-1] != K:
                return f'Target tensor for DirichletRegression must have shape (\\*, {K}), got {target.shape}.'
            if torch.any(target <= 0) or torch.any(target >= 1):
                return 'Target tensor for DirichletRegression must be in the range (0, 1).'
            if not torch.allclose(
                target.sum(dim=-1), target.new_ones(target.shape[:-1]),
                atol=1e-6, rtol=1e-6
            ):
                return 'Target tensor for DirichletRegression must sum to 1 along the last dimension.'

        def inverse_link(self, predictors: t) -> torch.Tensor:
            alpha = torch.stack(predictors, dim=-1)
            return alpha / alpha.sum(dim=-1, keepdim=True)

        def sample(self, predictors: t) -> torch.Tensor:
            alpha = torch.stack(predictors, dim=-1)
            gamma_samples = torch.distributions.Gamma(alpha, 1).sample()
            return gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

        def negative_log_likelihood(self, predictors, target) -> torch.Tensor:
            alpha = torch.stack(predictors, dim=-1)
            return -(
                torch.lgamma(alpha.sum(dim=-1)) +
                -torch.lgamma(alpha).sum(dim=-1) +
                +((alpha - 1) * target.log()).sum(dim=-1)
            )

    return DirichletRegressionImpl()


class PoissonRegression(BaseGLM):
    '''
    Poisson regression model with Poisson likelihood and log link function. This serves as a prediction head with Poisson loss.

    Note that the output of the model is the predicted mean of the Poisson distribution, obtained by applying the exponential function to the linear predictor (log-mean).
    '''
    predictor_tuple_type = NamedTuple('PoissonRegressionPredictors', [
        ('log_mu', torch.Tensor),
    ])

    def __init__(self):
        super().__init__()
        self._register_predictor('log_mu', Predictor(PredictorMode.SAMPLEWISE))
        self._finalize_register_predictors()

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if torch.any(target < 0):
            return 'Target tensor for PoissonRegression must be non-negative.'

    def inverse_link(self, predictors) -> torch.Tensor:
        return predictors.log_mu.exp()

    def sample(self, predictors) -> torch.Tensor:
        return torch.poisson(predictors.log_mu.exp())

    def negative_log_likelihood(self, predictors, target) -> torch.Tensor:
        return predictors.log_mu.exp() - target * predictors.log_mu + torch.lgamma(target + 1)


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

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.mu <= 0):
            return 'Predictor mu must be positive.'
        if torch.any(predictors.alpha <= 0):
            return 'Predictor alpha must be positive.'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if torch.any(target < 0):
            return 'Target tensor for NegativeBinomial must be non-negative.'

    def inverse_link(self, predictors) -> torch.Tensor:
        return predictors.mu

    def sample(self, predictors) -> torch.Tensor:
        mu, alpha = predictors.mu, predictors.alpha
        p = mu / (mu + alpha)
        return torch.distributions.NegativeBinomial(alpha, p).sample()

    def negative_log_likelihood(self, predictors, target) -> torch.Tensor:
        log_mu, log_alpha = predictors.mu.log(), predictors.alpha.log()
        log_mu_alpha = torch.logaddexp(log_mu, log_alpha)
        alpha_exp = predictors.alpha
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

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.sigma <= 0):
            return 'Predictor sigma must be positive.'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if torch.any(target < self.lb) or torch.any(target > self.ub):
            return f'Target tensor for StackedTruncatedNormal must be in the range [{self.lb.item()}, {self.ub.item()}].'

    def inverse_link(self, predictors) -> torch.Tensor:
        mu, sigma = predictors.mu, predictors.sigma

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

    def sample(self, predictors) -> torch.Tensor:
        # Sample then clamp
        mu, sigma = predictors.mu, predictors.sigma
        samples = mu + sigma * torch.randn_like(mu, device=mu.device)
        return samples.clamp(self.lb, self.ub)

    def negative_log_likelihood(self, predictors, target: torch.Tensor) -> torch.Tensor:
        mu, sigma = predictors.mu, predictors.sigma

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

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.sigma <= 0):
            return 'Predictor sigma must be positive.'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if torch.any(target <= 0):
            return 'Target tensor for LogStackedTruncatedNormal must be positive.'
        log_target = target.log()
        if torch.any(log_target < self.lb) or torch.any(log_target > self.ub):
            return f'Target tensor for LogStackedTruncatedNormal must have log-values in the range [{self.lb.item()}, {self.ub.item()}].'

    def inverse_link(self, predictors) -> torch.Tensor:
        mu, sigma = predictors.mu, predictors.sigma

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

    def sample(self, predictors) -> torch.Tensor:
        return super().sample(predictors).exp()

    def negative_log_likelihood(self, predictors, target: torch.Tensor) -> torch.Tensor:
        log_tgt = target.log()
        jacobian_mask = (log_tgt > self.lb) & (log_tgt < self.ub)
        return super().negative_log_likelihood(predictors, log_tgt) + log_tgt * jacobian_mask.float()

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

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.mu <= 0):
            return 'Predictor mu must be positive.'
        if torch.any(predictors.phi <= 0):
            return 'Predictor phi must be positive.'
        if torch.any((predictors.power <= 1) | (predictors.power >= 2)):
            return 'Predictor power must be in the range (1, 2).'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if torch.any(target < 0):
            return 'Target tensor for Tweedie must be non-negative.'

    def inverse_link(self, predictors) -> torch.Tensor:
        return predictors.mu

    def sample(self, predictors) -> torch.Tensor:
        mu, phi, p = predictors.mu, predictors.phi, predictors.power

        lambda_ = mu.pow(2.0 - p) / (phi * (2.0 - p))
        alpha = (2.0 - p) / (p - 1.0)
        gamma = phi * (p - 1.0) * mu.pow(p - 1.0)

        count = torch.poisson(lambda_)
        out = lambda_.new_zeros(lambda_.shape)

        mask = count > 0
        if mask.any():
            concentration = count[mask] * alpha[mask]
            rate = 1.0 / gamma[mask]

            gamma_dist = torch.distributions.Gamma(
                concentration=concentration,
                rate=rate,
            )
            out[mask] = gamma_dist.sample()

        return out

    def negative_log_likelihood(self, predictors, target: torch.Tensor) -> torch.Tensor:
        mu, phi, p = predictors.mu, predictors.phi, predictors.power

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

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.sigma <= 0):
            return 'Predictor sigma must be positive.'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if torch.any(target <= 0):
            return 'Target tensor for ZeroInflatedLogNormal must be positive.'

    def inverse_link(self, predictors) -> torch.Tensor:
        logit, mu, sigma = predictors.logit, predictors.mu, predictors.sigma

        prob = torch.sigmoid(logit)
        return prob * torch.exp(mu + 0.5 * sigma.pow(2))

    def sample(self, predictors) -> torch.Tensor:
        # Zero * sample from log-normal
        logit, mu, sigma = predictors.logit, predictors.mu, predictors.sigma
        prob = torch.sigmoid(logit)
        bern_sample = (torch.rand_like(prob) < prob).float()
        log_normal_sample = mu + sigma * torch.randn_like(mu, device=mu.device)
        return bern_sample * log_normal_sample.exp()

    def negative_log_likelihood(self, predictors, target):
        logit, mu, sigma = predictors.logit, predictors.mu, predictors.sigma

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

class PoissonGamma(BaseGLM):
    '''
    Poisson-Gamma regression model for modeling the sum of a series of i.i.d Gamma random variables where the number of terms in the sum follows a Poisson distribution.

    .. math::
        \\begin{aligned}
            Y &= \\sum_{i=1}^N X_i \\\\
            N &\\sim \\text{Poisson}(\\boldsymbol{\\lambda}) \\\\
            X_i &\\sim \\text{Gamma}(\\alpha, \\beta) \\\\
            \\alpha &= \\frac{1}{\\boldsymbol{\\phi}} \\\\
            \\beta &= \\frac{1}{\\boldsymbol{\\phi} \\boldsymbol{\\mu}}
        \\end{aligned}

    Target Shape:

    - (\\*, 2): the first column is the sum of observed response variable over all events, and the second column is counts of events.
    '''
    predictor_tuple_type = NamedTuple('PoissonGammaPredictors', [
        ('lambda_', torch.Tensor),
        ('mu', torch.Tensor),
        ('phi', torch.Tensor),
    ])

    def __init__(
        self, lambda_transform: Transform, mu_transform: Transform,
        phi: Predictor
    ):
        super().__init__()
        self._register_predictor('lambda_', Predictor(
            PredictorMode.SAMPLEWISE, raw_transform=lambda_transform
        ))
        self._register_predictor('mu', Predictor(
            PredictorMode.SAMPLEWISE, raw_transform=mu_transform
        ))
        self._register_predictor('phi', phi)
        self._finalize_register_predictors()

    def _guard_predictors(self, predictors) -> str | None:
        if torch.any(predictors.lambda_ <= 0):
            return 'Predictor lambda_ must be positive.'
        if torch.any(predictors.mu <= 0):
            return 'Predictor mu must be positive.'
        if torch.any(predictors.phi <= 0):
            return 'Predictor phi must be positive.'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        value, count = map(lambda x: x.squeeze(-1), torch.chunk(target, 2, dim=-1))
        if torch.any(value < 0):
            return 'Value tensor for PoissonGamma must be positive.'
        if torch.any(count < 0):
            return 'Count tensor for PoissonGamma must be positive.'
        if torch.any((count == 0) & (value != 0)):
            return 'If count == 0, aggregate value must be 0.'

    def inverse_link(self, predictors) -> torch.Tensor:
        return torch.stack([
            predictors.mu * predictors.lambda_,
            predictors.lambda_
        ], dim=-1)

    def sample(self, predictors) -> torch.Tensor:
        lambda_, mu, phi = predictors.lambda_, predictors.mu, predictors.phi
        alpha, beta = 1 / phi, 1 / (phi * mu)

        count = torch.poisson(lambda_)
        value = lambda_.new_zeros(lambda_.shape)

        mask = count > 0
        if mask.any():
            concentration = count[mask] * alpha[mask]
            rate = beta[mask]
            gamma_dist = torch.distributions.Gamma(
                concentration=concentration,
                rate=rate,
            )
            value[mask] = gamma_dist.sample()

        return torch.stack([value, count], dim=-1)

    def negative_log_likelihood(self, predictors, target) -> torch.Tensor:
        shape = predictors.mu.shape
        value, count = map(lambda x: x.squeeze(-1), torch.chunk(target, 2, dim=-1))
        mask = count > 0

        lambda_, mu, phi = predictors.lambda_, predictors.mu, predictors.phi
        alpha, beta = (1 / phi)[mask], (1 / (phi * mu))[mask]
        value_m, count_m = value[mask], count[mask]

        p = -lambda_ + torch.xlogy(count, lambda_) - torch.lgamma(count + 1)
        g = p.new_zeros(p.shape)
        g[mask] = (
            torch.xlogy(count_m * alpha, beta) - torch.lgamma(count_m * alpha)
            + torch.xlogy(count_m * alpha - 1, value_m) - beta * value_m
        )

        return -(p + g)

class NegativeBinomialGamma(BaseGLM):
    '''
    Implements the negative log-likelihood loss for modeling the sum of a series of i.i.d Gamma random variables where the number of terms in the sum follows a Negative Binomial distribution.

    Shapes:
        * Input shape: (\\*, in_features).
        * Output shape: (\\*)
    '''
    predictor_tuple_type = NamedTuple('NegativeBinomialGammaPredictors', [
        ('mu', torch.Tensor),      # aggregate mean E[Y|x]
        ('m', torch.Tensor),       # NB mean E[N|x]
        ('r', torch.Tensor),       # NB shape / dispersion
        ('alpha', torch.Tensor),   # Gamma shape
    ])

    def __init__(
        self,
        mu_transform: Transform, m_transform: Transform,
        r: Predictor, alpha: Predictor,
        n_terms: int = 16
    ):
        super().__init__()

        self.n_terms = n_terms
        self._register_predictor('mu', Predictor(PredictorMode.SAMPLEWISE, raw_transform=mu_transform))
        self._register_predictor('m', Predictor(PredictorMode.SAMPLEWISE, raw_transform=m_transform))
        self._register_predictor('r', r)
        self._register_predictor('alpha', alpha)
        self._finalize_register_predictors()

    def _guard_predictors(self, predictors) -> str | None:
        for name, value in predictors._asdict().items():
            if not (value >= 0).all():
                return f'Predictor {name} must be non-negative, but got min={value.min().item()}.'

    def _guard_target(self, target: torch.Tensor) -> str | None:
        if not (target >= 0).all():
            return f'Target tensor must be non-negative, but got min={target.min().item()}.'

    def inverse_link(self, predictors) -> torch.Tensor:
        return predictors.mu

    def sample(self, predictors) -> torch.Tensor:
        mu, m, r, alpha = predictors.mu, predictors.m, \
            predictors.r, predictors.alpha

        p = m / (m + r)
        count = torch.distributions.NegativeBinomial(r, p).sample()

        mask = count > 0
        value = mu.new_zeros(mu.shape)
        if mask.any():
            concentration = count[mask] * alpha[mask]
            rate = alpha[mask] * m[mask] / mu[mask]
            gamma_dist = torch.distributions.Gamma(
                concentration=concentration,
                rate=rate,
            )
            value[mask] = gamma_dist.sample()
        return value

    def negative_log_likelihood(self, predictors, target: torch.Tensor):
        mu = predictors.mu.unsqueeze(-1)
        m = predictors.m.unsqueeze(-1)
        r = predictors.r.unsqueeze(-1)
        alpha = predictors.alpha.unsqueeze(-1)

        non_zero_mask = target > 0
        nll_zero = (r * torch.log1p(m / r)).squeeze(-1)
        beta = (alpha * m / mu)

        n = torch.arange(
            1, self.n_terms + 1, device=target.device, dtype=target.dtype
        ).view(*([1] * target.ndim), self.n_terms)
        safe_target = torch.where(
            non_zero_mask, target, torch.ones_like(target)
        ).unsqueeze(-1)

        # log NB(n; mean=m, shape=r)
        log_nb = (
            torch.lgamma(n + r)
            - torch.lgamma(r)
            - torch.lgamma(n + 1.0)
            + r * (torch.log(r) - torch.log(r + m))
            + n * (torch.log(m) - torch.log(r + m))
        )

        n_alpha = n * alpha

        # log Gamma(y; shape=n*alpha, rate=beta)
        log_gamma = (
            n_alpha * torch.log(beta)
            - torch.lgamma(n_alpha)
            + (n_alpha - 1.0) * torch.log(safe_target)
            - beta * safe_target
        )

        log_mix = torch.logsumexp(log_nb + log_gamma, dim=-1)
        nll_pos = -log_mix

        return torch.where(non_zero_mask, nll_pos, nll_zero)
