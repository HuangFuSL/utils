'''
`utils.ctorch.functional` - Functional utilities for PyTorch tensors.
'''
import math
from typing import Callable, Literal, Tuple

import torch
import torch.linalg

from . import ops

def logit_product(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    '''
    Calculate the logit of the product of two probabilities given their logits.

    .. math::
        \\sigma^{-1}(\\sigma(x) \\cdot \\sigma(y)) = x + y - \\log(1 + e^x + e^y)

    Args:
        x (torch.Tensor): Logit of the first probability.
        y (torch.Tensor): Logit of the second probability.

    Returns:
        torch.Tensor: Logit of the product of the two probabilities.
    '''
    logaddexp = torch.logaddexp(x, y)
    return x + y - torch.logaddexp(
        logaddexp, logaddexp.new_zeros(logaddexp.size())
    )

def log_norm_pdf(
    x: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor | None = None,
    logSigma: torch.Tensor | None = None, batch_first: bool | None = None
) -> torch.Tensor:
    """
    Calculate the log probability density function of a normal distribution.

    .. math::
        \\log p(x) = -\\frac{1}{2} \\left( D \\log(2\\pi) + \\log |\\Sigma| + (x - \\mu)^T \\Sigma^{-1} (x - \\mu) \\right)

    Args:
        x (torch.Tensor): Input tensor, shape (N, D), where N is the number of samples and D is the number of dimensions.
        mean (torch.Tensor): Mean of the normal distribution, shape (D,), or (N, D)
        Sigma (torch.Tensor | None): Covariance matrix of the normal distribution, shape (N, D, D), (N, D), (N,), (D, D), (D,), or a scalar.
        logSigma (torch.Tensor | None): Logarithm of the covariance matrix, same shape as Sigma.
        batch_first (bool | None): If True, indicates that the first dimension of Sigma is the batch size (N).

    Returns:
        torch.Tensor: Tensor containing the log PDF values.
    """
    N, D = x.shape
    PI = x.new_tensor(torch.pi)

    # Check mean shape
    if mean.dim() == 1:
        if mean.shape != (D,):
            raise ValueError(
                f'Mean shape {mean.shape} does not match input dimension {D}.'
            )
    elif mean.dim() == 2:
        if mean.shape != (N, D):
            raise ValueError(
                f'Mean shape {mean.shape} does not match input dimension {(N, D)}.'
            )
    else:
        raise ValueError(f'Mean must be a vector or matrix, but got shape {mean.shape}.')

    # Check Sigma and logSigma
    if Sigma is not None:
        if N == D and Sigma.dim() in (1, 2) and batch_first is None:
            raise ValueError(
                'Ambiguous Sigma shape (D,) (D, D) or (N,) (N, D) with N == D'
            )
        if logSigma is not None:
            raise ValueError(
                'Cannot provide both Sigma and logSigma. Use one of them.'
            )
    else:
        if logSigma is None:
            raise ValueError('Either Sigma or logSigma must be provided.')
        if N == D and logSigma.dim() in (1, 2) and batch_first is None:
            raise ValueError(
                'Ambiguous logSigma shape (D,) (D, D) or (N,) (N, D) with N == D'
            )
        Sigma = torch.exp(logSigma)

    CONST = D * torch.log(2 * PI)
    if Sigma.dim() == 0:
        # Sigma is a scalar
        return -(
            CONST + D * torch.log(Sigma) +
            torch.sum((x - mean) ** 2, dim=1) / Sigma
        ) / 2
    if Sigma.dim() == 1:
        if not torch.all(Sigma > 0):
            raise ValueError('Sigma must be positive.')
        if batch_first is True:
            # When Sigma is (N,)
            if Sigma.shape != (N,):
                raise ValueError(
                    f'Sigma shape {Sigma.shape} does not match input dimension '
                    f'{N}.'
                )
            return -(
                CONST + D * torch.log(Sigma) +
                torch.sum((x - mean) ** 2 / Sigma.unsqueeze(1), dim=1)
            ) / 2
        elif batch_first is False:
            # When Sigma is (D,)
            if Sigma.shape != (D,):
                raise ValueError(
                    f'Sigma shape {Sigma.shape} does not match input dimension '
                    f'{D}.'
                )
            return -(
                CONST + torch.sum(torch.log(Sigma))+
                torch.sum((x - mean) ** 2 / Sigma, dim=1)
            ) / 2
        raise ValueError('When N == D, batch_first must be provided.')
    if Sigma.dim() == 2:
        if batch_first is True:
            # When Sigma is (N, D)
            if Sigma.shape != (N, D):
                raise ValueError(
                    f'Sigma shape {Sigma.shape} does not match input dimension '
                    f'{(N, D)}.'
                )
            if not torch.all(Sigma > 0):
                raise ValueError('Sigma must be positive.')
            return -(
                CONST + torch.sum(torch.log(Sigma), dim=1) +
                torch.sum((x - mean) ** 2 / Sigma, dim=1)
            ) / 2
        elif batch_first is False:
            # When Sigma is (D, D)
            if Sigma.shape != (D, D):
                raise ValueError(f"Sigma shape {Sigma.shape} does not match input dimension {D}.")
            if torch.det(Sigma) <= 0:
                raise ValueError("Covariance matrix Sigma must be positive definite.")
            if not torch.allclose(Sigma, Sigma.T):
                raise ValueError("Covariance matrix Sigma must be symmetric.")
            # Calculate exponential term using Triangular Solve method
            L = torch.linalg.cholesky(Sigma)
            diff = x - mean
            z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
            q = torch.sum(z.squeeze(-1) ** 2, dim=1)
            return -(CONST + torch.logdet(Sigma) + q) / 2
        raise ValueError('When N == D, batch_first must be provided.')
    if Sigma.dim() == 3:
        # When Sigma is (N, D, D)
        # x: (N, D), mean: (N, D)
        if Sigma.shape[1:] != (D, D):
            raise ValueError(
                f'Sigma shape {Sigma.shape} does not match input dimension {(D, D)}.'
            )
        if not torch.allclose(Sigma, Sigma.transpose(-1, -2)):
            raise ValueError("Covariance matrix Sigma must be symmetric.")
        if torch.any(torch.linalg.det(Sigma) <= 0):
            raise ValueError("Covariance matrix Sigma must be positive definite.")
        # Calculate exponential term using Triangular Solve method
        L = torch.linalg.cholesky(Sigma)
        diff = (x - mean).unsqueeze(-1)  # (N, D, 1)
        z = torch.linalg.solve_triangular(L, diff, upper=False)
        z = z.squeeze(-1)  # (N, D)
        q = torch.sum(z ** 2, dim=-1)  # (N,)
        return -(CONST + torch.logdet(Sigma) + q) / 2

    raise ValueError(f"Sigma must be a scalar, vector, or matrix, but got shape {Sigma.shape}.")

def norm_pdf(
    x: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor | None = None,
    logSigma: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Calculate the log probability density function of a normal distribution.

    Args:
        x (torch.Tensor): Input tensor, shape (N, D), where N is the number of samples and D is the number of dimensions.
        mean (torch.Tensor): Mean of the normal distribution, shape (D,), or (N, D)
        Sigma (torch.Tensor | None): Covariance matrix of the normal distribution, shape (N, D, D), (N, D), (N,), (D, D), (D,), or a scalar.
        logSigma (torch.Tensor | None): Logarithm of the covariance matrix, same shape as Sigma.
        batch_first (bool): If True, indicates that the first dimension of Sigma is the batch size (N).

    Returns:
        torch.Tensor: Tensor containing the PDF values.
    """
    return torch.exp(log_norm_pdf(x, mean, Sigma, logSigma))

def gradient_reversal(
    x: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    '''
    Apply a gradient reversal layer to the input tensor. The forward pass is the identity function, but during backpropagation, the gradient is multiplied by -alpha.

    Args:
        x (torch.Tensor): Input tensor.
        alpha (float): Scaling factor for the gradient reversal. Default is 1.0.

    Returns:
        torch.Tensor: The input tensor with the gradient reversed during backpropagation.
    '''
    return ops.GradientReversalOp.apply(x, alpha)

def linear_kernel(
    x: torch.Tensor, y: torch.Tensor | None = None,
):
    '''
    Compute the linear kernel between two sets of tensors.

    A linear kernel is given by:

    .. math::
        K(x, y) = x^T y

    Args:
        x (torch.Tensor): First tensor, shape (M, D).
        y (torch.Tensor | None): Second tensor, shape (N, D), defaults to x.

    Returns:
        torch.Tensor: Tensor containing the linear kernel values, shape (M, N).
    '''
    if y is None:
        y = x
    return torch.einsum('ik,jk->ij', x, y)

def rbf_kernel(
    x: torch.Tensor, y: torch.Tensor | None = None, *,
    sigma: torch.Tensor | int | float | None = None,
    gamma: torch.Tensor | int | float | None = None,
    reduce: torch.Tensor | bool = False
):
    '''
    Compute the Radial Basis Function (RBF) kernel between two sets of tensors.

    The RBF kernel is given by:

    .. math::
        K(x, y) = \\exp(-\\gamma ||x - y||^2)

    or equivalently,

    .. math::
        K(x, y) = \\exp\\left(-\\frac{||x - y||^2}{2 \\sigma^2}\\right)

    Args:
        x (torch.Tensor): First tensor, shape (N, D), where N is the number of samples and D is the number of features.
        y (torch.Tensor | None): Second tensor, shape (M, D), where M is the number of samples and D is the number of features.
        sigma (torch.Tensor | int | float | None): Bandwidth parameter for the RBF kernel, scalar, or shape (K,), where K is the number of kernels.
        gamma (torch.Tensor | int | float | None): 1 / (2 * sigma^2) parameter for the RBF kernel, scalar, or shape (K,), where K is the number of kernels.
        reduce (torch.Tensor | bool): Whether to reduce the output.

            * If True, returns the mean of RBF kernel values under different bandwidths.
            * If False, returns the RBF kernel values for each bandwidth.
            * If a tensor, it should have shape (K,) and will be used as mean weight.

    Returns:
        torch.Tensor: Tensor containing the RBF kernel values, shape (N, M) or (K, N, M) if multiple kernels are used.
    '''
    # Sanity checks
    if sigma is not None and gamma is not None:
        raise ValueError('Only one of sigma or gamma should be provided.')
    if sigma is None and gamma is None:
        raise ValueError('Either sigma or gamma must be provided.')
    if gamma is None:
        assert sigma is not None
        gamma = 1 / (2 * sigma ** 2)
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=x.dtype, device=x.device)
    if y is None:
        y = x

    # Shape checks
    if x.shape[1] != y.shape[1]:
        raise ValueError(f'Input tensors must have the same number of features, but got {x.shape[1]} and {y.shape[1]}.')
    M, D = x.shape
    N, _ = y.shape

    squeeze = gamma.dim() == 0
    if squeeze:
        gamma = gamma.unsqueeze(0)  # Scalar gamma -> (1,)
    K, = gamma.shape

    # Build the squared distance matrix
    x, y = x.unsqueeze(0), y.unsqueeze(0) # (1, M, D), (1, N, D)
    norm_dist = (torch.cdist(x, y, p=2) ** 2).squeeze(0)  # (M, N)
    result = torch.exp(-torch.einsum('mn,k->kmn', norm_dist, gamma))

    # Reduce the result
    if reduce is True:
        return result.mean(dim=0)
    elif reduce is False:
        if squeeze:
            return result.squeeze(0)  # (M, N)
        return result
    elif isinstance(reduce, torch.Tensor):
        if reduce.shape != (K,):
            raise ValueError(f'Reduce tensor must have shape (K,), but got {reduce.shape}.')
        return torch.einsum('kmn,k->mn', result, reduce)
    raise ValueError(f'Reduce must be a boolean or a tensor, but got {type(reduce)}.')

def modified_bessel_kn(
    nu: int, x: torch.Tensor
):
    nu = abs(nu)

    k0 = torch.special.modified_bessel_k0(x)
    if nu == 0:
        return k0

    k1 = torch.special.modified_bessel_k1(x)
    if nu == 1:
        return k1

    km1, kn = k0, k1
    for m in range(1, nu):
        kp1 = km1 + (2.0 * m / x) * kn
        km1, kn = kn, kp1

    return kn

def matern_kernel(
    x: torch.Tensor, y: torch.Tensor | None = None, *,
    l: torch.Tensor | int | float = 1, n: int = 1
) -> torch.Tensor:
    '''
    Compute the Matérn kernel between two sets of tensors.

    The Matérn kernel is given by:

    .. math::
        K(x, y) = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)} \\left( \\sqrt{2\\nu} \\frac{||x - y||}{l} \\right)^\\nu K_\\nu \\left( \\sqrt{2\\nu} \\frac{||x - y||}{l} \\right)

    where :math:`K_\\nu` is the modified Bessel function of the second kind.

    Args:
        x (torch.Tensor): First tensor, shape (M, D).
        y (torch.Tensor | None): Second tensor, shape (M, D), defaults to x.
        l (torch.Tensor | int | float): Length scale parameter.
        n (int): Smoothness parameter for the Matérn kernel.
    '''
    # Input: a, b: (num_a, num_b, num_hidden)
    if y is None:
        y = x
    nu = torch.as_tensor(n).long().to(x.device)
    l = torch.as_tensor(l).to(x.device)
    log_l = torch.log(l)

    log_a = (2 * nu).log() / 2 + torch.norm(
        x.unsqueeze(1) - y.unsqueeze(0), dim=-1
    ).log() - log_l
    log_b = (1 - nu) * torch.log(x.new_tensor(2.0)) - \
        torch.lgamma(nu) + \
        log_a * nu + \
        modified_bessel_kn(n, log_a.exp()).log()
    return torch.where(log_a < -6, torch.zeros_like(log_b), log_b).exp()

def periodic_kernel(
    x: torch.Tensor,
    y: torch.Tensor | None = None, *,
    l: torch.Tensor | int | float = 1,
    p: torch.Tensor | int | float = 1
) -> torch.Tensor:
    '''
    Compute the periodic kernel between two sets of tensors.

    The periodic kernel is given by:

    .. math::
        K(x, y) = \\exp\\left(-\\frac{2 \\sin^2(\\pi ||x - y|| / p)}{l^2}\\right)

    Args:
        x (torch.Tensor): First tensor, shape (M, D).
        y (torch.Tensor): Second tensor, shape (N, D).
        l (torch.Tensor | int | float): Length scale parameter.
        p (torch.Tensor | int | float): Periodicity parameter.

    Returns:
        torch.Tensor: Tensor containing the periodic kernel values, shape (M, N).
    '''
    if y is None:
        y = x
    l = torch.as_tensor(l).to(x.device)
    p = torch.as_tensor(p).to(x.device)
    return torch.exp(
        -2 * torch.sin(torch.pi * torch.norm(
            x.unsqueeze(1) - y.unsqueeze(0),
            dim=-1
        ) / p) ** 2 / l ** 2
    )

def noise_kernel(
    x: torch.Tensor, y: torch.Tensor | None = None
) -> torch.Tensor:
    '''
    Compute the noise kernel between two sets of tensors.

    The noise kernel is given by:

    .. math::
        K(x, y) = \\delta(x, y)

    where :math:`\\delta` is the Kronecker delta function.

    Args:
        x (torch.Tensor): First tensor, shape (M, D).
        y (torch.Tensor | None): Second tensor, shape (N, D), defaults to x.

    Returns:
        torch.Tensor: Tensor containing the noise kernel values, shape (M, N).
    '''
    if y is None:
        y = x
    return torch.all(x.unsqueeze(1) == y.unsqueeze(0), dim=-1).float()

def mmd_distance(
    x: torch.Tensor, y: torch.Tensor, *,
    sigma: torch.Tensor | int | float | None = None,
    gamma: torch.Tensor | int | float | None = None,
    reduce: bool | torch.Tensor = False
) -> torch.Tensor:
    """
    Compute the Maximum Mean Discrepancy (MMD) distance between two sets of tensors.

    The MMD distance is given by:

    .. math::
        \\text{MMD}(x, y) = K(x, x) - 2 K(x, y) + K(y, y)

    Args:
        x (torch.Tensor): First tensor, shape (N, D), where N is the number of samples and D is the number of features.
        y (torch.Tensor): Second tensor, shape (M, D), where M is the number of samples and D is the number of features.
        sigma (torch.Tensor | int | float | None): Bandwidth parameter for the RBF kernel, scalar, or shape (K,), where K is the number of kernels.
        gamma (torch.Tensor | int | float | None): 1 / (2 * sigma^2) parameter for the RBF kernel, scalar, or shape (K,), where K is the number of kernels.
        reduce (bool | torch.Tensor): Whether to reduce the output.
            * If True, returns the mean MMD distance under different bandwidths.
            * If False, returns the MMD distance for each bandwidth.
            * If a tensor, it should have shape (K,) and will be used as mean weight.

    Returns:
        torch.Tensor: Tensor containing the MMD distance values, shape (K,) if reduce is False, or a scalar otherwise,
    """
    rbf_xx = rbf_kernel(x, x, sigma=sigma, gamma=gamma, reduce=reduce).mean(dim=(0, 1))
    rbf_yy = rbf_kernel(y, y, sigma=sigma, gamma=gamma, reduce=reduce).mean(dim=(0, 1))
    rbf_xy = rbf_kernel(x, y, sigma=sigma, gamma=gamma, reduce=reduce).mean(dim=(0, 1))

    return rbf_xx - 2 * rbf_xy + rbf_yy

def wasserstein_distance(
    x: torch.Tensor, y: torch.Tensor, p: float = 2.0, eps: float = 1e-6,
    wasser_iters: int = 20, wasser_eps: float = 1e-3
) -> torch.Tensor:
    """
    Compute the Wasserstein distance between two sets of tensors using the Sinkhorn algorithm.

    The Wasserstein distance is given by:

    .. math::
        W_p(x, y) = \\left( \\inf_{\\gamma \\in \\Gamma(x, y)} \\int ||x - y||^p d\\gamma(x, y) \\right)^{1/p}

    Args:
        x (torch.Tensor): First tensor, shape (N, D), where N is the number of samples and D is the number of features.
        y (torch.Tensor): Second tensor, shape (M, D), where M is the number of samples and D is the number of features.
        p (float): Order of the norm to use for the distance calculation.
        eps (float): Small value to avoid division by zero.
        wasser_iters (int): Number of iterations for the Sinkhorn algorithm.
        wasser_eps (float): Epsilon value for the Sinkhorn algorithm.

    Returns:
        torch.Tensor: Tensor containing the Wasserstein distance value.
    """
    # Sanity checks
    if p <= 0:
        raise ValueError('p must be greater than 0.')
    if x.shape[1] != y.shape[1]:
        raise ValueError(f'Input tensors must have the same number of features, but got {x.shape[1]} and {y.shape[1]}.')
    M, D = x.shape
    N, _ = y.shape


    cost = torch.cdist(x, y, p=p).pow(p)  # (M, N)
    cost = cost / cost.max().clamp(min=eps)  # Normalize cost
    log_K = -cost / wasser_eps
    log_u, log_v = torch.zeros(M, device=x.device), torch.zeros(N, device=y.device)
    log_a, log_b = log_u - math.log(M), log_v - math.log(N)

    # Sinkhorn iteration
    for _ in range(wasser_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    log_pi = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return (torch.exp(log_pi) * cost).sum().pow(1 / p)

def gaussian_process(
    x: torch.Tensor, # M, D
    x_obs: torch.Tensor, # N, D
    y_obs: torch.Tensor, # N,
    kernel_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Perform Gaussian Process regression to predict the mean and covariance of the function values at the input locations `x` based on the observed data `(x_obs, y_obs)` and a specified kernel function.

    Args:
        x (torch.Tensor): Input locations where predictions are to be made, shape (M, D).
        x_obs (torch.Tensor): Observed input locations, shape (N, D).
        y_obs (torch.Tensor): Observed function values at `x_obs`, shape (N,).
        kernel_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Kernel function (M, D), (N, D) -> (M, N) that computes the covariance between input locations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mu (torch.Tensor): Predicted mean of the function values at `x`, shape (M,).
            - sigma (torch.Tensor): Predicted covariance matrix of the function values at `x`, shape (M, M).
    '''

    if x.shape[1] != x_obs.shape[1]:
        raise ValueError(f'Input tensors must have the same number of features, but got {x.shape[1]} and {x_obs.shape[1]}.')
    if x_obs.shape[0] != y_obs.shape[0]:
        raise ValueError(f'Number of observations in x_obs and y_obs must match, but got {x_obs.shape[0]} and {y_obs.shape[0]}.')
    if y_obs.dim() > 2 or (y_obs.dim() == 2 and y_obs.shape[1] != 1):
        raise ValueError(f'y_obs must be a vector, but got shape {y_obs.shape}.')
    elif y_obs.dim() == 1:
        y_obs = y_obs.unsqueeze(1)

    sigma_21 = kernel_func(x, x_obs)  # (M, N)
    sigma_22 = kernel_func(x, x)  # (M, M)
    sigma_11 = kernel_func(x_obs, x_obs)  # (N, N)

    L = torch.linalg.cholesky(sigma_11)
    alpha = torch.cholesky_solve(y_obs, L).squeeze(-1)
    mu = sigma_21 @ alpha
    v = torch.linalg.solve_triangular(L, sigma_21.T, upper=False)
    sigma = sigma_22 - v.T @ v
    return mu, sigma

def sliding_window_mask(
    length: int, left_size: int | None, right_size: int | None, is_causal: bool = False,
    output_ndim: int = 2, device: torch.device | str | None = None,
    dtype: Literal['bool', 'float'] = 'bool'
) -> torch.Tensor:
    '''
    Generate a sliding window mask for a sequence of given length.

    Args:
        length (int): Length of the sequence.
        left_size (int | None): Size of the left context window. None means no sliding.
        right_size (int | None): Size of the right context window. None means no sliding.
        is_causal (bool): If True, the mask will be causal, meaning that each position can only attend to previous positions. If False, the mask will be bi-directional.
        output_ndim (int): The number of dimensions of the output mask. Default is 2. When output_ndim > 2, the mask will be broadcasted to (1, ..., 1, length, length).
        device (torch.device | str | None): The device on which to create the mask. If None, the mask will be created on the default device.
        dtype (Literal['bool', 'float']): The data type of the output mask.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (..., length, length). Masked positions are False or -inf, while unmasked positions are True or 0.
    '''
    if is_causal:
        right_size = 0
    x, y = torch.arange(length, device=device), torch.arange(length, device=device)
    x, y = x.unsqueeze(1), y.unsqueeze(0)
    match left_size, right_size:
        case None, None:
            ret = torch.ones(length, length, dtype=torch.bool, device=device)
        case None, _:
            ret = (y <= x + right_size)
        case _, None:
            ret = (y >= x - left_size)
        case _, _:
            ret = (y >= x - left_size) & (y <= x + right_size)
    if dtype == 'float':
        masked_out = ~ret
        ret = masked_out.to(torch.float32)
        ret[masked_out] = float('-inf')
    return ret.reshape(*([1] * (output_ndim - 2)), length, length)
