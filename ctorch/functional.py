'''
`utils.ctorch.functional` - Functional utilities for PyTorch tensors.
'''
import math

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
