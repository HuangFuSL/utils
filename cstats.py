'''
cstats.py

Author: HuangFuSL
Date: 2025-06-30

This module provides functions related to distribution calculations in PyTorch.
'''

import torch
import torch.linalg

def log_norm_pdf(
    x: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor | None = None,
    logSigma: torch.Tensor | None = None, batch_first: bool | None = None
) -> torch.Tensor:
    """
    Calculate the log probability density function of a normal distribution.

    Parameters:
    - x: Input tensor, shape (N, D), where N is the number of samples and D is the number of dimensions.
    - mean: Mean of the normal distribution, shape (D,), or (N, D)
    - Sigma: Covariance matrix of the normal distribution, shape (N, D, D), (N, D), (N,), (D, D), (D,), or a scalar.
    - logSigma: Logarithm of the covariance matrix, same shape as Sigma.
    - batch_first: If True, indicates that the first dimension of Sigma is the batch size (N).

    Returns:
    - Tensor containing the log PDF values.
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

    Parameters:
    - x: Input tensor, shape (N, D), where N is the number of samples and D is the number of dimensions.
    - mean: Mean of the normal distribution, shape (D,), or (N, D)
    - Sigma: Covariance matrix of the normal distribution, shape (N, D, D), (N, D), (N,), (D, D), (D,), or a scalar.
    - logSigma: Logarithm of the covariance matrix, same shape as Sigma.
    - batch_first: If True, indicates that the first dimension of Sigma is the batch size (N).

    Returns:
    - Tensor containing the PDF values.
    """
    return torch.exp(log_norm_pdf(x, mean, Sigma, logSigma))
