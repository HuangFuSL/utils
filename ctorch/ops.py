'''
ops.py - Utilities Operators for PyTorch tensors

Author: HuangFuSL
Date: 2025-07-28
'''

from typing import TYPE_CHECKING

import torch


class GradientReversalOp(torch.autograd.Function):

    if TYPE_CHECKING:
        @classmethod
        def apply(cls, x: torch.Tensor, alpha: float) -> torch.Tensor:
            """
            Apply the gradient reversal operation.
            """
            ...

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
