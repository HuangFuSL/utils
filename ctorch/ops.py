'''
`utils.ctorch.ops` - Utilities Operators for PyTorch tensors
'''

from typing import TYPE_CHECKING

import torch


class GradientReversalOp(torch.autograd.Function):
    '''
    Gradient reversal operation for adversarial training.
    '''

    if TYPE_CHECKING:
        @classmethod
        def apply(cls, x: torch.Tensor, alpha: float) -> torch.Tensor:
            """
            Apply the gradient reversal operation.
            """
            ...

    @staticmethod
    def forward(ctx, x, alpha):
        '''
        Forward pass for the gradient reversal operation.

        Args:
            x (torch.Tensor): Input tensor.
            alpha (float): Scaling factor for the gradient reversal.

        Returns:
            torch.Tensor: The input tensor with the gradient reversal applied.
        '''
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Backward pass for the gradient reversal operation.

        Args:
            grad_output (torch.Tensor): Gradient from the next layer.

        Returns:
            torch.Tensor: The gradient with the reversal applied.
        '''
        output = grad_output.neg() * ctx.alpha

        return output, None
