import unittest

import torch

from ctorch.nn import Module, GradientReversalLayer

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class TestModule(unittest.TestCase):
    def test_Module(self):
        class _module(Module):
            def __init__(self):
                super().__init__()

        module = _module()
        self.assertIsInstance(module, Module)
        self.assertEqual(module.device.type, 'cpu')
        module.to(device)
        self.assertEqual(module.device.type, device.type)
        module.to('cpu')
        self.assertEqual(module.device.type, 'cpu')

class TestGradientReversalLayer(unittest.TestCase):
    def test_GradientReversalLayer(self):
        layer = GradientReversalLayer()

        x = torch.randn(10, 5, device=device, requires_grad=True)
        output = layer(x).sum()
        output.backward()

        self.assertTrue(x.grad is not None)
        assert x.grad is not None
        self.assertTrue(torch.allclose(x.grad, -torch.ones_like(x.grad)))
