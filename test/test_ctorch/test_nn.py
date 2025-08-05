import unittest

import torch

from ctorch.nn import Module, GradientReversalLayer, Activation

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

    def test_num_parameters(self):
        class _module(Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

        module = _module()
        self.assertEqual(module.num_parameters, 10 * 5 + 5)

    def test_nested_num_parameters(self):
        class _module1(Module):
            def __init__(self, a, b):
                super().__init__()
                self.linear1 = torch.nn.Linear(a, b)

        class _module2(Module):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(5, 2)
                self.module = _module1(10, 5)

        module = _module2()
        self.assertEqual(module.num_parameters, 5 * 2 + 2 + 10 * 5 + 5)

class TestGradientReversalLayer(unittest.TestCase):
    def test_GradientReversalLayer(self):
        layer = GradientReversalLayer()

        x = torch.randn(10, 5, device=device, requires_grad=True)
        output = layer(x).sum()
        output.backward()

        self.assertTrue(x.grad is not None)
        assert x.grad is not None
        self.assertTrue(torch.allclose(x.grad, -torch.ones_like(x.grad)))


class TestActivation(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(1000, 10)

    def test_relu(self):
        activation = Activation('relu')
        self.assertTrue(torch.allclose(activation(self.x), torch.relu(self.x)))

    def test_softmax(self):
        activation = Activation('softmax', dim=1)
        self.assertTrue(torch.allclose(activation(self.x), torch.softmax(self.x, dim=1)))
