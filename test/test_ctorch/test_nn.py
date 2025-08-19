import unittest

import torch

from ctorch.nn import *

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


class TestMonotonicLinear(unittest.TestCase):
    def test_monotonic_linear(self):
        for func in {'relu', 'softplus', 'sigmoid', 'elu', 'abs', 'square'}:
            with self.subTest(func=func):
                layer = MonotonicLinear(10, 5, non_neg_func=func)
                x = torch.rand(10, 10)
                x_2 = torch.rand(10, 10) + 2
                output = layer(x_2) - layer(x)
                self.assertTrue(torch.all(output >= 0), "Output should be non-decreasing with respect to input.")

class TestTemporalEmbedding(unittest.TestCase):
    def test_sinusoidal_temporal_embedding(self):
        import numpy as np
        class PositionalEncoding(torch.nn.Module):

            def __init__(self, d_hid, n_position=200):
                super(PositionalEncoding, self).__init__()

                # Not a parameter
                self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

            def _get_sinusoid_encoding_table(self, n_position, d_hid):
                ''' Sinusoid position encoding table '''
                # TODO: make it with torch instead of numpy

                def get_position_angle_vec(position):
                    return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

                sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
                sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
                sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

                return torch.FloatTensor(sinusoid_table).unsqueeze(0)

            def forward(self, x):
                return x + self.pos_table[:, :x.size(1)].clone().detach() # type: ignore

        teacher = PositionalEncoding(128, 200)
        student = SinusoidalTemporalEmbedding(128)
        x_tensor = torch.randn(10, 200, 128)
        input_tensor = torch.arange(0, 200).unsqueeze(0).expand(10, -1).float() # Batch size of 10
        self.assertTrue(torch.allclose(
            teacher(x_tensor) - x_tensor,
            student(input_tensor), atol=1e-4, rtol=1e-6
        ))

class TestFeatureEmbedding(unittest.TestCase):
    def test_feature_embedding_uniform(self):
        feature_sizes = [3, 4, 5]
        embedding_dim = 6
        batch_size = 10
        embedding = FeatureEmbedding(feature_sizes, embedding_dim)

        x = torch.randint(0, 2, (batch_size, len(feature_sizes)))
        output = embedding(x)
        self.assertEqual(output.shape, (batch_size, len(feature_sizes) * embedding_dim))

    def test_feature_embedding_different_dims(self):
        feature_sizes = [3, 4, 5]
        embedding_dims = [2, 3, 4]
        batch_size = 10
        embedding = FeatureEmbedding(feature_sizes, embedding_dims)

        x = torch.randint(0, 2, (batch_size, len(feature_sizes)))
        output = embedding(x)
        self.assertEqual(output.shape, (batch_size, sum(embedding_dims)))

    def test_out_of_bound_error(self):
        feature_sizes = [3, 4, 5]
        embedding_dim = 6
        batch_size = 10
        embedding = FeatureEmbedding(feature_sizes, embedding_dim)

        x = torch.arange(batch_size).unsqueeze(1).expand(-1, len(feature_sizes))
        with self.assertRaises(ValueError):
            _ = embedding(x)
