import random
import unittest

import torch

from ctorch.nn import Module

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
