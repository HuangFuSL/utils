
import random
import unittest

import torch
from ctorch.padding import *
from ctorch.nn import Module

class TestPadding(unittest.TestCase):
    def test_pad_packed_sequence_right(self):
        for _ in range(16):
            hidden_dim = random.randint(32, 256)
            num_sequences = random.randint(16, 128)
            lengths = torch.randint(32, 256, (num_sequences,), dtype=torch.long)
            lengths = lengths.sort(descending=True).values
            sequences = []
            for length in lengths.tolist():
                sequences.append(torch.randn(length, hidden_dim))
            packed = torch.nn.utils.rnn.pack_sequence(sequences)
            with self.subTest(hidden_dim=hidden_dim, num_sequences=num_sequences):
                padded, _length = pad_packed_sequence_right(packed, batch_first=True)
                # Check length
                self.assertEqual(_length.shape, (num_sequences,))
                self.assertTrue(torch.all(_length == lengths))
                # Check shape
                self.assertEqual(padded.shape, (num_sequences, lengths[0], hidden_dim))
                # Check values
                for i, seq in enumerate(sequences):
                    self.assertTrue(torch.all(padded[i, -lengths[i]:] == seq))
                    self.assertTrue(torch.all(padded[i, :-lengths[i]] == 0))

    def test_pack_padded_sequence_right(self):
        for _ in range(16):
            hidden_dim = random.randint(32, 256)
            num_sequences = random.randint(16, 128)
            lengths = torch.randint(32, 256, (num_sequences,), dtype=torch.long)
            lengths = lengths.sort(descending=True).values
            sequences = []
            for length in lengths.tolist():
                sequences.append(torch.randn(length, hidden_dim))
            packed = torch.nn.utils.rnn.pack_sequence(sequences)
            with self.subTest(hidden_dim=hidden_dim, num_sequences=num_sequences):
                padded, _length = pad_packed_sequence_right(
                    packed, batch_first=True
                )
                re_packed = pack_padded_sequence_right(
                    padded, _length, batch_first=True
                )
                unpacked = torch.nn.utils.rnn.unpack_sequence(re_packed)
                for old, new in zip(sequences, unpacked):
                    self.assertTrue(torch.allclose(old, new))


    def test_unpad_sequence_right(self):
        for _ in range(16):
            hidden_dim = random.randint(32, 256)
            num_sequences = random.randint(16, 128)
            lengths = torch.randint(32, 256, (num_sequences,), dtype=torch.long)
            lengths = lengths.sort(descending=True).values
            sequences = []
            for length in lengths.tolist():
                sequences.append(torch.randn(length, hidden_dim))
            padded = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_side='left'
            )
            with self.subTest(hidden_dim=hidden_dim, num_sequences=num_sequences):
                unpadded = unpad_sequence_right(padded, lengths, batch_first=True)
                for old, new in zip(sequences, unpadded):
                    self.assertTrue(torch.allclose(old, new))

class TestPackedOps(unittest.TestCase):
    def setUp(self):
        self.B = 64
        self.D = 32
        self.max_len = 128
        lengths = torch.randint(1, self.max_len + 1, (self.B,), dtype=torch.long)
        self.packed_a = torch.nn.utils.rnn.pack_padded_sequence(
            torch.randn(self.B, self.max_len, self.D),
            lengths=lengths,
            batch_first=True, enforce_sorted=False
        )
        self.packed_b = torch.nn.utils.rnn.pack_padded_sequence(
            torch.randn(self.B, self.max_len, self.D),
            lengths=lengths,
            batch_first=True, enforce_sorted=False
        )
        self.mask = get_key_padding_mask_left(lengths)
        self.padded_a, _ = torch.nn.utils.rnn.pad_packed_sequence(
            self.packed_a, batch_first=True, padding_value=0.0
        )
        self.padded_b, _ = torch.nn.utils.rnn.pad_packed_sequence(
            self.packed_b, batch_first=True, padding_value=0.0
        )

    def test_packed_forward(self):
        class _module(Module):
            def __init__(_self):
                super().__init__()
                _self.linear = torch.nn.Linear(self.D, self.D * 2)

            def forward(_self, x):
                return _self.linear(x)

        model = _module()
        model.eval()
        apply_then_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_forward(model, self.packed_a), batch_first=True
        )
        pad_then_apply = model(self.padded_a)
        self.assertTrue(torch.allclose(apply_then_pad[self.mask], pad_then_apply[self.mask]))

    def test_packed_unary(self):
        apply_then_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_unary_op(torch.sigmoid, self.packed_a), batch_first=True
        )
        pad_then_apply = torch.sigmoid(self.padded_a)
        self.assertTrue(torch.allclose(apply_then_pad[self.mask], pad_then_apply[self.mask]))

    def test_packed_binary(self):
        apply_then_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_binary_op(torch.add, self.packed_a, self.packed_b), batch_first=True
        )
        pad_then_apply = self.padded_a + self.padded_b
        self.assertTrue(torch.allclose(apply_then_pad[self.mask], pad_then_apply[self.mask]))

    def test_packed_concat(self):
        apply_then_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_concat([self.packed_a, self.packed_b]), batch_first=True
        )
        pad_then_apply = torch.cat((self.padded_a, self.padded_b), dim=-1)
        self.assertTrue(torch.allclose(apply_then_pad, pad_then_apply))
