
import random
import unittest

import torch
from ctorch.padding import pad_packed_sequence_right, pack_padded_sequence_right, unpad_sequence_right


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
