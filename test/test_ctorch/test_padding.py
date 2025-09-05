
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

class TestMaskedSelect(unittest.TestCase):
    def test_basic_tensor_with_feature_dim(self):
        values_list = [
            [[1., 10.], [2., 20.], [3., 30.]],
            [[4., 40.], [5., 50.], [0.,  0.]],
        ]
        mask_list = [
            [True, False, True],
            [False, True, False],
        ]
        values = torch.tensor(values_list, dtype=torch.float32)
        mask = torch.tensor(mask_list, dtype=torch.bool)

        out = masked_select(values, mask)
        out_padded, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        self.assertEqual(out_len.tolist(), [2, 1])
        self.assertEqual(
            out_padded.tolist(),
            [
                [[1.0, 10.0], [3.0, 30.0]],
                [[5.0, 50.0], [0.0,  0.0]],
            ]
        )

    def test_basic_tensor_no_feature_dim(self):
        # values: (B, L)
        values = torch.tensor([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
        ], dtype=torch.float32)
        mask = torch.tensor([
            [True, False, True, False],
            [False, True, False, True],
        ], dtype=torch.bool)

        out = masked_select(values, mask)
        out_padded, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        self.assertEqual(out_len.tolist(), [2, 2])
        self.assertEqual(
            out_padded.tolist(),
            [
                [1.0, 3.0],
                [6.0, 8.0],
            ]
        )

    def test_packed_inputs_both_packed(self):
        values_padded = torch.tensor([
            [[1., 10.], [2., 20.], [3., 30.], [4., 40.]],
            [[5., 50.], [6., 60.], [0.,  0.], [0.,  0.]],
        ], dtype=torch.float32)
        lengths = torch.tensor([4, 2], dtype=torch.long)
        values_packed = torch.nn.utils.rnn.pack_padded_sequence(values_padded, lengths, batch_first=True, enforce_sorted=False)

        mask_padded = torch.tensor([
            [True, True, False, False],
            [True, False, False, False],
        ], dtype=torch.bool)
        mask_packed = torch.nn.utils.rnn.pack_padded_sequence(mask_padded, lengths, batch_first=True, enforce_sorted=False)

        out = masked_select(values_packed, mask_packed)
        out_padded, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        self.assertEqual(out_len.tolist(), [2, 1])
        self.assertEqual(
            out_padded.tolist(),
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[5.0, 50.0], [0.0,  0.0]],
            ]
        )

    def test_mask_longer_than_values(self):
        values = torch.tensor([
            [1., 2., 3.],
            [4., 5., 6.],
        ], dtype=torch.float32)
        mask = torch.tensor([
            [True, False, True, True, True],
            [False, False, True, False, False],
        ], dtype=torch.bool)

        out = masked_select(values, mask)
        out_padded, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        self.assertEqual(out_len.tolist(), [2, 1])
        self.assertEqual(
            out_padded.tolist(),
            [
                [1.0, 3.0],
                [6.0, 0.0],
            ]
        )

    def test_mask_shorter_than_values(self):
        values = torch.tensor([
            [[1., 10.], [2., 20.], [3., 30.], [4., 40.]],
            [[5., 50.], [6., 60.], [7., 70.], [8., 80.]],
        ], dtype=torch.float32)
        mask = torch.tensor([
            [False, True],
            [True,  False],
        ], dtype=torch.bool)

        out = masked_select(values, mask)
        out_padded, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        self.assertEqual(out_len.tolist(), [1, 1])
        self.assertEqual(
            out_padded.tolist(),
            [
                [[2.0, 20.0]],
                [[5.0, 50.0]],
            ]
        )

    def test_raises_on_zero_length_after_mask(self):
        values = torch.tensor([
            [1., 2., 3.],
            [4., 5., 6.],
        ], dtype=torch.float32)
        mask = torch.tensor([
            [False, False, False],
            [True,  False, False],
        ], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'zero length'):
            _ = masked_select(values, mask)

    def test_gradient_flow_tensor_input(self):
        values = torch.tensor([
            [1., 2., 3.],
            [4., 5., 6.],
        ], dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([
            [True, False, True],
            [False, True, False],
        ], dtype=torch.bool)

        out = masked_select(values, mask)
        loss = out.data.sum()
        loss.backward()

        expected_grad = torch.tensor([
            [1., 0., 1.],
            [0., 1., 0.],
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(values.grad, expected_grad))

    def test_cross_device_optional(self):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            return
        values = torch.tensor([
            [[1., 10.], [2., 20.], [3., 30.]],
            [[4., 40.], [5., 50.], [0.,  0.]],
        ], dtype=torch.float32, device=device)
        mask = torch.tensor([
            [True, False, True],
            [False, True, False],
        ], dtype=torch.bool, device=device)

        out = masked_select(values, mask)
        out_padded, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        self.assertEqual(out_len.tolist(), [2, 1])
        self.assertTrue(out_padded.device.type == device)
        self.assertEqual(
            out_padded.cpu().tolist(),
            [
                [[1.0, 10.0], [3.0, 30.0]],
                [[5.0, 50.0], [0.0,  0.0]],
            ]
        )

class TestPrependLeft(unittest.TestCase):
    def make_packed_from_lengths(
        self, lengths, feat_shape=(4,), dtype=torch.float32,
        device="cpu", enforce_sorted=True
    ):
        torch.manual_seed(0)
        B = len(lengths)
        max_len = max(lengths)
        total = max_len * B * int(torch.tensor(feat_shape).prod().item())
        base = torch.arange(total, dtype=dtype, device=device)
        padded = base.view(max_len, B, *feat_shape).clone()

        for b, L in enumerate(lengths):
            if L < max_len:
                padded[L:, b] = 0

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths=lengths, batch_first=False, enforce_sorted=enforce_sorted
        )
        return packed, padded

    def setUp(self):
        self.devices = ["cpu"]
        if torch.cuda.is_available():
            self.devices.append("cuda")
        if torch.backends.mps.is_available():
            self.devices.append("mps")

    def test_basic_prepend_zero_padding(self):
        for device in self.devices:
            lengths = [5, 3, 2]  # B=3
            ps, _ = self.make_packed_from_lengths(lengths, feat_shape=(4,), dtype=torch.float32, device=device)
            B = int(ps.batch_sizes[0].item())
            self.assertEqual(B, 3)

            k = 2
            new_ps = prepend_left(ps, append_value=0.0, num_steps=k)
            expected_bs = torch.cat([
                torch.full((k,), B, dtype=ps.batch_sizes.dtype, device=ps.batch_sizes.device),
                ps.batch_sizes
            ], dim=0)
            self.assertTrue(torch.equal(new_ps.batch_sizes, expected_bs))

            self.assertEqual(new_ps.data.size(0), ps.data.size(0) + k * B)
            self.assertTrue(torch.all(new_ps.data[:k*B] == 0))
            torch.testing.assert_close(new_ps.data[k*B:], ps.data)

            self.assertIs(new_ps.sorted_indices, ps.sorted_indices)
            self.assertIs(new_ps.unsorted_indices, ps.unsorted_indices)

            self.assertEqual(int(new_ps.batch_sizes.sum().item()), new_ps.data.size(0))

            self.assertEqual(new_ps.data.dtype, ps.data.dtype)
            self.assertEqual(new_ps.data.device, ps.data.device)
            self.assertEqual(new_ps.batch_sizes.device, ps.batch_sizes.device)

    def test_nonzero_padding_value(self):
        for device in self.devices:
            lengths = [4, 4, 4]
            ps, _ = self.make_packed_from_lengths(
                lengths, feat_shape=(3,), dtype=torch.float32, device=device
            )
            B = int(ps.batch_sizes[0].item())
            val = 5.5
            k = 1
            new_ps = prepend_left(ps, append_value=val, num_steps=k)

            self.assertTrue(torch.all(new_ps.data[:k*B] == val))
            torch.testing.assert_close(new_ps.data[k*B:], ps.data)

    def test_zero_steps_returns_same_object(self):
        lengths = [3, 2]
        ps, _ = self.make_packed_from_lengths(lengths)
        self.assertIs(prepend_left(ps, num_steps=0), ps)

    def test_negative_steps_raises(self):
        lengths = [3, 2, 1]
        ps, _ = self.make_packed_from_lengths(lengths)
        with self.assertRaises(ValueError):
            prepend_left(ps, num_steps=-1)

    def test_indices_preserved_when_unsorted(self):
        lengths = [3, 5, 2]
        ps, _ = self.make_packed_from_lengths(lengths, enforce_sorted=False)
        self.assertIsNotNone(ps.sorted_indices)
        self.assertIsNotNone(ps.unsorted_indices)

        new_ps = prepend_left(ps, num_steps=3)
        self.assertIs(new_ps.sorted_indices, ps.sorted_indices)
        self.assertIs(new_ps.unsorted_indices, ps.unsorted_indices)

    def test_multi_feature_dims(self):
        lengths = [5, 3, 2]
        feat_shape = (2, 3)
        ps, _ = self.make_packed_from_lengths(lengths, feat_shape=feat_shape, dtype=torch.float32)
        B = int(ps.batch_sizes[0].item())
        k = 2
        new_ps = prepend_left(ps, append_value=0.0, num_steps=k)

        self.assertEqual(new_ps.data.shape, (ps.data.shape[0] + k * B, *feat_shape))
        self.assertTrue(torch.all(new_ps.data[:k*B] == 0.0))
        torch.testing.assert_close(new_ps.data[k*B:], ps.data)

    def test_integer_dtype_padding(self):
        lengths = [4, 2]
        ps, _ = self.make_packed_from_lengths(lengths, feat_shape=(3,), dtype=torch.long)
        B = int(ps.batch_sizes[0].item())
        k = 1
        val = 7
        new_ps = prepend_left(ps, append_value=val, num_steps=k)

        self.assertEqual(new_ps.data.dtype, torch.long)
        self.assertTrue(torch.all(new_ps.data[:k*B] == val))
        self.assertTrue(torch.equal(new_ps.data[k*B:], ps.data))

class TestAppendRight(unittest.TestCase):
    def make_packed_from_lengths(
        self, lengths, feat_shape=(4,), dtype=torch.float32,
        device="cpu", enforce_sorted=True
    ):
        torch.manual_seed(0)
        B = len(lengths)
        max_len = max(lengths)
        total = max_len * B * int(torch.tensor(feat_shape).prod().item())
        base = torch.arange(total, dtype=dtype, device=device)
        padded = base.view(max_len, B, *feat_shape).clone()

        for b, L in enumerate(lengths):
            if L < max_len:
                padded[L:, b] = 0

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths=lengths, batch_first=False, enforce_sorted=enforce_sorted
        )
        return packed, padded
    def setUp(self):
        self.devices = ["cpu"]
        if torch.cuda.is_available():
            self.devices.append("cuda")
        if torch.backends.mps.is_available():
            self.devices.append("mps")

    def trivial_append(self, seq, value, num_steps, enforce_sorted=True):
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            seq, batch_first=True, padding_value=value
        )
        B, _, *H = padded.size()
        padded = torch.cat([
            padded,
            padded.new_full((B, num_steps, *H), value)
        ], dim=1)
        lengths = lengths + num_steps
        return torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )

    def check_packed_sequence(self, a: PackedSequence, b: PackedSequence):
        self.assertTrue(torch.equal(a.batch_sizes, b.batch_sizes))
        self.assertTrue(torch.allclose(a.data, b.data))
        self.assertIs(a.sorted_indices, b.sorted_indices)
        self.assertIs(a.unsorted_indices, b.unsorted_indices)

    def test_basic_append_zero_padding(self):
        for device in self.devices:
            lengths = [5, 3, 2]  # B=3
            ps, _ = self.make_packed_from_lengths(lengths, feat_shape=(4,), device=device)
            B = int(ps.batch_sizes[0].item())
            self.assertEqual(B, 3)

            k = 2
            test = append_right(ps, append_value=0.0, num_steps=k)
            target = self.trivial_append(ps, value=0.0, num_steps=k)
            self.check_packed_sequence(test, target)

    def test_nonzero_padding_value(self):
        for device in self.devices:
            lengths = [4, 4, 4]
            ps, _ = self.make_packed_from_lengths(lengths, feat_shape=(3,), device=device)
            val = 5.5
            k = 2
            test = append_right(ps, append_value=val, num_steps=k)
            target = self.trivial_append(ps, value=val, num_steps=k)
            self.check_packed_sequence(test, target)

    def test_zero_steps_returns_same_object(self):
        lengths = [3, 2]
        ps, _ = self.make_packed_from_lengths(lengths)
        self.assertIs(append_right(ps, num_steps=0), ps)

    def test_negative_steps_raises(self):
        lengths = [3, 2, 1]
        ps, _ = self.make_packed_from_lengths(lengths)
        with self.assertRaises(ValueError):
            append_right(ps, num_steps=-1)

    def test_indices_preserved_when_unsorted(self):
        lengths = [3, 5, 2]
        ps, _ = self.make_packed_from_lengths(lengths, enforce_sorted=False)
        self.assertIsNotNone(ps.sorted_indices)
        self.assertIsNotNone(ps.unsorted_indices)

        new_ps = append_right(ps, num_steps=3)
        self.assertIs(new_ps.sorted_indices, ps.sorted_indices)
        self.assertIs(new_ps.unsorted_indices, ps.unsorted_indices)

    def test_multi_feature_dims(self):
        lengths = [5, 3, 2]
        feat_shape = (2, 3)
        ps, _ = self.make_packed_from_lengths(lengths, feat_shape=feat_shape, dtype=torch.float32)
        k = 2
        test = append_right(ps, append_value=0.0, num_steps=k)
        target = self.trivial_append(ps, value=0.0, num_steps=k)
        self.check_packed_sequence(test, target)

    def test_integer_dtype_padding(self):
        lengths = [4, 2]
        ps, _ = self.make_packed_from_lengths(lengths, feat_shape=(3,), dtype=torch.long)
        k = 1
        val = 7
        test = append_right(ps, append_value=val, num_steps=k)
        target = self.trivial_append(ps, value=val, num_steps=k)
        self.check_packed_sequence(test, target)
