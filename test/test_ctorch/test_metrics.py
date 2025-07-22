import time
import unittest

import torch
import numpy as np
from sklearn import metrics as sk_metrics

from ctorch import metrics

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class TestAUC(unittest.TestCase):
    def test_auc(self):
        for _ in range(32):
            # Generate random binary labels and scores
            y_true = torch.randint(0, 2, (100,))
            y_score = torch.rand(100)
            sk_auc_value = sk_metrics.roc_auc_score(y_true.numpy(), y_score.numpy())

            with self.subTest(num_pass=_):
                # Compute AUC using the custom function
                auc_value = metrics.auc_score(y_true, y_score)

                # Compute AUC using sklearn for comparison

                # Check if the values are close enough
                self.assertAlmostEqual(auc_value, sk_auc_value, places=5)

    def test_auc_large(self):
        # Test with larger tensors
        np_data = [(
            np.random.randint(0, 2, (1000000,)),
            np.random.rand(1000000).astype(np.float32)
        ) for _ in range(32)]
        sk_auc_values = []
        start_time = time.time()
        for y_true, y_score in np_data:
            sk_auc_values.append(sk_metrics.roc_auc_score(y_true, y_score))
        end_time = time.time()
        sk_time = end_time - start_time

        torch_data = [
            (torch.tensor(y_true, device=device), torch.tensor(y_score, device=device))
            for y_true, y_score in np_data
        ]
        torch_auc_values = []
        start_time = time.time()
        for y_true, y_score in torch_data:
            torch_auc_values.append(metrics.auc_score(y_true, y_score))
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"sklearn AUC time: {sk_time:.4f}s, Custom AUC time: {custom_time:.4f}s")
        self.assertLess(custom_time, sk_time, "Custom AUC should be faster than sklearn AUC")
        for sk_value, custom_value in zip(sk_auc_values, torch_auc_values):
            self.assertAlmostEqual(custom_value, sk_value, places=4)

    def test_auc_shape_mismatch(self):
        y_true = torch.randint(0, 2, (100,))
        y_score = torch.rand(99)
        with self.assertRaises(ValueError):
            metrics.auc_score(y_true, y_score)

    def test_auc_no_positive(self):
        y_true = torch.zeros(100, dtype=torch.int)
        y_score = torch.rand(100)
        auc_value = metrics.auc_score(y_true, y_score)
        self.assertEqual(auc_value, 0.0)

    def test_auc_no_negative(self):
        y_true = torch.ones(100, dtype=torch.int)
        y_score = torch.rand(100)
        auc_value = metrics.auc_score(y_true, y_score)
        self.assertEqual(auc_value, 0.0)

    def test_auc_stream(self):
        # Test with larger tensors
        get_score = lambda x: (x, (np.random.rand(10000) * 2 + x * 2 - 1).astype(np.float32))
        np_data = [
            [
                get_score(np.random.randint(0, 2, (10000,)))
                for batch in range(100)
            ] for epochs in range(32)
        ]
        sk_auc_values = []
        start_time = time.time()
        for epoch in np_data:
            y_trues, y_scores = zip(*epoch)
            full_y_true = np.concatenate(y_trues)
            full_y_score = np.concatenate(y_scores)
            sk_auc_values.append(sk_metrics.roc_auc_score(full_y_true, full_y_score))
        end_time = time.time()
        sk_time = end_time - start_time

        torch_data = [
            [(torch.tensor(y_true, device=device), torch.tensor(y_score, device=device))
            for y_true, y_score in epoch]
            for epoch in np_data
        ]
        torch_auc_values = []
        start_time = time.time()
        for epoch in torch_data:
            auc_accumulator = metrics.BatchedAUC(nbins=1000, device=device, logit=True)
            for y_true, y_score in epoch:
                auc_accumulator(y_true, y_score)
            torch_auc_values.append(auc_accumulator.finalize())
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"sklearn AUC time: {sk_time:.4f}s, Custom AUC time: {custom_time:.4f}s")
        self.assertLess(custom_time, sk_time, "Custom AUC should be faster than sklearn AUC")
        for sk_value, custom_value in zip(sk_auc_values, torch_auc_values):
            self.assertAlmostEqual(custom_value, sk_value, places=5)

class TestNDCG(unittest.TestCase):
    def test_ndcg(self):
        for _ in range(32):
            # Generate random binary labels and scores
            y_true = torch.randint(0, 2, (10000,)).reshape(100, 100)
            y_score = torch.rand(10000).reshape(100, 100)
            k = 10
            sk_ndcg_value = sk_metrics.ndcg_score(y_true.numpy(), y_score.numpy(), k=k)

            with self.subTest(num_pass=_):
                # Compute NDCG using the custom function
                ndcg_value = metrics.ndcg_score(y_true, y_score, k)

                # Check if the values are close enough
                self.assertAlmostEqual(ndcg_value, sk_ndcg_value, places=5)

    def test_ndcg_large(self):
        # Test with larger tensors
        np_data = [(
            np.random.randint(0, 2, (1000000,)).reshape(1000, 1000),
            np.random.rand(1000000).astype(np.float32).reshape(1000, 1000)
        ) for _ in range(32)]
        k = 10
        sk_ndcg_values = []
        start_time = time.time()
        for y_true, y_score in np_data:
            sk_ndcg_values.append(sk_metrics.ndcg_score(y_true, y_score, k=k))
        end_time = time.time()
        sk_time = end_time - start_time

        torch_data = [
            (torch.tensor(y_true, device=device), torch.tensor(y_score, device=device))
            for y_true, y_score in np_data
        ]
        torch_ndcg_values = []
        start_time = time.time()
        for y_true, y_score in torch_data:
            torch_ndcg_values.append(metrics.ndcg_score(y_true, y_score, k))
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"sklearn NDCG time: {sk_time:.4f}s, Custom NDCG time: {custom_time:.4f}s")
        self.assertLess(custom_time, sk_time, "Custom NDCG should be faster than sklearn NDCG")
        for sk_value, custom_value in zip(sk_ndcg_values, torch_ndcg_values):
            self.assertAlmostEqual(custom_value, sk_value, places=4)

    def test_ndcg_stream(self):
        # Test with larger tensors
        get_score = lambda x: (x, (np.random.rand(*x.shape) * 2 + x * 0.2 - 1).astype(np.float32))
        np_data = [
            [
                get_score(np.random.randint(0, 2, (100, 100)))
                for batch in range(100)
            ] for epochs in range(32)
        ]
        k = 10
        sk_ndcg_values = []
        start_time = time.time()
        for epoch in np_data:
            y_trues, y_scores = zip(*epoch)
            full_y_true = np.concatenate(y_trues).reshape(-1, 100)
            full_y_score = np.concatenate(y_scores).reshape(-1, 100)
            sk_ndcg_values.append(sk_metrics.ndcg_score(full_y_true, full_y_score, k=k))
        end_time = time.time()
        sk_time = end_time - start_time

        torch_data = [
            [(torch.tensor(y_true, device=device), torch.tensor(y_score, device=device))
            for y_true, y_score in epoch]
            for epoch in np_data
        ]
        torch_ndcg_values = []
        start_time = time.time()
        for epoch in torch_data:
            ndcg_accumulator = metrics.BatchedNDCG(k=k)
            for y_true, y_score in epoch:
                ndcg_accumulator(y_true, y_score)
            torch_ndcg_values.append(ndcg_accumulator.finalize())
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"sklearn NDCG time: {sk_time:.4f}s, Custom NDCG time: {custom_time:.4f}s")
        self.assertLess(custom_time, sk_time, "Custom NDCG should be faster than sklearn NDCG")
        for sk_value, custom_value in zip(sk_ndcg_values, torch_ndcg_values):
            self.assertAlmostEqual(custom_value, sk_value, places=5)
