import collections
import itertools
import os
import time
import unittest

import numpy as np
from scipy import stats

# Remove existing pre-built libraries if they exist
for _ in os.listdir(os.path.join(os.path.dirname(__file__), '../sampler')):
    for ext in ['.so', '.pyd', '.dll', '.dylib']:
        if _.endswith(ext):
            os.remove(os.path.join(os.path.dirname(__file__), '../sampler', _))

from sampler import *


class TestSampler(unittest.TestCase):
    def setUp(self):
        setup_threads(8)

    def test_batched_choices_with_negative(self):
        max_item = 10000
        N = [1000, 10000]
        k = [1, 10, 100, 1000]
        for n in N:
            for ki in k:
                with self.subTest(n=n, k=ki):
                    range_ = np.arange(max_item)
                    exclude = np.random.randint(0, max_item, (n, 500))
                    begin = time.time()
                    result = batched_choices_with_negative(range_, exclude, ki)
                    end = time.time()

                    # Check result shape
                    self.assertEqual(result.shape, (n, ki))
                    # Assert exclude values are not in the result
                    for i in range(n):
                        for j in range(ki):
                            self.assertNotIn(result[i, j], exclude[i])
                    # Assert all values are within the range
                    self.assertTrue(np.all(result >= 0))
                    self.assertTrue(np.all(result < max_item))
                    # Assert avg time per sample is reasonable
                    avg_time_per_sample = (end - begin) / n
                    print(
                        f'Time taken for n={n}, k={ki}: {end - begin} seconds, '
                        f'Average time per sample: {avg_time_per_sample} seconds'
                    )
                    self.assertLess(avg_time_per_sample, 1e-4)

    def test_weighted_choices_with_negative(self):
        max_item = 10000
        N = [1000, 10000]
        k = [1, 10, 100, 1000]
        for n in N:
            for ki in k:
                with self.subTest(n=n, k=ki):
                    range_ = np.arange(max_item)
                    weights = np.random.rand(max_item)
                    exclude = np.random.randint(0, max_item, (n, 500))
                    begin = time.time()
                    result = weighted_batched_choices_with_negative(range_, weights, exclude, ki)
                    end = time.time()

                    # Check result shape
                    self.assertEqual(result.shape, (n, ki))
                    # Assert exclude values are not in the result
                    for i in range(n):
                        for j in range(ki):
                            self.assertNotIn(result[i, j], exclude[i])
                    # Assert all values are within the range
                    self.assertTrue(np.all(result >= 0))
                    self.assertTrue(np.all(result < max_item))
                    # Assert avg time per sample is reasonable
                    avg_time_per_sample = (end - begin) / n
                    print(
                        f'Time taken for n={n}, k={ki}: {end - begin} seconds, '
                        f'Average time per sample: {avg_time_per_sample} seconds'
                    )
                    self.assertLess(avg_time_per_sample, 1e-4)

    def test_batched_sample_with_negative(self):
        max_item = 10000
        N = [1000, 10000]
        k = [1, 10, 100, 1000]
        for n in N:
            for ki in k:
                with self.subTest(n=n, k=ki):
                    range_ = np.arange(max_item)
                    exclude = np.random.randint(0, max_item, (n, 500))
                    begin = time.time()
                    result = batched_sample_with_negative(range_, exclude, ki)
                    end = time.time()

                    # Check result shape
                    self.assertEqual(result.shape, (n, ki))
                    # Assert exclude values are not in the result
                    for i in range(n):
                        for j in range(ki):
                            self.assertNotIn(result[i, j], exclude[i])
                    # Assert all values are within the range
                    self.assertTrue(np.all(result >= 0))
                    self.assertTrue(np.all(result < max_item))
                    # Assert avg time per sample is reasonable
                    avg_time_per_sample = (end - begin) / n
                    print(
                        f'Time taken for n={n}, k={ki}: {end - begin} seconds, '
                        f'Average time per sample: {avg_time_per_sample} seconds'
                    )
                    self.assertLess(avg_time_per_sample, 1e-4)

    def test_weighted_batched_sample_with_negative(self):
        max_item = 10000
        N = [1000, 10000]
        k = [10, 100]
        for n in N:
            for ki in k:
                with self.subTest(n=n, k=ki):
                    range_ = np.arange(max_item)
                    weights = np.random.rand(max_item)

                    p = 1 - (1 - 0.99 ** (1 / n)) ** 1 / ki
                    weights[0] = p / (1 - p) # 99% of the samples should include the first item
                    exclude = np.random.randint(1, max_item, (n, 500))
                    begin = time.time()
                    result = weighted_batched_sample_with_negative(range_, weights, exclude, ki)
                    end = time.time()

                    # Check result shape
                    self.assertEqual(result.shape, (n, ki))
                    # Assert exclude values are not in the result
                    for i in range(n):
                        for j in range(ki):
                            self.assertNotIn(result[i, j], exclude[i])
                    # Assert all values are within the range
                    self.assertTrue(np.all(result >= 0))
                    self.assertTrue(np.all(result < max_item))
                    # Assert weights are respected
                    for i in range(n):
                        # The first item should be (definitely) exist in the result
                        self.assertIn(0, result[i])
                    # Assert avg time per sample is reasonable
                    avg_time_per_sample = (end - begin) / n
                    print(
                        f'Time taken for n={n}, k={ki}: {end - begin} seconds, '
                        f'Average time per sample: {avg_time_per_sample} seconds'
                    )
                    self.assertLess(avg_time_per_sample, 1e-4)

    def test_weighted_batched_sample_with_negative_err(self):
        # When the rest items is less than k, it should raise ValueError
        range_ = np.arange(10)
        weights = np.random.rand(10)
        exclude = np.tile(np.arange(8), (5, 1))
        k = 5

        with self.assertRaises(ValueError):
            weighted_batched_sample_with_negative(range_, weights, exclude, k)

    def test_randomness(self):
        range_ = np.arange(1000)
        exclude = np.random.randint(0, 1000, (30))
        exclude_list = exclude.tolist()
        exclude_matrix = np.tile(exclude, (10000, 1))

        k = 10
        results = [batched_sample_with_negative(range_, exclude_matrix, k).tolist() for _ in range(10)]
        counter = collections.Counter()
        for result in results:
            counter.update(itertools.chain.from_iterable(result))
        # Evaluate the distribution of sampled numbers
        for item in exclude_list:
            self.assertNotIn(item, counter)
        self.assertGreater(len(counter), 100)  # Ensure diversity in samples
        self.assertLess(len(counter), 10000)  # Ensure not all numbers are sampled

        # Chi-squared test for uniformity
        numbers, counts = zip(*sorted(counter.items()))
        values, freqs = np.array(numbers), np.array(counts)
        n = freqs.sum()
        k = len(values)
        expected = np.full(k, n / k)

        chi2, p = stats.chisquare(freqs, expected)
        self.assertGreater(p, 0.05)
