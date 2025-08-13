from typing import Tuple
import unittest

import numpy as np
from scipy import stats
from sampler import point_process

def fit_expon(data: np.ndarray, lambda_: float = 1) -> Tuple[float, float, float]:
    sample_quantile = 1 - np.exp(-lambda_ * data)
    sample_quantile.sort()
    real_quantile = np.linspace(0, 1, data.shape[0])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        real_quantile, sample_quantile
    )
    return intercept, slope, r_value # type: ignore


def hawkes(t, h, mu, a, b):
    return mu + np.sum(a * np.exp(-b * (t - h[t >= h])))

class TestPointProcess(unittest.TestCase):
    def test_poisson(self):
        lambda_ = 1
        t_start = 0
        t_end = 200
        event_times = point_process.sample_hpp(lambda_, t_start, t_end)

        # Perform regression and test intercept, slope and R^2
        i, s, r = fit_expon(np.diff(event_times), lambda_)
        self.assertAlmostEqual(i, 0, places=1) # type: ignore
        self.assertAlmostEqual(s, 1, places=1)  # type: ignore
        self.assertGreater(r ** 2, 0.95)  # type: ignore

    def test_ogata_thinning(self):
        a = 0.8
        b = 0.8
        mu = 0.5
        lambda_ = lambda t, h: hawkes(t, h, mu, a, b)
        lambda_ub = 120

        t_start = 0
        t_end = 100

        event_times = point_process.ogata_thinning(lambda_, lambda_ub, t_start, t_end)
        deltas = np.diff(point_process.integrated_intensity(
            lambda_, event_times
        ))

        i, s, r = fit_expon(deltas)
        self.assertAlmostEqual(i, 0, places=1)
        self.assertAlmostEqual(s, 1, places=1)
        self.assertGreater(r ** 2, 0.95)

    def test_ogata_thinning_adaptive(self):
        a = 0.8
        b = 0.8
        mu = 0.5
        lambda_ = lambda t, h: hawkes(t, h, mu, a, b)
        hawkes_lambda_ub = lambda t, h: (hawkes(
            t, h, mu, a, b
        ), t + 5)

        t_start = 0
        t_end = 100

        event_times = point_process.ogata_thinning_adaptive(
            lambda_, hawkes_lambda_ub, t_start, t_end
        )
        deltas = np.diff(point_process.integrated_intensity(
            lambda_, event_times
        ))
        i, s, r = fit_expon(deltas)
        self.assertAlmostEqual(i, 0, places=1)
        self.assertAlmostEqual(s, 1, places=1)
        self.assertGreater(r ** 2, 0.95)
