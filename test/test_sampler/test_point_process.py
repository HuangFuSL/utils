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


class TestPointProcess(unittest.TestCase):
    def setUp(self):
        self.mu = 1
        self.alpha = 1
        self.beta = 1

        self.hawkes = point_process.HawkesIntensity(self.mu, self.alpha, self.beta)

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
        lambda_ub = 300

        t_start = 0
        t_end = 100

        event_times = point_process.ogata_thinning(
            self.hawkes.lambda_, lambda_ub, t_start, t_end
        )
        deltas = np.diff(point_process.integrated_intensity(
            self.hawkes.lambda_, event_times, n_points=50
        ))

        i, s, r = fit_expon(deltas)
        self.assertAlmostEqual(i, 0, places=1)
        self.assertAlmostEqual(s, 1, places=1)
        self.assertGreater(r ** 2, 0.95)

    def test_ogata_thinning_adaptive(self):
        t_start = 0
        t_end = 100

        event_times = point_process.ogata_thinning_adaptive(
            self.hawkes.lambda_, self.hawkes.lambda_upperbound, t_start, t_end
        )
        deltas = np.diff(point_process.integrated_intensity(
            self.hawkes.lambda_, event_times, n_points=50
        ))
        i, s, r = fit_expon(deltas)
        self.assertAlmostEqual(i, 0, places=1)
        self.assertAlmostEqual(s, 1, places=1)
        self.assertGreater(r ** 2, 0.95)
