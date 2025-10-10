import importlib.util
import unittest

import sklearn.metrics
import torch

from ctorch.functional import log_norm_pdf, rbf_kernel, mmd_distance, wasserstein_distance, logit_product

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None and \
    importlib.util.find_spec("numpy") is not None
if SCIPY_AVAILABLE:
    import numpy as np
    from scipy.stats import multivariate_normal as mvn

class TestLogitProduct(unittest.TestCase):
    def setUp(self) -> None:
        self.N, self.D = 4, 3

        self.x = torch.randn(self.N, self.D)
        self.y = torch.randn(self.N, self.D)

    def test_logit_product(self):
        new_prob = torch.sigmoid(self.x) * torch.sigmoid(self.y)
        new_logit = torch.log(new_prob) - torch.log1p(-new_prob)
        res = logit_product(self.x, self.y)
        self.assertTrue(torch.allclose(res, new_logit))

class TestLogNormPdf(unittest.TestCase):
    def setUp(self):
        self.N, self.D = 4, 3

        self.x = torch.randn(self.N, self.D)
        self.mean_vec = torch.zeros(self.D)
        self.mean_mat = torch.zeros(self.N, self.D)

    def test_scalar_sigma(self):
        sigma = torch.tensor(2.0)
        out = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma)
        self.assertEqual(out.shape, (self.N,))

        x = torch.tensor([[1.5]])
        expected = -0.5 * (torch.log(2 * torch.pi * sigma) + (1.5 ** 2) / sigma)
        res = log_norm_pdf(x, torch.tensor([0.0]), Sigma=sigma)
        self.assertTrue(torch.isclose(res.squeeze(), expected))

    def test_vector_sigma_dimension(self):
        sigma_d = torch.full((self.D,), 1.5)
        out = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma_d, batch_first=False)
        self.assertEqual(out.shape, (self.N,))

    def test_vector_sigma_batch(self):
        sigma_n = torch.full((self.N,), 1.2)
        out = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma_n, batch_first=True)
        self.assertEqual(out.shape, (self.N,))

    def test_matrix_diag_batch(self):
        sigma_nd = torch.full((self.N, self.D), 0.9)
        out = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma_nd, batch_first=True)
        self.assertEqual(out.shape, (self.N,))

    def test_matrix_full_shared(self):
        cov = torch.eye(self.D) * 1.1 + 0.1
        out = log_norm_pdf(self.x, self.mean_vec, Sigma=cov, batch_first=False)
        self.assertEqual(out.shape, (self.N,))

    def test_matrix_full_per_sample(self):
        cov_ndd = torch.stack([torch.eye(self.D) * (i + 1) for i in range(self.N)])
        out = log_norm_pdf(self.x, self.mean_mat, Sigma=cov_ndd)
        self.assertEqual(out.shape, (self.N,))

    def test_log_sigma(self):
        log_sigma = torch.log(torch.full((self.D,), 0.7))
        out = log_norm_pdf(self.x, self.mean_vec, logSigma=log_sigma, batch_first=False)
        self.assertEqual(out.shape, (self.N,))

    def test_ambiguous_shape_error(self):
        x = torch.randn(3, 3)
        sigma = torch.ones(3)
        with self.assertRaises(ValueError):
            _ = log_norm_pdf(x, torch.zeros(3), Sigma=sigma)

    def test_shape_mismatch_error(self):
        sigma_bad = torch.ones(self.N - 1)
        with self.assertRaises(ValueError):
            _ = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma_bad, batch_first=True)

    def test_negative_sigma_error(self):
        sigma_neg = torch.tensor([-1.0])
        with self.assertRaises(ValueError):
            _ = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma_neg)

    def test_nonsymmetric_cov_error(self):
        cov = torch.eye(self.D)
        cov[0, 1] = 0.3
        with self.assertRaises(ValueError):
            _ = log_norm_pdf(self.x, self.mean_vec, Sigma=cov, batch_first=False)

@unittest.skipUnless(SCIPY_AVAILABLE, "SciPy is not installed; skipping value tests.")
class TestLogNormPdfValues(unittest.TestCase):
    def setUp(self) -> None:
        self.N, self.D = 5, 4
        self.x = torch.randn(self.N, self.D, dtype=torch.double)
        self.mean_vec = torch.randn(self.D, dtype=torch.double)
        self.mean_mat = torch.randn(self.N, self.D, dtype=torch.double)

    def test_scalar_sigma_value(self):
        sigma = torch.tensor(1.7, dtype=torch.double)
        torch_out = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma).double()

        cov = sigma.item() * np.eye(self.D)
        ref = np.array([
            mvn.logpdf(self.x[i].numpy(), mean=self.mean_vec.numpy(), cov=cov)
            for i in range(self.N)
        ])
        self.assertTrue(np.allclose(torch_out.numpy(), ref, rtol=1e-6, atol=1e-6))

    def test_diag_shared_sigma_value(self):
        sigma_d = torch.full((self.D,), 0.8, dtype=torch.double)
        torch_out = log_norm_pdf(self.x, self.mean_vec, Sigma=sigma_d, batch_first=False).double()

        cov = np.diag(sigma_d.numpy())
        ref = np.array([
            mvn.logpdf(self.x[i].numpy(), mean=self.mean_vec.numpy(), cov=cov)
            for i in range(self.N)
        ])
        self.assertTrue(np.allclose(torch_out.numpy(), ref, rtol=1e-6, atol=1e-6))

    def test_diag_batch_sigma_value(self):
        sigma_nd = torch.full((self.N, self.D), 1.1, dtype=torch.double)
        torch_out = log_norm_pdf(self.x, self.mean_mat, Sigma=sigma_nd, batch_first=True).double()

        ref = np.array([
            mvn.logpdf(
                self.x[i].numpy(),
                mean=self.mean_mat[i].numpy(),
                cov=np.diag(sigma_nd[i].numpy())
            )
            for i in range(self.N)
        ])
        self.assertTrue(np.allclose(torch_out.numpy(), ref, rtol=1e-6, atol=1e-6))

    def test_full_shared_sigma_value(self):
        A = torch.randn(self.D, self.D, dtype=torch.double)
        cov_t = (A @ A.T) + 0.5 * torch.eye(self.D)      # 保证正定
        torch_out = log_norm_pdf(self.x, self.mean_vec, Sigma=cov_t, batch_first=False).double()

        cov = cov_t.numpy()
        ref = np.array([
            mvn.logpdf(self.x[i].numpy(), mean=self.mean_vec.numpy(), cov=cov)
            for i in range(self.N)
        ])
        self.assertTrue(np.allclose(torch_out.numpy(), ref, rtol=1e-6, atol=1e-6))

    def test_full_batch_sigma_value(self):
        cov_ndd = torch.stack([
            torch.eye(self.D, dtype=torch.double) * (i + 1) * 0.6
            for i in range(self.N)
        ])
        torch_out = log_norm_pdf(self.x, self.mean_mat, Sigma=cov_ndd).double()

        ref = np.array([
            mvn.logpdf(
                self.x[i].numpy(),
                mean=self.mean_mat[i].numpy(),
                cov=cov_ndd[i].numpy()
            )
            for i in range(self.N)
        ])
        self.assertTrue(np.allclose(torch_out.numpy(), ref, rtol=1e-6, atol=1e-6))

    def test_log_sigma_value(self):
        log_sigma_d = torch.log(torch.full((self.D,), 0.9, dtype=torch.double))
        torch_out = log_norm_pdf(self.x, self.mean_vec, logSigma=log_sigma_d, batch_first=False).double()

        cov = np.diag(np.exp(log_sigma_d.numpy()))
        ref = np.array([
            mvn.logpdf(self.x[i].numpy(), mean=self.mean_vec.numpy(), cov=cov)
            for i in range(self.N)
        ])
        self.assertTrue(np.allclose(torch_out.numpy(), ref, rtol=1e-6, atol=1e-6))

class TestIPMDistance(unittest.TestCase):
    def setUp(self):
        self.M = 1000
        self.x = torch.randn(self.M, 10)
        self.y = torch.randn(self.M, 10) # Same distribution, same shape
        self.z = torch.rand(self.M, 10) # Different distribution, same shape
        self.w = torch.randn(self.M - 1, 10) # Same shape, different shape

    def test_rbf_kernel_shape(self):
        kernel = rbf_kernel(self.x, self.y, sigma=1.0)
        self.assertEqual(kernel.shape, (self.M, self.M))

    def test_rbf_kernel_values(self):
        length_scale = 1.0
        expected = torch.tensor(sklearn.metrics.pairwise.rbf_kernel(
            self.x.numpy(), self.y.numpy(), gamma=1.0 / (2 * length_scale ** 2)
        ))
        kernel = rbf_kernel(self.x, self.y, sigma=length_scale)
        self.assertTrue(torch.allclose(kernel, expected))

    def test_mmd_distance(self):
        distance_xx = mmd_distance(self.x, self.x, sigma=1.0).item()
        distance_xy = mmd_distance(self.x, self.y, sigma=1.0).item()
        distance_yx = mmd_distance(self.y, self.x, sigma=1.0).item()
        distance_xz = mmd_distance(self.x, self.z, sigma=1.0).item()
        distance_xw = mmd_distance(self.x, self.w, sigma=1.0).item()
        self.assertLess(distance_xx, 1e-3)
        self.assertGreaterEqual(distance_xy, 0)
        self.assertAlmostEqual(distance_xy, distance_yx, places=3)
        self.assertAlmostEqual(distance_xy, distance_xw, places=1)
        self.assertGreater(distance_xz, distance_xy)

    def test_wasserstein_distance(self):
        distance_xx = wasserstein_distance(self.x, self.x).item()
        distance_xy = wasserstein_distance(self.x, self.y).item()
        distance_yx = wasserstein_distance(self.y, self.x).item()
        distance_xz = wasserstein_distance(self.x, self.z).item()
        distance_xw = wasserstein_distance(self.x, self.w).item()
        self.assertLess(distance_xx, 1e-3)
        self.assertGreaterEqual(distance_xy, 0)
        self.assertAlmostEqual(distance_xy, distance_yx, 3)
        self.assertAlmostEqual(distance_xw, distance_xy, 1)
        self.assertGreater(distance_xz, distance_xy)