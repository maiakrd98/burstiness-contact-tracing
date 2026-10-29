import os
import sys
import unittest
import random
import numpy as np
import pandas as pd
sys.path.append("src")
import infection_process


class TestGetGi(unittest.TestCase):

    def test_alpha_good(self):
        # alpha must be a postive number
        for good_alpha in [1.0, 2, 4, 5.5, np.float64(2.0)]:
            with self.subTest(alpha=good_alpha):
                result = infection_process.get_gi(
                    alpha=good_alpha, beta=1.0, psi=0.5
                )

                # args[0] = alpha*psi, check that this is true alpha > 0
                psi = 0.5
                alpha = (result.args[0])/psi
                self.assertAlmostEqual(result.args[0], good_alpha*psi)
                self.assertGreater(alpha, 0)

    def test_alpha_bad(self):
        # alpha must be a postive number
        for bad_alpha in [-2, "blah", -150.3, None, True, 0]:
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(bad_alpha, 1.0, 0.5)

    def test_beta_good(self):
        # beta must be a postive number
        for good_beta in [1.0, 2, 4, 5.5, np.float64(2.0)]:
            with self.subTest(beta=good_beta):
                result = infection_process.get_gi(
                    alpha=1.0, beta=good_beta, psi=0.5)

                # args[2] = 1.0/beta, check true and that beta > than zero
                beta = 1.0/(result.args[2])
                self.assertAlmostEqual(result.args[2], 1.0/good_beta)
                self.assertGreater(beta, 0)

    def test_beta_bad(self):
        # beta must be a postive number
        for bad_beta in [-2, "blah", -150.3, None, True, 0]:
            with self.subTest(beta=bad_beta):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(1.0, bad_beta, 0.5)

    def test_psi_good(self):
        # psi must be a number [0,1]
        for good_psi in [0, 0.0, 1.0, 1, 0.5, 0.75]:
            with self.subTest(psi=good_psi):
                result = infection_process.get_gi(
                    alpha=1.0, beta=1.0, psi=good_psi)

                # args[0] = alpha*psi, check true and that the psi [0,1]
                alpha = 1.0
                psi = (result.args[0])/alpha
                self.assertAlmostEqual(result.args[0], good_psi*alpha)
                self.assertGreaterEqual(psi, 0)
                self.assertLessEqual(psi, 1)

    def test_psi_bad(self):
        # psi must be a number [0,1]
        for bad_psi in [-0.1, 5.3, -100, 45.3]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(alpha=1.0, beta=1.0, psi=bad_psi)

    def test_returns_gamma_distribution(self):
        # get_gi must return a SciPy gamma distribution object
        result = infection_process.get_gi(alpha=2.0, beta=1.0, psi=0.5)
        self.assertEqual(result.dist.name, 'gamma')

    def test_lags_distribution(self):
        # Test that the lag follow the expected gamma dist over 1000 trials
        alpha = 2.0
        beta = 1.0
        psi = 0.4

        # calc theoretical mean and sd for the lag distribution
        shape_lag = (1.0 - psi) * alpha
        scale = 1.0 / beta

        expected_mean = shape_lag * scale
        expected_std = np.sqrt(shape_lag) * scale

        # Run the function 1000 times and extract the lag from result.args[1]
        n_trials = 1000
        lags = []
        for _ in range(n_trials):
            result = infection_process.get_gi(alpha=alpha, beta=beta, psi=psi)
            lag_value = result.args[1]  # Extracts the 'lag' used as loc
            lags.append(lag_value)

        lags = np.array(lags)

        # Compute sample statistics
        sample_mean = np.mean(lags)
        sample_std = np.std(lags, ddof=1)  # Sample standard deviation

        # Assert that sample mean and std are close to theoretical expectations
        self.assertAlmostEqual(sample_mean, expected_mean, 1)
        self.assertAlmostEqual(sample_std, expected_std, 1)


if __name__ == "__main__":
    unittest.main()
