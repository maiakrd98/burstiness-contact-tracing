import sys
import unittest
import random
import numpy as np
import pandas as pd
from scipy.stats import gamma
import scipy.special as sc
sys.path.append("src")  # noqa
import infection_process


class TestGetGi(unittest.TestCase):

    def test_alpha_good(self):
        # alpha must be a postive number
        rng = np.random.default_rng(0)
        for good_alpha in [1.0, 2, 4, 5.5, np.float64(2.0)]:
            with self.subTest(alpha=good_alpha):
                psi = 0.5
                result = infection_process.get_gi(
                    alpha=good_alpha, beta=1.0, psi=psi, rng=rng
                )

                # result.kwds['a'] = alpha*psi, check that this is true
                # alpha > 0
                alpha = (result.kwds['a'])/psi
                self.assertAlmostEqual(result.kwds['a'], good_alpha*psi)
                self.assertGreater(alpha, 0)

    def test_alpha_bad(self):
        # alpha must be a postive number
        rng = np.random.default_rng(0)
        for bad_alpha in [-2, "blah", -150.3, None, True, 0]:
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(bad_alpha, 1.0, 0.5, rng=rng)

    def test_beta_good(self):
        # beta must be a postive number
        rng = np.random.default_rng(0)
        for good_beta in [1.0, 2, 4, 5.5, np.float64(2.0)]:
            with self.subTest(beta=good_beta):
                result = infection_process.get_gi(
                    alpha=1.0, beta=good_beta, psi=0.5, rng=rng)

                # result.kwds['scale'] = 1.0/beta, check true and that beta
                # greater than zero
                beta = 1.0/(result.kwds['scale'])
                self.assertAlmostEqual(result.kwds['scale'], 1.0/good_beta)
                self.assertGreater(beta, 0)

    def test_beta_bad(self):
        # beta must be a postive number
        rng = np.random.default_rng(0)
        for bad_beta in [-2, "blah", -150.3, None, True, 0]:
            with self.subTest(beta=bad_beta):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(1.0, bad_beta, 0.5, rng=rng)

    def test_psi_good(self):
        # psi must be a number (0,1]
        rng = np.random.default_rng(0)
        for good_psi in [1.0, 1, 0.25, 0.5, 0.75]:
            with self.subTest(psi=good_psi):
                result = infection_process.get_gi(
                    alpha=1.0, beta=1.0, psi=good_psi, rng=rng)

                # result.kwds['a'] = alpha*psi, check true and that the
                # psi (0,1]
                alpha = 1.0
                psi = (result.kwds['a'])/alpha
                self.assertAlmostEqual(result.kwds['a'], good_psi*alpha)
                self.assertGreaterEqual(psi, 0)
                self.assertLessEqual(psi, 1)

    def test_psi_bad(self):
        # psi must be a number [0,1]
        rng = np.random.default_rng(0)
        for bad_psi in [-0.1, 0, 5.3, -100, 45.3]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(alpha=1.0,
                                             beta=1.0,
                                             psi=bad_psi,
                                             rng=rng)

    def test_returns_gamma_distribution(self):
        # get_gi must return a SciPy gamma distribution object
        rng = np.random.default_rng(0)
        result = infection_process.get_gi(alpha=2.0,
                                          beta=1.0,
                                          psi=0.5,
                                          rng=rng)
        self.assertEqual(result.dist.name, 'gamma')

    def test_lags_distribution(self):
        rng = np.random.default_rng(0)

        # Test that the lag follow the expected gamma dist over 1000 trials
        alpha = 2.0
        beta = 1.0
        psi = 0.4

        # calc theoretical mean and sd for the lag distribution
        shape_lag = (1.0 - psi) * alpha
        scale = 1.0 / beta

        expected_mean = shape_lag * scale
        expected_std = np.sqrt(shape_lag) * scale

        # Run the function 1000 times and extract the lag from
        # result.kwds['loc']
        n_trials = 1000
        lags = []
        for _ in range(n_trials):
            result = infection_process.get_gi(alpha=alpha,
                                              beta=beta,
                                              psi=psi,
                                              rng=rng)
            lag_value = result.kwds['loc']  # Extracts the 'lag' used as loc
            lags.append(lag_value)

        lags = np.array(lags)

        # Compute sample statistics
        sample_mean = np.mean(lags)
        sample_std = np.std(lags, ddof=1)  # Sample standard deviation

        # Assert that sample mean and std are close to theoretical expectations
        self.assertAlmostEqual(sample_mean, expected_mean, 1)
        self.assertAlmostEqual(sample_std, expected_std, 1)


class TestCalcForceOfInf(unittest.TestCase):

    def test_known_gamma_cdfs(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(0, 5)
            loc = rng.uniform(0, 1)
            scale = rng.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            start_t = loc + rng.uniform(0, 5)
            end_t = start_t + rng.uniform(0, 5)
            b = rng.uniform(0, 5)

            force = infection_process.calc_force_of_inf(start_t,
                                                        end_t,
                                                        gamma_dist,
                                                        b)

            exp_force = b*(sc.gammainc(a, (end_t-loc)/scale) -
                           sc.gammainc(a, (start_t-loc)/scale))

            self.assertAlmostEqual(force, exp_force, places=5)

    def test_end_time_less_than_loc(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(0, 5)
            loc = rng.uniform(5, 10)
            scale = rng.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            start_t = rng.uniform(0, 2)
            end_t = start_t + rng.uniform(0, 2)
            b = rng.uniform(0, 5)

            force = infection_process.calc_force_of_inf(start_t,
                                                        end_t,
                                                        gamma_dist,
                                                        b)

            self.assertAlmostEqual(force, 0, places=5)

    def test_bad_start_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        end_t = 5
        b = 2
        for bad_start_t in [-1, False, "hi"]:
            with self.subTest(start_t=bad_start_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_force_of_inf(bad_start_t,
                                                        end_t,
                                                        gamma_dist,
                                                        b)
                self.assertEqual("Invalid input: start_t must be a "
                                 "nonnegative number.",
                                 str(cm.exception))

    def test_bad_end_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 5
        b = 2
        for bad_end_t in [-1, False, "hi"]:
            with self.subTest(end_t=bad_end_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_force_of_inf(start_t,
                                                        bad_end_t,
                                                        gamma_dist,
                                                        b)
                self.assertEqual("Invalid input: end_t must be a nonnegative "
                                 "number.",
                                 str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            infection_process.calc_force_of_inf(start_t,
                                                4,
                                                gamma_dist,
                                                b)
        self.assertEqual("Invalid input: end_t must be greater than or equal "
                         "to start_t.",
                         str(cm.exception))

    def test_bad_b(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 2
        end_t = 5
        for bad_b in [-1, 0, False, "hi"]:
            with self.subTest(b=bad_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_force_of_inf(start_t,
                                                        end_t,
                                                        gamma_dist,
                                                        bad_b)
                self.assertEqual("Invalid input: b must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_gi_dist(self):
        start_t = 2
        end_t = 5
        b = 2
        for bad_gi_dist in [-1, 1.3, False, "hi"]:
            with self.subTest(gi_dist=bad_gi_dist):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_force_of_inf(start_t,
                                                        end_t,
                                                        bad_gi_dist,
                                                        b)
                self.assertEqual("Invalid input: gi_dist must be a "
                                 "scipy.stats frozen continuous random "
                                 "variable object.",
                                 str(cm.exception))


class TestCalcProbOfInf(unittest.TestCase):

    def test_increasing_end_time(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(0, 5)
            loc = rng.uniform(0, 5)
            scale = rng.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)

            start_t = rng.uniform(0, 5)
            end_t_1 = start_t + rng.uniform(0, 5)
            end_t_2 = end_t_1 + 1

            b = rng.uniform(0, 5)

            prob_1 = infection_process.calc_prob_of_inf(start_t,
                                                        end_t_1,
                                                        gamma_dist,
                                                        b)

            prob_2 = infection_process.calc_prob_of_inf(start_t,
                                                        end_t_2,
                                                        gamma_dist,
                                                        b)

            self.assertTrue(prob_1 <= prob_2)

    def test_decreasing_start_time(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(0, 5)
            loc = rng.uniform(0, 5)
            scale = rng.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)

            start_t_1 = rng.uniform(1, 5)
            start_t_2 = start_t_1 - 1
            end_t = start_t_1 + rng.uniform(0, 5)

            b = rng.uniform(0, 5)

            prob_1 = infection_process.calc_prob_of_inf(start_t_1,
                                                        end_t,
                                                        gamma_dist,
                                                        b)

            prob_2 = infection_process.calc_prob_of_inf(start_t_2,
                                                        end_t,
                                                        gamma_dist,
                                                        b)

            self.assertTrue(prob_1 <= prob_2)

    def test_prob_approaches_correct_value(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(0, 5)
            loc = rng.uniform(0, 5)
            scale = rng.uniform(0, 1)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            b = rng.uniform(0, 5)

            prob = infection_process.calc_prob_of_inf(0, 100000, gamma_dist, b)

            # probability should approach 1 - exp(-b)
            self.assertAlmostEqual(prob, 1 - np.exp(-b))

    def test_bad_start_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        end_t = 5
        b = 2
        for bad_start_t in [-1, False, "hi"]:
            with self.subTest(start_t=bad_start_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_prob_of_inf(bad_start_t,
                                                       end_t,
                                                       gamma_dist,
                                                       b)
                self.assertEqual("Invalid input: start_t must be a "
                                 "nonnegative number.",
                                 str(cm.exception))

    def test_bad_end_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 5
        b = 2
        for bad_end_t in [-1, False, "hi"]:
            with self.subTest(end_t=bad_end_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_prob_of_inf(start_t,
                                                       bad_end_t,
                                                       gamma_dist,
                                                       b)
                self.assertEqual("Invalid input: end_t must be a nonnegative "
                                 "number.",
                                 str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            infection_process.calc_prob_of_inf(start_t,
                                               4,
                                               gamma_dist,
                                               b)
        self.assertEqual("Invalid input: end_t must be greater than or equal "
                         "to start_t.",
                         str(cm.exception))

    def test_bad_b(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 2
        end_t = 5
        for bad_b in [-1, 0, False, "hi"]:
            with self.subTest(b=bad_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_prob_of_inf(start_t,
                                                       end_t,
                                                       gamma_dist,
                                                       bad_b)
                self.assertEqual("Invalid input: b must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_gi_dist(self):
        start_t = 2
        end_t = 5
        b = 2
        for bad_gi_dist in [-1, 1.3, False, "hi"]:
            with self.subTest(gi_dist=bad_gi_dist):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_prob_of_inf(start_t,
                                                       end_t,
                                                       bad_gi_dist,
                                                       b)
                self.assertEqual("Invalid input: gi_dist must be a "
                                 "scipy.stats frozen continuous random "
                                 "variable object.",
                                 str(cm.exception))


class TestCalcInfTime(unittest.TestCase):

    def test_times_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(1, 5)
            scale = rng.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_t = rng.uniform(0, 2)
            end_t = start_t + rng.uniform(1, 2)

            inf_time = infection_process.calc_inf_time(start_t,
                                                       end_t,
                                                       gamma_dist,
                                                       rng=rng)

            self.assertTrue(inf_time >= start_t)
            self.assertTrue(inf_time <= end_t)

    def test_proportions_correct(self):
        rng = np.random.default_rng(0)
        inf_times = np.zeros(shape=10000)

        a = 2
        loc = 1
        scale = 2
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_t = 2
        end_t = 5

        for i in range(10000):
            inf_times[i] = infection_process.calc_inf_time(start_t,
                                                           end_t,
                                                           gamma_dist,
                                                           rng=rng)

        self.assertAlmostEqual(np.sum(inf_times <= 3)/10000,
                               0.3455,
                               delta=0.02)
        self.assertAlmostEqual(np.sum((inf_times >= 3) &
                                      (inf_times <= 4))/10000,
                               0.3532,
                               delta=0.02)
        self.assertAlmostEqual(np.sum(inf_times >= 4)/10000,
                               0.3014,
                               delta=0.02)

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = rng.uniform(1, 5)
            scale = rng.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_t = rng.uniform(0, 2)
            end_t = start_t + rng.uniform(1, 2)

            inf_time_1 = infection_process.calc_inf_time(start_t,
                                                         end_t,
                                                         gamma_dist,
                                                         rng=0)
            inf_time_2 = infection_process.calc_inf_time(start_t,
                                                         end_t,
                                                         gamma_dist,
                                                         rng=0)

            self.assertEqual(inf_time_1, inf_time_2)

    def test_diff_with_diff_seed(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            seed_1 = rng.integers(1, 100)
            seed_2 = seed_1 + rng.integers(1, 100)

            a = rng.uniform(1, 5)
            scale = rng.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_t = rng.uniform(0, 2)
            end_t = start_t + rng.uniform(1, 2)

            inf_time_1 = infection_process.calc_inf_time(start_t,
                                                         end_t,
                                                         gamma_dist,
                                                         rng=seed_1)
            inf_time_2 = infection_process.calc_inf_time(start_t,
                                                         end_t,
                                                         gamma_dist,
                                                         rng=seed_2)

            self.assertNotEqual(inf_time_1, inf_time_2)

    def test_bad_start_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        end_t = 5
        for bad_start_t in [-1, False, "hi"]:
            with self.subTest(start_t=bad_start_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_inf_time(bad_start_t,
                                                    end_t,
                                                    gamma_dist)
                self.assertEqual("Invalid input: start_t must be a "
                                 "nonnegative number.",
                                 str(cm.exception))

    def test_bad_end_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 5
        for bad_end_t in [-1, False, "hi"]:
            with self.subTest(end_t=bad_end_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_inf_time(start_t,
                                                    bad_end_t,
                                                    gamma_dist)
                self.assertEqual("Invalid input: end_t must be a nonnegative "
                                 "number.",
                                 str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            infection_process.calc_inf_time(start_t,
                                            4,
                                            gamma_dist)
        self.assertEqual("Invalid input: end_t must be greater than or equal "
                         "to start_t.",
                         str(cm.exception))

    def test_bad_gi_dist(self):
        start_t = 2
        end_t = 5
        for bad_gi_dist in [-1, 1.3, False, "hi"]:
            with self.subTest(gi_dist=bad_gi_dist):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_inf_time(start_t,
                                                    end_t,
                                                    bad_gi_dist)
                self.assertEqual("Invalid input: gi_dist must be a "
                                 "scipy.stats frozen continuous random "
                                 "variable object.",
                                 str(cm.exception))


class TestSimOneContact(unittest.TestCase):

    def test_inf_time_is_none_or_not(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            a = rng.uniform(1, 5)
            scale = rng.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_t = rng.uniform(0, 2)
            end_t = start_t + rng.uniform(1, 2)

            b = rng.uniform(1, 6)

            is_inf, inf_time = infection_process.sim_one_contact(start_t,
                                                                 end_t,
                                                                 gamma_dist,
                                                                 b,
                                                                 rng=rng)

            if is_inf:
                self.assertTrue(inf_time is not None)

            else:
                self.assertTrue(inf_time is None)

    def test_correct_prob_high(self):
        rng = np.random.default_rng(0)
        infections = np.zeros(shape=10000)

        a = 2
        loc = 1
        scale = 2
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_t = 2
        end_t = 5
        b = 2

        for i in range(10000):
            is_inf, inf_time = infection_process.sim_one_contact(start_t,
                                                                 end_t,
                                                                 gamma_dist,
                                                                 b,
                                                                 rng=rng)
            infections[i] = is_inf

        self.assertAlmostEqual(np.sum(infections)/10000, 0.6349, delta=0.01)

    def test_correct_prob_low(self):
        rng = np.random.default_rng(0)
        infections = np.zeros(shape=10000)

        a = 1
        loc = 2
        scale = 0.5
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_t = 3
        end_t = 6
        b = 2

        for i in range(10000):
            is_inf, inf_time = infection_process.sim_one_contact(start_t,
                                                                 end_t,
                                                                 gamma_dist,
                                                                 b,
                                                                 rng=rng)
            infections[i] = is_inf

        self.assertAlmostEqual(np.sum(infections)/10000, 0.2366, delta=0.01)

    def test_time_dist_prob_high(self):
        rng = np.random.default_rng(0)
        inf_times = np.zeros(shape=10000)

        a = 2
        loc = 1
        scale = 2
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_t = 2
        end_t = 5
        b = 2

        for i in range(10000):
            is_inf, inf_time = infection_process.sim_one_contact(start_t,
                                                                 end_t,
                                                                 gamma_dist,
                                                                 b,
                                                                 rng=rng)
            inf_times[i] = inf_time

        num_infs = 10000 - np.sum(np.isnan(inf_times))

        self.assertAlmostEqual(np.sum(inf_times <= 3)/num_infs,
                               0.3455,
                               delta=0.02)
        self.assertAlmostEqual(np.sum((inf_times >= 3) &
                                      (inf_times <= 4))/num_infs,
                               0.3532,
                               delta=0.02)
        self.assertAlmostEqual(np.sum(inf_times >= 4)/num_infs,
                               0.3014,
                               delta=0.02)

    def test_time_dist_prob_low(self):
        rng = np.random.default_rng(0)
        inf_times = np.zeros(shape=10000)

        a = 1
        loc = 2
        scale = 0.5
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_t = 3
        end_t = 6
        b = 2

        for i in range(10000):
            is_inf, inf_time = infection_process.sim_one_contact(start_t,
                                                                 end_t,
                                                                 gamma_dist,
                                                                 b,
                                                                 rng=rng)
            inf_times[i] = inf_time

        num_infs = 10000 - np.sum(np.isnan(inf_times))

        self.assertAlmostEqual(np.sum(inf_times <= 4)/num_infs,
                               0.8668,
                               delta=0.02)
        self.assertAlmostEqual(np.sum((inf_times >= 4) &
                                      (inf_times <= 5))/num_infs,
                               0.1173,
                               delta=0.02)
        self.assertAlmostEqual(np.sum(inf_times >= 5)/num_infs,
                               0.0159,
                               delta=0.02)

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            a = rng.uniform(1, 5)
            scale = rng.uniform(1, 5)
            g_dist = gamma(a=a, loc=0, scale=scale)

            start_t = rng.uniform(0, 2)
            end_t = start_t + rng.uniform(1, 2)
            b = rng.uniform(1, 3)

            is_inf_1, inf_time_1 = infection_process.sim_one_contact(start_t,
                                                                     end_t,
                                                                     g_dist,
                                                                     b,
                                                                     rng=0)
            is_inf_2, inf_time_2 = infection_process.sim_one_contact(start_t,
                                                                     end_t,
                                                                     g_dist,
                                                                     b,
                                                                     rng=0)

            self.assertEqual(is_inf_1, is_inf_2)
            self.assertEqual(inf_time_1, inf_time_2)

    def test_bad_start_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        end_t = 5
        b = 2
        for bad_start_t in [-1, False, "hi"]:
            with self.subTest(start_t=bad_start_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_contact(bad_start_t,
                                                      end_t,
                                                      gamma_dist,
                                                      b)
                self.assertEqual("Invalid input: start_t must be a "
                                 "nonnegative number.",
                                 str(cm.exception))

    def test_bad_end_t(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 5
        b = 2
        for bad_end_t in [-1, False, "hi"]:
            with self.subTest(end_t=bad_end_t):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_contact(start_t,
                                                      bad_end_t,
                                                      gamma_dist,
                                                      b)
                self.assertEqual("Invalid input: end_t must be a nonnegative "
                                 "number.",
                                 str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            infection_process.sim_one_contact(start_t,
                                              4,
                                              gamma_dist,
                                              b)
        self.assertEqual("Invalid input: end_t must be greater than or equal "
                         "to start_t.",
                         str(cm.exception))

    def test_bad_gi_dist(self):
        start_t = 2
        end_t = 5
        b = 2
        for bad_gi_dist in [-1, 1.3, False, "hi"]:
            with self.subTest(gi_dist=bad_gi_dist):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_contact(start_t,
                                                      end_t,
                                                      bad_gi_dist,
                                                      b)
                self.assertEqual("Invalid input: gi_dist must be a "
                                 "scipy.stats frozen continuous random "
                                 "variable object.",
                                 str(cm.exception))

    def test_bad_b(self):
        gamma_dist = gamma(a=2, loc=1, scale=1)
        start_t = 2
        end_t = 5
        for bad_b in [-1, 0, False, "hi"]:
            with self.subTest(b=bad_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_contact(start_t,
                                                      end_t,
                                                      gamma_dist,
                                                      bad_b)
                self.assertEqual("Invalid input: b must be a positive "
                                 "number.",
                                 str(cm.exception))


class TestSimOneIndexCase(unittest.TestCase):

    def _get_df(self):
        '''Generates a contact data frame to use in tests'''
        contacts = [[1, 0, 2.3],
                    [2, 0, 3.1],
                    [3, 2.3, 4.7],
                    [4, 3.1, 4.6],
                    [5, 4.6, 6],
                    [6, 4.7, 6]]
        df = pd.DataFrame(contacts,
                          columns=["id", "start_contact", "end_contact"])
        return df

    def test_correct_columns(self):
        rng = np.random.default_rng(0)
        df = self._get_df()
        for _ in range(10):
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            new_df = infection_process.sim_one_index_case(df,
                                                          alpha,
                                                          beta,
                                                          psi,
                                                          b,
                                                          rng=rng)

            self.assertTrue("start_contact" in new_df.columns.values.tolist())
            self.assertTrue("end_contact" in new_df.columns.values.tolist())
            self.assertTrue("is_inf" in new_df.columns.values.tolist())
            self.assertTrue("inf_time" in new_df.columns.values.tolist())

    def test_correct_data_types(self):
        rng = np.random.default_rng(0)
        df = self._get_df()
        for _ in range(10):
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            new_df = infection_process.sim_one_index_case(df,
                                                          alpha,
                                                          beta,
                                                          psi,
                                                          b,
                                                          rng=rng)

            self.assertTrue(new_df['inf_time'].dtype == float)
            self.assertTrue(new_df['is_inf'].dtype == bool)

    def test_inf_times_in_contact(self):
        rng = np.random.default_rng(0)
        df = self._get_df()
        for _ in range(10):
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            f = infection_process.sim_one_index_case(df,
                                                     alpha,
                                                     beta,
                                                     psi,
                                                     b,
                                                     rng=rng)

            for i in range(len(f)):
                if f['is_inf'][i]:
                    self.assertTrue(f['start_contact'][i] <= f['inf_time'][i])
                    self.assertTrue(f['end_contact'][i] >= f['inf_time'][i])

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        df = self._get_df()
        for _ in range(10):
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            new_df_1 = infection_process.sim_one_index_case(df,
                                                            alpha,
                                                            beta,
                                                            psi,
                                                            b,
                                                            rng=0)
            new_df_2 = infection_process.sim_one_index_case(df,
                                                            alpha,
                                                            beta,
                                                            psi,
                                                            b,
                                                            rng=0)

            self.assertTrue(new_df_1.equals(new_df_2))

    def test_bad_df(self):
        contacts = [[1, 0, 2.3],
                    [2, 0, 3.1]]
        df_no_start = pd.DataFrame(contacts,
                                   columns=["id", "hi", "end_contact"])
        df_no_end = pd.DataFrame(contacts,
                                 columns=["id", "start_contact", "hi"])
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2

        with self.assertRaises(SystemExit) as cm:
            infection_process.sim_one_index_case(7,
                                                 alpha,
                                                 beta,
                                                 psi,
                                                 b)
        self.assertEqual("Invalid input: df must be a data frame.",
                         str(cm.exception))

        with self.assertRaises(SystemExit) as cm:
            infection_process.sim_one_index_case(df_no_start,
                                                 alpha,
                                                 beta,
                                                 psi,
                                                 b)
        self.assertEqual("Invalid input: df must contain column "
                         "'start_contact'",
                         str(cm.exception))

        with self.assertRaises(SystemExit) as cm:
            infection_process.sim_one_index_case(df_no_end,
                                                 alpha,
                                                 beta,
                                                 psi,
                                                 b)
        self.assertEqual("Invalid input: df must contain column 'end_contact'",
                         str(cm.exception))

    def test_bad_alpha(self):
        df = self._get_df()
        beta = 1
        psi = 0.5
        b = 2
        for bad_alpha in [-1, 0, False, "hi"]:
            with self.subTest(psi=bad_alpha):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_index_case(df,
                                                         bad_alpha,
                                                         beta,
                                                         psi,
                                                         b)
                self.assertEqual("Invalid input: alpha must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_beta(self):
        df = self._get_df()
        alpha = 2
        psi = 0.5
        b = 2
        for bad_beta in [-1, 0, False, "hi"]:
            with self.subTest(psi=bad_beta):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_index_case(df,
                                                         alpha,
                                                         bad_beta,
                                                         psi,
                                                         b)
                self.assertEqual("Invalid input: beta must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_psi(self):
        df = self._get_df()
        alpha = 2
        beta = 1
        b = 2
        for bad_psi in [-1, 0, False, 4, "hi"]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_index_case(df,
                                                         alpha,
                                                         beta,
                                                         bad_psi,
                                                         b)
                self.assertEqual("Invalid input: psi must be a number "
                                 "in the interval (0,1].",
                                 str(cm.exception))

    def test_bad_b(self):
        df = self._get_df()
        alpha = 2
        beta = 1
        psi = 0.5
        for bad_b in [-1, 0, False, "hi"]:
            with self.subTest(b=bad_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_one_index_case(df,
                                                         alpha,
                                                         beta,
                                                         psi,
                                                         bad_b)
                self.assertEqual("Invalid input: b must be a positive "
                                 "number.",
                                 str(cm.exception))


class TestSimNIndexCases(unittest.TestCase):

    def test_correct_number_of_index_cases(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            N = rng.integers(10, 100)
            d = rng.integers(1, 10)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            new_df = infection_process.sim_N_index_cases(N,
                                                         d,
                                                         lam,
                                                         alpha,
                                                         beta,
                                                         psi,
                                                         b,
                                                         rng=rng)

            self.assertEqual(len(set(new_df['index_case_id'])), N)

    def test_correct_column_names(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            N = rng.integers(10, 100)
            d = rng.integers(1, 10)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            new_df = infection_process.sim_N_index_cases(N,
                                                         d,
                                                         lam,
                                                         alpha,
                                                         beta,
                                                         psi,
                                                         b,
                                                         rng=rng)

            self.assertTrue("index_case_id" in new_df.columns.values.tolist())
            self.assertTrue("contact_id" in new_df.columns.values.tolist())
            self.assertTrue("start_contact" in new_df.columns.values.tolist())
            self.assertTrue("end_contact" in new_df.columns.values.tolist())
            self.assertTrue("is_inf" in new_df.columns.values.tolist())
            self.assertTrue("inf_time" in new_df.columns.values.tolist())

    def test_t_max_correct(self):
        rng = np.random.default_rng(0)
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 3
        psi = 0.5
        b = 2

        new_df = infection_process.sim_N_index_cases(N,
                                                     d,
                                                     lam,
                                                     alpha,
                                                     beta,
                                                     psi,
                                                     b,
                                                     rng=rng)

        self.assertAlmostEqual(max(new_df['end_contact']), 3.0778, places=4)

    def test_stops_at_t_max(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            N = rng.integers(10, 100)
            d = rng.integers(1, 10)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)
            t_max = rng.uniform(1, 10)

            new_df = infection_process.sim_N_index_cases(N,
                                                         d,
                                                         lam,
                                                         alpha,
                                                         beta,
                                                         psi,
                                                         b,
                                                         t_max=t_max,
                                                         rng=rng)

            self.assertAlmostEqual(max(new_df['end_contact']), t_max)

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            N = rng.integers(10, 100)
            d = rng.integers(1, 10)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            new_df_1 = infection_process.sim_N_index_cases(N,
                                                           d,
                                                           lam,
                                                           alpha,
                                                           beta,
                                                           psi,
                                                           b,
                                                           rng=0)
            new_df_2 = infection_process.sim_N_index_cases(N,
                                                           d,
                                                           lam,
                                                           alpha,
                                                           beta,
                                                           psi,
                                                           b,
                                                           rng=0)

            self.assertTrue(new_df_1.equals(new_df_2))

    def test_bad_N(self):
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_N in [-1, 0, 1.3, False, "hi"]:
            with self.subTest(N=bad_N):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(bad_N,
                                                        d,
                                                        lam,
                                                        alpha,
                                                        beta,
                                                        psi,
                                                        b)
                self.assertEqual("Invalid input: N must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_d(self):
        N = 10
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_d in [-1, 0, 1.3, False, "hi"]:
            with self.subTest(d=bad_d):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        bad_d,
                                                        lam,
                                                        alpha,
                                                        beta,
                                                        psi,
                                                        b)
                self.assertEqual("Invalid input: d must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_lam(self):
        N = 10
        d = 4
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_lam in [-1, 0, False, "hi"]:
            with self.subTest(lam=bad_lam):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        d,
                                                        bad_lam,
                                                        alpha,
                                                        beta,
                                                        psi,
                                                        b)
                self.assertEqual("Invalid input: lam must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_alpha(self):
        N = 10
        d = 4
        lam = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_alpha in [-1, 0, False, "hi"]:
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        d,
                                                        lam,
                                                        bad_alpha,
                                                        beta,
                                                        psi,
                                                        b)
                self.assertEqual("Invalid input: alpha must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_beta(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        psi = 0.5
        b = 2
        for bad_beta in [-1, 0, False, "hi"]:
            with self.subTest(beta=bad_beta):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        d,
                                                        lam,
                                                        alpha,
                                                        bad_beta,
                                                        psi,
                                                        b)
                self.assertEqual("Invalid input: beta must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_psi(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        b = 2
        for bad_psi in [-1, 0, 4.1, False, "hi"]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        d,
                                                        lam,
                                                        alpha,
                                                        beta,
                                                        bad_psi,
                                                        b)
                self.assertEqual("Invalid input: psi must be a number "
                                 "in the interval (0,1].",
                                 str(cm.exception))

    def test_bad_b(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        for bad_b in [-1, 0, False, "hi"]:
            with self.subTest(b=bad_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        d,
                                                        lam,
                                                        alpha,
                                                        beta,
                                                        psi,
                                                        bad_b)
                self.assertEqual("Invalid input: b must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_t_max(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_t_max in [-1, 0, False, "hi"]:
            with self.subTest(t_max=bad_t_max):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.sim_N_index_cases(N,
                                                        d,
                                                        lam,
                                                        alpha,
                                                        beta,
                                                        psi,
                                                        b,
                                                        bad_t_max)
                self.assertEqual("Invalid input: t_max must be a positive "
                                 "number.",
                                 str(cm.exception))


class TestCalcRFromDf(unittest.TestCase):

    def test_known_R(self):
        data = [[0, True],
                [0, False],
                [1, True],
                [1, False],
                [1, True],
                [2, True],
                [2, True],
                [3, True]]
        df = pd.DataFrame(data, columns=['index_case_id', 'is_inf'])
        R = infection_process.calc_R_from_df(df)
        self.assertAlmostEqual(R, 1.5)

    def test_known_prob_R(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            data = []
            prob = rng.uniform(0, 1)
            for i in range(1000):
                is_inf = rng.choice([True, False], p=[prob, 1-prob])
                data.append([i, is_inf])
            df = pd.DataFrame(data, columns=['index_case_id', 'is_inf'])
            R = infection_process.calc_R_from_df(df)
            self.assertAlmostEqual(R, prob, places=1)

    def test_bad_df(self):
        for bad_df in [-1, 0, False, "hi", [1, 2, 3]]:
            with self.subTest(df=bad_df):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R_from_df(bad_df)
                self.assertEqual("Invalid input: df must be a data frame.",
                                 str(cm.exception))

        data = [[1, True, 2.3],
                [2, False, 3.1]]
        df_no_id = pd.DataFrame(data, columns=['h', 'is_inf', 'inf_time'])
        df_no_is_inf = pd.DataFrame(data,
                                    columns=['index_case_id', 'h', 'inf_time'])
        df_is_inf_not_bool = pd.DataFrame(data,
                                          columns=['index_case_id',
                                                   'h',
                                                   'is_inf'])

        with self.assertRaises(SystemExit) as cm:
            infection_process.calc_R_from_df(df_no_id)
        self.assertEqual("Invalid input: df must contain column "
                         "'index_case_id'",
                         str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            infection_process.calc_R_from_df(df_no_is_inf)
        self.assertEqual("Invalid input: df must contain column 'is_inf'",
                         str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            infection_process.calc_R_from_df(df_is_inf_not_bool)
        self.assertEqual("Invalid input: 'is_inf' column must contain "
                         "booleans",
                         str(cm.exception))


class TestCalcR(unittest.TestCase):

    def test_R_increases_with_b(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            N = 100
            d = rng.integers(1, 10)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b_1 = rng.uniform(0, 1)
            b_2 = b_1 + rng.uniform(2, 3)

            R_1 = infection_process.calc_R(N,
                                           d,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b_1,
                                           rng=rng)
            R_2 = infection_process.calc_R(N,
                                           d,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b_2,
                                           rng=rng)

            self.assertTrue(R_1 < R_2)

    def test_R_increases_with_d(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            N = 100
            d_1 = rng.integers(1, 3)
            d_2 = d_1 + rng.integers(2, 5)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            R_1 = infection_process.calc_R(N,
                                           d_1,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b,
                                           rng=rng)
            R_2 = infection_process.calc_R(N,
                                           d_2,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b,
                                           rng=rng)

            self.assertTrue(R_1 < R_2)

    def test_matches_R_from_df(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            N = 200
            d = rng.integers(1, 7)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            R_1 = infection_process.calc_R(N,
                                           d,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b,
                                           rng=rng)

            df = infection_process.sim_N_index_cases(N,
                                                     d,
                                                     lam,
                                                     alpha,
                                                     beta,
                                                     psi,
                                                     b,
                                                     rng=rng)

            R_from_df = infection_process.calc_R_from_df(df)

            self.assertAlmostEqual(R_1, R_from_df, delta=0.2)

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            N = rng.integers(10, 100)
            d = rng.integers(1, 10)
            lam = rng.uniform(0, 3)
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            b = rng.uniform(1, 3)

            R_1 = infection_process.calc_R(N,
                                           d,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b,
                                           rng=0)
            R_2 = infection_process.calc_R(N,
                                           d,
                                           lam,
                                           alpha,
                                           beta,
                                           psi,
                                           b,
                                           rng=0)

            self.assertEqual(R_1, R_2)

    def test_bad_N(self):
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_N in [-1, 0, 1.3, False, "hi"]:
            with self.subTest(N=bad_N):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(bad_N,
                                             d,
                                             lam,
                                             alpha,
                                             beta,
                                             psi,
                                             b)
                self.assertEqual("Invalid input: N must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_d(self):
        N = 10
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_d in [-1, 0, 1.3, False, "hi"]:
            with self.subTest(d=bad_d):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             bad_d,
                                             lam,
                                             alpha,
                                             beta,
                                             psi,
                                             b)
                self.assertEqual("Invalid input: d must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_lam(self):
        N = 10
        d = 4
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_lam in [-1, 0, False, "hi"]:
            with self.subTest(lam=bad_lam):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             d,
                                             bad_lam,
                                             alpha,
                                             beta,
                                             psi,
                                             b)
                self.assertEqual("Invalid input: lam must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_alpha(self):
        N = 10
        d = 4
        lam = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_alpha in [-1, 0, False, "hi"]:
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             d,
                                             lam,
                                             bad_alpha,
                                             beta,
                                             psi,
                                             b)
                self.assertEqual("Invalid input: alpha must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_beta(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        psi = 0.5
        b = 2
        for bad_beta in [-1, 0, False, "hi"]:
            with self.subTest(beta=bad_beta):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             d,
                                             lam,
                                             alpha,
                                             bad_beta,
                                             psi,
                                             b)
                self.assertEqual("Invalid input: beta must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_psi(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        b = 2
        for bad_psi in [-1, 0, 4.1, False, "hi"]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             d,
                                             lam,
                                             alpha,
                                             beta,
                                             bad_psi,
                                             b)
                self.assertEqual("Invalid input: psi must be a number "
                                 "in the interval (0,1].",
                                 str(cm.exception))

    def test_bad_b(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        for bad_b in [-1, 0, False, "hi"]:
            with self.subTest(b=bad_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             d,
                                             lam,
                                             alpha,
                                             beta,
                                             psi,
                                             bad_b)
                self.assertEqual("Invalid input: b must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_t_max(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        b = 2
        for bad_t_max in [-1, 0, False, "hi"]:
            with self.subTest(t_max=bad_t_max):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.calc_R(N,
                                             d,
                                             lam,
                                             alpha,
                                             beta,
                                             psi,
                                             b,
                                             bad_t_max)
                self.assertEqual("Invalid input: t_max must be a positive "
                                 "number.",
                                 str(cm.exception))


class TestDetermineB(unittest.TestCase):

    def test_b_increases_with_R(self):
        rng = np.random.default_rng(0)
        N = 100
        d = rng.integers(1, 10)
        lam = rng.uniform(0, 3)
        alpha = rng.uniform(1, 5)
        beta = rng.uniform(1, 5)
        psi = rng.uniform(0.01, 1)
        R_1 = rng.uniform(0, 1)
        R_2 = R_1 + rng.uniform(2, 3)

        b_1 = infection_process.determine_b(N,
                                            d,
                                            lam,
                                            alpha,
                                            beta,
                                            psi,
                                            R_1,
                                            rng=rng)
        b_2 = infection_process.determine_b(N,
                                            d,
                                            lam,
                                            alpha,
                                            beta,
                                            psi,
                                            R_2,
                                            rng=rng)

        self.assertTrue(b_1 < b_2)

    def test_lam_very_small(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            N = 200
            d = rng.integers(1, 10)
            lam = 0.0001
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            psi = rng.uniform(0.01, 1)
            R = rng.uniform(1, 2)

            b = infection_process.determine_b(N,
                                              d,
                                              lam,
                                              alpha,
                                              beta,
                                              psi,
                                              R,
                                              start_b=4,
                                              rng=rng)

            self.assertAlmostEqual(b, -np.log(1-R/d), places=1)

    def test_known_b(self):
        rng = np.random.default_rng(0)
        N = 100
        d = 4
        lam = 2
        alpha = 2.39
        beta = 0.336
        psi = 0.1
        R = 6

        b = infection_process.determine_b(N,
                                          d,
                                          lam,
                                          alpha,
                                          beta,
                                          psi,
                                          R,
                                          start_b=0.01,
                                          rng=rng)

        self.assertAlmostEqual(b, 4, delta=0.2)

    def test_start_b_too_small(self):
        rng = np.random.default_rng(0)
        N = 100
        d = 4
        lam = 2
        alpha = 2.39
        beta = 0.336
        psi = 0.1
        R = 6

        with self.assertRaises(SystemExit) as cm:
            infection_process.determine_b(N,
                                          d,
                                          lam,
                                          alpha,
                                          beta,
                                          psi,
                                          R,
                                          start_b=0.0001,
                                          rng=rng)
        self.assertEqual("Could not get a large enough R value in the alloted "
                         "time. Try entering a larger value for start_b.",
                         str(cm.exception))

    def test_maxruns_too_small(self):
        rng = np.random.default_rng(0)
        N = 100
        d = 4
        lam = 2
        alpha = 2.39
        beta = 0.336
        psi = 0.1
        R = 6

        with self.assertRaises(SystemExit) as cm:
            infection_process.determine_b(N,
                                          d,
                                          lam,
                                          alpha,
                                          beta,
                                          psi,
                                          R,
                                          maxruns=1,
                                          rng=rng)
        self.assertEqual("Could not find b in the allotted runs. "
                         "Try increasing maxruns or increading tol.",
                         str(cm.exception))

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        N = rng.integers(10, 100)
        d = rng.integers(1, 10)
        lam = rng.uniform(0, 3)
        alpha = rng.uniform(1, 5)
        beta = rng.uniform(1, 5)
        psi = rng.uniform(0.01, 1)
        R = rng.uniform(1, 3)

        b_1 = infection_process.determine_b(N,
                                            d,
                                            lam,
                                            alpha,
                                            beta,
                                            psi,
                                            R,
                                            rng=0)
        b_2 = infection_process.determine_b(N,
                                            d,
                                            lam,
                                            alpha,
                                            beta,
                                            psi,
                                            R,
                                            rng=0)

        self.assertEqual(b_1, b_2)

    def test_bad_N(self):
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_N in [-1, 0, 1.3, False, "hi"]:
            with self.subTest(N=bad_N):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(bad_N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R)
                self.assertEqual("Invalid input: N must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_d(self):
        N = 10
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_d in [-1, 0, 1.3, False, "hi"]:
            with self.subTest(d=bad_d):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  bad_d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R)
                self.assertEqual("Invalid input: d must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_lam(self):
        N = 10
        d = 4
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_lam in [-1, 0, False, "hi"]:
            with self.subTest(lam=bad_lam):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  bad_lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R)
                self.assertEqual("Invalid input: lam must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_alpha(self):
        N = 10
        d = 4
        lam = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_alpha in [-1, 0, False, "hi"]:
            with self.subTest(alpha=bad_alpha):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  bad_alpha,
                                                  beta,
                                                  psi,
                                                  R)
                self.assertEqual("Invalid input: alpha must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_beta(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        psi = 0.5
        R = 2
        for bad_beta in [-1, 0, False, "hi"]:
            with self.subTest(beta=bad_beta):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  bad_beta,
                                                  psi,
                                                  R)
                self.assertEqual("Invalid input: beta must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_psi(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        R = 2
        for bad_psi in [-1, 0, 4.1, False, "hi"]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  bad_psi,
                                                  R)
                self.assertEqual("Invalid input: psi must be a number "
                                 "in the interval (0,1].",
                                 str(cm.exception))

    def test_bad_R(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        for bad_R in [-1, 0, False, "hi"]:
            with self.subTest(R=bad_R):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  bad_R)
                self.assertEqual("Invalid input: R must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_start_b(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_start_b in [-1, 0, False, "hi"]:
            with self.subTest(start_b=bad_start_b):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R,
                                                  start_b=bad_start_b)
                self.assertEqual("Invalid input: start_b must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_tol(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_tol in [-1, 0, False, "hi"]:
            with self.subTest(tol=bad_tol):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R,
                                                  tol=bad_tol)
                self.assertEqual("Invalid input: tol must be a positive "
                                 "number.",
                                 str(cm.exception))

    def test_bad_maxruns(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_maxruns in [-1, 0, 2.5, False, "hi"]:
            with self.subTest(maxruns=bad_maxruns):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R,
                                                  maxruns=bad_maxruns)
                self.assertEqual("Invalid input: maxruns must be a positive "
                                 "integer.",
                                 str(cm.exception))

    def test_bad_t_max(self):
        N = 10
        d = 4
        lam = 2
        alpha = 2
        beta = 1
        psi = 0.5
        R = 2
        for bad_t_max in [-1, 0, False, "hi"]:
            with self.subTest(t_max=bad_t_max):
                with self.assertRaises(SystemExit) as cm:
                    infection_process.determine_b(N,
                                                  d,
                                                  lam,
                                                  alpha,
                                                  beta,
                                                  psi,
                                                  R,
                                                  t_max=bad_t_max)
                self.assertEqual("Invalid input: t_max must be a positive "
                                 "number.",
                                 str(cm.exception))


if __name__ == "__main__":
    unittest.main()
