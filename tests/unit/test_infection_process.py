import sys
import unittest
import random
import numpy as np
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

                # result.kwds['a'] = alpha*psi, check that this is true alpha > 0
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

                # result.kwds['scale'] = 1.0/beta, check true and that beta > than zero
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
        # psi must be a number [0,1]
        rng = np.random.default_rng(0)
        for good_psi in [0, 0.0, 1.0, 1, 0.5, 0.75]:
            with self.subTest(psi=good_psi):
                result = infection_process.get_gi(
                    alpha=1.0, beta=1.0, psi=good_psi, rng=rng)

                # result.kwds['a'] = alpha*psi, check true and that the psi [0,1]
                alpha = 1.0
                psi = (result.kwds['a'])/alpha
                self.assertAlmostEqual(result.kwds['a'], good_psi*alpha)
                self.assertGreaterEqual(psi, 0)
                self.assertLessEqual(psi, 1)

    def test_psi_bad(self):
        # psi must be a number [0,1]
        rng = np.random.default_rng(0)
        for bad_psi in [-0.1, 5.3, -100, 45.3]:
            with self.subTest(psi=bad_psi):
                with self.assertRaises((TypeError, ValueError)):
                    infection_process.get_gi(alpha=1.0, beta=1.0, psi=bad_psi, rng=rng)

    def test_returns_gamma_distribution(self):
        # get_gi must return a SciPy gamma distribution object
        rng = np.random.default_rng(0)
        result = infection_process.get_gi(alpha=2.0, beta=1.0, psi=0.5, rng=rng)
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

        # Run the function 1000 times and extract the lag from result.kwds['loc']
        n_trials = 1000
        lags = []
        for _ in range(n_trials):
            result = infection_process.get_gi(alpha=alpha, beta=beta, psi=psi, rng=rng)
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
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 1)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            start_t = loc + random.uniform(0, 5)
            end_t = start_t + random.uniform(0, 5)
            b = random.uniform(0, 5)

            force = infection_process.calc_force_of_inf(start_t,
                                                        end_t,
                                                        gamma_dist,
                                                        b)

            exp_force = b*(sc.gammainc(a, (end_t-loc)/scale) -
                           sc.gammainc(a, (start_t-loc)/scale))

            self.assertAlmostEqual(force, exp_force, places=5)

    def test_end_time_less_than_loc(self):
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(5, 10)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            start_t = random.uniform(0, 2)
            end_t = start_t + random.uniform(0, 2)
            b = random.uniform(0, 5)

            force = infection_process.calc_force_of_inf(start_t,
                                                        end_t,
                                                        gamma_dist,
                                                        b)

            self.assertAlmostEqual(force, 0, places=5)


class TestCalcProbOfInf(unittest.TestCase):

    def test_increasing_end_time(self):
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 5)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)

            start_t = random.uniform(0, 5)
            end_t_1 = start_t + random.uniform(0, 5)
            end_t_2 = end_t_1 + 1

            b = random.uniform(0, 5)

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
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 5)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)

            start_t_1 = random.uniform(1, 5)
            start_t_2 = start_t_1 - 1
            end_t = start_t_1 + random.uniform(0, 5)

            b = random.uniform(0, 5)

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
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 5)
            scale = random.uniform(0, 1)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            b = random.uniform(0, 5)

            prob = infection_process.calc_prob_of_inf(0, 100000, gamma_dist, b)

            # probability should approach 1 - exp(-b)
            self.assertAlmostEqual(prob, 1 - np.exp(-b))


class TestCalcInfTime(unittest.TestCase):

    def test_times_in_range(self):
        rng = np.random.default_rng(0)
        for i in range(100):
            a = random.uniform(1, 5)
            scale = random.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_t = random.uniform(0, 2)
            end_t = start_t + random.uniform(1, 2)

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


class TestSimOneContact(unittest.TestCase):

    def test_inf_time_is_none_or_not(self):
        rng = np.random.default_rng(0)
        for i in range(100):
            a = random.uniform(1, 5)
            scale = random.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_t = random.uniform(0, 2)
            end_t = start_t + random.uniform(1, 2)

            b = random.uniform(1, 6)

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

# TODO: test results are the same with same rng (and results diff in loop?)
# TODO: test bad inputs
# TODO: test sim one index case

if __name__ == "__main__":
    unittest.main()
