import sys
import unittest
import random
import numpy as np
from scipy.stats import gamma
import scipy.special as sc
sys.path.append("src")  # noqa
import infection_process


class TestCalcForceOfInf(unittest.TestCase):

    def test_known_gamma_cdfs(self):
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 1)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
            start_time = loc + random.uniform(0, 5)
            end_time = start_time + random.uniform(0, 5)
            b = random.uniform(0, 5)

            force = infection_process.calc_force_of_inf(start_time,
                                                        end_time,
                                                        gamma_dist,
                                                        b)

            exp_force = b*(sc.gammainc(a, (end_time-loc)/scale) - sc.gammainc(a, (start_time-loc)/scale))

            self.assertAlmostEqual(force, exp_force, places=5)

    def test_end_time_less_than_loc(self):
            for i in range(100):
                a = random.uniform(0, 5)
                loc = random.uniform(5, 10)
                scale = random.uniform(0, 5)
                gamma_dist = gamma(a=a, loc=loc, scale=scale)
                start_time = random.uniform(0, 2)
                end_time = start_time + random.uniform(0, 2)
                b = random.uniform(0, 5)
    
                force = infection_process.calc_force_of_inf(start_time,
                                                            end_time,
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
                            
            start_time = random.uniform(0, 5)
            end_time_1 = start_time + random.uniform(0, 5)
            end_time_2 = end_time_1 + 1

            b = random.uniform(0, 5)

            prob_1 = infection_process.calc_prob_of_inf(start_time,
                                                        end_time_1,
                                                        gamma_dist,
                                                        b)

            prob_2 = infection_process.calc_prob_of_inf(start_time,
                                                        end_time_2,
                                                        gamma_dist,
                                                        b)

            self.assertTrue(prob_1 <= prob_2)

    def test_decreasing_start_time(self):
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 5)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a=a, loc=loc, scale=scale)
                            
            start_time_1 = random.uniform(1, 5)
            start_time_2 = start_time_1 - 1
            end_time = start_time_1 + random.uniform(0, 5)

            b = random.uniform(0, 5)

            prob_1 = infection_process.calc_prob_of_inf(start_time_1,
                                                        end_time,
                                                        gamma_dist,
                                                        b)

            prob_2 = infection_process.calc_prob_of_inf(start_time_2,
                                                        end_time,
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
        for i in range(100):
            a = random.uniform(1, 5)
            scale = random.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_time = random.uniform(0, 2)
            end_time = start_time + random.uniform(1, 2)

            inf_time = infection_process.calc_inf_time(start_time, end_time, gamma_dist)

            self.assertTrue(inf_time >= start_time)
            self.assertTrue(inf_time <= end_time)

    def test_proportions_correct(self):
        inf_times = np.zeros(shape=10000)
        
        a = 2
        loc = 1
        scale = 2
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_time = 2
        end_time = 5

        for i in range(10000):
            inf_times[i] = infection_process.calc_inf_time(start_time, end_time, gamma_dist)

        self.assertAlmostEqual(np.sum(inf_times<=3)/10000, 0.3455, delta = 0.02)
        self.assertAlmostEqual(np.sum((inf_times>=3) & (inf_times<=4))/10000, 0.3532, delta = 0.02)
        self.assertAlmostEqual(np.sum(inf_times>=4)/10000, 0.3014, delta = 0.02)


class TestSimOneContact(unittest.TestCase):

    def test_inf_time_is_none_or_not(self):
       for i in range(100):
            a = random.uniform(1, 5)
            scale = random.uniform(1, 5)
            gamma_dist = gamma(a=a, loc=0, scale=scale)

            start_time = random.uniform(0, 2)
            end_time = start_time + random.uniform(1, 2)

            b = random.uniform(1, 6)

            is_infected, inf_time = infection_process.sim_one_contact(start_time, end_time, gamma_dist, b)

            if is_infected:
                self.assertTrue(inf_time is not None)

            else:
                self.assertTrue(inf_time is None)

    def test_correct_prob_high(self):
        infections = np.zeros(shape=10000)

        a = 2
        loc = 1
        scale = 2
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_time = 2
        end_time = 5
        b = 2

        for i in range(10000):
            is_infected, inf_time = infection_process.sim_one_contact(start_time, end_time, gamma_dist, b)
            infections[i] = is_infected

        self.assertAlmostEqual(np.sum(infections)/10000, 0.6349, delta=0.01)

    def test_correct_prob_low(self):
        infections = np.zeros(shape=10000)

        a = 1
        loc = 2
        scale = 0.5
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_time = 3
        end_time = 6
        b = 2

        for i in range(10000):
            is_infected, inf_time = infection_process.sim_one_contact(start_time, end_time, gamma_dist, b)
            infections[i] = is_infected

        self.assertAlmostEqual(np.sum(infections)/10000, 0.2366, delta=0.01)

    def test_time_dist_prob_high(self):
        inf_times = np.zeros(shape=10000)

        a = 2
        loc = 1
        scale = 2
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_time = 2
        end_time = 5
        b = 2

        for i in range(10000):
            is_infected, inf_time = infection_process.sim_one_contact(start_time, end_time, gamma_dist, b)
            inf_times[i] = inf_time

        num_infs = 10000 - np.sum(np.isnan(inf_times))

        self.assertAlmostEqual(np.sum(inf_times<=3)/num_infs, 0.3455, delta = 0.02)
        self.assertAlmostEqual(np.sum((inf_times>=3) & (inf_times<=4))/num_infs, 0.3532, delta = 0.02)
        self.assertAlmostEqual(np.sum(inf_times>=4)/num_infs, 0.3014, delta = 0.02)

    def test_time_dist_prob_low(self):
        inf_times = np.zeros(shape=10000)

        a = 1
        loc = 2
        scale = 0.5
        gamma_dist = gamma(a=a, loc=loc, scale=scale)

        start_time = 3
        end_time = 6
        b = 2

        for i in range(10000):
            is_infected, inf_time = infection_process.sim_one_contact(start_time, end_time, gamma_dist, b)
            inf_times[i] = inf_time

        num_infs = 10000 - np.sum(np.isnan(inf_times))

        self.assertAlmostEqual(np.sum(inf_times<=4)/num_infs, 0.8668, delta = 0.02)
        self.assertAlmostEqual(np.sum((inf_times>=4) & (inf_times<=5))/num_infs, 0.1173, delta = 0.02)
        self.assertAlmostEqual(np.sum(inf_times>=5)/num_infs, 0.0159, delta = 0.02)
