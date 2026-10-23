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
