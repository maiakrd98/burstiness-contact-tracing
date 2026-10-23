import sys
import unittest
import random
from scipy.stats import gamma
sys.path.append("src")  # noqa
import infection_process

class TestCalcForceOfInf(unittest.TestCase):

    def test_known_gamma_cdfs(self):
        for i in range(100):
            a = random.uniform(0, 5)
            loc = random.uniform(0, 5)
            scale = random.uniform(0, 5)
            gamma_dist = gamma(a = 2, loc = 0, scale = 1)
            start_time = random.uniform(0, 5)
            end_time = start_time + random.uniform(0, 5)
            b = random.uniform(0, 5)

            force = infection_process.calc_force_of_inf(start_time,
                                                        end_time,
                                                        gamma_dist,
                                                        b)

            self.assertAlmostEqual(force, b*(gamma_dist.cdf(end_time)-gamma_dist.cdf(start_time)))
