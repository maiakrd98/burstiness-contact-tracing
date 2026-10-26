import os
import sys
import unittest
import random
import numpy as np
import pandas as pd
sys.path.append("src")
import contact_process

class TestContactProcess(unittest.TestCase):
    def test_exponential_turnover(self):
        # random generator
        rng = np.random.default_rng(seed=0)

        # perform this test 100 times
        for trial in range(100):
            # select random inputs for the contact process function
            d = int(rng.integers(2, 9)) # d can be a value between 2 and 9
            lam = float(rng.uniform(0.5, 5.0)) # lam can be a vlaue between 0.5 and 5.0
            t_max = 100000 / lam # tmax is 100000 turnovers per run

            with self.subTest(trial=trial, d=d, lam=lam):
                # run the simulation
                contacts = contact_process.contact_simulation(d, lam, t_max, seed=0)

                # exclude contacts remaining at end of simulation
                filtered_contacts = contacts[(contacts["end_contact"] != t_max)]
        
                # calculate the contact length for each node
                contact_lengths = filtered_contacts["end_contact"] - filtered_contacts["start_contact"]

                # calculate sampling error of the mean and std
                expected = d / lam
                n = len(contact_lengths)
                se_mean = expected / np.sqrt(n)
                se_std = expected * np.sqrt(2 / n)

                # check that mean and standard deviation both equal d/lambda
                # which verifies its an exponential distribution
                self.assertAlmostEqual(np.mean(contact_lengths), expected, delta=5*se_mean)
                self.assertAlmostEqual(np.std(contact_lengths), expected, delta=5*se_std)

    def test_d_good(self):
        # d must be a positive integer
        for good_d in [1, 4, 8, np.int64(4)]:
            with self.subTest(d=good_d):
                df = contact_process.contact_simulation(good_d, 2.0, 10)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreaterEqual(len(df), good_d)
                
    def test_d_bad(self):
        # d must be a positive integer
        for bad_d in [0, -3, 4.0, 2.5, "4", None, True]:
            with self.subTest(d=bad_d):
                with self.assertRaises((TypeError, ValueError)):
                    contact_process.contact_simulation(bad_d, 2.0, 10)

    def test_lam_good(self):
        # lam must be a positive number
        for good_lam in [0.5, 1, 2.0, 5]:
            with self.subTest(lam=good_lam):
                df = contact_process.contact_simulation(4, good_lam, 10)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreaterEqual(len(df), 4)

    def test_lam_bad(self):
        # lam must be a positive number
        for bad_lam in [0, -1.5, float("nan"), float("inf"), "2", None, True]:
            with self.subTest(lam=bad_lam):
                with self.assertRaises((TypeError, ValueError)):
                    contact_process.contact_simulation(4, bad_lam, 10)

    def test_t_max_good(self):
        # t_max must be a positive number 
        for good_t in [0.5, 10, 10.5, 100]:
            with self.subTest(t_max=good_t):
                df = contact_process.contact_simulation(4, 2.0, good_t)
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreaterEqual(len(df), 4)

    def test_t_max_bad(self):
        # t_max must be a positive numbre
        for bad_t in [0, -10, float("nan"), float("inf"), "10", None, True]:
            with self.subTest(t_max=bad_t):
                with self.assertRaises((TypeError, ValueError)):
                    contact_process.contact_simulation(4, 2.0, bad_t)

    def test_d_contacts_end_at_t_max(self):
        # node 0 must end with d contacts
        test_parameters = [(4, 2.0, 50), (2, 0.5, 30), (7, 3.0, 20), (4, 2.0, 0.01)]

        for d, lam, t_max in test_parameters:
            for seed in range(5):
                with self.subTest(d=d, lam=lam, t_max=t_max, seed=seed):
                    contacts = contact_process.contact_simulation(d, lam, t_max, seed=seed)

                    # count contacts still active when the simulation stops
                    n_at_end = (contacts["end_contact"] == t_max).sum()

                    self.assertEqual(n_at_end, d)

    def test_end_times_match_start_times(self):
        # end times of nodes must match start times of new nodes
        t_max = 100

        for seed in range(10):
            with self.subTest(seed=seed):
                contacts = contact_process.contact_simulation(4, 2.0, t_max, seed=seed)

                # contacts that were replaced (not the d still active at t_max)
                replaced = contacts[contacts["end_contact"] < t_max]
                end_times = list(replaced["end_contact"])

                # contacts that replaced someone (not the d initial contacts at time 0)
                later = contacts[contacts["start_contact"] > 0]
                start_times = list(later["start_contact"])

                # same count, and every end time has a matching start time
                self.assertEqual(len(end_times), len(start_times))
                self.assertEqual(set(end_times), set(start_times))

if __name__ == "__main__":
    unittest.main()  