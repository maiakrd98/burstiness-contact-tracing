import numpy as np
import pandas as pd
#from scipy.stats import gamma
#from scipy.stats import truncate
from scipy import stats
# import scipy.integrate as integrate
import matplotlib.pyplot as plt


def calc_force_of_inf(start_time, end_time, gi_dist, b):
    # TODO: deal with edge cases (e.g. end time larger than start,
    # gi_dist wrong type, b<0)
    force = b*(gi_dist.cdf(end_time)-gi_dist.cdf(start_time))
    return force


def calc_prob_of_inf(start_time, end_time, gi_dist, b):
    force = calc_force_of_inf(start_time, end_time, gi_dist, b)
    prob = 1 - np.exp(-force)
    return prob


def calc_inf_time(start_contact, end_contact, gi_dist):
    while True:
        inf_time_proposal = gi_dist.rvs(size=1)
        if (inf_time_proposal >= start_contact and inf_time_proposal<= end_contact):
            break

    inf_time = inf_time_proposal
    
    return inf_time


def sim_one_contact():
    pass


def sim_one_index_case():
    pass


def main():

    gi_dist = stats.gamma(a=2.0, scale=1.0)

    inf_time = calc_inf_time(3, 4, gi_dist)

    print(inf_time)


if __name__ == "__main__":
    main()
