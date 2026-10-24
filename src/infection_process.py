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

# TODO: add seed!!

def calc_inf_time(start_contact, end_contact, gi_dist):
    while True:
        inf_time_proposal = gi_dist.rvs(size=1)
        if (inf_time_proposal >= start_contact and inf_time_proposal<= end_contact):
            break

    inf_time = inf_time_proposal
    
    return inf_time


def sim_one_contact(start_contact, end_contact, gi_dist, b):
    prob_of_inf = calc_prob_of_inf(start_contact, end_contact, gi_dist, b)
    is_infected = np.random.choice([True, False], p=[prob_of_inf, 1-prob_of_inf])

    if is_infected:
        inf_time = calc_inf_time(start_contact, end_contact, gi_dist)
        return is_infected, inf_time

    else:
        return is_infected, None


def sim_one_index_case():
    pass


def main():

    gi_dist = stats.gamma(a=2.0, scale=1.0)

    is_infected = sim_one_contact(1, 3, gi_dist, 1, seed = 34)

    print(is_infected)


if __name__ == "__main__":
    main()
