import numpy as np
import pandas as pd
from scipy.stats import gamma
import scipy.integrate as integrate
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
    pass


def calc_inf_time():
    pass


def sim_one_contact():
    pass


def sim_one_index_case():
    pass


def main():
    gi_dist = gamma(a=0.11504419425716839, loc=4.925310155292234, scale=2.247587519653209)
    force = calc_force_of_inf(3.0003090192607216, 7.588729997993129, gi_dist, 3.727677573564176)
    print(force)


if __name__ == "__main__":
    main()
