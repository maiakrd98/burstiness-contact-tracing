import numpy as np
import pandas as pd
from scipy.stats import gamma
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def calc_force_of_inf(start_time, end_time, gi_dist, b):
    force = b*integrate.quad(lambda x: gi_dist.pdf(x), start_time, end_time)
    return force[0]

def calc_prob_of_inf(start_time, end_time, gi_dist, b):
    pass

def calc_inf_time():
    pass

def sim_one_contact():
    pass

def sim_one_index_case():
    pass

def main():
    gi_dist = gamma(a = 2, loc = 0, scale = 1)
    force = calc_force_of_inf(0, 4, gi_dist, 1)
    print(force)

if __name__ =="__main__":
    main()