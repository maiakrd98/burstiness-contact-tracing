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

def calc_inf_time(start_contact, end_contact, gi_dist, rng=None):
    rng = np.random.default_rng(rng)
    while True:
        inf_time_proposal = gi_dist.rvs(size=1, random_state=rng)
        if (inf_time_proposal >= start_contact and inf_time_proposal<= end_contact):
            break

    inf_time = inf_time_proposal[0]
    
    return inf_time


def sim_one_contact(start_contact, end_contact, gi_dist, b, rng=None):
    rng = np.random.default_rng(rng)
    prob_of_inf = calc_prob_of_inf(start_contact, end_contact, gi_dist, b)
    is_infected = np.random.choice([True, False], p=[prob_of_inf, 1-prob_of_inf])

    if is_infected:
        inf_time = calc_inf_time(start_contact, end_contact, gi_dist, rng)
        return is_infected, inf_time

    else:
        return is_infected, None


def sim_one_index_case(df, alpha, beta, psi, b, rng=None):
    # TODO: add in get_gi call once merged
    # gi_dist = get_gi(alpha, beta, psi)

    # TODO: check that data frame columns are correct

    rng = np.random.default_rng(rng)

    gi_dist = stats.gamma(a=psi*alpha, loc=0, scale=1.0/beta)

    new_df = df
    is_inf_list = []
    inf_time_list = []

    for i in range(len(df)):
        is_inf, inf_time = sim_one_contact(df.start_contact[i], df.end_contact[i], gi_dist, b, rng)
        is_inf_list.append(is_inf)
        inf_time_list.append(inf_time)

    new_df['is_inf'] = is_inf_list
    new_df['inf_time'] = inf_time_list

    return new_df


def main():
    #np.random.seed(42)
    rng = np.random.default_rng(42)

    contacts = [[1, 0, 2.3],
                [2, 0, 3.1],
                [3, 2.3, 4.7],
                [4, 3.1, 4.6],
                [5, 4.6, 6],
                [6, 4.7, 6]]
    df = pd.DataFrame(contacts, columns=["id", "start_contact", "end_contact"])

    alpha = 2
    beta = 2
    psi = 0.5
    b = 10

    print(df)
    new_df = sim_one_index_case(df, alpha, beta, psi, b, rng)
    print(new_df)

    gi_dist = stats.gamma(a=2, loc=0, scale=2)
    for _ in range(5):
        inf_time = calc_inf_time(0, 5, gi_dist, rng)
        print(inf_time)


if __name__ == "__main__":
    main()
