import sys
import numpy as np
import pandas as pd
# from scipy.stats import gamma
# from scipy.stats import truncate
from scipy import stats
# import scipy.integrate as integrate
import matplotlib.pyplot as plt


def get_gi(alpha, beta, psi, rng=None):
    """Calculates the gamma distribution object

    Parameters
    ----------
    alpha: float
        Shape Parameter
        This value must be postive
    beta: float
        Rate Parameter, is equal to 1/theta
        This values must be positive
    psi: float
        Burstiness parameter
        This must be a value from 0-1
    Returns
    -------
    Continuous frozen distribution

    """
    # parameter validation
    # alpha must be a positive number
    if (isinstance(alpha, bool)
            or not isinstance(alpha, (int, float, np.number))):
        raise TypeError("alpha must be a number")
    if alpha <= 0:
        raise ValueError("alpha must be positive (>0)")

    # beta must be a positve number
    if isinstance(beta, bool) or not isinstance(beta, (int, float, np.number)):
        raise TypeError("beta must be a number")
    if beta <= 0:
        raise ValueError("beta must be positive (> 0)")

    # psi must be a positive number from [0,1]
    if isinstance(psi, bool) or not isinstance(psi, (int, float, np.number)):
        raise TypeError("psi must be a number")
    if not (0 <= psi <= 1):
        raise ValueError("psi must be between 0 and 1 inclusive")

    rng = np.random.default_rng(rng)

    # scale is used for both lag and distribution calculations
    scale = 1.0/beta

    # get the lag, drawn from the shape and scale defined as scale and shap_lag
    shape_lag = (1.0 - psi) * alpha
    lag = rng.gamma(shape_lag, scale, 1)[0]
    # why is this always zero??

    # uses shape value of psi*alpha, location of lag, and scale
    # defined above to calculate distribution
    alpha_dist = psi*alpha

    gi_dist = stats.gamma(a=alpha_dist, loc=lag, scale=scale)

    return gi_dist

# TODO: update documentation and code review


def calc_force_of_inf(start_t, end_t, gi_dist, b):
    """
    Calculates the force of infection for a given infectiousness distribution, \
    b value, and time interval.

    Parameters
    ----------
    start_t : float
        Start of time interval. Must be greater than or equal to zero.

    end_t : float
        End of time interval. Must be greater than or equal to start_t.

    gi_dist : scipy.stats frozen continuous random variable distribution
        Infectiousness distribution.

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    Returns
    -------
    force : float
        The force of infection over the given time interval.
    """

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        sys.exit("Invalid input: b must be a positive number.")
    if b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < 0:
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < start_t:
        sys.exit("Invalid input: end_t must be greater than or equal to"
                 " start_t.")
    if not isinstance(gi_dist,
                      stats._distn_infrastructure.rv_continuous_frozen):
        sys.exit("Invalid input: gi_dist must be a scipy.stats frozen "
                 "continuous random variable object.")

    force = b*(gi_dist.cdf(end_t)-gi_dist.cdf(start_t))
    return force


def calc_prob_of_inf(start_t, end_t, gi_dist, b):
    """
    Calculates the probability that a contact becomes infected over a given time \
    interval for a given infectiousness distribution and b value.

    Parameters
    ----------
    start_t : float
        Start of time interval. Must be greater than or equal to zero.

    end_t : float
        End of time interval. Must be greater than or equal to start_t.

    gi_dist : scipy.stats frozen continuous random variable distribution
        Infectiousness distribution.

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    Returns
    -------
    prob : float
        The probability of infection over the given time interval.
    """

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        sys.exit("Invalid input: b must be a positive number.")
    if b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < 0:
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < start_t:
        sys.exit("Invalid input: end_t must be greater than or equal to"
                 " start_t.")
    if not isinstance(gi_dist,
                      stats._distn_infrastructure.rv_continuous_frozen):
        sys.exit("Invalid input: gi_dist must be a scipy.stats frozen "
                 "continuous random variable object.")

    force = calc_force_of_inf(start_t, end_t, gi_dist, b)
    prob = 1 - np.exp(-force)
    return prob


def calc_inf_time(start_t, end_t, gi_dist, rng=None):
    """
    Generates the time at which a contact was infected by drawing from the \
    infectiousness distribution, truncated to the given time interval.

    Parameters
    ----------
    start_t : float
        Start time of the contact. Must be greater than or equal to zero.

    end_t : float
        End time of the contact. Must be greater than or equal to start_t.

    gi_dist : scipy.stats frozen continuous random variable distribution
        Infectiousness distribution.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to None, \
        in which case you will get different outputs each time you run the function.

    Returns
    -------
    inf_time : float
        Time at which the contact was infected.
    """

    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < 0:
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < start_t:
        sys.exit("Invalid input: end_t must be greater than or equal to"
                 " start_t.")
    if not isinstance(gi_dist,
                      stats._distn_infrastructure.rv_continuous_frozen):
        sys.exit("Invalid input: gi_dist must be a scipy.stats frozen "
                 "continuous random variable object.")

    rng = np.random.default_rng(rng)

    while True:
        inf_time_proposal = gi_dist.rvs(size=1, random_state=rng)
        if (inf_time_proposal >= start_t and inf_time_proposal <= end_t):
            break

    inf_time = inf_time_proposal[0]
    return inf_time


def sim_one_contact(start_t, end_t, gi_dist, b, rng=None):
    """
    Simulates one contact with a given start and end time by randomly choosing \
    whether they were infected or not based on their probability of infection. \
    If they were infected, also generates an infection time.

    Parameters
    ----------
    start_t : float
        Start time of the contact. Must be greater than or equal to zero.

    end_t : float
        End time of the contact. Must be greater than or equal to start_t.

    gi_dist : scipy.stats frozen continuous random variable distribution
        Infectiousness distribution.

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to None, \
        in which case you will get different outputs each time you run the function.

    Returns
    -------
    is_inf : boolean
        True if the contact was infected, False if not.
    
    inf_time : float
        Time at which the contact was infected. Returns None if is_inf is false.
    """

    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < 0:
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if end_t < start_t:
        sys.exit("Invalid input: end_t must be greater than or equal to"
                 " start_t.")
    if not isinstance(gi_dist,
                      stats._distn_infrastructure.rv_continuous_frozen):
        sys.exit("Invalid input: gi_dist must be a scipy.stats frozen "
                 "continuous random variable object.")

    rng = np.random.default_rng(rng)

    prob_of_inf = calc_prob_of_inf(start_t, end_t, gi_dist, b)
    is_inf = rng.choice([True, False], p=[prob_of_inf, 1-prob_of_inf])

    if is_inf:
        inf_time = calc_inf_time(start_t, end_t, gi_dist, rng)
        return is_inf, inf_time

    else:
        return is_inf, None


def sim_one_index_case(df, alpha, beta, psi, b, rng=None):
    """
    Simulates one index case, including whether or not each contact was infected \
    and their infection times.

    Parameters
    ----------
    df : Numpy data frame
        Must include the columns "id", "start_contact", and "end_contact". Each row \
        of this data frame represents a contact of the index case and lists their id, \
        what time their contact started and what time it ended. 

    alpha : float
        The shape parameter of the overall population generation interval distribution \
        (which is assumed to be gamma distributed). Must be positive.

    beta : float
        The rate parameter of the overall population generation interval distribution. \
        Note that beta is equal to 1/scale in the scale parameterization of the gamma \
        distribution. Must be positive.

    psi : float
        The burstiness parameter. Must be in the interval (0,1]. The smaller psi is, \
        the more bursty the infection is. 

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to None, \
        in which case you will get different outputs each time you run the function.

    Returns
    -------
    new_df : Numpy data frame
        Numpy data frame with columns "id", "start_contact", "end_contact", "is_inf", \
        and "inf_time". Each row of this data frame represents a contact of the index \
        case. The first three columns are inherited from df, "is_inf" indicates whether \
        that contact was infected or not, and "inf_time" gives their time of infection \
        if so (if not infected, "inf_time" is nan).
    """

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        sys.exit("Invalid input: b must be a positive number.")
    if b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(psi, bool) or not isinstance(psi, (int, float)):
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if psi <= 0 or psi > 1:
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        sys.exit("Invalid input: beta must be a positive number.")
    if beta <= 0:
        sys.exit("Invalid input: beta must be a positive number.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        sys.exit("Invalid input: alpha must be a positive number.")
    if alpha <= 0:
        sys.exit("Invalid input: alpha must be a positive number.")
    if not isinstance(df, pd.DataFrame):
        sys.exit("Invalid input: df must be a data frame.")
    if "start_contact" not in df.columns.values.tolist():
        sys.exit("Invalid input: df must contain column 'start_contact'")
    if "end_contact" not in df.columns.values.tolist():
        sys.exit("Invalid input: df must contain column 'end_contact'")

    rng = np.random.default_rng(rng)

    gi_dist = get_gi(alpha, beta, psi, rng=rng)

    new_df = df.copy()
    is_inf_list = []
    inf_time_list = []

    for i in range(len(df)):
        is_inf, inf_time = sim_one_contact(df.start_contact[i],
                                           df.end_contact[i],
                                           gi_dist,
                                           b,
                                           rng)
        is_inf_list.append(is_inf)
        inf_time_list.append(inf_time)

    new_df['is_inf'] = is_inf_list
    new_df['inf_time'] = inf_time_list

    return new_df


def main():
    rng = np.random.default_rng(456982309564557)

    contacts = [[1, 0, 2.3],
                [2, 0, 3.1],
                [3, 2.3, 4.7],
                [4, 3.1, 4.6],
                [5, 4.6, 6],
                [6, 4.7, 6]]
    df = pd.DataFrame(contacts, columns=["id", "start_contact", "end_contact"])

    alpha = 1
    beta = 2
    psi = 0.9
    b = 10

    
    new_df = sim_one_index_case(df, alpha, beta, psi, b, rng)
    print(new_df)



if __name__ == "__main__":
    main()

# TODO: add documentation