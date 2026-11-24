import sys
import numpy as np
import pandas as pd
# from scipy.stats import gamma
# from scipy.stats import truncate
from scipy import stats
# import scipy.integrate as integrate
import matplotlib.pyplot as plt
import contact_process


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
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive (>0)")

    # beta must be a positve number
    if isinstance(beta, bool) or not isinstance(beta, (int, float, np.number)):
        raise TypeError("beta must be a number")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive (> 0)")

    # psi must be a positive number from [0,1]
    if isinstance(psi, bool) or not isinstance(psi, (int, float, np.number)):
        raise TypeError("psi must be a number")
    if not (0 < psi <= 1):
        raise ValueError("psi must be a number in the interval (0,1]")

    rng = np.random.default_rng(rng)

    # scale is used for both lag and distribution calculations
    scale = 1.0/beta

    # get the lag, drawn from the shape and scale defined as scale and shap_lag
    shape_lag = (1.0 - psi) * alpha
    lag = rng.gamma(shape_lag, scale, 1)[0]

    # uses shape value of psi*alpha, location of lag, and scale
    # defined above to calculate distribution
    alpha_dist = psi*alpha

    gi_dist = stats.gamma(a=alpha_dist, loc=lag, scale=scale)

    return gi_dist


def calc_force_of_inf(start_t, end_t, gi_dist, b):
    """
    Calculates the force of infection for a given infectiousness \
    distribution, b value, and time interval.

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
    if not np.isfinite(b) or b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if not np.isfinite(start_t) or start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if not np.isfinite(end_t) or end_t < 0:
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
    Calculates the probability that a contact becomes infected over a given \
    time interval for a given infectiousness distribution and b value.

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
    if not np.isfinite(b) or b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if not np.isfinite(start_t) or start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if not np.isfinite(end_t) or end_t < 0:
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
        A random number generator to ensure reproducibility. Defaults to \
        None, in which case you will get different outputs each time you run \
        the function.

    Returns
    -------
    inf_time : float
        Time at which the contact was infected.
    """

    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if not np.isfinite(start_t) or start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if not np.isfinite(end_t) or end_t < 0:
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
    Simulates one contact with a given start and end time by randomly \
    choosing whether they were infected or not based on their probability \
    of infection. If they were infected, also generates an infection time.

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
        A random number generator to ensure reproducibility. Defaults to \
        None, in which case you will get different outputs each time you \
        run the function.

    Returns
    -------
    is_inf : boolean
        True if the contact was infected, False if not.

    inf_time : float
        Time at which the contact was infected. Returns None if is_inf is \
        false.
    """

    if isinstance(start_t, bool) or not isinstance(start_t, (int, float)):
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if not np.isfinite(start_t) or start_t < 0:
        sys.exit("Invalid input: start_t must be a nonnegative number.")
    if isinstance(end_t, bool) or not isinstance(end_t, (int, float)):
        sys.exit("Invalid input: end_t must be a nonnegative number.")
    if not np.isfinite(end_t) or end_t < 0:
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
    Simulates one index case, including whether or not each contact was \
    infected and their infection times.

    Parameters
    ----------
    df : Numpy data frame
        Must include the columns "id", "start_contact", and "end_contact". \
        Each row of this data frame represents a contact of the index case \
        and lists their id, what time their contact started and what time it \
        ended.

    alpha : float
        The shape parameter of the overall population generation interval \
        distribution (which is assumed to be gamma distributed). Must be \
        positive.

    beta : float
        The rate parameter of the overall population generation interval \
        distribution. Note that beta is equal to 1/scale in the scale \
        parameterization of the gamma distribution. Must be positive.

    psi : float
        The burstiness parameter. Must be in the interval (0,1]. The smaller \
        psi is, the more bursty the infection is.

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to \
        None, in which case you will get different outputs each time you run \
        the function.

    Returns
    -------
    new_df : Numpy data frame
        Numpy data frame with columns "id", "start_contact", "end_contact", \
        "is_inf", and "inf_time". Each row of this data frame represents a \
        contact of the index case. The first three columns are inherited \
        from df, "is_inf" indicates whether that contact was infected or not, \
        and "inf_time" gives their time of infection if so (if not infected, \
        "inf_time" is nan).
    """

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        sys.exit("Invalid input: b must be a positive number.")
    if not np.isfinite(b) or b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(psi, bool) or not isinstance(psi, (int, float)):
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if psi <= 0 or psi > 1:
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        sys.exit("Invalid input: beta must be a positive number.")
    if not np.isfinite(beta) or beta <= 0:
        sys.exit("Invalid input: beta must be a positive number.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        sys.exit("Invalid input: alpha must be a positive number.")
    if not np.isfinite(alpha) or alpha <= 0:
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


def sim_N_index_cases(N, d, lam, alpha, beta, psi, b, t_max=None, rng=None):
    """
    Simulates N index cases, including generating their contacts and \
    simulating whether or not each contact was infected and their infection \
    times.

    Parameters
    ----------
    N : int
        The number of index cases to simulate. Must be positive.

    d : int
        The number of contacts an index case has at any given time, which is \
        assumed to be constant. Must be positive.

    lam : float
        The turnover rate (lambda) at which contacts are replaced by new \
        contacts. Must be positive.

    alpha : float
        The shape parameter of the overall population generation interval \
        distribution (which is assumed to be gamma distributed). Must be \
        positive.

    beta : float
        The rate parameter of the overall population generation interval \
        distribution. Note that beta is equal to 1/scale in the scale \
        parameterization of the gamma distribution. Must be positive.

    psi : float
        The burstiness parameter. Must be in the interval (0,1]. The smaller \
        psi is, the more bursty the infection is.

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    t_max : float (optional)
        The length of time the simulation is run for each index case. Must be \
        positive. Defaults to None, in which case the function calculates the \
        time required to cover 99.9% of the infectiousness distribution and \
        uses that as t_max.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to \
        None, in which case you will get different outputs each time you run \
        the function.

    Returns
    -------
    all_cases : Numpy data frame
        Numpy data frame with columns "index_case_id", "contact_id", \
        "start_contact", "end_contact", "is_inf", and "inf_time". Each row of \
        this data frame represents a contact of an index case. The columns \
        "start_contact" and "end_contact" store the starting time and ending \
        time of the contact, "is_inf" indicates whether that contact was \
        infected or not, and "inf_time" gives their time of infection if so \
        (if not infected, "inf_time" is nan).
    """

    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        sys.exit("Invalid input: N must be a positive integer.")
    if N < 1:
        sys.exit("Invalid input: N must be a positive integer.")
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        sys.exit("Invalid input: d must be a positive integer.")
    if d < 1:
        sys.exit("Invalid input: d must be a positive integer.")
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        sys.exit("Invalid input: lam must be a positive number.")
    if not np.isfinite(lam) or lam <= 0:
        sys.exit("Invalid input: lam must be a positive number.")
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        sys.exit("Invalid input: b must be a positive number.")
    if not np.isfinite(b) or b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(psi, bool) or not isinstance(psi, (int, float)):
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if psi <= 0 or psi > 1:
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        sys.exit("Invalid input: beta must be a positive number.")
    if not np.isfinite(beta) or beta <= 0:
        sys.exit("Invalid input: beta must be a positive number.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        sys.exit("Invalid input: alpha must be a positive number.")
    if not np.isfinite(alpha) or alpha <= 0:
        sys.exit("Invalid input: alpha must be a positive number.")
    if t_max is not None:
        if isinstance(t_max, bool) or not isinstance(t_max, (int, float)):
            sys.exit("Invalid input: t_max must be a positive number.")
        if not np.isfinite(t_max) or t_max <= 0:
            sys.exit("Invalid input: t_max must be a positive number.")

    rng = np.random.default_rng(rng)

    if t_max is None:
        t_max = stats.gamma.ppf(0.999, a=alpha, scale=1/beta)

    all_cases_list = []

    for i in range(N):

        df = contact_process.contact_simulation(d, lam, t_max, rng=rng)
        new_df = sim_one_index_case(df, alpha, beta, psi, b, rng=rng)
        new_df.insert(loc=0, column='index_case_id', value=i)

        all_cases_list.append(new_df)

    all_cases = pd.concat(all_cases_list, ignore_index=True)
    all_cases = all_cases.rename(columns={"id": "contact_id"})

    return all_cases


def calc_R_from_df(df):
    '''
    Calculates the basic reproduction number R (average new infections caused \
    by one index case) from a data frame of index case contacts and whether \
    or not they were infected.

    Parameters
    ----------
    df : Numpy data frame
        Numpy data frame index cases and their contacts, including whether or \
        not they were infected. Must include the columns "index_case_id", \
        with ids of the index cases, and "is_inf", with booleans that \
        indicate whether or not each contact was infected.

    Returns
    -------
    R : float
        An estimate of the basic repoduction number, the mean number of new \
        infections caused by the index cases in df.
    '''

    if not isinstance(df, pd.DataFrame):
        sys.exit("Invalid input: df must be a data frame.")
    if "index_case_id" not in df.columns.values.tolist():
        sys.exit("Invalid input: df must contain column 'index_case_id'")
    if "is_inf" not in df.columns.values.tolist():
        sys.exit("Invalid input: df must contain column 'is_inf'")
    if df['is_inf'].dtype != bool:
        sys.exit("Invalid input: 'is_inf' column must contain booleans")

    N = len(set(df['index_case_id']))
    total_infs = sum(df['is_inf'])
    R = total_infs/N

    return R


def calc_R(N, d, lam, alpha, beta, psi, b, t_max=None, rng=None):
    '''
    Calculates the basic reproduction number R (average new infections caused \
    by one index case) for a set of parameters by simulating contacts and \
    infections.

    Parameters
    ----------
    N : int
        The number of index cases to simulate. Must be positive.

    d : int
        The number of contacts an index case has at any given time, which is \
        assumed to be constant. Must be positive.

    lam : float
        The turnover rate (lambda) at which contacts are replaced by new \
        contacts. Must be positive.

    alpha : float
        The shape parameter of the overall population generation interval \
        distribution (which is assumed to be gamma distributed). Must be \
        positive.

    beta : float
        The rate parameter of the overall population generation interval \
        distribution. Note that beta is equal to 1/scale in the scale \
        parameterization of the gamma distribution. Must be positive.

    psi : float
        The burstiness parameter. Must be in the interval (0,1]. The smaller \
        psi is, the more bursty the infection is.

    b : float
        Infectiousness scaling parameter, which you can tune to obtain the \
        desired reproduction number. Must be positive.

    t_max : float (optional)
        The length of time the simulation is run for each index case. Must be \
        positive. Defaults to None, in which case the function calculates the \
        time required to cover 99.9% of the infectiousness distribution and \
        uses that as t_max.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to \
        None, in which case you will get different outputs each time you run \
        the function.

    Returns
    -------
    R : float
        An estimate of the basic repoduction number, the mean number of new \
        infections caused by each index case.
    '''

    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        sys.exit("Invalid input: N must be a positive integer.")
    if N < 1:
        sys.exit("Invalid input: N must be a positive integer.")
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        sys.exit("Invalid input: d must be a positive integer.")
    if d < 1:
        sys.exit("Invalid input: d must be a positive integer.")
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        sys.exit("Invalid input: lam must be a positive number.")
    if not np.isfinite(lam) or lam <= 0:
        sys.exit("Invalid input: lam must be a positive number.")
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        sys.exit("Invalid input: b must be a positive number.")
    if not np.isfinite(b) or b <= 0:
        sys.exit("Invalid input: b must be a positive number.")
    if isinstance(psi, bool) or not isinstance(psi, (int, float)):
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if psi <= 0 or psi > 1:
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        sys.exit("Invalid input: beta must be a positive number.")
    if not np.isfinite(beta) or beta <= 0:
        sys.exit("Invalid input: beta must be a positive number.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        sys.exit("Invalid input: alpha must be a positive number.")
    if not np.isfinite(alpha) or alpha <= 0:
        sys.exit("Invalid input: alpha must be a positive number.")
    if t_max is not None:
        if isinstance(t_max, bool) or not isinstance(t_max, (int, float)):
            sys.exit("Invalid input: t_max must be a positive number.")
        if not np.isfinite(t_max) or t_max <= 0:
            sys.exit("Invalid input: t_max must be a positive number.")

    cases = sim_N_index_cases(N, d, lam, alpha, beta, psi, b, t_max, rng)
    R = calc_R_from_df(cases)

    return R


def determine_b(N, d, lam, alpha, beta, psi, R,
                start_b=2, tol=0.1, maxruns=10,
                t_max=None, rng=None):
    '''
    Estimates the infectiousness scaling parameter b required to obtain the \
    desired basic reproduction number R (with the other given parameters) \
    using a binary search algorithm and the function calc_R. Since there is \
    some randomness in calc_R, this estimate will not be precise, but it will \
    generally work better for larger N.

    Parameters
    ----------
    N : int
        The number of index cases to simulate. Must be positive.

    d : int
        The number of contacts an index case has at any given time, which is \
        assumed to be constant. Must be positive.

    lam : float
        The turnover rate (lambda) at which contacts are replaced by new \
        contacts. Must be positive.

    alpha : float
        The shape parameter of the overall population generation interval \
        distribution (which is assumed to be gamma distributed). Must be \
        positive.

    beta : float
        The rate parameter of the overall population generation interval \
        distribution. Note that beta is equal to 1/scale in the scale \
        parameterization of the gamma distribution. Must be positive.

    psi : float
        The burstiness parameter. Must be in the interval (0,1]. The smaller \
        psi is, the more bursty the infection is.

    R : float
        Desired basic reproduction number. Must be positive.

    start_b : float (optional)
        Initial guess for the value of b. Must be positive. Defaults to 2,

    tol : float (optional)
        The tolerance for how close the value of calc_R must be to the \
        desired value of R. Must be positive. Defaults to 0.1.

    maxruns : int (optional)
        Maximum steps of the binary search algorithm to run. Must be \
        positive. Defaults to 10.

    t_max : float (optional)
        The length of time the simulation is run for each index case. Must be \
        positive. Defaults to None, in which case the function calculates the \
        time required to cover 99.9% of the infectiousness distribution and \
        uses that as t_max.

    rng : Numpy Random Generator (optional)
        A random number generator to ensure reproducibility. Defaults to \
        None, in which case you will get different outputs each time you run \
        the function.

    Returns
    -------
    b : float
        An estimate of the infectiousness scaling parameter b required to \
        obtain a basic reproduction number of R, given the other parameters.
    '''

    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        sys.exit("Invalid input: N must be a positive integer.")
    if N < 1:
        sys.exit("Invalid input: N must be a positive integer.")
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        sys.exit("Invalid input: d must be a positive integer.")
    if d < 1:
        sys.exit("Invalid input: d must be a positive integer.")
    if isinstance(lam, bool) or not isinstance(lam, (int, float)):
        sys.exit("Invalid input: lam must be a positive number.")
    if not np.isfinite(lam) or lam <= 0:
        sys.exit("Invalid input: lam must be a positive number.")
    if isinstance(R, bool) or not isinstance(R, (int, float)):
        sys.exit("Invalid input: R must be a positive number.")
    if not np.isfinite(R) or R <= 0:
        sys.exit("Invalid input: R must be a positive number.")
    if isinstance(psi, bool) or not isinstance(psi, (int, float)):
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if psi <= 0 or psi > 1:
        sys.exit("Invalid input: psi must be a number in the interval (0,1].")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        sys.exit("Invalid input: beta must be a positive number.")
    if not np.isfinite(beta) or beta <= 0:
        sys.exit("Invalid input: beta must be a positive number.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        sys.exit("Invalid input: alpha must be a positive number.")
    if not np.isfinite(alpha) or alpha <= 0:
        sys.exit("Invalid input: alpha must be a positive number.")
    if isinstance(start_b, bool) or not isinstance(start_b, (int, float)):
        sys.exit("Invalid input: start_b must be a positive number.")
    if not np.isfinite(start_b) or start_b <= 0:
        sys.exit("Invalid input: start_b must be a positive number.")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        sys.exit("Invalid input: tol must be a positive number.")
    if not np.isfinite(tol) or tol <= 0:
        sys.exit("Invalid input: tol must be a positive number.")
    if isinstance(maxruns, bool) or not isinstance(maxruns, (int, np.integer)):
        sys.exit("Invalid input: maxruns must be a positive integer.")
    if maxruns < 1:
        sys.exit("Invalid input: maxruns must be a positive integer.")
    if t_max is not None:
        if isinstance(t_max, bool) or not isinstance(t_max, (int, float)):
            sys.exit("Invalid input: t_max must be a positive number.")
        if not np.isfinite(t_max) or t_max <= 0:
            sys.exit("Invalid input: t_max must be a positive number.")

    # first find value that overshoots R
    for n in range(10):
        R_over = calc_R(N, d, lam, alpha, beta, psi, start_b*2**n, t_max, rng)
        if R_over > R:
            break

    if n == 9 and R_over <= R:
        sys.exit("Could not get a large enough R value in the alloted time. "
                 "Try entering a larger value for start_b.")

    # binary search
    low_b = 0
    high_b = start_b*2**n
    runs = 0

    while runs < maxruns:
        mid_b = (low_b+high_b)/2
        mid_R = calc_R(N, d, lam, alpha, beta, psi, mid_b, t_max, rng)

        if abs(mid_R-R) < tol:
            return mid_b
        elif R < mid_R:
            high_b = mid_b
        else:
            low_b = mid_b

        runs += 1

    sys.exit("Could not find b in the allotted runs. "
             "Try increasing maxruns or increading tol.")


def main():
    rng = np.random.default_rng(45698380564557)

    contacts = [[1, 0, 2.3],
                [2, 0, 3.1],
                [3, 2.3, 4.7],
                [4, 3.1, 4.6],
                [5, 4.6, 6],
                [6, 4.7, 6]]
    df = pd.DataFrame(contacts, columns=["id", "start_contact", "end_contact"])

    alpha = 2.39
    beta = 0.336
    psi = 0.1
    b = 2

    N = 100
    d = 4
    lam = 2

    R = 5

    cases = sim_N_index_cases(N, d, lam, alpha, beta, psi, b, rng=rng)
    print(cases)

    b = determine_b(N, d, lam, alpha, beta, psi, R, tol=0.05, rng=rng)
    print(b)


'''
    b_vals = [0.1*x for x in range(1, 100)]
    R_vals = []
    for b in b_vals:
        R_vals.append(calc_R(N, d, lam, alpha, beta, psi, b, rng=rng))

    print(b_vals)
    print(R_vals)
    plt.scatter(b_vals, R_vals)
    plt.show()

    d_vals = range(1, 20)
    R_vals = []
    for d in d_vals:
        cases = sim_N_index_cases(N, d, lam, alpha, beta, psi, b, rng=rng)
        R_vals.append(calc_R_from_df(cases))

    print(d_vals)
    print(R_vals)
    plt.scatter(d_vals, R_vals)
    plt.show()

    psi_vals = [0.025*x for x in range(1, 40)]
    R_vals = []
    for psi in psi_vals:
        R_vals.append(calc_R(N, d, lam, alpha, beta, psi, b, rng=rng))

    print(psi_vals)
    print(R_vals)
    plt.scatter(psi_vals, R_vals)
    plt.show()

'''


if __name__ == "__main__":
    main()
