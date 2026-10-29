import numpy as np
from scipy.stats import gamma


def get_gi(alpha, beta, psi):
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

    # scale is used for both lag and distribution calculations
    scale = 1.0/beta

    # get the lag, drawn from the shape and scale defined as scale and shap_lag
    shape_lag = (1.0 - psi) * alpha
    lag = np.random.gamma(shape_lag, scale, 1)

    # uses shape value of psi*alpha, location of lag, and scale
    # defined above to calculate distribution
    alpha_dist = psi*alpha
    gi_dist = gamma(alpha_dist, lag, scale)

    return gi_dist


if __name__ == "__main__":
    val = get_gi(alpha=2, beta=1, psi=0.1)
    print(val)
