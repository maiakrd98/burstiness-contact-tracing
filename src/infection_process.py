""" Funciton get_gi takes in alpha, beta, and sigh and outputs individual gamma dist object.
     alpha: shape parameter
     beta: rate parameter
     sigh: burstiness parameter (between 0 and 1)

     """

import numpy as np
from scipy.stats import gamma

def get_gi(alpha, beta, psi):
    
    # scale is used for both lag and distribution calculations
    scale = 1.0/beta
    
    # get the lag which is drawn from the shape and scale as defined as scale and shap_lag
    shape_lag = (1.0 - psi) * alpha
    lag = np.random.gamma(shape_lag, scale, 1)

    # uses shape value of psi*alpha, location of lag, and scale as defined above to calculate distribution
    alpha_dist = psi*alpha
    gi_dist = gamma(alpha_dist, lag, scale)
    
    return gi_dist

    

if __name__ =="__main__":
    val = get_gi(alpha=2.0, beta=1.0, psi=0.1)
    print(val)
