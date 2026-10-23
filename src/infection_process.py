""" Funciton get_gi takes in alpha, beta, and sigh and outputs individual gamma dist object.
     alpha: shape parameter
     beta: rate parameter
     sigh: burstiness parameter (between 0 and 1)

     """
# from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)

def get_gi(alpha, beta, psi):
    shape = (1.0 - psi) * alpha
    scale = 1.0/beta

    # get the lag (
    lag = np.random.gamma(shape, scale,1)

    # return lag

    
    gi_dist = gamma(a = psi*alpha, loc= lag, scale=scale)

    a= psi*alpha
    
    x = np.linspace(gamma.ppf(0.01, a),
                gamma.ppf(0.99, a), 100)
    ax.plot(x, gi_dist.pdf(x))
    plt.show()
    
    return gi_dist

    

if __name__ =="__main__":
    # Test the function a few times
    print("Testing get_gi function:")
    val = get_gi(alpha=2.0, beta=1.0, psi=0.1)
    print(val)