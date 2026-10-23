""" Funciton get_gi takes in alpha, beta, and sigh and outputs individual gamma dist object.
     alpha: shape parameter
     beta: rate parameter
     sigh: burstiness parameter (between 0 and 1)

     """

import numpy as np

def get_gi(alpha, beta, sigh):
    shape = (1.0 - sigh) * alpha
    scale = 1.0/beta
    gi = np.random.gamma(shape = shape, scale=scale)
    
 