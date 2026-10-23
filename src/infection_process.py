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

    return gi
    
if __name__ =="__main__":
    # Test the function a few times
    print("Testing get_gi function:")
    for i in range(5):
        val = get_gi(alpha=2.0, beta=1.0, sigh=0.1)
        print(f"Sample {i+1}: {val}")