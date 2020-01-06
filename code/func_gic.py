###############################################################################
# func_gic.py: Functions that compute the glaciers and ice caps contribution
###############################################################################
import numpy as np

def fett13(Td_gic, NormDs)
    '''Contribution based on equation 2 of Fettweis et al. 2013 as used in AR5.
    Converts the global temperature to glacier contribution.'''

    N = len(NormDs)
    
    # Values of f and p, fitting parameter from formula B.1 (de Vries et al. 2014)
    f    = np.array([3.02,4.96,5.45,3.44])
    p    = np.array([0.733,0.685,0.676,0.742])
    
    Icum = Td_gic.cumsum(axis=1)

    # The following formula is not defined for negative values of Icum, which happens for
    # a few extreme values of the distribution.
    Icum = np.where(Icum < 0, 0, Icum)

    DEL_gic   = (f[0] * Icum**p[0] + f[1] * Icum**p[1] + f[2] * Icum**p[2] + \
                 f[3] * Icum**p[3])/4
    DEL_gic   = DEL_gic * .1   # Convert from mm to cm

    X_gic   = np.zeros([N,nb_y2])
    NormDl  = 1 + NormDs * 0.2
    for t in range(0,nb_y2):         # Use broadcasting?
        X_gic[:,t] = DEL_gic[:,t] * NormDl

    X_gic = X_gic + 0.95 # Add 0.95 cm for the changes between 1996 to 2006

    return X_gic