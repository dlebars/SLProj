###############################################################################
# func_gic.py: Functions that compute the glaciers and ice caps contribution
###############################################################################
import numpy as np

def gic_ipcc(Td_gic, NormDs, ar):
    '''Contribution based on IPCC AR5 method using four global glacier models'''

    N = len(NormDs)
    nb_y2 = Td_gic.shape[1]
    
    if ar == 'AR5':
        # Values of f and p, fitting parameter from formula B.1 (de Vries et al. 2014)
        f = np.array([3.02,4.96,5.45,3.44])
        p = np.array([0.733,0.685,0.676,0.742])
        ref = 0.95 # Add 0.95 cm for the changes between 1996 to 2006
        
    elif ar == 'AR6':
        # Values from AR6 table 9.A.4
        f    = np.array([3.7, 4.08, 5.5, 4.89, 4.26, 5.18, 2.66])
        p    = np.array([0.66, 0.72, 0.56, 0.65, 0.72, 0.71, 0.73])
        ref = 0.74 # Add this for the changes between 1996 to 2006 (unit:cm)
        # Average between Zemp et al. 2019 and Marzeion et al. 2015
    
    Icum = Td_gic.cumsum(axis=1)

    # The following formula is not defined for negative values of Icum, which happens for
    # a few extreme values of the distribution.
    Icum = np.where(Icum < 0, 0, Icum)

    DEL_gic = 0
    for i in range(len(f)):
        DEL_gic += f[i] * Icum**p[i]
    DEL_gic = DEL_gic/len(f)
    
    DEL_gic = DEL_gic * .1   # Convert from mm to cm

    X_gic   = np.zeros([N,nb_y2])
    NormDl  = 1 + NormDs * 0.2
    for t in range(0,nb_y2):         # Use broadcasting?
        X_gic[:,t] = DEL_gic[:,t] * NormDl

    X_gic = X_gic + ref

    return X_gic