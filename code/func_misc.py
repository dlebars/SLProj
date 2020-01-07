###############################################################################
# func_misc.py: Miscellaneous functions
###############################################################################
import numpy as np
from scipy.stats import norm

def TempDist(TGLOBs, Tref, GAM, NormD):
    '''Build a distribution of global temperature for a contributor (reference periods 
     are different of each contributors)'''
    N        = len(NormD)
    nb_MOD   = TGLOBs.shape[0]
    nb_y2    = TGLOBs.shape[1]

    TGLOBl   = np.zeros([nb_MOD, nb_y2])
    for m in range(0, nb_MOD):
        TGLOBl[m, :]    = TGLOBs[m, :] - Tref[m]

    TGLOB_m  = TGLOBl.mean(axis=0) # Compute the inter-model mean for each time
    TGLOB_sd = TGLOBl.std(axis=0)  # Compute the inter-model standard deviation

    Td       = np.zeros([N, nb_y2])
    for t in range(0, nb_y2):
        Td[:,t]  = TGLOB_m[t] + GAM * NormD * TGLOB_sd[t]

    return Td

def landw_ar5(ys, TIME2, N):
    '''Land water contribution to sea level as in IPCC AR5. Second order polynomial to 
    reach a target value.'''

    nb_y2 = len(TIME2)
    alpha_95 = norm.ppf(0.95)
    alpha_05 = norm.ppf(0.05)
    
    a1_up             = 0.049 # First order term (cm/y), equal to observations in 2006
    a1_lo             = 0.026
    Delta_grw_up_2100 = 9
    Delta_grw_lo_2100 = -1

    # Compute the second order coefficient of the equations:
    a2_up  = (Delta_grw_up_2100 - a1_up*(2100-ys))/(2100 - ys)**2
    a2_lo  = (Delta_grw_lo_2100 - a1_lo*(2100-ys))/(2100 - ys)**2

    Delta_grw_up = a1_up*(TIME2-ys) + a2_up*(TIME2-ys)**2
    Delta_grw_lo = a1_lo*(TIME2-ys) + a2_lo*(TIME2-ys)**2

    # Build the distibution including other uncertainties
    sig2   = (Delta_grw_up - Delta_grw_lo)/(alpha_95 - alpha_05)
    NormD2 = np.random.normal(0, 1, N)       # The standard deviations sould be scaled.

    X_landw = np.zeros([N,nb_y2])  # Independent of the scenario, size is to add up easily.
    for t in range(0, nb_y2):
        X_landw[:,t] = (Delta_grw_up[t] - Delta_grw_lo[t])/2 + sig2[t]*NormD2
        
    return X_landw