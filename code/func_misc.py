###############################################################################
# func_misc.py: Miscellaneous functions
###############################################################################

import numpy as np

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