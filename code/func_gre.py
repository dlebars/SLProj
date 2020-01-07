################################################################################
# func_gre.py: Define functions to project Greenland contribution to sea level
################################################################################
import numpy as np

def fett13( fac, Td_g, NormDl, GRE):
    '''Function used by AR5 and de Vries et al. 2014 for the KNMI scenarios based 
    on Fettweis et al. 2013'''

    N = len(NormDl)
    nb_y2 = Td_g.shape[1]
    
    #Values of the 3 parameters from formula B.2 (de Vries et al. 2014)
    a = -71.5
    b = -20.4
    c = -2.8

    #Convert the global temperature to Greenland SMB changes
    G_SMB     = fac * (a * Td_g + b * Td_g**2 + c * Td_g**3)
    G_SMB_cum = G_SMB.cumsum(axis=1)
        
    #Build the distibution including other uncertainties
    Unif   = np.random.uniform(1, 1.15, N)
    LogNor = np.exp(NormDl * 0.4)
    X_gsmb = np.zeros([N,nb_y2])
    for t in range(0, nb_y2):
        X_gsmb[:,t] = G_SMB_cum[:,t] * Unif * LogNor

    X_gsmb = X_gsmb * 100 # Convert from m to cm

    if GRE == 'KNMI14':
        X_gsmb = X_gsmb + 0.291 # 0.291 is used for KNMI14 because nothing is added to 
                                # the dynamics while IPCC uses 0.15 cm as well there
    elif GRE == 'IPCC':
        X_gsmb = X_gsmb + 0.15  # Add 0.15 cm for contribution between 1995-2005, 
    else:
        print("ERROR: Wrong GRE option")

    return X_gsmb

