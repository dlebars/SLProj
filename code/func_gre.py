################################################################################
# func_gre.py: Define functions to project Greenland contribution to sea level
################################################################################
import numpy as np
import func_misc as misc

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

def gre_ar6(TIME_loc, a1_up, a1_lo, sce, NormD):
    '''Total Greenland contribution as in AR6 table 9.9.
    Compute a discontinuous two sided half-normal distribution from the likely range.
    These numbers in 2100 are referenced to the period 1995-2014
    while the code uses 1986-2005 as a reference period but since there is only
    1 mm between these two periods this is neglected.'''
    
    if sce == 'ssp126':
        l_range = [1., 6., 10.] # 17pc, med, 83pc
    elif sce == 'ssp245':
        l_range = [4., 8., 13.]
    elif sce == 'ssp585':
        l_range = [9., 13., 18.]
    elif sce == 'ssp585_hpp': # Low confidence in AR6
        l_range = [9., 15., 59.]
    else:
        print('Scenario not supported by ant_ar6')
    
    std_lo_2100 = l_range[1]-l_range[0]
    std_up_2100 = l_range[2]-l_range[1]
    
    X_gre = misc.proj2order_normal_assym(TIME_loc, a1_up, a1_lo, l_range[1], 
                                         std_lo_2100, std_up_2100, NormD)
    
    return X_gre