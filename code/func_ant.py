################################################################################
# func_ant.py: Defines functions to return probabilistic Antarctic contribution 
#             to sea level
################################################################################
#import func_misc as misc
import numpy as np
from scipy.stats import norm

def ant_smb_ar5(NormDl, fac, Td_a):
    '''Define Antarctic surface mass balance contribution to sea level as in IPCC 
    AR5'''
    N = len(NormDl)
    nb_y2 = Td_a.shape[1]
    
    Cref     = 1923         # Reference accumulation (Gt/year)
    Delta_C  = 5.1/100      # Increase in accumulation with local temperature (%/degC)
    Delta_Ce = 1.5/100      # Error bar
    AmpA     = 1.1          # Antarctic amplification of global temperature
    AmpAe    = 0.2          # Amplification uncertainty
    
    N = len(NormDl)
    
    NormD2 = np.random.normal(AmpA, AmpAe, N)
    NormDl = Delta_C + NormDl*Delta_Ce

    X_asmb = np.zeros([N,nb_y2])
    for t in range(0, nb_y2):
        X_asmb[:,t] = fac * Cref * Td_a[:,t] * NormD2 * NormDl

    X_asmb = X_asmb.cumsum(axis=1) # Cumulate over time
    X_asmb = X_asmb*100            # Convert from m to cm
    
    return X_asmb

def ant_dyn_knmi14(SCE, a1_up_a, a1_lo_a, start_date2, end_date2, TIME_loc, N):
    '''Antarctic dyncamis projection as implemented by the KNMI14 scenarios '''

    nb_y_loc   = len(TIME_loc)
    alpha_95   = norm.ppf(0.95)
    alpha_98   = norm.ppf(0.98)

    # Local values
    Delta_ant_up_KNMI_2100 = 40.
    Delta_ant_lo_2100      = -3.4   # Same as de Vries et al. 2014
    Delta_ant_up_2100      = 16.7

    # Compute the second order coefficient of the equations:
    a2_up_a  = (Delta_ant_up_2100 - a1_up_a*(2100-start_date2))/(2100 - start_date2)**2
    a2_lo_a  = (Delta_ant_lo_2100 - a1_lo_a*(2100-start_date2))/(2100 - start_date2)**2
    Delta_ant_up = a1_up_a*(TIME_loc-start_date2) + a2_up_a*(TIME_loc-start_date2)**2
    Delta_ant_lo = a1_lo_a*(TIME_loc-start_date2) + a2_lo_a*(TIME_loc-start_date2)**2

    a2_up_a_KNMI  = (Delta_ant_up_KNMI_2100 \
                     - a1_up_a*(2100-start_date2))/(2100 - start_date2)**2
    Delta_ant_up_KNMI = a1_up_a*(TIME_loc-start_date2) \
    + a2_up_a_KNMI*(TIME_loc-start_date2)**2

    # Build distribution that conserves the mode of the KNMI distribution
    Delta_ant_cen = (Delta_ant_up-Delta_ant_lo)/2
    tau_ant  = Delta_ant_lo
    Diff     = Delta_ant_cen-Delta_ant_lo
    # Avoid 0's to divide later, this happens only during the first years
    Diff     = np.where(Diff <= 0.1, 0.1, Diff)     
    mu_ant   = np.log(Diff)
    sig_ant  = 1/alpha_98*np.log((Delta_ant_up_KNMI-Delta_ant_lo)/Diff)

    NormD2 = np.random.normal(0, 1, N)
    X_ant = np.zeros([N, nb_y_loc])
    for t in range(0, nb_y_loc):
        X_ant[:,t] = tau_ant[t] + np.exp(mu_ant[t] + sig_ant[t]*NormD2) 

    return X_ant

def ant_dyn_srocc(SCE, a1_up_a, a1_lo_a, TIME_loc, N):
    '''Compute the antarctic dynamics contribution to global sea level based on
    the SROCC report. 
    Assume here a normal distribution for simplicity, this is not the case in 
    the SROCC report so it should be modified. Also only works for RCP8.5 for now. 
    Assume that this contribution is independent from the others.'''
    
    if SCE == 'rcp85':
        nb_y_loc = len(TIME_loc)
        X_ant_m = 16
        X_ant_sig = 14
        NormD_loc = np.random.normal(0, 1, N)

        X_anti = (a1_up_a+a1_lo_a)/2 + NormD_loc*(a1_up_a-a1_lo_a)/2
        X_antf = np.zeros(N)
        X_ant = np.zeros([N,nb_y_loc])
        alp = 2  # Use a second order polynomial as for AR5
        X_antf = X_ant_m + NormD_loc*X_ant_sig
        
        for t in range(nb_y_loc):
            X_ant[:,t] =  ( X_anti*(TIME_loc[t]-TIME_loc[0]) + 
                           ((X_antf - X_anti*(2100-TIME_loc[0])) / 
                            (2100-TIME_loc[0])**alp) 
                           * (TIME_loc[t]-TIME_loc[0])**alp )
        
    else:
        print('ERROR: The ant_dyn_srocc function is only supported for rcp8.5')
        
    return X_ant