################################################################################
# func_ant.py: Defines functions to return probabilistic Antarctic contribution 
#             to sea level
################################################################################
import numpy as np
from scipy.stats import norm
import xarray as xr

import func_misc as misc

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

def read_larmip_lrf(data_dir):
    '''Read LARMIP Linear Response Functions'''
    
    f = xr.open_dataset(f'{data_dir}LFR_Lev14/RFunctions.nc', decode_times=False)
    ic_models = ['AIF', 'PS', 'PISM', 'SICO', 'UMISM']

    for icm in ic_models:
        RF_tmp = f[f'RF_{icm}'].assign_coords({'model' : icm})
        RF_tmp = RF_tmp.expand_dims('model', axis=0)
        try:
            RF = xr.concat([RF, RF_tmp], dim='model', combine_attrs='drop')
        except:
            RF = RF_tmp

    RF.name = 'RF'
    
    return RF

def read_larmip_coeff(data_dir):
    '''Read the coefficient scalling GMST to temperature around Antarctica'''
    
    f = xr.open_dataset(f'{data_dir}LFR_Lev14/RFunctions.nc', decode_times=False)
    
    return f.coeff


def read_larmip2_lrf(data_dir):
    '''Read LAMIP 2 Linear Response Functions downloaded from:
    https://github.com/ALevermann/Larmip2019'''
    
    
    return RF

def ant_dyn_larmip(SCE, MOD, start_date2, GAM, NormD, UnifDd, data_dir, 
                   temp_files, larmip_v, LowPass):
    '''Compute the antarctic dynamics contribution to global sea level as in 
    Levermann et al 2014, using linear response functions.'''

    model_corr = False # Introduces a correlation between input distribution 
                       # UnifDd and the LRF model. Only implemented for the 
                       # three LARMIP ice sheet models with ice shelves

    nb_MOD = len(MOD)
    N = len(NormD)
    ye = 2100
    start_date = 1861 # This is different from other runs
    Beta_low = 7 # Bounds to use for the basal melt rate,
    Beta_high = 16 # units are m.a^(-1).K^(-1)
    nb_y = ye - start_date + 1
    TIME = np.arange(start_date,ye+1)
    i_ys = np.where(TIME == start_date2)[0][0]
    
    if larmip_v == 'LARMIP':
        RF = read_larmip_lrf(data_dir)
        coeff = read_larmip_coeff(data_dir)
        nbLRF = 3 # Number of LRF to use. 3 or 5 for LARMIP
    elif larmip_v == 'LARMIP2':
        print('To do')
    else:
        print(f'ERROR: {larmip_v} value of larmip_v not implemented')

    nb_bass = len(RF.bass)

    TGLOB = misc.tglob_cmip5(temp_files, SCE, start_date, ye, LowPass, False)    
    TGLOBs = TGLOB.sel(time=slice(start_date,None))
    Tref_Lev = TGLOBs - TGLOB.sel(time=slice(start_date,1880)).mean(dim='time')
    Td_Lev = misc.normal_distrib(Tref_Lev, GAM, NormD)

    # Random model number: 1-19
    RMod = np.random.randint(0, 19, N) # Select random model indice (0 to 18)
    AlpCoeff = coeff[:,RMod,0] # dim: bassin, N
    del(RMod)   
        
    # Use following line if Beta should have some external dependence
    #  Beta = Beta_low + UnifDd*(Beta_high - Beta_low) # Modify to obtain random_uniform(7,16,N)
    Beta = np.random.uniform(Beta_low, Beta_high, N)

    BMelt = np.zeros([nb_bass, N, nb_y])
    for b in range(nb_bass):
        for t in range(nb_y):
            BMelt[b,:,t] = AlpCoeff[b,:]*Td_Lev[:,t]*Beta
    
    del(AlpCoeff)
    del(Td_Lev)
    del(Beta)
    
    if model_corr:
        Rdist = np.zeros([N], dtype=int)
        Rdist = 2
        Rdist = np.where(UnifDd >= 0.33, 1, Rdist)
        Rdist = np.where(UnifDd >= 0.67, 0, Rdist)
        modelsel = Rdist # Select model
    else:
        modelsel = np.random.randint(0, nbLRF, N) # Select models

    RF = RF.transpose('bass', 'model', 'time')
    RF = RF[:,modelsel,::-1]
    
    del(modelsel)

    X_ant_b = np.zeros([nb_bass, N, nb_y])
    for t in range(nb_y):
        X_ant_b[:,:,t] = np.sum(RF[:,:,-1-t:]*BMelt[:,:,:t+1],2)

    del(RF)
    del(BMelt)

    X_ant_b = X_ant_b*100 # Convert from m to cm
    X_ant = np.sum(X_ant_b, 0) # Sum 4 bassins

    # Remove the uncertainty at the beginning of the projection
    ref = X_ant[:,i_ys-1]
    for t in range(nb_y):
        X_ant[:,t] = X_ant[:,t] - ref

    return X_ant[:,i_ys:]

def ant_dyn_larmip_t(SCE, MOD, start_date2, GAM, NormD, UnifDd, data_dir, 
                   temp_files, larmip_v, LowPass):
    '''Compute the antarctic dynamics contribution to global sea level as in 
    Levermann et al 2014, using linear response functions.'''

    model_corr = False # Introduces a correlation between input distribution 
                       # UnifDd and the LRF model. Only implemented for the 
                       # three LARMIP ice sheet models with ice shelves

    nb_MOD = len(MOD)
    N = len(NormD)
    ye = 2100
    start_date = 1861 # This is different from other runs
    Beta_low = 7 # Bounds to use for the basal melt rate,
    Beta_high = 16 # units are m.a^(-1).K^(-1)
    nb_y = ye - start_date + 1
    TIME = np.arange(start_date,ye+1)
    i_ys = np.where(TIME == start_date2)[0][0]
    
    if larmip_v == 'LARMIP':
        RF = read_larmip_lrf(data_dir)
        coeff = read_larmip_coeff(data_dir)
        nbLRF = 3 # Number of LRF to use. 3 or 5 for LARMIP
    elif larmip_v == 'LARMIP2':
        print('To do')
    else:
        print(f'ERROR: {larmip_v} value of larmip_v not implemented')

    nb_bass = len(RF.bass)

    TGLOB = misc.tglob_cmip5(temp_files, SCE, start_date, ye, LowPass, False)    
    TGLOBs = TGLOB.sel(time=slice(start_date,None))
    Tref_Lev = TGLOBs - TGLOB.sel(time=slice(start_date,1880)).mean(dim='time')
    Td_Lev = misc.normal_distrib(Tref_Lev, GAM, NormD)

    # Random model number: 1-19
    RMod = np.random.randint(0, 19, N) # Select random model indice (0 to 18)
    AlpCoeff = coeff[:,RMod,0] # dim: bassin, N
    del(RMod)   
        
    # Use following line if Beta should have some external dependence
    #  Beta = Beta_low + UnifDd*(Beta_high - Beta_low) # Modify to obtain random_uniform(7,16,N)
    Beta = np.random.uniform(Beta_low, Beta_high, N)

    BMelt = np.zeros([nb_bass, N, nb_y])
    for b in range(nb_bass):
        for t in range(nb_y):
            BMelt[b,:,t] = AlpCoeff[b,:]*Td_Lev[:,t]*Beta
    
    del(AlpCoeff)
    del(Td_Lev)
    del(Beta)
    
    if model_corr:
        Rdist = np.zeros([N], dtype=int)
        Rdist = 2
        Rdist = np.where(UnifDd >= 0.33, 1, Rdist)
        Rdist = np.where(UnifDd >= 0.67, 0, Rdist)
        modelsel = Rdist # Select model
    else:
        modelsel = np.random.randint(0, nbLRF, N) # Select models

    RF = RF.transpose('bass', 'model', 'time')
    RF = RF[:,modelsel,::-1]
    
    del(modelsel)

    X_ant_b = np.zeros([nb_bass, N, nb_y])
    for t in range(nb_y):
        X_ant_b[:,:,t] = np.sum(RF[:,:,-1-t:]*BMelt[:,:,:t+1],2)

    del(RF)
    del(BMelt)

    X_ant_b = X_ant_b*100 # Convert from m to cm
    X_ant = np.sum(X_ant_b, 0) # Sum 4 bassins

    # Remove the uncertainty at the beginning of the projection
    ref = X_ant[:,i_ys-1]
    for t in range(nb_y):
        X_ant[:,t] = X_ant[:,t] - ref

    return X_ant[:,i_ys:]