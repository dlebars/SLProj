################################################################################
# func_ant.py: Defines functions to return probabilistic Antarctic contribution 
#             to sea level
################################################################################
import glob

import numpy as np
from scipy.stats import norm
import xarray as xr
from scipy import signal
import pandas as pd

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
    
    f = xr.open_dataset(f'{data_dir}LRF_Lev14/RFunctions.nc', decode_times=False)
    ic_models = ['AIF', 'PS', 'PISM', 'SICO', 'UMISM']

    for icm in ic_models:
        RF_tmp = f[f'RF_{icm}'].assign_coords({'model' : icm})
        RF_tmp = RF_tmp.expand_dims('model', axis=0)
        try:
            RF = xr.concat([RF, RF_tmp], dim='model', combine_attrs='drop')
        except:
            RF = RF_tmp

    RF.name = 'RF'
    RF = RF.rename({'bass': 'region'})
    RF = RF.assign_coords(region = (['Amundsen', 'Ross', 'Weddell', 'EAIS']))
    RF = RF.transpose('region', 'model', 'time')
    
    return RF

def read_larmip_coeff(data_dir):
    '''Read the coefficient scalling GMST to temperature around Antarctica'''
    
    coeff = xr.open_dataset(f'{data_dir}LRF_Lev14/RFunctions.nc', decode_times=False).coeff
    
    coeff = coeff.assign_coords(model = ([
        "ACCESS1-0", "ACCESS1-3", "BNU-ESM", "CanESM2", "CCSM4", "CESM1-BGC", 
        "CESM1-CAM5", "CSIRO-Mk3-6-0", "FGOALS-s2", "GFDL-CM3", "HadGEM2-ES", 
        "INMCM4", "IPSL-CM5A-MR", "MIROC-ESM-CHEM", "MIROC-ESM", "MPI-ESM-LR", 
        "MRI-CGCM3", "NorESM1-M", "NorESM1-ME"]))
    
    coeff = coeff.rename({'bass': 'region'})
    coeff = coeff.assign_coords(region = (['Amundsen', 'Ross', 'Weddell', 'EAIS']))
    coeff = coeff.assign_coords(result = (['NoDelay', 'Rsquared', 'Delay', 'WithDelay', 'Rsquared']))
    
    return coeff


def read_larmip2_lrf(data_dir, basal_melt):
    '''Read LARMIP2 Linear Response Functions downloaded from:
    https://github.com/ALevermann/Larmip2019.
    Basal melt is in m.y-1. BM02, BM04, BM08, BM16 are available. 
    basal_melt = BM08 is used in Levermann et al. 2020.'''
    
    reg_names = {'R1':'EAIS', 'R2':'Ross', 'R3':'Amundsen', 
                 'R4':'Weddell', 'R5':'Peninsula'}

    for idb, reg in enumerate(reg_names):
        path = f'{data_dir}RF_*_{basal_melt}_{reg}.dat'
        files = glob.glob(path)

        for idf, f in enumerate(files):
            ds = pd.read_csv(f, names=['RF']).to_xarray()
            f2 = f.split('/')[-1]
            ds = ds.expand_dims({'model': [f2[3:-12]]})
            
            if idf ==0:
                ds2 = ds
            else:
                ds2 = xr.concat([ds2, ds], dim='model')

        ds2 = ds2.expand_dims({'region': [reg_names[reg]]})
        if idb == 0:
            RF = ds2
        else:
            RF = xr.concat([RF, ds2], dim='region')

    RF = RF.rename({'index' : 'time'})
    RF = RF.transpose('region', 'model', 'time')
    
    return RF.RF

def ant_dyn_larmip(SCE, start_date2, ye, GAM, NormD, UnifDd, data_dir, 
                   temp_opt, larmip_v, delay, LowPass):
    '''Compute the antarctic dynamics contribution to global sea level as in 
    Levermann et al 2014, using linear response functions.'''

    model_corr = False # Introduces a correlation between input distribution 
                       # UnifDd and the LRF model. Only implemented for the 
                       # three LARMIP ice sheet models with ice shelves

    N = len(NormD)
    start_date = 1861 # This is different from other runs
    Beta_low = 7 # Bounds to use for the basal melt rate,
    Beta_high = 16 # units are m.a^(-1).K^(-1)
    nb_y = ye - start_date + 1
    TIME = np.arange(start_date,ye+1)
    i_ys = np.where(TIME == start_date2)[0][0]
    # TODO change the way this time reference is handled
    i_ys_ref = np.where(TIME == 1995)[0][0]
    
    if larmip_v == 'LARMIP':
        RF = read_larmip_lrf(data_dir)
        nbLRF = 3 # Number of LRF to use. 3 or 5 for LARMIP
        coeff = read_larmip_coeff(data_dir).values
    elif larmip_v == 'LARMIP2':
        RF = read_larmip2_lrf(f'{data_dir}LRF_Lev20/RFunctions/', 'BM08')
        # Exclude a model? There are two BISI_LBL...
        nbLRF = len(RF.model)
        coeff = read_larmip_coeff(data_dir)
        # The coefficents need to be reordered to fit the RF
        coeff = xr.concat([coeff.sel(region='EAIS'), 
                    coeff.sel(region='Ross'),
                    coeff.sel(region='Amundsen'),
                    coeff.sel(region='Weddell'),
                    coeff.sel(region='Amundsen').assign_coords(
                        region = 'Peninsula')], dim='region')
        coeff = coeff.values
        
    else:
        print(f'ERROR: {larmip_v} value of larmip_v not implemented')
        
    nb_bass = len(RF.region)

    TGLOB = misc.make_tglob_array(data_dir, temp_opt, SCE, start_date, ye , LowPass)
    TGLOBs = TGLOB.sel(time=slice(start_date,None))
    Tref_Lev = TGLOBs - TGLOB.sel(time=slice(start_date,start_date+19)).mean(dim='time')
    Td_Lev = misc.normal_distrib(Tref_Lev, GAM, NormD)

    # Random climate model number: 1-19
    RMod = np.random.randint(0, 19, N) # Select random model indice (0 to 18)

    if delay:
        AlpCoeff = coeff[:,RMod,3] # dim: region, N
        TimeDelay = np.array(coeff[:,RMod,2], dtype='int')
        
        # That could be made faster using fancy indexing
        # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
        Td_Lev_d = np.zeros([AlpCoeff.shape[0], Td_Lev.shape[0], Td_Lev.shape[1]])
        
        for r in range(AlpCoeff.shape[0]):
            for t in range(N):
                Td_Lev_d[r,t,TimeDelay[r,t]:] = Td_Lev[t,:nb_y-TimeDelay[r,t]]
        
    else:
        AlpCoeff = coeff[:,RMod,0] # dim: region, N
        Td_Lev_d = Td_Lev[np.newaxis,:,:]
        
    # Use following line if Beta should have some external dependence
    #  Beta = Beta_low + UnifDd*(Beta_high - Beta_low) # Modify to obtain random_uniform(7,16,N)
    Beta = np.random.uniform(Beta_low, Beta_high, N)
    BMelt = AlpCoeff[:,:,np.newaxis] * Td_Lev_d * Beta[np.newaxis,:,np.newaxis]
    
    if model_corr:
        Rdist = np.zeros([N], dtype=int)
        Rdist = 2
        Rdist = np.where(UnifDd >= 0.33, 1, Rdist)
        Rdist = np.where(UnifDd >= 0.67, 0, Rdist)
        modelsel = Rdist # Select model
    else:
        modelsel = np.random.randint(0, nbLRF, N) # Select models

    RF = RF[:,modelsel,:]

    X_ant_b = signal.fftconvolve(RF[:,:,:nb_y],BMelt, mode='full', axes=2)[:,:,:nb_y]

    X_ant_b = X_ant_b*100 # Convert from m to cm
    X_ant = np.sum(X_ant_b, 0) # Sum 4 bassins

    # Remove the uncertainty at the beginning of the projection
    X_ant -= X_ant[:,[i_ys_ref]]
    
    return X_ant[:,i_ys:]

def ant_ar6(TIME_loc, a1_up, a1_lo, sce, NormD, ANT_DYN):
    '''Total antarctic contribution as in AR6 table 9.9.
    Compute a discontinuous two sided half-normal distribution from the likely range.
    These numbers in 2100 are referenced to the period 1995-2014
    while the code uses 1986-2005 as a reference period but since there is only
    1 mm between these two periods this is neglected.'''
    
    if sce == 'ssp126':
        l_range = [3., 11., 27.] # 17pc, med, 83pc
    elif sce == 'ssp245':
        l_range = [3., 11., 29.]
    elif sce == 'ssp585':
        l_range = [3., 12., 34.]
    elif sce == 'ssp585_hpp': # Low confidence in AR6
        l_range = [2., 19., 56.]
    else:
        print('Scenario not supported by ant_ar6')
    
    std_lo_2100 = l_range[1]-l_range[0]
    std_up_2100 = l_range[2]-l_range[1]
    
    if ANT_DYN == 'KS21':
        X_ant = misc.proj2order_normal_assym_ks21(
            TIME_loc, a1_up, a1_lo, l_range[1], std_lo_2100, std_up_2100, NormD)
    elif ANT_DYN == 'KNMI23':
        X_ant = misc.proj2order_normal_assym_knmi23(
            TIME_loc, a1_up, a1_lo, l_range[1], std_lo_2100, std_up_2100, NormD)
    
    return X_ant

def ant_dyn_vdl23(ROOT, SCE, N, cal_reg, select, ref_per):
    # Use projections from van der Linden et al. 2023
    # Options:
    # cal_reg: The region used to calibrate basal melt sensitivity to increased 
    #          ocean temperature temperature
    #         'AMUN' for Amundsen Sea, 'SU' for full Antarctica
    # select: 'intersection' use only four climate models also selected for 
    #         ODSL in KNMI23
    #         'best10pc' for the 10% of best performing models over the 
    #         historical period
    #         'noCAS' removed the CAS-ESM2-0 climate model that is an outlier
    # ref_per: 'usual' for 1986-2005
    #          'present' for 2008-2028 complemented with observations 
    #          from 1995 to 2018
    
    path_vdl = f"{ROOT}/DataAntarctica_vdl/"
    opt1 = 'calibrated_quadM'
    opt2 = '_thetao_merged_biasadj_shelfbasedepth'
    vdl_ds = xr.open_dataset(f"{path_vdl}slr_{cal_reg}{opt1}{opt2}_historical+{SCE}_1850-2100.nc")
    vdl_ds = vdl_ds.rename_vars({"__xarray_dataarray_variable__":"AA_dyn"})
    vdl_da = vdl_ds.AA_dyn.sel(region = "SU")
    
    # Convert from m to cm
    vdl_da = vdl_da*100
    
    # For one model values in 2100 are 0
    vdl_da.loc[:,'CAMS-CSM1-0', 2100] = vdl_da.loc[:,'CAMS-CSM1-0', 2099]
    
    if select == 'intersection':
        vdl_da = vdl_da.sel(model=['EC-Earth3', 'INM-CM4-8', 'NorESM2-MM', 'MPI-ESM1-2-LR'])
    
    vdl_da = vdl_da.stack(model_pairs=['model', 'ism'])

    if select == 'best10pc':
        # Select the best 10% of model pairs
        # Only works for Amundsen Sea calibration
        aa_t10 = pd.read_csv(f"{path_vdl}top10pct_models_{cal_reg}{opt1}_shelfbasedepth.txt", 
                             names = ["ESM", "ISM"],
                             delim_whitespace=True)
        na = np.transpose(np.array(aa_t10))
        mi = list(zip(*na))
        vdl_da = vdl_da.sel(model_pairs=mi )
        
    elif select == 'noCAS':
        vdl_da.loc[:,'CAS-ESM2-0'] = np.nan
    
    vdl_da = vdl_da.dropna(dim="model_pairs")

    if ref_per == 'usual':
        # Use the usual reference period
        vdl_da = vdl_da - vdl_da.sel(year=slice(1986,2005)).mean(dim="year")
        vdl_da = vdl_da.sel(year=slice(2006,2100))
        
    elif ref_per == 'present':
        # Take a later reference period
        vdl_da = vdl_da - vdl_da.sel(year=slice(2008,2028)).mean(dim="year")
        vdl_da = vdl_da.sel(year=slice(2006,2100))
    
        # Add Antarctic contribution from 1995 to 2018 based on Frederikse et al. 2020
        obs = 0.79
        vdl_da = vdl_da + obs
    
        # Reconstruct values back to 2006
        lin_ar = np.linspace(0, obs, 13)
        vdl_da.loc[2006:2018] =  lin_ar[:,np.newaxis]
    
    vdl_na = np.swapaxes(vdl_da.data, 0, 1)
    modelsel = np.random.randint(0, len(vdl_da.model_pairs), N)
    vdl_na_big = vdl_na[modelsel, :]
    
    return vdl_na_big