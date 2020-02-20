###############################################################################
# func_misc.py: Miscellaneous functions
###############################################################################
import numpy as np
from scipy.stats import norm
import pandas as pd
import glob

def temp_path_AR5(MOD, DIR_T, SCE):
    files     = []
    nb_MOD_AR5 = len(MOD)
    for m in range(0, nb_MOD_AR5-1):
        if MOD[m] == 'BCC-CSM1-1':
            loc_mod = "bcc-csm1-1"
        else:
            loc_mod = MOD[m]
        path = DIR_T+'global_tas_Amon_'+loc_mod+'_'+SCE+'_r1i1p1.dat'
        file_sel = glob.glob(path)
        if file_sel: # Make sure the list is not empty
            files.append(file_sel[0])
    return files

def tglob_cmip5(INFO, files, SCE, nb_y, start_date, ye):
    '''Read the text files of monthly temperature for each CMIP5 model and store
    yearly averged values in and array'''
    nb_MOD    = len(files)
    if INFO:
        print('Number of models used for scenario '+ SCE + ' : ' + str(nb_MOD))
        print('Models path: ')
        print("\n".join(files))
    
    TGLOB    = np.zeros([nb_MOD, nb_y])
    col_names = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', \
                 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(0,nb_MOD):
        TEMP     = pd.read_csv(files[m], comment='#', delim_whitespace=True, \
                               names=col_names)
        time     = TEMP['Year'][:]
        dim_t    = len(time)
        i_start  = np.where(time == start_date)[0][0]
        i_end    = np.where(time == ye)[0][0]
        TGLOB[m, :i_end + 1 - i_start] = TEMP.iloc[i_start:i_end+1, 1:].mean(axis=1)   
        # !Data is in degree Kelvin
        #### Issue of missing temperature value for rcp26 after 2100 for this scenario
        # it is ok to assume it is constant after 2100
        if (SCE == 'rcp26') and (ye > 2100):
            i2100 = np.where(time == 2099)
            print(i2100)
            TGLOB[m,i2100-i_start : ] = TGLOB[m,i2100-i_start]
        del(TEMP)
        del(time)
    return TGLOB

def Tref(ys, ye, TGLOB, TIME):
    '''Compute the reference temperature for a specific contributor'''
    nb_MOD = TGLOB.shape[0]
    i_ysr  = np.where(TIME == ys)[0][0]
    i_yer  = np.where(TIME == ye)[0][0]
    Tref   = np.zeros(nb_MOD)
    for m in range(0,nb_MOD):
        Tref[m] = TGLOB[m,i_ysr:i_yer+1].mean()
    return Tref

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

    X_landw = np.zeros([N,nb_y2])  # Independent of the scenario, size is to 
                                   # add up easily.
    for t in range(0, nb_y2):
        X_landw[:,t] = (Delta_grw_up[t] - Delta_grw_lo[t])/2 + sig2[t]*NormD2
        
    return X_landw

def proj2order(TIME_loc, a1_up, a1_lo, Delta_up_2100, Delta_lo_2100, Unif):
    '''Project future values of sea level using present day uncertanty range of 
    the contribution in cm/year and uncertainty of total contribution in 2100 
    in cm. The uncertainty is represented by a uniform distribution.'''

    nb_y_loc   = len(TIME_loc)
    N          = len(Unif)

    # Compute the second order coefficient of the equations:
    a2_up  = (Delta_up_2100 - a1_up * (2100-TIME_loc[0]))/(2100 - TIME_loc[0])**2
    a2_lo  = (Delta_lo_2100 - a1_lo * (2100-TIME_loc[0]))/(2100 - TIME_loc[0])**2

    Delta_up = a1_up * (TIME_loc-TIME_loc[0]) + a2_up * (TIME_loc-TIME_loc[0])**2
    Delta_lo = a1_lo * (TIME_loc-TIME_loc[0]) + a2_lo * (TIME_loc-TIME_loc[0])**2

    X_out = np.zeros([N, nb_y_loc])  # Independent of the scenario, size is to add up easily.
    for t in range(0, nb_y_loc):
        X_out[:,t] = Unif * Delta_up[t] + (1-Unif)*Delta_lo[t]

    return X_out

def perc_df(InPDF, Perc, bin_centers):
    '''Compute percentiles from a PDF without time dimension.
     Inputs:
     InPDF : A pdf computed from the np.histogram function
     Perc  : The percentiles to compute'''
    PDF_cum = InPDF.cumsum(axis=0)*100*(bin_centers[1] - bin_centers[0])
    perc_ar = np.zeros(len(Perc))
    for i in range(0, len(Perc)):
        indi =  np.abs(PDF_cum - Perc[i]).argmin()
        perc_ar[i] = bin_centers[indi]
    perc_df = pd.DataFrame(data= {'percentiles': Perc, 'values': perc_ar})
    perc_df = perc_df.set_index('percentiles')
    return perc_df
        
def perc_df_2d(InPDF, Perc, bin_centers, time_ar):
    '''Compute percentiles from a PDF with time dimension. '''
    PDF_cum = InPDF.cumsum(axis=1)*100*(bin_centers[1] - bin_centers[0])
    perc_ar = np.zeros([InPDF.shape[0], len(Perc)])
    for t in range(0, InPDF.shape[0]):
        for i in range(0, len(Perc)):
            indi =  np.abs(PDF_cum[t,:] - Perc[i]).argmin()
            perc_ar[t,i] = bin_centers[indi]
    perc_df = pd.DataFrame(perc_ar)
    perc_df.columns = [str(i)+'pc' for i in Perc]
    perc_df.index = time_ar
    return perc_df

def finger1D(lats, lons, lat1D, lon1D, fingerprint):
    '''Select a fingerprint value at a lat/lon point from 2D or 3D array. Make 
    sure to select a point that is not on land otherwise the fingerprint value 
    it 0 there.'''
    dim_f = fingerprint.shape
    ndim  = len(dim_f)
    if ndim == 2:
        output = np.zeros(1)
        mask2D = np.where(fingerprint == 0, 0, 1)
    elif ndim == 3:
        output = np.zeros(dim_f[0])
        mask2D = np.where(fingerprint[1,:,:] == 0, 0, 1)

    lon_ind = np.abs(lon1D - lons).argmin()
    lat_ind = np.abs(lat1D - lats).argmin()
    lon_dist, lat_dist = np.meshgrid(np.arange(dim_f[-1]) - np.asscalar(lon_ind), 
                                np.arange(dim_f[-2]) - np.asscalar(lat_ind))
    tot_dist = np.abs(lon_dist) + np.abs(lat_dist)
    tot_dist = np.where(mask2D != 0, tot_dist, tot_dist.max() +1 )
    ind = np.unravel_index(np.argmin(tot_dist, axis=None), tot_dist.shape)

    if ndim == 2:
        output = fingerprint[ind[0], ind[1]]
    elif ndim == 3:
        output = fingerprint[:, ind[0], ind[1]]

    return output
