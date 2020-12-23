###############################################################################
# func_misc.py: Miscellaneous functions
###############################################################################
import glob

import numpy as np
from scipy.stats import norm
import pandas as pd
import xarray as xr

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

def tglob_cmip5( files, SCE, start_date, ye, INFO):
    '''Read the text files of monthly temperature for each CMIP5 model and store
    yearly averged values in and array.
    Output data is in degree Kelvin'''
    
    nb_y = ye-start_date+1

    if INFO:
        print('Number of models used for scenario '+ SCE + ' : ' + str(len(files)))
        print('Models path: ')
        print("\n".join(files))

    col_names = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', \
                 'Sep', 'Oct', 'Nov', 'Dec']
    
    for m in range(0,len(files)):
        TEMP     = pd.read_csv(files[m], comment='#', delim_whitespace=True, \
                               names=col_names)
        TEMP = TEMP.set_index('Year')
        TGLOBi = xr.DataArray(TEMP.mean(axis=1))
        mod = files[m][86:-17] # Select model names from path
        TGLOBi = TGLOBi.expand_dims({'model':[mod]})

        if m==0:
            TGLOB = TGLOBi
        else:
            TGLOB = xr.concat([TGLOB, TGLOBi], dim='model')

    TGLOB = TGLOB.rename({'Year':'time'})
    TGLOB = TGLOB.sel(time=slice(start_date,ye))

    return TGLOB

def normal_distrib(model_ts, GAM, NormD):
    '''Build a normal distribution for a contributor'''
    
    N = len(NormD)
    nb_y2 = len(model_ts.time)

    em  = model_ts.mean(dim='model').values
    esd = model_ts.std(dim='model').values

    em = em[np.newaxis, :]
    esd = esd[np.newaxis, :]
    NormD = NormD[:, np.newaxis]
    
    nd = em + GAM * NormD * esd

    return nd

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
    if PDF_cum.any() != 0:
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
        if PDF_cum[t,:].any() != 0:
            for i in range(0, len(Perc)):
                indi =  np.abs(PDF_cum[t,:] - Perc[i]).argmin()
                if indi == 0:
                    # Select the last "argmin()" instead of first as default element:
                    indi =  InPDF.shape[1] - 1 \
                    - np.abs(PDF_cum[t,::-1] - Perc[i]).argmin()
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

def rotate_longitude(ds, name_lon):

    ds = ds.assign_coords({name_lon:(((ds[name_lon] + 180 ) % 360) - 180)})
    ds = ds.sortby(ds[name_lon])

    return ds

def which_mip(sce):
    '''From input scenario return the MIP is corresponds to'''
    
    mip_dic = {'ssp119':'cmip6',
           'ssp126':'cmip6',
           'ssp245':'cmip6',
           'ssp370':'cmip6',
           'ssp585':'cmip6', 
           'rcp26':'cmip5', 
           'rcp45':'cmip5',
           'rcp60':'cmip5',
           'rcp85':'cmip5'}
    
    return mip_dic[sce]