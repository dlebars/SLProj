###############################################################################
# func_misc.py: Miscellaneous functions
###############################################################################
import glob
import os

import numpy as np
from scipy.stats import norm
import pandas as pd
import xarray as xr
from numpy.polynomial import Polynomial as P

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

def tglob_cmip5(files, SCE, start_date, ye, LowPass, INFO):
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

    if LowPass:
        new_time = xr.DataArray( np.arange(start_date,ye+1), dims='time', 
                coords=[np.arange(start_date,ye+1)], name='time' )
        fit_coeff = TGLOB.polyfit('time', 2)
        TGLOB = xr.polyval(coord=new_time, coeffs=fit_coeff.polyfit_coefficients) 
    
    return TGLOB

def tglob_cmip6(DIR_T, SCE, start_date, ye, LowPass, INFO):
    '''Read the NetCDF files of monthly temperature for each CMIP6 model and 
    store yearly averged values in and array
    DOES NOT WORK YET: New coode epxects a dataframe, not a numpy array
    '''
    
    ENS = 'r1i1p1f1'
    nb_y = ye-start_date+1
    files_all = os.listdir(DIR_T)
    
    try:
        files_all.remove('README')
    except:
        print('No README file in folder')
    
    info_df = pd.DataFrame(columns=['variable','table_id','model', 'experiment', 'ensemble', 'grid', 'date'])
    
    for i in range(len(files_all)):
        st = files_all[i].split('_')
        if len(st) != 7:
            print('WARNING!!! This file name is not standard')
            print(i)
            print(st)
        info_df.loc[i] = st
    sel_df = info_df[info_df.ensemble.eq(ENS) & info_df.experiment.eq(SCE) &
                     info_df.table_id.eq('Amon') & info_df.model.ne('BCC-ESM1')]    
    # Notes:
    # Removed 'BCC-ESM1' because projections stop in 2055
    nb_MOD    = len(sel_df)
    nb_ind_MOD = len(set(sel_df['model']))
    if nb_MOD != nb_ind_MOD:
        print('WARNING: Some models are used multiple times')
        print()
    if INFO:
        print('Number of models used for scenario '+ SCE + ' : ' + str(nb_MOD))
        print('Models: ')
        display(sel_df)
    
    TGLOB    = np.zeros([nb_MOD, nb_y])
    for m in range(0,nb_MOD):
        file_name = "_".join(sel_df.iloc[m])
        # Corresponding historical file (EC-Earth3 has no historical r1i1p1f1)
        if (sel_df.iloc[m][2] == 'EC-Earth3') and (ENS == 'r1i1p1f1'):
            hist_df = info_df[info_df.ensemble.eq('r2i1p1f1') &
                              info_df.experiment.eq('historical') & 
                              info_df.table_id.eq('Amon') &
                              info_df.model.eq(sel_df.iloc[m][2])]
        else:
            hist_df = info_df[info_df.ensemble.eq(ENS) &
                              info_df.experiment.eq('historical') & 
                              info_df.table_id.eq('Amon') &
                              info_df.model.eq(sel_df.iloc[m][2])]
        if len(hist_df)  != 1:
            print('ERROR: There is more/less than one corresponding historical' 
                  +' file')
            print(hist_df)
        hist_name = "_".join(hist_df.iloc[0])
        TEMP_SCE_ds = xr.open_dataset(DIR_T+file_name)
        TEMP_HIST_ds = xr.open_dataset(DIR_T+hist_name)
        TEMP_ALL_ds = xr.concat([TEMP_SCE_ds, TEMP_HIST_ds], 'time')
        TEMP_ALL_y_ds = TEMP_ALL_ds.groupby('year').mean(dim='time')
        nb_y_avail = len(TEMP_ALL_y_ds.year.loc[start_date:ye])
        TGLOB[m,:nb_y_avail] = TEMP_ALL_y_ds.tas.loc[start_date:ye]
        if (nb_y_avail != nb_y): #and (TEMP_ALL_y_ds.year[0] == start_date)
            print('Some years are missing for '+sel_df.iloc[m][0]+' '+sel_df.iloc[m][2])
            print(TEMP_ALL_y_ds.year.loc[start_date:ye].values)
            print('Filing up the gap by extrapolation')
            TGLOB[m,-1] = TGLOB[m,-2]
        
    return TGLOB

def get_ar6_temp():
    '''Make a dataframe from the temperature values of table 4.1 (or 4.6?) 
    of AR6. Normaly relative to 1995–2014 but modify to make it relative to
    the AR5 reference period 1986-2005.
    Data in the table refer to the following periods: 
    2021–2040, 2041–2060, 2081–2100 that we attribute to their middle years.'''
    
    central_years = [2005, 2030, 2050, 2090]
    ssp126 = pd.DataFrame({'mean':[0, 0.6, 0.9, 0.9], '5pc':[0, 0.4, 0.5, 0.5], '95pc':[0, 0.9, 1.3, 1.5]},
                       index=central_years)
    ssp126.columns = pd.MultiIndex.from_product([['ssp126'], ssp126.columns])
    ssp245 = pd.DataFrame({'mean':[0, 0.7, 1.1, 1.8], '5pc':[0, 0.4, 0.8, 1.2], '95pc':[0, 0.9, 1.6, 2.6]},
                       index=central_years)
    ssp245.columns = pd.MultiIndex.from_product([['ssp245'], ssp245.columns])
    ssp585 = pd.DataFrame({'mean':[0, 0.8, 1.5, 3.5], '5pc':[0, 0.5, 1.1, 2.4], '95pc':[0, 1.0, 2.1, 4.8]},
                       index=central_years)
    ssp585.columns = pd.MultiIndex.from_product([['ssp585'], ssp585.columns])

    all_df = pd.concat([ssp126, ssp245, ssp585], axis = 1)
    # Add temperature difference between 1995–2014 and 1986-2005
    all_df = all_df + 0.156
    all_df.loc[1995] = 0
    all_df = all_df.sort_index()
    
    return all_df

def tglob_ar6(sce, start_date, ye):
    '''Provides a few time series of temperature consistent with the AR6 
    temperature assessment.
    Assumes normal distribution which is not the case in AR6. Needs to be 
    revised later.
    Export a data array.'''
    
    if sce == 'ssp585_hpp':
        # hpp scenario from AR6 uses the same temperature as 
        sce = 'ssp585'
    
    N = 50 # Number of time series to generate
    ar6_temp_df = get_ar6_temp()
    df = ar6_temp_df[sce].copy()
    # Assuming a normal distribution (it is not!) the standard devation is:
    df['sigma'] = (df['95pc'] - df['5pc'])/(2*1.64)
    
    NormD = np.random.normal(0, 1, N)
    years = np.arange(start_date,ye+1)
    paths = np.zeros([N,len(years)])
    
    for i in range(N):
        p = P.fit(df.index, df['mean']+NormD[i]*df['sigma'], 3)
        paths[i,:] = p(years)
    
    ds = xr.DataArray(paths, coords=[np.arange(N), years] , dims=['model', 'time'])
    
    return ds

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
    '''Project future values of sea level using present day uncertainty range of 
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

def proj2order_normal(TIME_loc, a1_up, a1_lo, Delta_mean_2100, Delta_std_2100, NormD):
    '''Project future values of sea level using present day uncertainty range of 
    the contribution in cm/year and uncertainty of total contribution in 2100 
    in cm. The uncertainty is represented by a normal distribution.'''

    nb_y_loc = len(TIME_loc)
    N = len(NormD)
    
    speed_t0 = (a1_up+a1_lo)/2

    # Compute the second order coefficient of the equations:
    a2_mean  = (Delta_mean_2100 - speed_t0 * (2100-TIME_loc[0]))/(2100 - TIME_loc[0])**2
    a2_std  = (Delta_std_2100 - a1_up * (2100-TIME_loc[0]))/(2100 - TIME_loc[0])**2

    Delta_mean = speed_t0 * (TIME_loc-TIME_loc[0]) + a2_mean * (TIME_loc-TIME_loc[0])**2
    Delta_std = Delta_std_2100/(2100-TIME_loc[0])*(TIME_loc-TIME_loc[0])
    
    X_out = Delta_mean[np.newaxis,:] + Delta_std[np.newaxis,:]*NormD[:,np.newaxis]

    return X_out

def proj2order_normal_assym(TIME_loc, a1_up, a1_lo, med_2100, 
                            std_lo_2100, std_up_2100, NormD):
    '''Project future values of sea level using present day uncertainty range of 
    the contribution in cm/year and uncertainty of total contribution in 2100 
    in cm. The uncertainty is represented by a two half-normal distributions,
    one above the median and the other one below.
    The median grows as a 2nd order polynomial and the standard deviations grow 
    linearly'''

    nb_y_loc = len(TIME_loc)
    N = len(NormD)
    
    speed_t0 = (a1_up+a1_lo)/2

    # Compute the second order coefficient of the equations:
    a2_med  = (med_2100 - speed_t0 * (2100-TIME_loc[0]))/(2100 - TIME_loc[0])**2

    med = speed_t0 * (TIME_loc-TIME_loc[0]) + a2_med * (TIME_loc-TIME_loc[0])**2
    std_lo = std_lo_2100/(2100-TIME_loc[0])*(TIME_loc-TIME_loc[0])
    std_up = std_up_2100/(2100-TIME_loc[0])*(TIME_loc-TIME_loc[0])
    
    NormD_up = np.where(NormD>0, NormD, 0)
    NormD_lo = np.where(NormD<0, NormD, 0)
    
    X_out = (med[np.newaxis,:] + 
             std_lo[np.newaxis,:]*NormD_lo[:,np.newaxis] + 
             std_up[np.newaxis,:]*NormD_up[:,np.newaxis])

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
    '''Returns the MIP that corresponds to the input scenario '''
    
    mip_dic = {'ssp119':'cmip6',
               'ssp126':'cmip6',
               'ssp245':'cmip6',
               'ssp370':'cmip6',
               'ssp585':'cmip6',
               'ssp585_hpp':'cmip6',
               'rcp26':'cmip5', 
               'rcp45':'cmip5',
               'rcp60':'cmip5',
               'rcp85':'cmip5'}
    
    return mip_dic[sce]

def read_zos_ds(data_dir, mip, sce):
    '''Read both historical and scenario datasets, select the intersecting 
    models and concatenate the two datasets'''
    
    hist_ds = xr.open_mfdataset(
        f'{data_dir}/Data_{mip}/{mip}_zos_historical/{mip}_zos_historical_*.nc')
    sce_ds = xr.open_mfdataset(
        f'{data_dir}/Data_{mip}/{mip}_zos_{sce}/{mip}_zos_{sce}_*.nc')
    model_intersection = list(set(hist_ds.model.values) & 
                              set(sce_ds.model.values))
    model_intersection.sort()
    tot_ds = xr.concat([hist_ds,sce_ds],'time').sel(model=model_intersection)
    
    return tot_ds

def read_zostoga_ds(data_dir, mip, sce):
    '''Read both historical and scenario datasets, select the intersecting 
    models and concatenate the two datasets'''
    
    hist_ds = xr.open_mfdataset(
        f'{data_dir}/Data_{mip}/{mip}_zostoga/{mip}_zostoga_historical_*.nc')
    sce_ds = xr.open_mfdataset(
        f'{data_dir}/Data_{mip}/{mip}_zostoga/{mip}_zostoga_{sce}_*.nc')

    model_intersection = list(set(hist_ds.model.values) & 
                              set(sce_ds.model.values))
    model_intersection.sort()
    tot_ds = xr.concat([hist_ds,sce_ds],'time').sel(model=model_intersection)
    
    return tot_ds