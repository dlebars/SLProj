################################################################################
# func_odyn.py: Defines functions returning ocean dynamics probabilistic 
#               contribution to sea level
################################################################################
import numpy as np
import xarray as xr
from scipy.stats import norm
import regionmask

import func_misc as misc

#Change inputs -> make it easy to change the reference period
def odyn_loc(SCE, MOD, DIR_O, DIR_OG, LOC, \
             ref_steric, ye, N, ys, Gam, NormD, LowPass):
    '''Compute the ocean dynamics and thermal expansion contribution to local sea
    level using KNMI14 files.'''

    nb_MOD = len(MOD)
    nb_y2 = ye - ys +1
    lat_N, lat_S, lon_W, lon_E = LOC
    
    #For KNMI files the SSH has a different name for each scenario
    SSH_VAR_dic = {'rcp45':'ZOSH45', 'rcp85':'ZOS85'}
    SSH_VAR = SSH_VAR_dic[SCE]

    # Read sterodynamics and global steric contributions
    for m in range(nb_MOD):
        fi = xr.open_dataset(f'{DIR_O}{MOD[m]}_{SCE}.nc', use_cftime=True)
        fig = xr.open_dataset(f'{DIR_OG}{MOD[m]}_{SCE}.nc', use_cftime=True)
        fi = fi.assign_coords(TIME=fi['TIME.year'])
        fig = fig.assign_coords(time=fig['time.year']).squeeze()
        sd_da = fi[SSH_VAR].assign_coords(model=MOD[m])
        st_da = fig[SSH_VAR].assign_coords(model=MOD[m])
        
        if m==0:
            full_sd_da = sd_da
            full_st_da = st_da
        else:
            full_sd_da = xr.concat([full_sd_da, sd_da], dim='model')
            full_st_da = xr.concat([full_st_da, st_da], dim='model')
    
    full_sd_da = full_sd_da.rename({'TIME':'time'})
    full_sd_da = full_sd_da.sel(time=slice(ref_steric[0],ye), 
                            latitude=slice(lat_S,lat_N), 
                            longitude=slice(lon_W,lon_E))
    
    MAT = full_sd_da.mean(dim=['latitude', 'longitude'])
    MAT_G = full_st_da.sel(time=slice(ref_steric[0],ye))

    MAT = MAT - MAT.sel(time=slice(ref_steric[0],ref_steric[1])).mean(dim='time')
    
    if LowPass:
        fit_coeff = MAT.polyfit('time', 3)
        MAT = xr.polyval(coord=MAT.time, coeffs=fit_coeff.polyfit_coefficients)
    
    MAT_G = MAT_G - MAT_G.sel(time=slice(ref_steric[0],ref_steric[1])).mean(dim='time')               
    MAT_A = MAT - MAT_G

    list_da = [MAT, MAT_G, MAT_A]
    
    X_O_out = np.zeros([3,N,nb_y2])
    
    for idx, da in enumerate(list_da):
        # Select years after the reference period and convert from m to cm
        da = da.sel(time=slice(ys,None))*100
        X_O_out[idx,:,:] = misc.normal_distrib(da, Gam, NormD)

    return X_O_out

# odyn_glob_knmi14


def odyn_glob_ipcc(SCE, DIR_IPCC, N, nb_y2, Gam, NormD):
    '''Compute thermal expansion contribution to global sea level from IPCC AR5 data.'''

    X_O_med   = np.zeros(nb_y2-1) # In the files 2007 means the average over 2006
    X_O_up    = np.zeros(nb_y2-1)

    f_med     = xr.open_dataset(DIR_IPCC+SCE+'_expansionmid.nc')
    f_up      = xr.open_dataset(DIR_IPCC+SCE+'_expansionupper.nc')
    # Read the variables in the files. Use .values to convert the xarray 
    # DataArrays to numpy arrays that can be multiplied more easily.
    X_O_med   = f_med.global_average_sea_level_change.values * 100
    X_O_up    = f_up.global_average_sea_level_change.values * 100

    X_O    = np.zeros([N,nb_y2])
    X_O_sd = (X_O_up - X_O_med) / norm.ppf(0.95)  # ~1.64 

    for t in range(1,nb_y2):
        X_O[:,t]  = X_O_med[t-1] + Gam*NormD * X_O_sd[t-1]

    X_O[:,0]      = X_O[:,1]

    X_O_out        = np.zeros([3, N, nb_y2])
    X_O_out[0,:,:] = X_O
    X_O_out[1,:,:] = X_O   # In this case global is the same as total
    X_O_out[2,:,:] = 0     # and anomaly is 0
    
    return X_O_out

def odyn_cmip(SCE, data_dir, LOC, ref_steric, ye, N, ys, Gam, NormD, LowPass, BiasCorr, ODSL_LIST):
    '''Read the CMIP5 and CMIP6 global steric and ocean dynamics contribution
    and compute a probability distribution.'''
    
    nb_y2 = ye - ys +1
    
    # Read global steric
    full_st_da = misc.read_zostoga_ds(data_dir, SCE)
    
    # Convert from m to cm
    MAT_G = full_st_da['zostoga_corrected'].sel(time=slice(ref_steric[0],ye))*100
    
    if not(LOC):
        MAT = MAT_G
        MAT_A = xr.zeros_like(MAT_G)
        
    else:
        if len(LOC) == 4:
            # The region is a box
            lat_N, lat_S, lon_W, lon_E = LOC

            zos_ds = misc.read_zos_ds(data_dir, SCE)

            full_sd_da = zos_ds['CorrectedReggrided_zos'].sel(
                time=slice(ref_steric[0],ye), 
                lat=slice(lat_S,lat_N), 
                lon=slice(lon_W,lon_E))

            full_sd_da = full_sd_da.mean(dim=['lat', 'lon'])

        elif len(LOC) > 4:
            # The region is a polygon
            zos_ds = misc.read_zos_ds(data_dir, SCE)
            full_sd_da = zos_ds['CorrectedReggrided_zos'].sel(
                time=slice(ref_steric[0],ye))
    
            region = regionmask.Regions([LOC], names=['Reg'], abbrevs=['Reg'])
    
            # Define the mask and change its value from 0 to 1
            mask_cmip = region.mask_3D(full_sd_da.lon, full_sd_da.lat)

            weights = np.cos(np.deg2rad(full_sd_da.lat))

            full_sd_da = full_sd_da.weighted(mask_cmip * weights).mean(dim=('lat', 'lon'))
            full_sd_da = full_sd_da.sel(region=0)
            
        # There are more models available for zos than for zostoga
        # Here we select the intersection
        #model_list = list(set(full_sd_da.model.values) & set(MAT_G.model.values))
        
        if ODSL_LIST:
            # Select a list of models chosen in the namelist
            print("### Sub-selecting models")
            print("Before selection:")
            model_list = full_sd_da.model.values
            print(model_list)
            model_list = list(set(model_list) & set(ODSL_LIST))
            print("After selection")
            print(model_list)
        
        MAT_A = full_sd_da.sel(model=model_list)
        
        if BiasCorr:
            # Use a bias correction based on the comparison of model and 
            # observations between 1979 and 2018
            MAT_A = MAT_A*BiasCorr
        
        # This sum of dara arrays selects the intersection of models between 
        # MAT_A and MAT_G
        MAT = MAT_A + MAT_G
        
        print("Final list of models used:")
        print(MAT.model.values)

    if LowPass:
        new_time = xr.DataArray( np.arange(ys,ye+1)+0.5, dims='time', 
                        coords=[np.arange(ys,ye+1)+0.5], name='time' )
        fit_coeff = MAT.polyfit('time', 2)
        MAT = xr.polyval(coord=new_time, coeffs=fit_coeff.polyfit_coefficients)       
        fit_coeff = MAT_G.polyfit('time', 2)
        MAT_G = xr.polyval(coord=new_time, coeffs=fit_coeff.polyfit_coefficients)
        MAT_A = MAT - MAT_G
    
    list_da = [MAT, MAT_G, MAT_A]
    
    X_O_out = np.zeros([3,N,nb_y2])
    
    for idx, da in enumerate(list_da):
        # Select years after the reference period
        da = da.sel(time=slice(ys,None))
        if not(LowPass):
            # Add a year at the end because data is available until 2099 while code 
            # goes up to 2100
            dal = da.isel(time=-1).assign_coords(time=da.time[-1]+1) # Last year with new coords
            da = xr.concat([da, dal], dim='time')
            
        X_O_out[idx,:,:] = misc.normal_distrib(da, Gam, NormD)

    return X_O_out