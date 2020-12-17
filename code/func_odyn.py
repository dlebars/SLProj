################################################################################
# func_odyn.py: Defines functions returning ocean dynamics probabilistic 
#               contribution to sea level
################################################################################
import numpy as np
import xarray as xr
from scipy.stats import norm

#Change inputs -> make it easy to change the reference period
def odyn_loc(SCE, MOD, nb_y2, DIR_O, DIR_OG, lat_N, lat_S, lon_W, lon_E, \
             start_date, ye, SSH_VAR, N, ys, Gam, NormD, LowPass):
    '''Compute the ocean dynamics and thermal expansion contribution to local sea
    level.'''

    nb_MOD = len(MOD)
    nb_y = ye-start_date+1

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
    full_sd_da = full_sd_da.sel(time=slice(start_date,ye), 
                            latitude=slice(lat_S,lat_N), 
                            longitude=slice(lon_W,lon_E))
    
    MAT = full_sd_da.mean(dim=['latitude', 'longitude'])
    MAT_G = full_st_da.sel(time=slice(start_date,ye))

    MAT = MAT - MAT.sel(time=slice(start_date,start_date+20)).mean(dim='time')
    
    if LowPass:
        fit_coeff = MAT.polyfit('time', 3)
        MAT = xr.polyval(coord=MAT.time, coeffs=fit_coeff.polyfit_coefficients)
    
    MAT_G = MAT_G - MAT_G.sel(time=slice(start_date,start_date+20)).mean(dim='time')               
    MAT_A = MAT - MAT_G

    # Select years after the reference period and convert from m to cm
    MAT = MAT.sel(time=slice(ys,None))*100
    MAT_Gs = MAT_G.sel(time=slice(ys,None))*100
    MAT_As = MAT_A.sel(time=slice(ys,None))*100
    
    # Build the distributions
    # If a model has missing data it is not included in the mean and standard deviation.
    # Another possibility would be to fill the missing data first.
    X_O_m = np.array(MAT.mean(dim='model')) # Compute the inter-model mean for each time
    X_O_sd = np.array(MAT.std(dim='model')) # Compute the inter-model standard deviation
    X_O_G_m = np.array(MAT_Gs.mean(dim='model'))
    X_O_G_sd = np.array(MAT_Gs.std(dim='model'))
    X_O_A_m = np.array(MAT_As.mean(dim='model'))
    X_O_A_sd = np.array(MAT_As.std(dim='model'))

    X_O = np.zeros([N,nb_y2])
    X_O_G = np.zeros([N,nb_y2])
    X_O_A = np.zeros([N,nb_y2])
    for t in range(nb_y2):
        X_O[:,t]   = X_O_m[t] + Gam * NormD * X_O_sd[t]
        X_O_G[:,t] = X_O_G_m[t] + Gam * NormD * X_O_G_sd[t]
        X_O_A[:,t] = X_O_A_m[t] + Gam * NormD * X_O_A_sd[t]

    X_O_out = np.zeros([3,N,nb_y2])
    X_O_out[0,:,:] = X_O
    X_O_out[1,:,:] = X_O_G
    X_O_out[2,:,:] = X_O_A

    return X_O_out

# odyn_glob_knmi14



def odyn_glob_ipcc(SCE, DIR_IPCC, N, nb_y2, Gam, NormD):
    '''Compute thermal expansion contribution to global sea level from IPCC data.'''

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



# read_odyn_cmip5

# odyn_cmip5