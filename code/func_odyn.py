################################################################################
# func_odyn.py: Defines functions returning ocean dynamics probabilistic 
#               contribution to sea level
################################################################################
import numpy as np
import xarray as xr
from scipy.stats import norm

def odyn_loc(SCE, MOD, nb_y2, DIR_O, DIR_OG, lat_N, lat_S, lon_W, lon_E, \
             start_date, ye, SSH_VAR, N, i_ys, Gam, NormD):
    '''Compute the ocean dynamics and thermal expansion contribution to local sea
    level.'''

    nb_MOD = len(MOD)
    nb_y = ye-start_date+1

    # Initialize the SSH matrix: (scenario, model, time (years))
    MAT   = np.zeros([nb_MOD,nb_y])
    MAT_G = np.zeros([nb_MOD,nb_y]) # Global mean steric effect
    MAT_A = np.zeros([nb_MOD,nb_y]) # Local dynamics

    for m in range(nb_MOD):
        fi = xr.open_dataset(f'{DIR_O}{MOD[m]}_{SCE}.nc', use_cftime=True)
        fig = xr.open_dataset(f'{DIR_OG}{MOD[m]}_{SCE}.nc', use_cftime=True)
        fi = fi.assign_coords(TIME=fi['TIME.year'])
        fig = fig.assign_coords(time=fig['time.year']).squeeze()
        SSH = fi[SSH_VAR].sel(TIME=slice(start_date,ye), 
                              latitude=slice(lat_S,lat_N), 
                              longitude=slice(lon_W,lon_E))
        print(SSH.shape)

        if SSH.TIME[-1].values.item() == ye:
            MAT[m,:] = SSH.mean(dim=['latitude', 'longitude'])
            #RQ: No scaling depending on the area, gives more weight to the southern
            #cell
            MAT_G[m,:] = fig[SSH_VAR].sel(time=slice(start_date,ye))
        else: # For models that stop in 2099
            MAT[m, :nb_y-1] = SSH.mean(dim=['latitude', 'longitude'])
            MAT[m,nb_y-1] = MAT[m,nb_y-2]
            MAT_G[m,:nb_y-1] = fig[SSH_VAR].sel(time=slice(start_date,ye))
            MAT_G[m,nb_y-1] = MAT_G[m,nb_y-2]

        # Remove the average SSH of the first 20 years from all models
        MAT[m,:] =  MAT[m,:] - MAT[m,:21].mean()

        MAT_G[m,:] = MAT_G[m,:] - MAT_G[m,:21].mean()
        MAT_A[m,:] = MAT[m,:] - MAT_G[m,:]
        
        del(fi)
        del(SSH)

    MATs     = MAT[:,i_ys:]*100   # Convert from m to cm
    MAT_Gs   = MAT_G[:,i_ys:]*100 # Convert from m to cm
    MAT_As   = MAT_A[:,i_ys:]*100 # Convert from m to cm
    #Build the distribution
    X_O_m    = MATs.mean(axis=0) # Compute the inter-model mean for each time
    X_O_sd   = MATs.std(axis=0)  # Compute the inter-model standard deviation
    X_O_G_m  = MAT_Gs.mean(axis=0)
    X_O_G_sd = MAT_Gs.std(axis=0)
    X_O_A_m  = MAT_As.mean(axis=0)
    X_O_A_sd = MAT_As.std(axis=0)

    X_O    = np.zeros([N,nb_y2])
    X_O_G  = np.zeros([N,nb_y2])
    X_O_A  = np.zeros([N,nb_y2])
    for t in range(nb_y2):
        X_O[:,t]    = X_O_m[t] + Gam * NormD * X_O_sd[t]
        X_O_G[:,t]  = X_O_G_m[t] + Gam * NormD * X_O_G_sd[t]
        X_O_A[:,t]  = X_O_A_m[t] + Gam * NormD * X_O_A_sd[t]

    X_O_out = np.zeros([3,N,nb_y2])
    X_O_out[0,:,:] = X_O
    X_O_out[1,:,:] = X_O_G
    X_O_out[2,:,:] = X_O_A

    return X_O_out

# odyn_glob_knmi14



def odyn_glob_ipcc(SCE, DIR_IPCC, N, nb_y2, Gam, NormD):
    '''Compute thermal expansion contribution to global sea level from IPCC data.'''

    X_O_med   = np.zeros(nb_y2-1) # These start in 2007 instead of 2006
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