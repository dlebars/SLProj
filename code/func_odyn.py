################################################################################
# func_odyn.py: Defines functions returning ocean dynamics probabilistic 
#               contribution to sea level
################################################################################

import xarray as xr
from scipy.stats import norm


# odyn_loc

# odyn_glob_knmi


def odyn_glob_ipcc(SCE:string ,DIR_IPCC:string, N[*]:numeric, \
                        nb_y2[*]:numeric, Gam[*]:numeric, NormD[*]:numeric):
    '''Compute thermal expansion contribution to global sea level from IPCC data.'''

    X_O_med   = np.zeros(nb_y2-1)     # These start in 2007 instead of 2006
    X_O_up    = np.zeros(nb_y2-1)

    f_med     = xr.open_dataset(DIR_IPCC+SCE+'_expansionmid.nc')
    f_up      = xr.open_dataset(DIR_IPCC+SCE+'_expansionupper.nc')
    X_O_med   = f_med.global_average_sea_level_change * 100
    X_O_up    = f_up.global_average_sea_level_change * 100

    X_O    = np.zeros([N,nb_y2])
    X_O_sd = (X_O_up - X_O_med) / norm.ppf(0.95)  # ~1.64 

    for t in range(1,nb_y2):
        X_O[:,t]  = X_O_med[t-1] + Gam*NormD(:) * X_O_sd[t-1]

    X_O[:,0]      = X_O[:,1]

    X_O_out        = np.zeros([3, N, nb_y2])
    X_O_out(0,:,:) = X_O
    X_O_out(1,:,:) = X_O   # In this case global is the same as total
    X_O_out(2,:,:) = 0     # and anomaly is 0
    return X_O_out



# read_odyn_cmip5

# odyn_cmip5