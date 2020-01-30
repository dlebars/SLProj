################################################################################
# func_odyn.py: Defines functions returning ocean dynamics probabilistic 
#               contribution to sea level
################################################################################
import numpy as np
import xarray as xr
from scipy.stats import norm

def odyn_loc(SCE, MOD, nb_y, nb_y2, DIR_O, lat_N, lat_S, lon_W, lon_E, start_date, \
             end_date2, VAR, N, i_ys, Gam, NormD):
    '''Compute the ocean dynamics and thermal expansion contribution to local sea level.'''

    nb_MOD = len(MOD)

    # Initialize the SSH matrix: (scenario, model, time (years))
    MAT   = np.zeros([nb_MOD,nb_y])
    MAT_G = np.zeros([nb_MOD,nb_y]) # Global mean steric effect
    MAT_A = np.zeros([nb_MOD,nb_y]) # Local dynamics

    for m in range(0,nb_MOD):
        fi      = xr.open_dataset(DIR_O + MOD[m] + '_' + SCE + '.nc')
        lon     = np.array(fi.longitude)  # Here we do not assume the models all 
        lat     = np.array(fi.latitude)   # have the same grid, even though they DO
        lat_Ni  = np.abs(lat_N - lat).argmin()
        lat_Si  = np.abs(lat_S - lat).argmin()
        lon_Wi  = np.abs(lon_W - lon).argmin()
        lon_Ei  = np.abs(lon_E - lon).argmin()
        TIMEt   = np.array(fi.TIME)
        TIMEt2  = fi['TIME.year'] # Select the years
        i_start = np.abs(start_date - TIMEt2[:,0]).argmin()
        i_end   = np.abs(end_date2 - TIMEt2[:,0]).argmin()
        SSH     = fi[VAR][i_start:i_end,lat_Si:lat_Ni,lon_Wi:lon_Ei]
        nb_y_loop = i_end - i_start + 1
        if nb_y_loop == nb_y:
            MAT[m,:] = SSH.mean(axis=1).mean(axis=2)
            #RQ: No scaling depending on the area, gives more weight to the southern cell
            MAT_G[m,:] = fi[VAR][i_start:i_end,0,0]
        else:
            MAT[m, :nb_y-2] = SSH.mean(axis=1).mean(axis=2)
            MAT[m,nb_y-1]    = MAT[m,nb_y-2]
            # Is this to initialize the size of the array? Strange method...
            MAT_G[m,:nb_y-2] = fi[VAR][i_start:i_end,0,0]
            MAT_G[m,nb_y-1]  = MAT_G[m,nb_y-2]

        # Remove the average SSH of the first 20 years from all models
        MAT[m,:] =  MAT[m,:] - [MAT[m,:20]].mean()

        MAT_G[m,:] = MAT_G[m,:] - MAT_G[m,:20].mean()
        MAT_A[m,:] = MAT[m,:] - MAT_G[m,:]
        del(fi)
        del(lon)
        del(lat)
        del(SSH)
        del(TIMEt)
        del(TIMEt2)

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
    for t in range(0,nb_y2):
        X_O[:,t]    = X_O_m[t] + Gam * NormD * X_O_sd[t]
        X_O_G[:,t]  = X_O_G_m[t] + Gam * NormD * X_O_G_sd[t]
        X_O_A[:,t]  = X_O_A_m[t] + Gam * NormD * X_O_A_sd[t]

    X_O_out = np.zeros([3,N,nb_y2])
    X_O_out[0,:,:] = X_O
    X_O_out[1,:,:] = X_O_G
    X_O_out[2,:,:] = X_O_A

    return X_O_out


# def odyn_glob_knmi14(SCE, MOD, nb_y, nb_y2, DIR_O, DIR_OG, start_date, end_date2, \
#       VAR, N, i_ys, Gam, NormD)
#     '''Compute thermal expansion contribution to global sea level from KNMI'14 
#     data. The data was processed from the CMIP5 dataset.'''
#     nb_MOD = len(MOD)

#     # Initialize the SSH matrix: (scenario, model, time (years))
#     MAT   = np.zeros([nb_MOD,nb_y])
#     print("WARNING !!!!!!! There seem to be an mistake in this script, variable fig"+ \
#         " is not used, should be used instead of fi in loop?")

#   do m=0,nb_MOD-1
#     ;###
#     fi      = addfile(DIR_O+MOD(m)+"_"+SCE+".nc","r")
#     TIMEt   = fi->TIME
#     TIMEt2  = cd_calendar(TIMEt,0)
#     i_start = closest_val(start_date,TIMEt2(:,0))
#     i_end   = closest_val(end_date2,TIMEt2(:,0))
#     fig          = addfile(DIR_OG+MOD(m)+"_"+SCE+".nc","r")
#     nb_y_loop = i_end - i_start +1
#     if nb_y_loop.eq.nb_y then
#       MAT(m,:) = fi->$VAR$(i_start:i_end,0,0)
#       else
#       MAT(m,:nb_y-2) = fi->$VAR$(i_start:i_end,0,0)
#       MAT(m,nb_y-1)  = MAT(m,nb_y-2)
#     end if
#     MAT(m,:) = MAT(m,:) - avg(MAT(m,:20))
#     delete(fi)
#     delete(fig)
#     delete(TIMEt)
#     delete(TIMEt2)
#   end do

#   MATs     = MAT(:,i_ys:)*100   ; Convert from m to cm, and select dates after 2006
#   ;Build the distribution
#   X_O_m    = dim_avg_n(MATs,0)    ; Compute the inter-model mean for each time
#   X_O_sd   = dim_stddev_n(MATs,0) ; Compute the inter-model standard deviation

#   X_O    = new((/N,nb_y2/),float)
#   do t=0,nb_y2-1
#     X_O(s,:,t)    = X_O_m(s,t) + Gam*NormD(:)*X_O_sd(s,t)
#   end do

#   X_O_out = new((/3,N,nb_y2/),float)
#   X_O_out(0,:,:) = X_O
#   X_O_out(1,:,:) = X_O   ; In this case global is the same as total
#   X_O_out(2,:,:) = 0     ; and anomaly is 0

#   return X_O_out

# end


def odyn_glob_ipcc(SCE, DIR_IPCC, N, nb_y2, Gam, NormD):
    '''Compute thermal expansion contribution to global sea level from IPCC data.'''

    X_O_med   = np.zeros(nb_y2-1)     # These start in 2007 instead of 2006
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