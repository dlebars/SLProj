################################################################################
# BuildTotalSeaLevelMaps.py: Read input files from a probabilistic sea level 
#   projection, from zos fields and from GIA and build a total sea level field.
#   Plots individual components and save outputs in a NetCDF file.
#
# Steps to make total sea level maps:
# - Run the SLP code with the option "Decomp = True" this outputs a 
# decomposition of total sea level into each contributor.
# - Prepare the ocean dynamical part. In the CMIP_SeaLevel repo.
# - You are ready to run BuildTotalSeaLevelMaps.py
################################################################################

import os
from datetime import datetime

import numpy as np
import xarray as xr

import func_misc as misc

SCE = 'rcp85'
PercS = 95 # Percentile of interest
namelist_name = 'CMIP5_glo_LEV20' #RECEIPT_D73

# !!! Could be made more efficient for dates after 2100 by extrapolating dynamics only once
for Year in [2020, 2040, 2060, 2080, 2100, 2120]:

    print('#######################################################################')
    print(f'Computing maps for scenario: {SCE}, percentile: {PercS}')

    GAM   = 1.0  #!!! This depends on the projection, ideally it should be read from 
                 # the results of the probabilistic projection
                 # Not used at the moment because otherwise maps of uncertainty 
                 # have a very high average.
                 # This depends on the meaning of the uncertainty map.

    nb_years = Year - (2005+1986)/2 # Should be read from inputs
    data_dir = '../../Data_Proj/'
    DIR_F = data_dir+'Data_AR5/Fingerprints/'
    DIR_GIA = data_dir+'GIA/ICE-6G_VM5a/'
    DIR_OUT = '../outputs/'

    fgia = xr.open_dataset(DIR_GIA+'dsea.1grid_O512_regridded.nc')
    fgia = fgia.rename_dims({'longitude' : 'lon', 'latitude' : 'lat'})
    X_gia = (fgia.Dsea_250)*nb_years/10 # Convert from mm/year to cm
    X_gia = X_gia.assign_coords({'proc_s': 'GIA'})

    fcomp = xr.open_dataset(f'../outputs/ref_proj/SeaLevelPerc_{namelist_name}_{SCE}.nc')

    print('Processes: ')
    print(fcomp.proc_s.values)

    decomp = fcomp.decomp.sel(time_s = Year)

    print('Contributors')
    print(decomp.sel(percentiles=PercS).values)
    print('Sum of contributors')
    print(sum(decomp.sel(percentiles=PercS).values))

    ### Ocean dynamics effects
    DIR_CDE  = '/Users/dewilebars/Projects/Project_ProbSLR/CMIP_SeaLevel/outputs/'
#    fcde     = xr.open_mfdataset(f'{DIR_CDE}cmip5_zos_{SCE}/*_zos_{SCE}*')
    fcde = misc.read_zos_ds(data_dir, SCE)
    CDE_mean = fcde.CorrectedReggrided_zos.mean(dim='model') #.sel(time=Year+0.5)

    if Year > 2100.5:
        new_time = xr.DataArray( np.array([Year+0.5]), dims='time', 
                    coords=[np.array([Year+0.5])], name='time' )
        fit_coeff = CDE_mean.polyfit('time', 2)
        CDE_mean = xr.polyval(coord=new_time, coeffs=fit_coeff.polyfit_coefficients)
    
    CDE_mean = CDE_mean.sel(time=Year+0.5)
    CDE_mean = xr.where( CDE_mean==0, np.nan, CDE_mean)
    #CDE_std = fcde.sel(time=Year+0.5).std(dim='model')

    #### Read fingerprints
    f_gic = xr.open_dataset(f'{DIR_F}Relative_GLACIERS.nc', decode_times=False)
    f_gic = f_gic.assign_coords({'time': np.arange(1986,2101)})
    f_gic = f_gic.rename_dims({'longitude' : 'lon', 'latitude' : 'lat'})
    finger_gic = f_gic.RSL.sel(time=Year, method='nearest')/100 # Convert from % to fraction

    f_ic = xr.open_dataset(f'{DIR_F}Relative_icesheets.nc')/100
    f_ic = f_ic.rename_dims({'longitude' : 'lon', 'latitude' : 'lat'})

    f_gw = xr.open_dataset(f'{DIR_F}Relative_GROUNDWATER.nc', decode_times=False)
    f_gw = f_gw.assign_coords({'time': np.arange(1986,2101)})
    f_gw = f_gw.rename_dims({'longitude' : 'lon', 'latitude' : 'lat'})
    finger_gw = f_gw.GROUND.sel(time=Year, method='nearest')/100 # Convert from % to fraction

    #### Compute each component
    #!!! Only valid for the De Conto and Pollard projections, the assumption about 
    # the fingerprint of Antarctica is 2/3 dynamics and 1/3 smb, this is not the 
    # case but since East Antarctica also looses a lot of mass in this scenario
    # it is more accurate to use a mixture of smb and dyn fingerprints since dyn
    # is only located in west Antarctica

    X_gic = decomp.sel(proc_s='Glaciers', percentiles=PercS) * finger_gic
    X_gsmb = decomp.sel(proc_s='Green. SMB', percentiles=PercS) * f_ic.SMB_GRE
    X_asmb = decomp.sel(proc_s='Ant. SMB', percentiles=PercS) * f_ic.SMB_ANT
    X_gw = decomp.sel(proc_s='Land water', percentiles=PercS) * finger_gw
    X_adyn = decomp.sel(proc_s='Ant. dyn.', percentiles=PercS) * (
        2./3. * f_ic.DYN_ANT + 1./3. * f_ic.SMB_ANT)
    X_gdyn = decomp.sel(proc_s='Green dyn.', percentiles=PercS) * f_ic.DYN_GRE
    X_te   = decomp.sel(proc_s='Thermal exp.', percentiles=PercS)
    # Adding uncertainty or not in climate dynamics is a difficult choice
    X_cde  = CDE_mean #+ CDE_std*GAM*cdfnor_x(tofloat(PercS)/100.,0,1)
    X_sd = X_te + X_cde
    X_sd = X_sd.assign_coords({'proc_s': 'Stero-dynamics'})

    # Add nan values on continents
    X_gic.values = X_gic.where(~np.isnan(X_sd))
    X_gsmb.values = X_gsmb.where(~np.isnan(X_sd))
    X_asmb.values = X_asmb.where(~np.isnan(X_sd))
    X_gw.values = X_gw.where(~np.isnan(X_sd))
    X_adyn.values = X_adyn.where(~np.isnan(X_sd))
    X_gdyn.values = X_gdyn.where(~np.isnan(X_sd))
    X_gia.values = X_gia.where(~np.isnan(X_sd))

    # Concatenate all sea level contributors into one data array
    da = xr.concat([X_sd, X_gic, X_gsmb, X_asmb, X_gw, X_adyn, X_gdyn, X_gia], 
                   dim='proc_s', coords='minimal', compat='override')
    ds = xr.Dataset({'slc': da.reset_coords(drop=True)})

    ### Compute total sea level
    ds['TotalSL'] = ds.slc.sum(dim='proc_s')
    # The sum of NaN become 0, why is that? Need to make it NaN manually
    ds['TotalSL'].values = ds['TotalSL'].where(~np.isnan(X_sd))

    ds['slc'].attrs['long_name'] = 'Sea level contibutors'
    ds['TotalSL'].attrs['long_name'] = (f'Total relative sea level change in {Year}'+
                                        ' relative to the period 1986-2005')

    ### Compute the weigthed average as a test
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = 'weights'
    area_mean = ds['TotalSL'].weighted(weights).mean(('lon', 'lat'))
    print('Weighted average of TotalSL to test:')
    print(area_mean.values)

    ds['area_weighted_mean'] = area_mean

    ##### Export outputs as a NetCDF file

    ds.attrs['source_file'] = 'This NetCDF file was built from BuildTotalSeaLevelMaps.py'
    ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    NameOutput = f'{DIR_OUT}SeaLevelMap_{namelist_name}_{SCE}_Perc{PercS}_{Year}.nc'
    if os.path.isfile(NameOutput):
        os.remove(NameOutput)
    ds.to_netcdf(NameOutput)

