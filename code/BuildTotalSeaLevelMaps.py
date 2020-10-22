################################################################################
# BuildTotalSeaLevelMaps.py: Read input files from a probabilistic sea level 
#   projection, from zos fields and from GIA and build a total sea level field.
#   Plots individual components and save outputs in a NetCDF file.
#
# Steps to make total sea level maps:
# - Run the SLP code with the option "Decomp = True" this outputs a 
# decomposition of total sea level into each contributor.
# - Prepare the ocean dynamical part. In the CMIP_SeaLevel repo.
# - Run BuildTotalSeaLevelMaps.ncl to produce a plot of the total relative sea level
# change and its contributions and export the results in a netCDF file.
################################################################################

import xarray as xr

import func_misc as misc

#load "../SLP_v1.2/func_misc.ncl"

SCE = 'rcp85'
PercS = 95 # Percentile of interest
CASE = 'MapsDC16' # MapsDC16 or MapsAR5
Year = 2100

print('#######################################################################')
print(f'Computing maps for scenario: {SCE}, percentile: {PercS}')

GAM   = 1.0  #!!! This depends on the projection, ideally it should be read from 
             # the results of the probabilistic projection
             # Not used at the moment because otherwise maps of uncertainty have a very high average.
             # This depends on the meaning of the uncertainty map.

nb_years = Year - (2005+1986)/2 # Could be read from inputs
data_dir = '../../Data_Proj/'
DIR_F = data_dir+'Data_AR5/Fingerprints/'
DIR_GIA = data_dir+'GIA/ICE-6G_VM5a/'

fgia = xr.open_dataset(DIR_GIA+'dsea.1grid_O512_regridded.nc')
X_gia = (fgia.Dsea_250)*nb_years/10 # Convert from mm/year to cm

fcomp = xr.open_dataset(f'../outputs/SeaLevelPDF_AR5_glo_decomp_{SCE}.nc')

MAT_RES = fcomp.MAT_RES.sel(time = Year)

print('Processes: ')
print(fcomp.proc3.values)

X_Decomp = fcomp.decomp.sel(time_s = Year)

PDF_cum = MAT_RES[-1,:].cumsum()*100*(MAT_RES.bins[1] - MAT_RES.bins[0])

p     = misc.perc_df(MAT_RES[-1,:], [PercS], fcomp.bins).loc[PercS].values[0]

print('Contributors')
print(X_Decomp.sel(bins=p, method='nearest').values)
print(sum(X_Decomp.sel(bins=p, method='nearest').values))

;### Climate dynamics effects
DIR_CDE  = "/usr/people/bars/Project_ProbSLR/CMIP5_ThermalExp/CorrectedZOS/"
fcde     = addfile(DIR_CDE+"CorrectedZOS_EXP"+SCE+"_mean_stddev.nc","r")
CDE_mean = fcde->CorrectedZOS_reg_mean
CDE_std  = fcde->CorrectedZOS_reg_std
FillVal  = default_fillvalue(typeof(CDE_mean))
CDE_mean = where(CDE_mean.eq.CDE_mean@_FillValue,FillVal,CDE_mean)
CDE_std  = where(CDE_std.eq.CDE_std@_FillValue,FillVal,CDE_std)
CDE_mean@_FillValue = FillVal
CDE_std@_FillValue  = FillVal

;#### Read fingerprints
; For time evolving fingerprints, take the time average as a good approximation
f_gic      = addfile(DIR_F+"Relative_GLACIERS.nc","r")
lat        = f_gic->latitude
lat@units  = "degrees_north"
lon        = f_gic->longitude
lon@units  = "degrees_east"
finger_gic_t = f_gic->RSL
finger_gic = dim_avg_n(finger_gic_t(1:,:,:),0)
delete(finger_gic_t)
f_ic       = addfile(DIR_F+"Relative_icesheets.nc","r")
finger_gsmb = f_ic->SMB_GRE
finger_asmb = f_ic->SMB_ANT
finger_gdyn = f_ic->DYN_GRE
finger_adyn = f_ic->DYN_ANT
f_gw       = addfile(DIR_F+"Relative_GROUNDWATER.nc","r")
finger_gw_t  = f_gw->GROUND
finger_gw  = dim_avg_n(finger_gw_t(1:,:,:),0)
delete(finger_gw_t)

printVarSummary(finger_gsmb)

;### Compute each component
;!!! Only valid for the De Conto and Pollard projections, the assumption about 
; the fingerprint of Antarctica is 2/3 dynamics and 1/3 smb, this is not the 
; case but since East Antarctica also looses a lot of mass in this scenario
; it is more accurate to use a mixture of smb and dyn fingerprints since dyn
; is only located in west Antarctica

X_gic  = X_Decomp(1,indi)*finger_gic/100 ; Divide by 100 because fingerprints are percentages
X_gsmb = X_Decomp(2,indi)*finger_gsmb/100
X_asmb = X_Decomp(3,indi)*finger_asmb/100
X_gw   = X_Decomp(4,indi)*finger_gw/100
X_adyn = X_Decomp(5,indi)*(2./3.*finger_adyn+1./3.*finger_asmb)/100
X_gdyn = X_Decomp(6,indi)*finger_gdyn/100
; Adding uncertainty or not in the climate dynamics is a difficult choice
X_cde  = CDE_mean ;+ CDE_std*GAM*cdfnor_x(tofloat(PercS)/100.,0,1)

X_g    = X_gsmb+X_gdyn
X_a    = X_asmb+X_adyn
X_sd   = X_Decomp(0,indi) + X_cde

X_g    = where(X_g.eq.0,X_g@_FillValue,X_g)
X_a    = where(X_a.eq.0,X_a@_FillValue,X_a)
X_gic  = where(X_gic.eq.0,X_gic@_FillValue,X_gic)
X_gia  = where(X_gia.eq.0,X_gia@_FillValue,X_gia)

;### Add metadata to plot
X_gic!0   = "lat"
X_gic!1   = "lon"
X_gic&lat = lat
X_gic&lon = lon
copy_VarMeta(X_gic,X_g)
copy_VarMeta(X_gic,X_a)
copy_VarMeta(X_gic,X_sd)
copy_VarMeta(X_gic,X_gia)
copy_VarMeta(X_gic,X_gw)

printVarSummary(X_gic)
printVarSummary(X_gsmb)

test = X_gsmb+X_gdyn
printVarSummary(test)

;### Compute total sea level
TotSL = X_Decomp(0,indi) + X_gic + X_gsmb + X_asmb + X_gw + X_adyn + X_gdyn + \
        X_gia + X_cde
copy_VarMeta(X_gic,TotSL)

clat        = cos(lat*rad)
area_mean   = wgt_areaave_Wrap(TotSL, clat, 1.0, 0)
print("###### Area weigthed averaged ######")
print(area_mean)
print("###### Non weighted averaged ######")
print(avg(TotSL))

