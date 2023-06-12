################################################################################
#Set variables to make a sea level projections with "main.ncl"
#
# Description of variables:
# INFO: Set to true to get more info in output
# LastYear: Last year of the computation 2100 or 2125.
# LOC: If local give either the box coordinates [lat_N, lat_S, lon_W, lon_E] or
# a polygon that will be used with regionmask for ocean
# dynamics averaging. Otherwise False for global sea level projections.
# Original box from KNMI14: 51-60N, -3.5,7.5E
# For local projections the fingerprint location is also used:
# LOC_FP = [lat, lon] (Point where the fingerprints are read)
# IBarE: If True, include the inverse barometer effect, only used for local 
#        projections. But not necessary (generally small effect).
# TEMP: Choose option for global surface air temperature. 
# Possible choices: "AR5": same models as for IPCC AR5
#                   "CMIP5": all CMIP5 models available (used to be called 'all')
#                   "AR6": Result of the AR6 assessment, mixing emulator and 
#                          corrected CMIP6
#                   "CMIP6": all CMIP6 models available
# GRE: Greenland dynamics and surface mass balance
# Possible choices: "IPCC": RCP8.5 is different from other scenarios
#                   "KNMI14": Dynamics for all scenarios the same (average of both 
#                             IPCC values)
#                   "B19": Greenland contribution from Bamber et al. 2019 SEJ, this
#                   method does not split between SMB and dynamics so in the code it
#                   is all attributed to SMB and dynamics is 0 in this case.
#                   "AR6": Uses the numbers from IPCC AR6 (with half-normal distribution 
#                   assumption). Also defined for ssp585_hpp.
# ODYN: Option for Ocean Dynamics
# Possible choices: "IPCC" can only be used for global analysis (read file 
#                   distributed by AR5)
#                   "KNMI" use data from KNMI14 scenarios. Can be used for both 
#                   local and global but only for RCP4.5 and RCP.8.5. For global
#                   "IPCC" option is preferable.
#                   "CMIP5" or "CMIP6" use data computed directly from the CMIP databases.
#                   It can be used for both local and global computations. These 
#                   input files should be computed before running the sea level 
#                   projection.
# ODSL_LIST: Option to select the models for ODSL
# Possible choices: A list of models or False to use all models
# ANT_DYN: Antarctic dynamics
# Possible choices: "IPCC" for AR5 process based 
#                   "KNMI14" as implemented by de Vries et al. 2014 for the KNMI'14 
#                   scenarios
#                   "KNMI16" for Rijkswaterstaat project
#                   "DC16"   DeConto and Pollard 2016
#                   "DC16T"  DeConto and Pollard 2016 with with temperature sensitivity
#                   "LEV14"  Levermann et al. 2014, only the 3 ice shelves models
#                   "LEV20"  Levermann et al. 2020, 17 models
#                   "B19":   Antarctic contribution from Bamber et al. 2019 SEJ, this
#                   method does not split between SMB and dynamics so in the code it
#                   is all attributed to SMB and dynamics is 0 in this case.
#                   "SROCC"  IPCC SROCC report, only defined for RCP8.5. For other 
#                            scenarios SROCC is similar to AR5
#                   "KS21": Uses the numbers from IPCC AR6 (with half-normal distribution 
#                   assumption). Also defined for ssp585_hpp.
#                   "KNMI23": Similar to KS21 but with a slower increase of the uncertainty
#                   "VDL23": Results from van Der Linden 2023
# GIC: Glaciers and ice caps
# Possible choices: "AR5" or "AR6"
# LWS: Land water storage
# Possible choices: "AR5" or "AR6". Remark: AR6 option is an approximation of 
#                   the AR6 results based on a 2nd order polynomial.
# GAM: Uncertainty of climate models
# Possible choices: 1 for IPCC AR5
#                   1.64 to convert the IPCC expert judgement that the 5-95 
#                   percentiles from climate models is the likely range.
# COMB: Combination of sea level contributors 
# Possible choices: "IPCC" Follows the normal IPCC way to combine processes
#                   "IND"  Assume complete independence
#                   "DEP"  Assumes complete dependence (correlation = 1 )
# SaveAllSamples: If True save all samples in a big array until the end of the 
#                 computation. Increases the memory use. Necessary for "Corr" and 
#                 "Decomp"
# Corr: Set True to compute the Pearson and Spearman correlations between sea level 
#       contributors. Increases the computing time significantly. Correlation between
#       contributors is always used in the model but not necessarily computed. This
#       option does not change the results but adds some diagnostics.
# CorrGT: Correlation between the GMST and thermal expansion, default is 1 for 
#       IPCC AR5. Around 0.6 for local Dutch Coast. Global between 0.3 (RCP4.5) and
#       0.47 (RCP2.6 and RCP8.5), see 2017 report for RWS about these correlations.
#       The value of CorrGT is overwritten by COMB if COMB is IND or DEP.
# CorrM: Correlation measure of CorrGT. Can be either "Pearson" or "Spearman"
# CorrSMB: If True, correlate some model uncertainty from the glaciers, Greenland 
#          SMB and Antactic SMB.
# CorrDYN: If True, correlate the Antarctic and Greenland dynamics, only applies
#          to "LEV14" case
# NoU_O: If True, removes ocean uncertainty. Replace the distribution by its 
#        expected value.
# NoU_G: If True, remove Greenland uncertainty.
# NoU_A: If True, remove Antartic uncertainty.
# NoU_Gl: If True, remove Glacier and ice caps uncertainty
# Decomp: Save the decomposition of each total sea level value into the average 
#         components that form it. SaveAllSamples needs to be true.
# LowPass: If integer then filter the time series of stero-dynamics and GMST with a 
#          polynomial fit of the order of the integre provided.
#          Default option: False
# BiasCorr: There is a possibility to include a bias correction on the local ocean
#           dynamics term
# InterpBack: The reference period of the code is 1986-2005 (including 2005). 
#          Data is usually provided from 2006 but can be interpolated back to the 
#          middle of the reference period if this option is true. Then data starts 
#          in 1995.
################################################################################
Example of namelist file:
INFO = True
LastYear = 2100
LOC = [60, 51, -3.5, 7.5]
LOC_FP = [53, 5]
IBarE = False
TEMPf = 'AR5'
GRE = 'IPCC'
ODYN = 'KNMI'
ODSL_LIST = False
ANT_DYN = 'IPCC'
GAM = 1
COMB = 'IPCC'
SaveAllSamples = True
Corr = False
CorrGT = 1
CorrM = "Spearman"
CorrSMB = False
CorrDYN = False
NoU_O = False
NoU_G = False
NoU_A = False
NoU_Gl = False
Decomp = False
LowPass = False
BiasCorr = False
InterpBack = False