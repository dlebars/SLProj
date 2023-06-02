INFO = True
LastYear = 2100 # Last year of the computation 2100 or 2125.
# Original box from Hylke: 51-60N, -3.5,7.5E
#LOC = [60, 51, -3.5, 7.5] # lat_N, lat_S, lon_W, lon_E
# Define a polygon for the Dutch coast
LOC = [[2.5, 53], [3.3, 51.5], [4.25, 52.25], [4.75, 53.3], [5.5, 53.6], 
       [7, 53.75], [7, 55], [4, 54.5]]
LOC_FP = [53, 5] # lat, lon
IBarE = False
TEMP = 'AR6'
GRE = 'IPCC'
ODYN = 'CMIP6'
ODSL_LIST = ['ACCESS-ESM1-5', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 
             'EC-Earth3', 'GISS-E2-1-G', 'INM-CM4-8', 'MIROC-ES2L', 
             'MPI-ESM1-2-LR', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL']
ANT_DYN = 'KNMI23'
GIC = 'AR6'
GAM = 1
COMB = 'IPCC'
SaveAllSamples = False
Corr = False
CorrGT = 0.6
CorrM = "Spearman"
CorrSMB = False
CorrDYN = False
NoU_O = False
NoU_G = False
NoU_A = False
NoU_Gl = False
Decomp = False
LowPass = 3
BiasCorr = False
InterpBack = True