#
import numpy as np
from scipy.stats import norm

def main(VER, N, MIN_IT, er, namelist_df, SCE):
    """Compute future total sea level distribution.
               Calls external functions for each contribution.
                List of processes considered:
                - Thermal expansion and SSH changes
                - Glaciers and Ice Caps
                - SMB Greenland and Antarctica
                - Landwater
                - Dynamics Antarctic Ice Sheet
                - Dynamics Greenland Ice Sheet
                - Iverse barometer effect 
    """
    
    ROOT = '/Users/dewi/Work/Project_ProbSLR/Data_Proj/'
    DIR_T = ROOT+'Data_AR5/Tglobal/'
    DIR_IPCC = ROOT+'Data_AR5/Final_Projections/'
    
    ProcessNames = ['Global steric', 'Local ocean', 'Inverse barometer', 'Glaciers',    \
                 'Greenland SMB', 'Antarctic SMB', 'Landwater', 'Antarctic dynamics',\
                 'Greenland dynamics', 'sum anta.', 'Total']
    
    #List of model names
    MOD = ["ACCESS1-0","BCC-CSM1-1","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G", \
        "GFDL-ESM2M","GISS-E2-R","HadGEM2-CC","HadGEM2-ES","inmcm4","IPSL-CM5A-LR", \
        "IPSL-CM5A-MR","MIROC5","MIROC-ESM-CHEM","MIROC-ESM","MPI-ESM-LR","MPI-ESM-MR", \
        "MRI-CGCM3","NorESM1-ME","NorESM1-M"]
    nb_MOD_AR5 = len(MOD)
    
    ### General parameters
    start_date = 1986    # Start reading data
    ys = 2006   # Starting point for the integration, if this is changed problems in functions
    ye = 2100   # End year for computation

    nb_y = ye-start_date+1       # Period where data needs to be read
    nb_y2 = ye - ys +1           # Period of integration of the model

    Aoc = 3.6704e14              # Ocean Area (m2)
    rho_w = 1e3                  # Water density (kg.m-3)
    fac = -1e12 / (Aoc * rho_w)  # Convert Giga tones to m sea level
    alpha_95 = norm.ppf(0.95)
    alpha_05 = norm.ppf(0.05)

    #### Specific parameters
    ## GIC: Values of f and p, fitting parameter from formula B.1 (de Vries et al. 2014)
    f    = np.array([3.02,4.96,5.45,3.44])
    p    = np.array([0.733,0.685,0.676,0.742])
    ## Antarctic SMB
    Cref     = 1923         # Reference accumulation (Gt/year)
    Delta_C  = 5.1/100      # Increase in accumulation with local temperature (%/degC)
    Delta_Ce = 1.5/100      # Error bar
    AmpA     = 1.1          # Antarctic amplification of global temperature
    AmpAe    = 0.2          # Amplification uncertainty
    ## Initial Antarctic dynamics contribution
    a1_up_a           = 0.061    # Unit is cm/y, equal to observations in 2006
    a1_lo_a           = 0.021
    ## Initial Greenland dynamics contribution
    a1_up_g           = 0.076    # Unit is cm/y, equal to observations in 2006
    a1_lo_g           = 0.043
    
    return




