################################################################################
# func_B19.py: Functions to read data and build distribution of ice sheet 
#               contribution to sea level based on Bamber et al. 2019 Sructured 
#               Expert Judgment.
################################################################################

import numpy as np
import pandas as pd
from scipy import stats

def ReadB19(path):
    '''Read distribution parameter data (shape, location and scale) that are 
    computed from a separate Notebook and convert csv file to an array with 
    convenient dimensions'''

    PARc = pd.read_csv(path, delim_whitespace=True)

    # Convert csv to an array with convenient dimensions
    PAR = np.zeros([3, 3, 2, 2]) # ICE SHEET, PARAMETER, SCE, TIME
    for i in range(0,3):
        for j in range(0,3):
            for s in range(0,2):
                for t in range(0,2):
                    PAR[i, j, s, t] = PARc.iloc[ i + t*3 + s*6, j]

    return PAR

def TempF(T0, T1, T2, t0, t1, t2):
    '''Define the high and low temperature scenarios from Bamber et al. 2019'''
    tf = np.arange(t0, t2+1)
    T = np.zeros(len(tf))
    T[0:t1-t0+1] = (T1 - T0)/(t1 - t0) * tf[0:t1-t0+1] + (t1*T0 - t0*T1)/(t1-t0)
    T[t1-t0+1:t2-t0+1] = (T2 - T1)/(t2 - t1) * tf[t1-t0+1:t2-t0+1] + (t2*T1 - t1*T2)/(t2-t1)
    return T


def TempInterp(IceSheet, Tdcum, Tlcum, Thcum, PAR, UnifP):
    '''Temperature interpolation function'''
    N = Tdcum.shape[0]
    ind0 = 0  
    if IceSheet == 'GIS':
        i = 0
    elif IceSheet == 'WAIS':
        i = 1
    elif IceSheet == 'EAIS':
        i = 2
    else:
        print('ERROR: Ice sheet name '+IceSheet+' not recognised')

    PARtT = np.zeros([3, 2, N]) # PARAM, TIME, DRAWS
    for j in range(0,3):
        for t in range(0,2):
            PARtT[j, t, :] = (PAR[i, j, 1, t] - PAR[i, j, 0, t])/ \
            (Thcum[t] - Tlcum[t]) * Tdcum[:, t] \
            + (Thcum[t]*PAR[i, j, 0, t] - Tlcum[t]*PAR[i, j, 1, t]) / \
            (Thcum[t] - Tlcum[t])
    PARtT[0,:,:] = np.where(PARtT[0,:,:] <= 0, 0.001, PARtT[0,:,:])
    ind0 = np.where(PARtT[0,:,:] == 0.001)
    if len(ind0[0]) > 0:
        print('Warning: The shape parameter was ' +\
              'set to 0.001 '+ str(len(ind0[0])) + ' times out of ' + str(2*N) + \
              ' samples to avoid negative values')
        
    # Convert the shape parameters to a x value from gamma using the percentiles
    # Could be much faster with categories of similar shape
    xg = np.zeros([2, N]) # DRAWS, TIME
    for t in range (0,2):
        xg[t, :] = stats.gamma.ppf( UnifP, PARtT[0, t, :], 0, 1)

    # Apply scale and location parameters
    IS_cont = xg * PARtT[2, :, :] + PARtT[1, :, :]

    return IS_cont


def TimeInterp(IS_cont, td, v0):
    '''Time interpolation between present day mass loss and future SEJ data'''

    N = IS_cont.shape[1]
    tf = np.arange(td[0],td[2]+1)

    t0 = td[0]
    t1 = td[1]
    t2 = td[2]

    ## Extrapollate using a third order polynomial:
    # X(t) = p3*(t-t0)^3 + p2*(t-t0)^2 + p1*(t-t0) + p0
    # Using three conditions:
    # X(t0) = 0
    # X'(t0) = v0
    # X(t1) = X1
    # X(t2) = X2

    X1 = IS_cont[0,:]
    X2 = IS_cont[1,:]

    p0 = 0
    p1u = np.linspace(v0[0], v0[1], num=N)
    # Change the order of p1u to fit the order of IS_cont    
    ISpv = np.argsort(IS_cont[0,:])
    ISpv2 = np.argsort(ISpv)
    p1 = p1u[ISpv2]
    # Check
#     print('Compute stats.spearmanr(p1, IS_cont[0,:], result should be 1:')
#     print(stats.spearmanr(p1, IS_cont[0,:]))
#     print('Compute stats.spearmanr(p1u, IS_cont[0,:], result should be 0:')
#     print(stats.spearmanr(p1u, IS_cont[0,:]))
    
    Pa = X1 - p1*(t1 - t0) - p0
    Pb = X2 - p1*(t2 - t0) - p0

    p3 = (Pa*(t2-t0)**2 - Pb*(t1-t0)**2) / \
    ( (t1-t0)**3 * (t2-t0)**2 - (t2-t0)**3 * (t1-t0)**2 )
    p2 = (Pa - p3 * (t1-t0)**3) / (t1-t0)**2

    IS_cont_t = np.zeros([N, len(tf)])
    for t in range(0, len(tf)):
        IS_cont_t[:, t] = p3 * t**3 + p2 * t**2 + p1*t + p0

    return IS_cont_t

def Bamber19(IceSheet, UnifP, v0, t0, Td):
    '''Combine all previous functions to provide the final ice sheet 
    contribution Works for IceSheet is 'GIS', 'WAIS' or 'EAIS' '''
    # Read the parameters data: Shape, location and scale
    # Option: parameters.csv and parameters_mean_std_opt.csv that differ in the
    # way that the optimisation is performed to fit the distribution
    path = "../../BamberDataDistribution/parameters_mean_std_opt.csv" 
    PAR = ReadB19(path)

    t1 = 2050
    t2 = 2100
    tf = np.arange(t0, t2+1)

    ## First order interpolation along temperature integral
    # Define the temperature functions for the low and high scenarios
    T0 = 0.8     # Should be floats, integers lead to problem in the computations
    Tl50 = 1.5
    Tl100 = 2.
    Th50 = 2.
    Th100 = 5.

    Tl = TempF(T0, Tl50, Tl100, 2000, t1, t2)
    Th = TempF(T0, Th50, Th100, 2000, t1, t2)
    Tlcum = Tl.cumsum()
    Thcum = Th.cumsum()

    Tdcum = Td.cumsum(axis=1)
    indt = np.zeros(2, dtype=int)
    indt[0] = abs(tf - t1).argmin()
    indt[1] = abs(tf - t2).argmin()

    ## Temperature interpollation
    IS_cont = TempInterp(IceSheet, Tdcum[:,indt], Tlcum[indt], Thcum[indt], PAR, UnifP)

    ## Time interpolation:
    # Start from present day observed rates of melt (or AR5 range)
    # Interpolate through the future using a 3rd order polynomial. 
    # To add data for latter dates a higher order polynomial might be used

    IS_cont_t = TimeInterp(IS_cont, [t0, t1, t2], v0)

    return IS_cont_t
