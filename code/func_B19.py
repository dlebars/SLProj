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
    tf = np.arange(t0, t2)
    T = np.zeros(len(tf))
    T[0:t1-t0+1] = (T1 - T0)/(t1 - t0) * tf[0:t1-t0+1] + (t1*T0 - t0*T1)/(t1-t0)
    T[t1-t0+1:t2-t0+1] = (T2 - T1)/(t2 - t1) * tf[t1-t0+1:t2-t0+1] + (t2*T1 - t1*T2)/(t2-t1)
    return T


def TempInterp(IceSheet, Tdcum, Tlcum, Thcum, PAR, UnifP):
    '''Temperature interpolation function'''
    dimTd = Tdcum.shape()
    N = dimTd[0]
    if IceSheet == 'GIS':
        i = 0
    else if IceSheet == 'WAIS':
        i = 1
    else if IceSheet == 'EAIS':
        i = 2
    else
        print('ERROR: Ice sheet name '+IceSheet+' not recognised')

    PARtT = np.zeros([3, 2, N]) # PARAM, TIME, DRAWS
    for j in range(0,3):
        for t in range(0,2):
            PARtT[j, t, :] = (PAR[i, j, 1, t] - PAR[i, j, 0, t])/ \
            (Thcum[t] - Tlcum[t]) * Tdcum[:, t] \
            + (Thcum[t]*PAR[i, j, 0, t] - Tlcum[t]*PAR[i, j, 1, t]) / \
            (Thcum[t] - Tlcum[t])

    # Convert the shape parameters to a x value from gamma using the percentiles
    # Could be much faster with categories of similar shape
    xg = np.zeros([2, N]) # DRAWS, TIME
    for r in range(0,N):
    for t in range (0,2):
        if PARtT[0, t, r] <= 0:
            print('Warning: PARtT[0, t, r] <= 0, setting to 0.01')
            PARtT[0, t, r] = 0.01
        xg[t, r] = stats.gamma.ppf( UnifP[r], PARtT[0, t, r], 0, 1)

    # Apply scale and location parameters
    ICcont = xg * PARtT[2, :, :] + PARtT[1, :, :]

    return ICcont


# def TimeInterp(IS_cont, td, v0):
#     '''Time interpolation between present day mass loss and future SEJ data'''

#     dimIS = dimsizes(IS_cont)
#     N = dimIS(1)
#     tf = ispan(td(0),td(2),1)
#     dimt = dimsizes(tf)

#     t0 = td(0)
#     t1 = td(1)
#     t2 = td(2)

#     ;# Extrapollate using a third order polynomial:
#     ; X(t) = p3*(t-t0)^3 + p2*(t-t0)^2 + p1*(t-t0) + p0
#     ; Using three conditions:
#     ; X(t0) = 0
#     ; X'(t0) = v0
#     ; X(t1) = X1
#     ; X(t2) = X2

#     X1 = IS_cont(0,:)
#     X2 = IS_cont(1,:)

#     p0 = 0
#     p1u = fspan(v0(0), v0(1), N)
#     ;Change the order of p1u to fit the order of IS_cont
#     ISpv  = dim_pqsort(IS_cont(0,:), 1)
#     ISpv2 = dim_pqsort(ISpv, 1)
#     p1 = p1u(ISpv2)

#     Pa = X1 - p1*(t1 - t0) - p0
#     Pb = X2 - p1*(t2 - t0) - p0

#     p3 = (Pa*(t2-t0)^2 - Pb*(t1-t0)^2) / ( (t1-t0)^3*(t2-t0)^2 - (t2-t0)^3*(t1-t0)^2 )
#     p2 = (Pa - p3*(t1-t0)^3) / (t1-t0)^2

#     IS_cont_t = new((/N, dimt/), float)
#     do t=0,dimt-1
#     IS_cont_t(:, t) = p3*t^3 + p2*t^2 + p1*t + p0
#     end do

#     return IS_cont_t