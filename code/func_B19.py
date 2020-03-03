################################################################################
# func_B19.py: Functions to read data and build distribution of ice sheet 
#               contribution to sea level based on Bamber et al. 2019 Sructured 
#               Expert Judgment.
################################################################################

import numpy as np
import pandas as pd


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

