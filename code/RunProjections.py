### Loop over a few cases of sea level projections, read namelist and call the 
# main function.

import numpy as np
import pandas as pd
import mod_main as mm
import sys
import os

##### User defined parameters #################################################
VER = 1.0             # File version to write in outputs
N = int(1e5)          # Number of sample in all distributions, normally: 1e4 (7*1e5)
MIN_IT = 5           # Minimum of iterations (bipasses the convergence parameter)
er = 0.1              # Convergence parameter, cm difference for the 99th percentile 
                      #of total ditribution

names_col = ('Keywords', 'Values')
# Add the namelist folder to the Python path so that namelists can be read as modules
namelists_dir = '/../namelists/'
sys.path.append(os.getcwd() + namelists_dir)

for namelist_name in ['loc', ]: # AR5_glo, loc
    
    for SCE in ['rcp85', ]:
        mm.main(VER, N, MIN_IT, er, namelist_name, SCE)
