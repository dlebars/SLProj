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
MIN_IT = 5            # Minimum of iterations (bipasses the convergence parameter)
er = 0.1              # Convergence parameter, cm difference for the 99th percentile 
                      #of total ditribution
RESOL = 0.1           # Resolution of the pdf in cm, tested: 1, 0.1, 0.0001
    
names_col = ('Keywords', 'Values')
# Add the namelist folder to the Python path so that namelists can be read as modules
namelists_dir = '/../namelists/'
sys.path.append(os.getcwd() + namelists_dir)

# Possible choices: AR5_glo, loc, B19_glo, KNMI14, B19_loc, AR5_glo_decomp 
for namelist_name in ['AR5_glo_decomp', ]:
    
    for SCE in ['rcp45', 'rcp85']: #rcp85, rcp45
        mm.main(VER, N, MIN_IT, er, RESOL, namelist_name, SCE)
