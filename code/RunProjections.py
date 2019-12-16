### Loop over a few cases of sea level projections, read namelist and call the 
# main function.

import numpy as np
import pandas as pd
import mod_main as mm

##### User defined parameters #################################################
VER = 1.3             # File version to write in outputs
N = int(1e4)          # Number of sample in all distributions, normally: 1e4 (7*1e5)
MIN_IT = 50           # Minimum of iterations (bipasses the convergence parameter)
er = 0.1              # Convergence parameter, cm difference for the 99th percentile 
                      #of total ditribution

names_col = ('Keywords', 'Values')
namelists_dir = '../namelists/'

for namelist in ['namelist_AR5_glo.txt', ]:
    print('### Reading namelist' + namelist)
    namelist_df = pd.read_csv(namelists_dir + namelist, sep='=', comment='#', names=names_col)
    
    for SCE in ['rcp85', ]:
        mm.main(VER, N, MIN_IT, er, namelist_df, SCE)
