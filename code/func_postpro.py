# Functions used to facilitate data analysis, plots, comparison between 
# projections and observations.

import numpy as np
import pandas as pd

def read_knmi14(add_subsidence=False):
    '''Read KNMI14 projections from csv file.
    Possibility to add subsidence with a value of 0.045 cm/y as in the 
    Zeespiegelmonitor.
    Export a handy dataframe for further analysis.'''
    
    path_KNMI14 = '/Users/dewilebars/Projects/Project_ProbSLR/KNMI14/'
    KNMI14_df = pd.read_csv(path_KNMI14 + 'K14_scenarios_by_year.csv')
    del(KNMI14_df['year.1'])
    KNMI14_df = KNMI14_df.set_index('year')

    if add_subsidence:
        subsidence = np.arange(KNMI14_df.shape[0]) * 0.045

        for i in KNMI14_df.columns:
            KNMI14_df[i + '_sub'] = KNMI14_df[i] + subsidence
        
    return KNMI14_df