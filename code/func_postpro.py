# Functions used to facilitate data analysis, plots, comparison between 
# projections and observations.

import numpy as np
import pandas as pd

def read_knmi14(add_subsidence=False, polyfit=False):
    '''Read KNMI14 projections from csv file.
    Possibility to add subsidence with a value in cm/y.
    'polyfit' provides a way to choose to read the original data or the data 
    after fitting a 3rd degree polynomial
    Export a handy dataframe for further analysis.'''
    
    path_KNMI14 = '/Users/dewilebars/Projects/Project_ProbSLR/KNMI14/'
    
    if polyfit:
        KNMI14_df = pd.read_csv(path_KNMI14 + 'KNMI14_sealevel_scenarios_by_1year_3polfit.csv')
        del(KNMI14_df['comment'])
    else:
        KNMI14_df = pd.read_csv(path_KNMI14 + 'K14_scenarios_by_year.csv')
        del(KNMI14_df['year.1'])
        
    KNMI14_df = KNMI14_df.set_index('year')

    if add_subsidence:
        subsidence = np.arange(KNMI14_df.shape[0]) * add_subsidence

        for i in KNMI14_df.columns:
            KNMI14_df[i + '_sub'] = KNMI14_df[i] + subsidence
        
    return KNMI14_df

def define_area(reg):
    '''Provides box coordinates given a region name'''
    
    if reg == 'dutch_coast':
        lon_min, lon_max = 3, 7
        lat_min, lat_max = 51, 54
    elif reg == 'north_sea':
        lon_min, lon_max = -2, 9
        lat_min, lat_max = 48, 60
    elif reg == 'knmi14_reg':
        lon_min, lon_max = -3.5, 7.5
        lat_min, lat_max = 51, 60
    
    return lon_min, lon_max, lat_min, lat_max