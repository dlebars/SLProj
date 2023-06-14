import numpy as np
import scipy.stats as stats
import pandas as pd
import xarray as xr
import glob
import importlib
from datetime import datetime
import sys
import os
#sys.path.append('../code')
import func_odyn as odyn
import func_misc as misc
import func_gic as gic
import func_gre as gre
import func_ant as ant
import func_B19 as b19

def main(VER, N, MIN_IT, er, namelist_name, SCE):
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
    
    print('#### Starting a new computation ##################################')
    print(f'Using namelist_{namelist_name} and scenario {SCE}')
    nl = importlib.import_module(f'namelist_{namelist_name}')

    ROOT = '/Users/dewilebars/Projects/Project_ProbSLR/Data_Proj/'
    DIR_IPCC = f'{ROOT}Data_AR5/Final_Projections/'
    DIR_OUT = '../outputs/'       # Output directory
    
    if nl.LOC:
        DIR_F       = ROOT+'Data_AR5/Fingerprints/'
        DIR_IBE     = ROOT+'Data_AR5/InvertedBarometer/1x1_reg/'
        DIR_IBEmean = ROOT+'Data_AR5/InvertedBarometer/globalmeans_from_1x1_glob/'

        if nl.ODYN == 'KNMI':
            DIR_O       = ROOT + 'Data_AR5/Ocean/1x1_reg/'
            DIR_OG      = ROOT + 'Data_AR5/Ocean/globalmeans_from_1x1_glob/'
        elif nl.ODYN == 'IPCC':
            sys.exit('ERROR: This option of ocean dynamics can only be used for' +
                     ' global computations')

    else:
        if nl.ODYN == 'KNMI':
            DIR_OG      = ROOT + 'Data_AR5/Ocean/globalmeans_from_1x1_glob/'

    
    ProcessNames = ['Global steric', 'Ocean Dynamic Sea Level', 'Inverse barometer', 'Glaciers',
                 'Greenland SMB', 'Antarctic SMB', 'Landwater', 'Antarctic dynamics',
                 'Greenland dynamics', 'Greenland', 'Antarctica', 'GIA','Total']
    
    # Percentiles to print at run time and store in outputs
    Perc = [1,5,10,17,20,50,80,83,90,95,99]
    nb_perc = len(Perc)
    
    if nl.SaveAllSamples:
        if nl.LOC:  # Number of components, used to compute correlations efficently.
            NameComponents = ["Glob. temp.", "Glob. thermal exp.", "ODSL",
                              "Barometer effect", "Glaciers", "Green. SMB",
                              "Ant. SMB", "Land water", "Ant. dyn.", "Green dyn.", ]
            print("!!! Warning: This combination of SaveAllSamples ="+
                  f" {nl.SaveAllSamples} and LOC= {nl.LOC} hasn't been tested")
        else:
            NameComponents = ["Glob. temp.", "Thermal exp.", "Glaciers", "Green. SMB",
                              "Ant. SMB", "Land water", "Ant. dyn.", "Green dyn."]
            
        nb_comp = len(NameComponents)
    
    #List of model names
    MOD = misc.model_selection_ar5()
    nb_MOD_AR5 = len(MOD)
    
    ### General parameters
    start_date = 1980    # Start reading data
    ys = 2006   # Starting point for the integration, if this is changed 
                # then expect problems in functions
    ye = nl.LastYear

    nb_y = ye-start_date+1       # Period where data needs to be read
    nb_y2 = ye - ys +1           # Period of integration of the model

    Aoc = 3.6704e14              # Ocean Area (m2)
    rho_w = 1e3                  # Water density (kg.m-3)
    fac = -1e12 / (Aoc * rho_w)  # Convert Giga tones to m sea level

    #### Specific parameters
    ## Initial Antarctic dynamics contribution
    a1_up_a           = 0.061    # Unit is cm/y, equal to observations in 2006
    a1_lo_a           = 0.021
    ## Initial Greenland dynamics contribution
    a1_up_g           = 0.076    # Unit is cm/y, equal to observations in 2006
    a1_lo_g           = 0.043
    
    ####
    TIME       = np.arange( start_date, ye + 1 )
    TIME2      = np.arange( ys, ye + 1, 1 )
    # This option keeps all years. It uses a lot of memory
    ind_d      = np.arange(len(TIME2)) 
    # This option selects 1 in 10 years to save memory
    #ind_d = np.where(TIME2 % 10 == 0)[0] # Select the indices of 2010, 2020... 2100
    nb_yd      = len(ind_d)
    
    #### Read fingerprints, some are time dependent so make all of them time 
    # dependent for easy mutliplication at the end.
    
    F_gic2  = np.ones(nb_y2)
    F_gsmb2 = np.ones(nb_y2)
    F_asmb2 = np.ones(nb_y2)
    F_gdyn2 = np.ones(nb_y2)
    F_adyn2 = np.ones(nb_y2)
    F_gw2   = np.ones(nb_y2)
    
    if nl.LOC:
        FPs    = 1986 # The fingerprints start in 1986 and end in 2100,
        FPe    = 2100 # but do not have a useful time vector so we create one here
        TIME_F = np.arange(FPs,FPe+1)
        ifs    = np.abs(ys - TIME_F).argmin()
        ife    = np.abs(ye - TIME_F).argmin()
        jfs    = np.abs(FPs - TIME2).argmin()
        jfe    = np.abs(FPe - TIME2).argmin()

        # Read fingerprint for Glaciers and Ice Caps
        f_gic = xr.open_dataset(DIR_F+'Relative_GLACIERS_reg.nc', \
                                     decode_times=False)
        F_gic = misc.finger1D(nl.LOC_FP[0], nl.LOC_FP[1], f_gic.latitude, f_gic.longitude, 
                              f_gic.RSL)
        F_gic2[jfs:jfe+1] =  F_gic[ifs:ife+1]/100 # Convert from % to fraction

        del(f_gic, TIME_F)

        f_ic        = xr.open_dataset(DIR_F+'Relative_icesheets_reg.nc')
        lat_ic      = f_ic.latitude #tofloat?
        lon_ic      = f_ic.longitude
        F_gsmb      = misc.finger1D(nl.LOC_FP[0], nl.LOC_FP[1], lat_ic, lon_ic, f_ic.SMB_GRE)
        F_gsmb2[jfs:jfe+1]  =  F_gsmb/100
        F_asmb      = misc.finger1D(nl.LOC_FP[0], nl.LOC_FP[1], lat_ic, lon_ic, f_ic.SMB_ANT)
        F_asmb2[jfs:jfe+1]  =  F_asmb/100
        F_gdyn      = misc.finger1D(nl.LOC_FP[0], nl.LOC_FP[1], lat_ic, lon_ic, f_ic.DYN_GRE)
        F_gdyn2[jfs:jfe+1]  =  F_gdyn/100
        F_adyn      = misc.finger1D(nl.LOC_FP[0], nl.LOC_FP[1], lat_ic, lon_ic, f_ic.DYN_ANT)
        F_adyn2[jfs:jfe+1]  =  F_adyn/100

        del(f_ic, lat_ic, lon_ic)

        f_gw       = xr.open_dataset(DIR_F+'Relative_GROUNDWATER_reg.nc', \
                                     decode_times=False)
        lat_gw     = f_gw.latitude #tofloat?
        lon_gw     = f_gw.longitude
        finger_gw  = f_gw.GROUND
        #finger_gw@_FillValue = 0
        F_gw       = misc.finger1D(nl.LOC_FP[0], nl.LOC_FP[1], lat_gw, lon_gw, finger_gw)
        F_gw2[jfs:jfe+1]  =  F_gw[ifs:ife+1]/100

        del(f_gw)
        del(lat_gw)
        del(lon_gw)
        del(finger_gw)

        # Extend the values of the fingerprints after 2100:
        if jfe < (nb_y2-1):
            print('Extending fingerprints using last available value')
            F_gic2[jfe+1:] = F_gic2[jfe]
            F_gsmb2[jfe+1:] = F_gsmb2[jfe]
            F_asmb2[jfe+1:] = F_asmb2[jfe]
            F_gdyn2[jfe+1:] = F_gdyn2[jfe]
            F_adyn2[jfe+1:] = F_adyn2[jfe]
            F_gw2[jfe+1:] = F_gw2[jfe]

        del(ifs)
        del(ife)
        del(jfs)
        del(jfe)
        
        if nl.INFO:
            print('Check all fingerprint values:')
            print('Glaciers and ice caps (F_gic2):')
            print(F_gic2)
            print('Greenland SMB (F_gsmb2):')
            print(F_gsmb2)
            print('Antarctic SMB (F_asmb2):')
            print(F_asmb2)
            print('Greenland dynamics (F_gdyn2):')
            print(F_gdyn2)
            print('Antarctic dynamics (F_adyn2):')
            print(F_adyn2)
            print('Groundwater change (F_gw2):')
            print(F_gw2)
        
    ###########################################################################
    if nl.INFO:
        print("### Read Tglob             #################")
        
    TGLOB = misc.make_tglob_array(ROOT, nl.TEMP, SCE, start_date, ye , nl.LowPass)
    TGLOBs = TGLOB.sel(time=slice(ys,None))

    # Compute the temperature anomalies for each process using a different 
    # reference temperature
    T_gic = TGLOBs - TGLOB.sel(time=2006) # Glaciers and Ice Caps
    T_g = TGLOBs - TGLOB.sel(time=slice(1980,1999)).mean(dim='time') # Greenland SMB
    T_a = TGLOBs - TGLOB.sel(time=slice(1985,2005)).mean(dim='time') # Antarctic SMB
    T_ad = TGLOBs - TGLOB.sel(time=2000) # Antarctic dynamics for DC16T
    T_b = TGLOBs - TGLOB.sel(time=2000) # Bamber et al. 2019
    
    ref_steric = [1986, 2005] # Reference period for steric sea level
    
    del(TGLOB)
    del(TGLOBs)
    
    ##############################################################################
    # Start loop 
    CONV  = [] # Check for convergence NCL: NewList("lifo")
    nb_it = 0
    END = False
    
    X_O_G_perc     = np.zeros([nb_perc+1,nb_y2])
    X_O_A_perc     = np.zeros([nb_perc+1,nb_y2])
    X_B_perc       = np.zeros([nb_perc+1,nb_y2])
    X_gic_perc     = np.zeros([nb_perc+1,nb_y2])
    X_gsmb_perc    = np.zeros([nb_perc+1,nb_y2])
    X_asmb_perc    = np.zeros([nb_perc+1,nb_y2])
    X_landw_perc   = np.zeros([nb_perc+1,nb_y2])
    X_ant_perc     = np.zeros([nb_perc+1,nb_y2])
    X_gre_perc     = np.zeros([nb_perc+1,nb_y2])
    X_ant_tot_perc = np.zeros([nb_perc+1,nb_y2])
    X_gre_tot_perc = np.zeros([nb_perc+1,nb_y2])
    X_gia_perc     = np.zeros([nb_perc+1,nb_y2])
    X_tot_perc     = np.zeros([nb_perc+1,nb_y2])

    if nl.SaveAllSamples:
        X_all       = np.zeros([nb_comp, N, nb_yd])

    if nl.Corr:
        nb_el       = nb_comp*(nb_comp-1)/2
        M_Corr_P    = np.zeros([nb_yd,nb_el])  # Pearson correlation matrice 
        M_Corr_S    = np.zeros([nb_yd,nb_el])  # Rank correlation matrice

    if nl.Decomp:
        X_Decomp = np.zeros([nb_comp-1,nb_perc,nb_yd])
    
    while not END:
        nb_it = nb_it + 1
        comp  = 1    # Re-initialize the component count for the Corr case

        #### Set seeds for the random number generators
        # Does not seem to be necessary in Python -> Check

        # Sample a normal distribution to use for temperature
        NormD  = np.random.normal(0, 1, N)

        X_tot = np.zeros([N,nb_y2])
        
        if nl.COMB:
            # Reorder contribution in ascending order
            NormD   = np.sort(NormD)

        if nl.SaveAllSamples:
            # NormD is used in the correlation with temperature because all of 
            # the temperature distributions (different for each process) are 
            # based on a linear combination of it. 
            ar1 = np.ones(nb_yd)
            NormDc = NormD[:, np.newaxis] * ar1
            X_all[0, :, :] = NormDc

        #######################################################################
        if nl.INFO:
            print("### Thermal expansion and ocean dynamics #################")

        CorrGT = nl.CorrGT
        if nl.COMB == 'IND':
            CorrGT   = 0 # Force a 0 correlation, even if the CorrGT coefficient
                         # has another value
        elif nl.COMB == 'DEP':
            CorrGT = 1

        NormDT1 = np.random.normal(0, 1, N)
        # Build NormDT as a combination of NormD (the distribution of GMST) and 
        # an independent normal distribution.
        if nl.CorrM == 'Pearson':
            rhoP  = CorrGT
        elif nl.CorrM == 'Spearman':
        # Convert correlation coefficient from Spearman to Pearson
            rhoP  = 2 * np.sin( np.pi / 6 * CorrGT)

        NormDT = NormD*rhoP + NormDT1*np.sqrt(1 - rhoP**2)
        
        # For the hpp scenario the thermal expansion and dynamics are the same as ssp585 
        if SCE == 'ssp585_hpp':
            SCE_loc = 'ssp585'
        else:
            SCE_loc = SCE

        if nl.LOC:
            if nl.ODYN == 'KNMI':
                X_Of = odyn.odyn_loc(SCE_loc, MOD, DIR_O, DIR_OG, nl.LOC, 
                                     ref_steric, ye, N, ys, nl.GAM, NormDT, 
                                     nl.LowPass)
            elif nl.ODYN in ['CMIP5', 'CMIP6']:                   
                X_Of = odyn.odyn_cmip(SCE_loc, ROOT, nl.LOC, ref_steric, ye, N, 
                                      ys, nl.GAM, NormDT, nl.LowPass, 
                                      nl.BiasCorr, nl.ODSL_LIST)
        else:
            if nl.ODYN == 'KNMI':
                X_Of = odyn.odyn_glob_knmi(SCE_loc, MOD, nb_y, nb_y2, DIR_O, DIR_OG,
                                      start_date, ye, N, i_ys, nl.GAM, NormDT)
            elif nl.ODYN == 'IPCC':
                X_Of = odyn.odyn_glob_ipcc(SCE_loc, DIR_IPCC, N, nb_y2, nl.GAM, NormDT)
                
            elif nl.ODYN in ['CMIP5', 'CMIP6']:
                X_Of = odyn.odyn_cmip(SCE_loc, ROOT, nl.LOC, ref_steric, ye, N, 
                                      ys, nl.GAM, NormDT, nl.LowPass, 
                                      nl.BiasCorr, nl.ODSL_LIST)

        X_O_G_perc += np.concatenate( (np.percentile(X_Of[1,:,:], Perc, axis=0), 
                                       X_Of[1,:,:].mean(axis=0, keepdims=True)), axis=0)
        
        X_O_A_perc += np.concatenate( (np.percentile(X_Of[2,:,:], Perc, axis=0), 
                                       X_Of[2,:,:].mean(axis=0, keepdims=True)), axis=0)
        
        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_Of   = np.sort(X_Of, 1)

        if nl.NoU_O:
            for t in range(0, nb_y2):
                X_tot[:, t] = X_Of[0, :, t].mean()
        else:
            X_tot = X_Of[0, :, :]

        if nl.SaveAllSamples:
            if nl.LOC:
                X_all[comp,:,:] = X_Of[1, :, ind_d].swapaxes(0,1)
                comp            = comp + 1
                X_all[comp,:,:] = X_Of[2, :, ind_d].swapaxes(0,1)
                comp            = comp + 1
            else:
                X_all[comp,:,:] = X_Of[0, :, ind_d].swapaxes(0,1)
                #X_all[comp,:,:] = np.squeze(X_Of[0:1, :, ind_d])
                #!!! swapaxes or squeeze are work arround a peculiar Numpy 
                # behaviour, see Tests.ipynb
                comp            = comp + 1

        del(X_Of)
        del(NormDT1)
        del(NormDT)
        
        #######################################################################
        if nl.IBarE:
            if nl.INFO:
                print('### Inverse Barometer effect #########################')
            # TO DO
        
        #######################################################################
        if nl.INFO:
            print('### Glaciers and Ice Caps (Not including Antarctica) #####')

        if nl.COMB == 'IND':
            # Redefine NormD to loose correlation
            NormD  = np.random.normal(0, 1, N)

        #Build the distribution of global temperature for this process
        Td_gic = misc.normal_distrib(T_gic, nl.GAM, NormD)

        NormDs  = np.random.normal(0, 1, N)   # This distribution is then kept 
                                              #for correlation
        X_gic = gic.gic_ipcc(Td_gic, NormDs, nl.GIC)

        for t in range(0,nb_y2): # Use broadcasting?
            X_gic[:,t] = X_gic[:,t]*F_gic2[t]
        
        X_gic_perc += np.concatenate( (np.percentile(X_gic, Perc, axis=0), 
                                       X_gic.mean(axis=0, keepdims=True)), axis=0)

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_gic   = np.sort(X_gic, 0)

        if nl.NoU_Gl:
            for t in range(0,nb_y2):
                X_tot[:,t] = X_tot[:,t] + X_gic[:,t].mean()
        else:
            X_tot = X_tot + X_gic

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_gic[:,ind_d]
            comp = comp + 1

        del(X_gic)
        del(Td_gic)
        
        ###############################################################################
        if nl.INFO:
            print("### SMB Greenland ################################################")

        if nl.COMB == 'IND':
            # Redefine NormD to loose correlation
            NormD  = np.random.normal(0, 1, N)

        if nl.GRE in ['IPCC', 'KNMI14']:
            #Build the distribution of global temperature for this contributor
            Td_g  = misc.normal_distrib(T_g, nl.GAM, NormD)
            if nl.CorrSMB:
                NormDl = NormDs
            else:
                NormDl = np.random.normal(0, 1, N)
            X_gsmb = gre.fett13(fac, Td_g, NormDl, nl.GRE)
            del(NormDl)
            del(Td_g)

        elif nl.GRE == 'B19':
            #Build the percentiles to follow over time in the distributions
            #can be used to correlate this uncertainty with others.
            UnifP_GIS = np.random.uniform(0, 1, N)

            Td_b  = misc.normal_disctib(T_b, nl.GAM, NormD)
            X_gsmb = b19.Bamber19('GIS', UnifP_GIS, [a1_lo_g, a1_up_g], ys, Td_b)
            X_gsmb = X_gsmb + 0.3    # Contribution between 1995 and 2005 in mm
            del(UnifP_GIS)
            
        elif nl.GRE == 'AR6':
            NormDG  = np.random.normal(0, 1, N)
            X_gsmb = gre.gre_ar6(TIME2, a1_up_g, a1_lo_g, SCE, NormDG)
            
        else:
            print(f'ERROR: GRE option {nl.GRE} is not defined. Check the namelist.')

        for t in range(0, nb_y2):
            X_gsmb[:,t] = X_gsmb[:,t] * F_gsmb2[t]
        
        X_gsmb_perc += np.concatenate( (np.percentile(X_gsmb, Perc, axis=0), 
                                        X_gsmb.mean(axis=0, keepdims=True)), axis=0)

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_gsmb = np.sort(X_gsmb, 0)

        if nl.NoU_G:
            for t in range(0, nb_y2):
                X_tot[:,t] = X_tot[:,t] + X_gsmb[:,t].mean()
        else:
            X_tot = X_tot + X_gsmb
            X_gre_tot = X_gsmb

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_gsmb[:,ind_d]
            comp = comp + 1
            
        del(X_gsmb)
    
        ###############################################################################
        if nl.INFO:
            print("### SMB Antarctica ###############################################")
            
        if nl.COMB == 'IND':
            # Redefine NormD to loose correlation
            NormD  = np.random.normal(0, 1, N)

        if nl.ANT_DYN in ['IPCC', 'KNMI14', 'KNMI16', 'LEV14', 'LEV20', 'SROCC', 'VDL23_AA', 'VDL23_AS']:
            # Build the distribution of global temperature for this contributor
            Td_a = misc.normal_distrib(T_a, nl.GAM, NormD)

            if nl.CorrSMB:
                NormDl = NormDs
            else:
                NormDl = np.random.normal(0, 1, N)

            X_asmb = ant.ant_smb_ar5(NormDl, fac, Td_a)

            del(Td_a)
            del(NormDl)

        elif nl.ANT_DYN in ['DC16', 'DC16T', 'B19', 'KS21', 'KNMI23']:
            # In these cases the SMB is included in the dynamics
            X_asmb = np.zeros([N,nb_y2])

        for t in range(0, nb_y2):
            X_asmb[:,t] = X_asmb[:,t] * F_asmb2[t]
        
        X_asmb_perc += np.concatenate( (np.percentile(X_asmb, Perc, axis=0), 
                                       X_asmb.mean(axis=0, keepdims=True)), axis=0)
    
        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_asmb = np.sort(X_asmb, 0)

        X_ant_tot  = np.zeros([N,nb_y2])
        if nl.NoU_A:
            for t in range(0, nb_y2): # Loop on the period
                X_tot[:,t] = X_tot[:,t] + X_asmb[:,t].mean()
                X_ant_tot[:,t] = X_asmb[:,t].mean()
        else:
            X_tot     = X_tot + X_asmb
            X_ant_tot = X_asmb

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_asmb[:,ind_d]
            comp = comp + 1
        
        ###############################################################################
        if nl.INFO:
            print("### Landwater changes ############################################")

        if nl.LWS == 'AR5':
            X_landw = misc.landw_ar5(ys, TIME2, N)
            
        elif nl.LWS == 'AR6':
            UnifLWS = np.random.uniform(0, 1, N)
            # Start with the same AR5 rates from 2006
            # Also same assumption as AR5 that this process does not contribute
            # before 2006.
            X_landw = misc.proj2order(TIME2, 0.049, 0.026, 4, 1, UnifLWS)

        for t in range(0,nb_y2):
            X_landw[:,t] = X_landw[:,t]*F_gw2[t]
        
        X_landw_perc += np.concatenate( (np.percentile(X_landw, Perc, axis=0), 
                                       X_landw.mean(axis=0, keepdims=True)), axis=0)

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_landw = np.sort(X_landw, 0)

        X_tot = X_tot + X_landw
    
        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_landw[:,ind_d]
            comp = comp + 1
    
        del(X_landw)
        
        ###############################################################################
        if nl.INFO:
            print("### Antarctic dynamics ###########################################")

        if nl.COMB == 'IND':
            # Redefine NormD to loose correlation
            NormD  = np.random.normal(0, 1, N)

        if nl.ANT_DYN == 'IPCC':
            Unif_AA   = np.random.uniform(0, 1, N)
            #### 2nd order projection starting from observations and ending 
            # between -2 and 18.5 cm
            X_ant = misc.proj2order(TIME2, a1_up_a, a1_lo_a, 18.5, -2, Unif_AA)

            #### This is the influence of increased SMB on an increase in Antactic 
            #dynamics
            if nl.COMB == 'IND':
                irg = np.random.permutation(N)
                for t in range(0, nb_y2):
                    X_ant[:,t] = X_ant[:,t] - Unif_AA * 0.35 * X_asmb[irg,t]
            else:
                for t in range(0,nb_y2):
                    X_ant[:,t] = X_ant[:,t] - Unif_AA * 0.35 * X_asmb[:,t]

        elif nl.ANT_DYN == 'KNMI14':
            X_ant = ant.ant_dyn_knmi14(SCE, a1_up_a, a1_lo_a, ys, ye, TIME2, N)
            
        elif nl.ANT_DYN == 'KNMI16':
            print('ERROR : ANT_DYN option '+ ANT_DYN + ' not yet implemented')
        elif nl.ANT_DYN == 'DC16':
            print('ERROR : ANT_DYN option '+ ANT_DYN + ' not yet implemented')
        elif nl.ANT_DYN == 'DC16T':
            print('ERROR : ANT_DYN option '+ ANT_DYN + ' not yet implemented')
            
        elif nl.ANT_DYN == 'LEV14':
            UnifDd = np.random.uniform(0, 1, N)
            X_ant  = ant.ant_dyn_larmip(SCE, ys, ye, nl.GAM, NormD, UnifDd, ROOT, 
                                        nl.TEMP, 'LARMIP', True, nl.LowPass)
        elif nl.ANT_DYN == 'LEV20':
            UnifDd = np.random.uniform(0, 1, N)
            X_ant  = ant.ant_dyn_larmip(SCE, ys, ye, nl.GAM, NormD, UnifDd, ROOT, 
                                        nl.TEMP, 'LARMIP2', True, nl.LowPass)
            
        elif nl.ANT_DYN == 'SROCC':
            X_ant = ant.ant_dyn_srocc(SCE, a1_up_a, a1_lo_a, TIME2, N)
            
        elif nl.ANT_DYN == 'B19':
            # Build the percentiles to follow over time in the distributions
            # can be used to correlate this uncertainty with others.
            UnifP_WAIS = np.random.uniform(0, 1, N)
            UnifP_EAIS = np.random.uniform(0, 1, N)

            Td_b  = misc.normal_distrib(T_b, nl.GAM, NormD)
            X_ant_wais = b19.Bamber19('WAIS', UnifP_WAIS, [a1_lo_a, a1_up_a], ys, Td_b)
            X_ant_eais = b19.Bamber19('EAIS', UnifP_EAIS, [0, 0], ys, Td_b)
            X_ant = X_ant_wais + X_ant_eais
            
        elif nl.ANT_DYN in ['KS21', 'KNMI23']:
            # Redefine an independent normal distribution
            NormDA  = np.random.normal(0, 1, N)
            X_ant = ant.ant_ar6(TIME2, a1_up_a, a1_lo_a, SCE, NormDA, nl.ANT_DYN)
            
        elif nl.ANT_DYN == 'VDL23_AS':
            cal_reg = 'AMUN'
            X_ant = ant.ant_dyn_vdl23(ROOT, SCE, N, cal_reg, 'noCAS', 'usual')
        
        elif nl.ANT_DYN == 'VDL23_AA':
            cal_reg = 'SU'
            X_ant = ant.ant_dyn_vdl23(ROOT, SCE, N, cal_reg, False, 'usual')
        
        # Add the conribution from 1995 to 2006
        if nl.ANT_DYN in ['KNMI23']:
            X_ant = X_ant + 0.17 # From Frederikse et al. 2020
        elif nl.ANT_DYN in ['VDL23_AS', 'VDL23_AA']:
            # Reference period already taken care of in ant_dyn_vdl23 function
            X_ant = X_ant
        else:
            X_ant = X_ant + 0.25 # From AR5

        for t in range(0, nb_y2):
            X_ant[:,t] = X_ant[:,t]*F_adyn2[t]
        
        X_ant_perc += np.concatenate( (np.percentile(X_ant, Perc, axis=0), 
                                       X_ant.mean(axis=0, keepdims=True)), axis=0)

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_ant = np.sort(X_ant, 0)

        if nl.NoU_A:
            for t in range(0, nb_y2):
                X_tot[:,t]     = X_tot[:,t] + X_ant[:,t].mean()
                X_ant_tot[:,t] = X_ant_tot[:,t] + X_ant[:,y].mean()
        else:
            X_tot     = X_tot + X_ant
            X_ant_tot = X_ant_tot + X_ant
        
        X_ant_tot_perc += np.concatenate( (np.percentile(X_ant_tot, Perc, axis=0), 
                                       X_ant_tot.mean(axis=0, keepdims=True)), axis=0)
            
        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_ant[:,ind_d]
            comp = comp + 1

        del(X_ant)
        del(X_ant_tot)
        del(X_asmb)
        
        ###############################################################################
        if nl.INFO:
            print("### Greenland dynamics ############################################")

        if nl.GRE in ['B19', 'AR6']:
            # This contribution is included in SMB in this case
            X_gre = np.zeros([N,nb_y2])
        else:
            # First order term (cm/y), equal to half of observations in 2006
            a1_up_gdyn        = 0.5 * a1_up_g
            a1_lo_gdyn        = 0.5 * a1_lo_g

            if not nl.CorrDYN:
                UnifDd = np.random.uniform(0, 1, N)  # Sample a new independent distrib.

            if nl.GRE == 'KNMI14':
                X_gre  = misc.proj2order(TIME2, a1_up_gdyn, a1_lo_gdyn, 7.4, 1.7, \
                                         UnifDd)
            elif nl.GRE == 'IPCC':
                if SCE in ['rcp26', 'rcp45', 'rcp60', 'ssp126', 'ssp245']:
                    Delta_gre_up_2100 = 6.3
                    Delta_gre_lo_2100 = 1.4
                elif SCE in ['rcp85', 'ssp585']:
                    Delta_gre_up_2100 = 8.5
                    Delta_gre_lo_2100 = 2
                X_gre  = misc.proj2order(TIME2, a1_up_gdyn, a1_lo_gdyn, \
                                         Delta_gre_up_2100, Delta_gre_lo_2100, UnifDd)

            del(UnifDd)
            X_gre = X_gre + 0.15  # Add 0.15cm, the contribution from 1995 to 2005

        # Multiply by the fingerprint
        for t in range(0, nb_y2):
            X_gre[:,t] = X_gre[:,t]*F_gdyn2[t]
        
        X_gre_perc += np.concatenate( (np.percentile(X_gre, Perc, axis=0), 
                                       X_gre.mean(axis=0, keepdims=True)), axis=0)

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_gre = np.sort(X_gre, 0)

        if nl.NoU_G:
            for t in range(0,nb_y2):
                X_tot[:,t] = X_tot[:,t] + X_gre[:,t].mean()
        else:
            X_tot = X_tot + X_gre
            X_gre_tot = X_gre_tot + X_gre
        
        X_gre_tot_perc += np.concatenate( (np.percentile(X_gre_tot, Perc, axis=0), 
                                           X_gre_tot.mean(axis=0, keepdims=True)), axis=0)

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_gre[:,ind_d]
            comp = comp + 1
            
        del(X_gre)
        del(X_gre_tot)
        
        ########################################################################
        if nl.INFO:
            print("### Glacial Isostatic Adjustment ##########################") 
        
        #!!! This process only works for the Dutch coast !!!
        # Total subisdence from Deltares: 0.045
        # GIA from ICE6G for 6 tide gauges: 0.037
        
        gia = (TIME2-1995) * 0.037 
        X_gia = np.repeat(gia[np.newaxis, :], repeats=N, axis=0)
                
        X_gia_perc += np.concatenate( (np.percentile(X_gia, Perc, axis=0), 
                                       X_gia.mean(axis=0, keepdims=True)), axis=0)
        X_tot = X_tot + X_gia
        
        del(X_gia)
        
        ########################################################################
        if nl.INFO:
            print("### Compute PDF of total SLR  #############################")

        # Tot = Thermal exp. and ocean dyn. + Glaciers and ice sheets + Greenland SMB +
        #       Antractic SMB + land water + antarctic dynamic + greenland dynamics + glacial isostatic adjustment
        
        X_tot_perc += np.concatenate( (np.percentile(X_tot, Perc, axis=0), 
                                       X_tot.mean(axis=0, keepdims=True)), axis=0)

        X_tot_perc_i = X_tot_perc/nb_it        

        # Compute the correlations (X_all: nb_comp,nb_SCE,N,nb_yd)
        # (M_Corr_x: nb_SCE,nb_yd,nb_el)
        # Could compute a matrix?
        if nl.Corr:
            print('Computing correlations')
            for t in range(0, nb_yd):
                pl = 0
                for i in range(0, nb_comp):
                    for j in range(i+1,nb_comp):
                        M_Corr_P[t,pl] = M_Corr_P[t,pl] + \
                        stats.pearsonr(X_all[i,:,t], X_all[j,:,t])
                        M_Corr_S[t,pl] = M_Corr_S[t,pl] + \
                        stats.spearmanr(X_all[i,:,t], X_all[j,:,t])
                        pl = pl+1

        if nl.Decomp:
            print('Decomp is on')
            accuracy = 0.1
            print(f'accuracy is {accuracy} cm')
            X_tot_sel = X_tot[:,ind_d]
            X_tot_perc_i_sel = X_tot_perc_i[:,ind_d]
            print(X_tot_sel.shape)

            for t in range(nb_yd):
                for perc in range(nb_perc):                    
                    ind_perc  = np.where( 
                        (X_tot_sel[:,t] > (X_tot_perc_i_sel[perc,t]-accuracy/2)) & 
                        (X_tot_sel[:,t] <= (X_tot_perc_i_sel[perc,t]+accuracy/2)) )[0]

                    if len(ind_perc) > 1:
                        X_Decomp[:,perc,t] = (X_Decomp[:,perc,t] + 
                                              X_all[1:,ind_perc,t].mean(axis=1))
                    else:
                        print('WARNING: X_Decomp does not have enough samples')

        # Check the convergence
        print('All precentiles:')
        print('Perc:')
        print(X_tot_perc_i[:-1,-1])
        
        CONV.append(X_tot_perc_i[-2,-1])
        print(f'{Perc[-1]} percentile: ' + str(CONV[-1]))
        del(X_tot_perc_i)

        if len(CONV) >= 4:
            dc1 = abs(CONV[-1]-CONV[-2])
            dc2 = abs(CONV[-2]-CONV[-3])
            dc3 = abs(CONV[-3]-CONV[-4])
            if (dc1 <= er) and (dc2 <= er) and (dc3 <= er) and (MIN_IT <= nb_it):
                END = True


        del(X_tot)
        print('Finished iteration ' + str(nb_it))
    
        #END = True # Just for testing
        ##### End of main loop
    
    # Scale percentiles and correlation matrices with number of iterations:
    X_O_G_perc     = X_O_G_perc/nb_it
    X_O_A_perc     = X_O_A_perc/nb_it
    X_B_perc       = X_B_perc/nb_it
    X_gic_perc     = X_gic_perc/nb_it
    X_gsmb_perc    = X_gsmb_perc/nb_it
    X_asmb_perc    = X_asmb_perc/nb_it
    X_landw_perc   = X_landw_perc/nb_it
    X_ant_perc     = X_ant_perc/nb_it
    X_gre_perc     = X_gre_perc/nb_it
    X_ant_tot_perc = X_ant_tot_perc/nb_it
    X_gre_tot_perc = X_gre_tot_perc/nb_it
    X_gia_perc     = X_gia_perc/nb_it
    X_tot_perc     = X_tot_perc/nb_it
    
    if nl.Corr:
        M_Corr_P      = M_Corr_P/nb_it
        M_Corr_S      = M_Corr_S/nb_it
        
    if nl.Decomp:
        X_Decomp      = X_Decomp/nb_it

    print('### Numbers for the total distribution ###')
    print(f'### Scenario {SCE} ###')
    print(X_tot_perc[:,-1])
    
    if nl.Corr:
        print('### Spearman correlations ###')
        print(M_Corr_S[-1,:])
    
    ############################################################################
    print("### Export data to a NetCDF file ##################################")

    perc_ar = np.array([X_O_G_perc, X_O_A_perc, X_B_perc, X_gic_perc, X_gsmb_perc, 
                        X_asmb_perc, X_landw_perc, X_ant_perc, X_gre_perc, 
                        X_gre_tot_perc, X_ant_tot_perc, X_gia_perc, X_tot_perc])
    
    # Add mean to list of percentiles
    Perc.append('mean')
    
    # Store data into a DataArray and add 0.5 to the years
    perc_da = xr.DataArray(perc_ar, 
                           coords=[ProcessNames, Perc, TIME2+0.5],
                           dims=['proc', 'percentiles', 'time'])
    
    if nl.InterpBack:
        #Assume 0 values in 1995.5, the middle of the reference period and 
        #interpolate linearly between 1995.5 and 2006.5'''
    
        sel_slice = perc_da.isel(time=0).copy()
        sel_slice.values = np.zeros(sel_slice.shape)
        sel_slice['time'] = 1995.5
    
        perc_da_ext = xr.concat([sel_slice, perc_da], dim='time')

        new_time = np.arange(perc_da_ext.time[0].values.item(),
                             perc_da_ext.time[-1].values.item()+1)

        perc_da = perc_da_ext.interp(time=new_time, method="linear")
    
    perc_da.attrs['units'] = 'cm'
    perc_da.attrs['long_name'] = 'Time series of percentiles.'
    
    if nl.Corr:
        print('Output for option np.Corr = '+ str(nl.Corr) +' is not supported yet')
    
    OUT_ds = xr.Dataset({'perc_ts' : perc_da})
    
    if nl.Corr:
        print('Not yet implemented')
        # Write PearsonCor and SpearmanCor out (MAT_OUTc2, MAT_OUTc2)
        # proc2 coordinate name
        
    if nl.Decomp:
        X_Decomp_da = xr.DataArray(X_Decomp, 
                                   coords=[NameComponents[1:], Perc, TIME2[ind_d]], 
                                   dims=['proc_s', 'percentiles', 'time_s'])
        X_Decomp_da.attrs['long_name'] = ('Provides the average decomposition '+
                                          'of total sea level into its individual '+ 
                                          'components')
        OUT_ds['decomp'] = X_Decomp_da
    
    OUT_ds.attrs['options'] = (
    "Computations were done with the following options:: " +
    f"Local computations? {nl.LOC}" +
    f", include Inverse Barometer effect: {nl.IBarE}" +
    f", GMST option: {nl.TEMP}" +
    f", Greenland SMB and dynamics is: {nl.GRE}" +
    f", Ocean dynamics is: {nl.ODYN}" +
    f", Antarctic dynamics is: {nl.ANT_DYN}" +
    f", Gamma is: {nl.GAM}" +
    f", combination of processes: {nl.COMB}" +
    f", save all samples: {nl.SaveAllSamples}" +
    f", compute correlation between processes: {nl.Corr}" +
    f", correlation between GMST and thermal expansion is: {nl.CorrGT}" +
    f", measure of correlation between GMST and thermal expansion is: {nl.CorrM}" +
    f", correlation between surface mass balances: {nl.CorrSMB}" +
    f", correlation between ice sheet dynamics: {nl.CorrDYN}" +
    f", remove ocean dynamics uncertainty: {nl.NoU_O}" +
    f", remove greenland uncertainty: {nl.NoU_G}" +
    f", remove Antarctic uncertainty: {nl.NoU_A}" +
    f", remove glaciers and ice caps uncertainty: {nl.NoU_Gl}" +
    f", decompose the total sea level into its contributors: {nl.Decomp}" +
    f", filter stero-dynamics and GMST with a polynomial fit: {nl.LowPass}" +
    f", interpolate backward to 1995: {nl.InterpBack}")
    
    OUT_ds.attrs['source_file'] = 'This NetCDF file was built from the ' + \
    'Probabilistic Sea Level Projection code version ' + str(VER)
    OUT_ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    NameOutput= DIR_OUT + 'SeaLevelPerc_' + namelist_name + '_' + SCE + '.nc'
    
    if os.path.isfile(NameOutput):
        os.remove(NameOutput)
    OUT_ds.to_netcdf(NameOutput) #mode='a' to append or overwrite
