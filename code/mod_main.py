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
    
    nl = importlib.import_module('namelist_'+namelist_name)

    ROOT = '/Users/dewilebars/Projects/Project_ProbSLR/Data_Proj/'
    DIR_T = ROOT+'Data_AR5/Tglobal/'
    DIR_IPCC = ROOT+'Data_AR5/Final_Projections/'
    DIR_OUT = '../outputs/'       # Output directory
    
    if nl.LOC:
        DIR_F       = ROOT+'Data_AR5/Fingerprints/'
        DIR_IBE     = ROOT+'Data_AR5/InvertedBarometer/1x1_reg/'
        DIR_IBEmean = ROOT+'Data_AR5/InvertedBarometer/globalmeans_from_1x1_glob/'

        # Box of interest, used for local computations:
        lat_N = 60    # Original box from Hylke: 51-60N, -3.5,7.5E
        lat_S = 51
        lon_W = -3.5
        lon_E = 7.5

        # Point of interest (Netherlands), used for local computations:
        lat_Neth = 53
        lon_Neth = 5
        if nl.ODYN == 'KNMI':
            DIR_O       = ROOT + 'Data_AR5/Ocean/1x1_reg/'
        elif nl.ODYN == 'IPCC':
            sys.exit('ERROR: This option of ocean dynamics can only be used for' + \
                     ' global computations')

    else:
        if nl.ODYN == 'KNMI':
            DIR_OG      = ROOT + 'Data_AR5/Ocean/globalmeans_from_1x1_glob/'

    
    ProcessNames = ['Global steric', 'Local ocean', 'Inverse barometer', 'Glaciers', \
                 'Greenland SMB', 'Antarctic SMB', 'Landwater', 'Antarctic dynamics',\
                 'Greenland dynamics', 'sum anta.', 'Total']
    # Percentiles to print at run time
    Perc  = (1,5,10,17,20,50,80,83,90,95,99,99.5,99.9)
    
    if nl.SaveAllSamples:
        if nl.LOC:  # Number of components, used to compute correlations efficently.
            NameComponents = ["Glob. temp.", "Glob. thermal exp.", "Local ocean", \
                              "Barometer effect", "Glaciers", "Green. SMB", \
                              "Ant. SMB", "Land water", "Ant. dyn.", "Green dyn."]
            print("!!! Warning: This combination of SaveAllSamples = "+str(nl.SaveAllSamples)+ \
                  " and LOC:"+str(nl.LOC)+" hasn't been tested")
        else:
            NameComponents = ["Glob. temp.", "Thermal exp.", "Glaciers", "Green. SMB", \
                              "Ant. SMB", "Land water", "Ant. dyn.", "Green dyn."]
        nb_comp = len(NameComponents)
    
    #List of model names
    MOD = ["ACCESS1-0","BCC-CSM1-1","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G", \
        "GFDL-ESM2M","GISS-E2-R","HadGEM2-CC","HadGEM2-ES","inmcm4","IPSL-CM5A-LR", \
        "IPSL-CM5A-MR","MIROC5","MIROC-ESM-CHEM","MIROC-ESM","MPI-ESM-LR","MPI-ESM-MR",\
        "MRI-CGCM3","NorESM1-ME","NorESM1-M"]
    nb_MOD_AR5 = len(MOD)
    
    #For KNMI files the SSH has a different name for each scenario
    if SCE  == 'rcp45':
        SSH_VAR  = 'ZOSH45'
    elif SCE == 'rcp85':
        SSH_VAR  = "ZOS85"
    
    ### General parameters
    start_date = 1980    # Start reading data
    ys = 2006   # Starting point for the integration, if this is changed 
                # then expect problems in functions
    ye = 2100   # End year for computation

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
    
    #### Parameters to produce PDF
    ACCURACY = 'MM'
    if ACCURACY == 'CM':
        bin_min = -20.5
        bin_max = 500.5
        bin_centers = np.arange(bin_min + 0.5, bin_max - 0.5 + 1, 1)
    elif ACCURACY == 'MM':
        bin_min = -20.05
        bin_max = 500.05
        bin_centers = np.arange(bin_min + 0.05, bin_max - 0.05 + 0.1, 0.1)
    nbin = len(bin_centers)
    
    ####
    TIME       = np.arange( start_date, ye + 1 )
    TIME2      = np.arange( ys, ye + 1, 1 )
    ind_d      = np.where(TIME2 % 10 == 0)[0] # Select the indices of 2010, 2020... 2100
    nb_yd      = len(ind_d)
    
    #### Read finger prints, some are time dependent so make all of them  time 
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
        f_gic      = xr.open_dataset(DIR_F+'Relative_GLACIERS_reg.nc', \
                                     decode_times=False)
        F_gic      = misc.finger1D(lat_Neth, lon_Neth, f_gic.latitude, f_gic.longitude, 
                              f_gic.RSL)
        F_gic2[jfs:jfe+1] =  F_gic[ifs:ife+1]/100 # Convert from % to fraction

        del(f_gic, TIME_F)

        f_ic        = xr.open_dataset(DIR_F+'Relative_icesheets_reg.nc')
        lat_ic      = f_ic.latitude #tofloat?
        lon_ic      = f_ic.longitude
        F_gsmb      = misc.finger1D(lat_Neth, lon_Neth, lat_ic, lon_ic, f_ic.SMB_GRE)
        F_gsmb2[jfs:jfe+1]  =  F_gsmb/100
        F_asmb      = misc.finger1D(lat_Neth, lon_Neth, lat_ic, lon_ic, f_ic.SMB_ANT)
        F_asmb2[jfs:jfe+1]  =  F_asmb/100
        F_gdyn      = misc.finger1D(lat_Neth, lon_Neth, lat_ic, lon_ic, f_ic.DYN_GRE)
        F_gdyn2[jfs:jfe+1]  =  F_gdyn/100
        F_adyn      = misc.finger1D(lat_Neth, lon_Neth, lat_ic, lon_ic, f_ic.DYN_ANT)
        F_adyn2[jfs:jfe+1]  =  F_adyn/100

        del(f_ic, lat_ic, lon_ic)

        f_gw       = xr.open_dataset(DIR_F+'Relative_GROUNDWATER_reg.nc', \
                                     decode_times=False)
        lat_gw     = f_gw.latitude #tofloat?
        lon_gw     = f_gw.longitude
        finger_gw  = f_gw.GROUND
        #finger_gw@_FillValue = 0
        F_gw       = misc.finger1D(lat_Neth, lon_Neth, lat_gw, lon_gw, finger_gw)
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
        
    ###############################################################################
    if nl.INFO:
        print("### Read Tglob             #################")
    
    if nl.TEMPf == 'all':
        path = DIR_T+'global_tas_Amon_*_'+SCE+'_r1i1p1.dat'
        if nl.INFO:
            print(path)
        files     = glob.glob(path)
    elif nl.TEMPf == 'AR5':
        files = misc.temp_path_AR5(MOD, DIR_T, SCE)
    else:
        print('Option TEMPf: ' + nl.TEMPf + ' is not supported')
    
    TGLOB = misc.tglob_cmip5(nl.INFO, files, SCE, nb_y, start_date, ye)
    del(files)

    # Read TGLOB and compute reference temperaure for each model
    # The first two numbers are the beginning and end of the reference time period
    # to take the reference global temperature
    Tref_gic = misc.Tref(1986, 2005, TGLOB, TIME)   # Glaciers and Ice Caps
    Tref_g   = misc.Tref(1980, 1999, TGLOB, TIME)   # Greenland SMB
    Tref_a   = misc.Tref(1985, 2005, TGLOB, TIME)   # Antarctic SMB
    Tref_ad  = misc.Tref(2000, 2000, TGLOB, TIME)   # Antarctic dynamics for DC16T
    Tref_b   = misc.Tref(2000, 2000, TGLOB, TIME)   # Reference for Bamber et al. 2019
    
    i_ys   = np.where(TIME == ys)[0][0]
    TGLOBs = TGLOB[:,i_ys:]
    del(TGLOB)
    
    ##############################################################################
    # Start loop 
    CONV  = [] # Check for convergence NCL: NewList("lifo")
    nb_it = 0
    END = False
    
    X_O_G_pdf     = np.zeros([nb_y2,nbin])
    X_O_A_pdf     = np.zeros([nb_y2,nbin])
    X_B_pdf       = np.zeros([nb_y2,nbin])
    X_gic_pdf     = np.zeros([nb_y2,nbin])
    X_gsmb_pdf    = np.zeros([nb_y2,nbin])
    X_asmb_pdf    = np.zeros([nb_y2,nbin])
    X_landw_pdf   = np.zeros([nb_y2,nbin])
    X_ant_pdf     = np.zeros([nb_y2,nbin])
    X_gre_pdf     = np.zeros([nb_y2,nbin])
    X_ant_tot_pdf = np.zeros([nb_y2,nbin])
    X_tot_pdf     = np.zeros([nb_y2,nbin])

    if nl.SaveAllSamples:
        X_all       = np.zeros([nb_comp, N, nb_yd])

    if nl.Corr:
        nb_el       = nb_comp*(nb_comp-1)/2
        M_Corr_P    = np.zeros([nb_yd,nb_el])  # Pearson correlation matrice 
        M_Corr_S    = np.zeros([nb_yd,nb_el])  # Rank correlation matrice

    if nl.Decomp:
        X_Decomp = np.zeros([nb_comp-1,nb_yd,nbin])
    
    while not END:
        nb_it = nb_it + 1
        comp  = 1    # Re-initialize the component count for the Corr case

        #### Set seeds for the random number generators
        # Does not need to be necessary in Python -> Check

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

        if nl.LOC:
            if nl.ODYN == 'KNMI':
                X_Of = odyn.odyn_loc(SCE, MOD, nb_y, nb_y2, DIR_O, lat_N, lat_S, \
                                     lon_W, lon_E, start_date, ye, SSH_VAR, N, \
                                     i_ys, nl.GAM, NormDT)
            elif nl.ODYN == 'CMIP5':
                X_Of = odyn.odyn_cmip5(SCE, LOC, DIR_OCMIP5, N, ys, ye, nl.GAM, NormDT)
        else:
            if nl.ODYN == 'KNMI':
                X_Of = odyn.odyn_glob_knmi(SCE, MOD, nb_y, nb_y2, DIR_O, DIR_OG, \
                                      start_date, ye, SSH_VAR, N, i_ys, nl.GAM, NormDT)
            elif nl.ODYN == 'IPCC':
                X_Of = odyn.odyn_glob_ipcc(SCE, DIR_IPCC, N, nb_y2, nl.GAM, NormDT)
            elif ODYN == 'CMIP5':
                X_Of = odyn.odyn_cmip5(SCE, LOC, DIR_OCMIP5, N, ys, ye, nl.GAM, NormDT)

        # Compute the pdfs based on the chosen periods
        for t in range(0, nb_y2):
            X_O_G_pdf[t,:] = X_O_G_pdf[t,:] + \
            np.histogram(X_Of[1,:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]
            X_O_A_pdf[t,:] = X_O_A_pdf[t,:] + \
            np.histogram(X_Of[2,:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]

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
        Td_gic = misc.TempDist(TGLOBs, Tref_gic, nl.GAM, NormD)

        NormDs  = np.random.normal(0, 1, N)   # This distribution is then kept for correlation
        X_gic = gic.gic_ar5(Td_gic, NormDs)

        for t in range(0,nb_y2): # Use broadcasting?
            X_gic[:,t] = X_gic[:,t]*F_gic2[t]

        # Compute the pdfs based on the chosen periods
        for t in range(0,nb_y2):  # Use broadcasting?
            X_gic_pdf[t,:]  = X_gic_pdf[t,:] + \
            np.histogram(X_gic[:,t], bins=nbin, range=(bin_min, bin_max), density=True)[0]

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
            comp        = comp + 1

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
            Td_g  = misc.TempDist(TGLOBs, Tref_g, nl.GAM, NormD)
            if nl.CorrSMB:
                NormDl = NormDs
            else:
                NormDl = np.random.normal(0, 1, N)
            X_gsmb = gre.fett13(fac, Td_g, NormDl, nl.GRE)
            del(NormDl)
            del(Td_g)

#        elif nl.GRE == 'B19':
            #Build the percentiles to follow over time in the distributions
            #can be used to correlate this uncertainty with others.
#             UnifP_GIS = np.random.uniform(0, 1, N)

#             Td_b  = misc.TempDist(TGLOBs, Tref_b, GAM, NormD)
#             X_gsmb = Bamber19("GIS", UnifP_GIS, (/a1_lo_g, a1_up_g/), ys, Td_b)
#             X_gsmb = X_gsmb + 0.3    # Contribution between 1995 and 2005 in mm
#             del(UnifP_GIS)

        for t in range(0, nb_y2):
            X_gsmb[:,t] = X_gsmb[:,t] * F_gsmb2[t]

        # Compute the pdfs based on the chosen periods
        for t in range(0, nb_y2):
            X_gsmb_pdf[t, :]  = X_gsmb_pdf[t,:] + \
            np.histogram(X_gsmb[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_gsmb = np.sort(X_gsmb, 0)

        if nl.NoU_G:
            for t in range(0, nb_y2):
                X_tot[:,t] = X_tot[:,t] + X_gsmb[:,t].mean()
        else:
            X_tot = X_tot + X_gsmb

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_gsmb[:,ind_d]
            comp            = comp + 1
            
        del(X_gsmb)
    
        ###############################################################################
        if nl.INFO:
            print("### SMB Antarctica ###############################################")
            
        if nl.COMB == 'IND':
            # Redefine NormD to loose correlation
            NormD  = np.random.normal(0, 1, N)

        if nl.ANT_DYN in ['IPCC', 'KNMI14', 'KNMI16', 'LEV14', 'SROCC']:
            #Build the distribution of global temperature for this contributor
            Td_a = misc.TempDist(TGLOBs, Tref_a, nl.GAM, NormD)

            if nl.CorrSMB:
                NormDl = NormDs
            else:
                NormDl = np.random.normal(0, 1, N)

            X_asmb = ant.ant_smb_ar5(NormDl, fac, Td_a)

            del(Td_a)
            del(NormDl)

        elif nl.ANT_DYN in ['DC16', 'DC16T', 'B19']:
            # In these cases the SMB is included in the dynamics
            X_asmb = np.zeros([N,nb_y2])

        for t in range(0, nb_y2):
            X_asmb[:,t] = X_asmb[:,t] * F_asmb2[t]

        # Compute the pdfs for the the chosen periods
        for t in range(0, nb_y2):
            X_asmb_pdf[t,:]  = X_asmb_pdf[t,:] + \
            np.histogram(X_asmb[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]
    
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
            comp            = comp + 1
        
        ###############################################################################
        if nl.INFO:
            print("### Landwater changes ############################################")

        X_landw = misc.landw_ar5(ys, TIME2, N)

        for t in range(0,nb_y2):
            X_landw[:,t] = X_landw[:,t]*F_gw2[t]

        # Compute the pdfs based on the chosen periods
        for t in range(0,nb_y2):        # Loop on the period
            X_landw_pdf[t,:]  = X_landw_pdf[t,:] + \
            np.histogram(X_landw[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_landw = np.sort(X_landw, 0)

        X_tot = X_tot + X_landw
    
        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_landw[:,ind_d]
            comp        = comp + 1
    
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
            print('ERROR : ANT_DYN option '+ ANT_DYN + ' not yet implemented')
        elif nl.ANT_DYN == 'SROCC':
            print('ERROR : ANT_DYN option '+ ANT_DYN + ' not yet implemented')
        elif nl.ANT_DYN == 'B19':
            print('ERROR : ANT_DYN option '+ ANT_DYN + ' not yet implemented')
        
        X_ant = X_ant + 0.25 # Add 0.25cm, the conribution from 1995 to 2005

        for t in range(0, nb_y2):
            X_ant[:,t] = X_ant[:,t]*F_adyn2[t]

        # Compute the pdfs based on user defined periods of time
        for t in range(0, nb_y2):
            X_ant_pdf[t,:]  = X_ant_pdf[t,:] + \
            np.histogram(X_ant[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]

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

          # Compute the pdfs based on user defined periods of time
        for t in range(0, nb_y2):
            X_ant_tot_pdf[t,:]  = X_ant_tot_pdf[t,:] + \
            np.histogram(X_ant_tot[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]
            
        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_ant[:,ind_d]
            comp            = comp + 1

        del(X_ant)
        del(X_ant_tot)
        del(X_asmb)
        
        ###############################################################################
        if nl.INFO:
            print("### Greenland dynamics ############################################")

        if nl.GRE == 'B19':
            # This contribution is included in SMB in this case
            X_gre = np.zeros([N,nb_y2])
            X_gre = 0
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
                if SCE in ['rcp26', 'rcp45']:
                    Delta_gre_up_2100 = 6.3
                    Delta_gre_lo_2100 = 1.4
                elif SCE == 'rcp85':
                    Delta_gre_up_2100 = 8.5
                    Delta_gre_lo_2100 = 2
            X_gre  = misc.proj2order(TIME2, a1_up_gdyn, a1_lo_gdyn, Delta_gre_up_2100, 
                                Delta_gre_lo_2100, UnifDd)

            del(UnifDd)
            X_gre = X_gre + 0.15  # Add 0.15cm, the contribution from 1995 to 2005

        # Multiply by the fingerprint
        for t in range(0, nb_y2):
            X_gre[:,t] = X_gre[:,t]*F_gdyn2[t]

        # Compute the pdfs based on the chosen periods
        for t in range(0, nb_y2):
            X_gre_pdf[t,:]  = X_gre_pdf[t,] + \
            np.histogram(X_gre[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]

        # Update X_tot, the sum of all contributions
        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_gre = np.sort(X_gre, 0)

        if nl.NoU_G:
            for t in range(0,nb_y2):
                X_tot[:,t] = X_tot[:,t] + X_gre[:,t].mean()
        else:
            X_tot = X_tot + X_gre

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_gre[:,ind_d]
            comp        = comp + 1
        del(X_gre)
    
        ########################################################################
        if nl.INFO:
            print("### Compute PDF of total SLR  #############################")

        # Tot = Thermal exp. and ocean dyn. + Glaciers and ice sheets + Greenland SMB +
        #       Antractic SMB + land water + antarctic dynamic + greenland dynamics

        # Compute the pdfs based on the chosen periods
        for t in range(0, nb_y2):
            X_tot_pdf[t,:]  = X_tot_pdf[t,:] + \
            np.histogram(X_tot[:,t], bins=nbin, range=(bin_min, bin_max), \
                         density=True)[0]

        # Check the convergence
        X_tot_pdf_i = X_tot_pdf/nb_it
        PDF_cum     = X_tot_pdf_i[-1,:].cumsum()*100
        indi        = np.abs(PDF_cum - 99.9).argmin()
        CONV.append(bin_centers[indi])
        print('99.9 percentile: ' + str(CONV[-1]))
        del(indi)
        del(PDF_cum)
        del(X_tot_pdf_i)

        if len(CONV) >= 4:
            dc1 = abs(CONV[-1]-CONV[-2])
            dc2 = abs(CONV[-2]-CONV[-3])
            dc3 = abs(CONV[-3]-CONV[-4])
            if (dc1 <= er) and (dc2 <= er) and (dc3 <= er) and (MIN_IT <= nb_it):
                END = True

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

        X_tot_sel = X_tot[:,ind_d]
        if nl.Decomp:
            for t in range(0, nb_yd):
                for bi in (0, nbin):
                    ind_bin  = np.where( (X_tot_sel[:,t] > bin_min+bi) and \
                                        (X_tot_sel[:,t] <= bin_min+bi+1) )
                    if len(ind_bin) > 1:
                        X_Decomp[:,t,bi] =  X_Decomp[:,t,bi] + \
                        X_all[1:,ind_bin,t].mean(axis=1)
                    else:
                        X_Decomp[:,t,bi] = np.NA
                    del(ind_bin)

        del(X_tot)
        print("Finished iteration " + str(nb_it))
    
        #END = True # Just for testing
        ##### End of main loop
    
    # Scale PDFs and correlation matrices with number of iterations:
    X_O_G_pdf     = X_O_G_pdf/nb_it
    X_O_A_pdf     = X_O_A_pdf/nb_it
    X_B_pdf       = X_B_pdf/nb_it
    X_gic_pdf     = X_gic_pdf/nb_it
    X_gsmb_pdf    = X_gsmb_pdf/nb_it
    X_asmb_pdf    = X_asmb_pdf/nb_it
    X_landw_pdf   = X_landw_pdf/nb_it
    X_ant_pdf     = X_ant_pdf/nb_it
    X_gre_pdf     = X_gre_pdf/nb_it
    X_ant_tot_pdf = X_ant_tot_pdf/nb_it
    X_tot_pdf     = X_tot_pdf/nb_it
    if nl.Corr:
        M_Corr_P      = M_Corr_P/nb_it
        M_Corr_S      = M_Corr_S/nb_it
    if nl.Decomp:
        X_Decomp      = X_Decomp/nb_it

    print('### Numbers for the total distribution ###')
    print("### Scenario " + SCE + " ###")
    p     = misc.perc_df(X_tot_pdf[-1,:], Perc, bin_centers)
    print(p)
    
    if nl.Corr:
        print("### Spearman correlations ###")
        print(M_Corr_S[-1,:])
    
    ############################################################################
    # Output the results: Choice between xarray and Netcdf4 libraries
    
    nb_proc = 9+2     # 9 different processes plus total antarctic and total

    MAT_OUTd         = np.zeros([nb_y2, nb_proc, nbin])
    MAT_OUTd[:,0,:]  = X_O_G_pdf
    MAT_OUTd[:,1,:]  = X_O_A_pdf
    MAT_OUTd[:,2,:]  = X_B_pdf
    MAT_OUTd[:,3,:]  = X_gic_pdf
    MAT_OUTd[:,4,:]  = X_gsmb_pdf
    MAT_OUTd[:,5,:]  = X_asmb_pdf
    MAT_OUTd[:,6,:]  = X_landw_pdf
    MAT_OUTd[:,7,:]  = X_ant_pdf
    MAT_OUTd[:,8,:]  = X_gre_pdf
    MAT_OUTd[:,9,:]  = X_ant_tot_pdf
    MAT_OUTd[:,10,:] = X_tot_pdf

    proc_coord = np.arange(nb_proc) # Can we have names here?
    MAT_OUT = xr.DataArray(MAT_OUTd, coords=[TIME2, ProcessNames, bin_centers] , 
                           dims=['time', 'proc', 'bin'])

    MAT_OUT.attrs['units'] = 'cm'
    MAT_OUT.attrs['long_name'] = 'This variable contains the pdfs of each sea level' + \
    'contributor and of the total sea level. The list of processes included here is' + \
    'written in variable ProcessNames'

    if nl.Corr:
        print('Output for option np.Corr = '+ str(nl.Corr) +' is not supported yet')

    print("### Export data to a NetCDF file ##################################")
    
    # Build a DataSet
    MAT_OUT_ds = xr.Dataset({'MAT_RES': MAT_OUT})
    
    MAT_OUT_ds.attrs['options'] = \
    "Computations were done with the following options:: " + \
    "Local computations?" + str(nl.LOC) + \
    ", include Inverse Barometer effect: "+ str(nl.IBarE) + \
    ", GMST option: "+ nl.TEMPf + \
    ", Greenland SMB and dynamics is: "+ nl.GRE + \
    ", Ocean dynamics is: " + nl.ODYN + \
    ", Antarctic dynamics is: " + nl.ANT_DYN + \
    ", Gamma is: " + str(nl.GAM)+ \
    ", combination of processes: " + nl.COMB + \
    ", save all samples: " + str(nl.SaveAllSamples) + \
    ", compute correlation between processes: " + str(nl.Corr) + \
    ", correlation between GMST and thermal expansion is: "+ str(nl.CorrGT) + \
    ", measure of correlation between GMST and thermal expansion is:"+ nl.CorrM + \
    ", correlation between surface mass balances: "+ str(nl.CorrSMB) + \
    ", correlation between ice sheet dynamics: "+ str(nl.CorrDYN) + \
    ", remove ocean dynamics uncertainty: "+ str(nl.NoU_O) + \
    ", remove greenland uncertainty: "+ str(nl.NoU_G) + \
    ", remove Antarctic uncertainty: "+ str(nl.NoU_A) + \
    ", remove glaciers and ice caps uncertainty: "+ str(nl.NoU_Gl) + \
    ", decompose the total sea level into its contributors: "+ str(nl.Decomp)
    
    MAT_OUT_ds.attrs['source_file'] = 'This NetCDF file was built from the ' + \
    'Probabilistic Sea Level Projection code version ' + str(VER)
    
    MAT_OUT_ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    NameOutput= DIR_OUT + 'SeaLevelPDF_' + namelist_name + '_' + SCE + '.nc'
    if os.path.isfile(NameOutput):
        os.remove(NameOutput)
    MAT_OUT_ds.to_netcdf(NameOutput) #mode='a' to append or overwrite
