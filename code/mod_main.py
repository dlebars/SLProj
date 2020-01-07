import numpy as np
import pandas as pd
from scipy.stats import norm
import glob
import importlib
#import sys
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
    
    nl = importlib.import_module(namelist_name)

    ROOT = '/Users/dewi/Work/Project_ProbSLR/Data_Proj/'
    DIR_T = ROOT+'Data_AR5/Tglobal/'
    DIR_IPCC = ROOT+'Data_AR5/Final_Projections/'
    
    ProcessNames = ['Global steric', 'Local ocean', 'Inverse barometer', 'Glaciers',    \
                 'Greenland SMB', 'Antarctic SMB', 'Landwater', 'Antarctic dynamics',\
                 'Greenland dynamics', 'sum anta.', 'Total']
    
    if nl.SaveAllSamples:
        if nl.LOC:  # Number of components, used to compute correlations efficently.
            NameComponents = ["Glob. temp.", "Glob. thermal exp.", "Local ocean", \
                              "Barometer effect", "Glaciers", "Green. SMB", \
                              "Ant. SMB", "Land water", "Ant. dyn.", "Green dyn."]
            print("!!! Warning: This combination of SaveAllSamples = "+nl.SaveAllSamples+ \
                  " and LOC:"+nl.LOC+" hasn't been tested")
        else:
            NameComponents = ["Glob. temp.", "Thermal exp.", "Glaciers", "Green. SMB", \
                              "Ant. SMB", "Land water", "Ant. dyn.", "Green dyn."]
        nb_comp = len(NameComponents)
    
    #List of model names
    MOD = ["ACCESS1-0","BCC-CSM1-1","CanESM2","CNRM-CM5","CSIRO-Mk3-6-0","GFDL-ESM2G", \
        "GFDL-ESM2M","GISS-E2-R","HadGEM2-CC","HadGEM2-ES","inmcm4","IPSL-CM5A-LR", \
        "IPSL-CM5A-MR","MIROC5","MIROC-ESM-CHEM","MIROC-ESM","MPI-ESM-LR","MPI-ESM-MR", \
        "MRI-CGCM3","NorESM1-ME","NorESM1-M"]
    nb_MOD_AR5 = len(MOD)
    
    #For KNMI files the SSH has a different name for each scenario
    if SCE  == 'rcp45':
        SSH_VAR  = 'ZOSH45'
    elif SCE == 'rcp85':
        SSH_VAR  = "ZOS85"
    
    ### General parameters
    start_date = 1980    # Start reading data
    ys = 2006   # Starting point for the integration, if this is changed problems in functions
    ye = 2100   # End year for computation

    nb_y = ye-start_date+1       # Period where data needs to be read
    nb_y2 = ye - ys +1           # Period of integration of the model

    Aoc = 3.6704e14              # Ocean Area (m2)
    rho_w = 1e3                  # Water density (kg.m-3)
    fac = -1e12 / (Aoc * rho_w)  # Convert Giga tones to m sea level
    alpha_95 = norm.ppf(0.95)
    alpha_05 = norm.ppf(0.05)

    #### Specific parameters
    ## Initial Antarctic dynamics contribution
    a1_up_a           = 0.061    # Unit is cm/y, equal to observations in 2006
    a1_lo_a           = 0.021
    ## Initial Greenland dynamics contribution
    a1_up_g           = 0.076    # Unit is cm/y, equal to observations in 2006
    a1_lo_g           = 0.043
    
    #### Reference time period to take the reference global temperature
    # Reference period for Glaciers and Ice Caps
    ysr_gic  = 1986          # Start of the reference period
    yer_gic  = 2005          # End of the reference period
    # Reference period for Greenland SMB
    ysr_g  = 1980          # Start of the reference period
    yer_g  = 1999          # End of the reference period
    # Reference period for Antarctic SMB
    ysr_a  = 1985          # Start of the reference period
    yer_a  = 2005          # End of the reference period
    # Reference period for Antarctic Dynamics, DC16T option
    ysr_ad = 2000          # Start of the reference period
    yer_ad = 2000          # End of the reference period
    # Reference period for Bamber et al. 2019
    yr_b = 2000            # Only one year
    
    #### Parameters to produce PDF
    bin_min = -20.5
    bin_max = 500.5
    nbin = int(bin_max - bin_min)
    
    ####
    TIME       = np.arange( start_date, ye + 1 )
    TIME2      = np.arange( ys, ye + 1, 1 )
    ind_d      = np.where(TIME2 % 10 == 0)[0] # Select the indices of 2010, 2020... 2100
    nb_yd      = len(ind_d)
    
    i_ys       = np.where(TIME == ys)[0][0]
    i_ye       = np.where(TIME == ye)[0][0]
    i_ysr_gic  = np.where(TIME == ysr_gic)[0][0]
    i_yer_gic  = np.where(TIME == yer_gic)[0][0]
    i_ysr_g    = np.where(TIME == ysr_g)[0][0]
    i_yer_g    = np.where(TIME == yer_g)[0][0]
    i_ysr_a    = np.where(TIME == ysr_a)[0][0]
    i_yer_a    = np.where(TIME == yer_a)[0][0]
    i_ysr_ad   = np.where(TIME == ysr_ad)[0][0]
    i_yer_ad   = np.where(TIME == yer_ad)[0][0]
    i_yr_b     = np.where(TIME == yr_b)[0][0]
    
    #### Read finger prints, some are time dependent so make all of them  time 
    # dependent for easy mutliplication at the end.
    
    F_gic2  = np.ones(nb_y2)
    F_gsmb2 = np.ones(nb_y2)
    F_asmb2 = np.ones(nb_y2)
    F_gdyn2 = np.ones(nb_y2)
    F_adyn2 = np.ones(nb_y2)
    F_gw2   = np.ones(nb_y2)
    
    if nl.LOC:
        # [Add fingerprints here later XXXX]
        print('Option not supported yet')
        
    ###############################################################################
    if nl.INFO:
        print("### Read Tglob             #################")
    
    if nl.TEMPf == 'all':
        path = DIR_T+'global_tas_Amon_*_'+SCE+'_r1i1p1.dat'
        print(path)
        files     = glob.glob(path)
    elif nl.TEMPf == 'AR5':
        files     = []
        for m in range(0, nb_MOD_AR5-1):
            if MOD[m] == 'BCC-CSM1-1':
                loc_mod = "bcc-csm1-1"
            else:
                loc_mod = MOD[m]
            path = DIR_T+'global_tas_Amon_'+loc_mod+'_'+SCE+'_r1i1p1.dat'
            file_sel = glob.glob(path)
            if file_sel: # Make sure the lits is not empty
                files.append(file_sel[0])
    else:
        print('Option TEMPf: ' + nl.TEMPf + ' is not supported')

    nb_MOD    = len(files)
    
    if nl.INFO:
        print('Number of models used for scenario '+ SCE + ' : ' + str(nb_MOD))
        print('Models path: ')
        print("\n".join(files))
    
    #Read Tglob
    TGLOB    = np.zeros([nb_MOD,nb_y])
    Tref_gic = np.zeros(nb_MOD)   # Each process needs a different time anomaly
    Tref_g   = np.zeros(nb_MOD)
    Tref_a   = np.zeros(nb_MOD)
    Tref_ad  = np.zeros(nb_MOD)   # Antarctic dynamics for DC16T option
    Tref_b   = np.zeros(nb_MOD)   # Reference for Bamber et al. 2019 option

    col_names = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', \
                 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(0,nb_MOD):
        TEMP     = pd.read_csv(files[m], comment='#', delim_whitespace=True, names=col_names)
        time     = TEMP['Year'][:]
        dim_t    = len(time)
        i_start  = np.where(time == start_date)[0][0]
        i_end    = np.where(time == ye)[0][0]
        TGLOB[m, :i_end + 1 - i_start] = TEMP.iloc[i_start:i_end+1, 1:].mean(axis=1)   #Data in degree Kelvin
        Tref_gic[m] = TGLOB[m,i_ysr_gic:i_yer_gic+1].mean()
        Tref_g[m]   = TGLOB[m,i_ysr_g:i_yer_g+1].mean()
        Tref_a[m]   = TGLOB[m,i_ysr_a:i_yer_a+1].mean()
        Tref_ad[m]  = TGLOB[m,i_ysr_ad:i_yer_ad+1].mean()
        Tref_b[m]   = TGLOB[m,i_yr_b].mean()
        #### Issue of missing temperature value for rcp26 after 2100 for this scenario
        # it is ok to assume it is constant after 2100
        if (SCE == 'rcp26') and (ye > 2100):
            i2100 = np.where(time == 2099)
            print(i2100)
            TGLOB[m,i2100-i_start : ] = TGLOB[m,i2100-i_start]
        del(TEMP)
        del(time)
    del(files)

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
            # NormD is used in the correlation with temperature because all of the temperature
            # distributions (different for each process) are based on a linear combination of it. 
            ar1 = np.ones(nb_yd)
            NormDc = NormD[:, np.newaxis] * ar1
            X_all[0, :, :] = NormDc

        #######################################################################
        if nl.INFO:
            print("### Thermal expansion and ocean dynamics #################")

        CorrGT = nl.CorrGT
        if nl.COMB == 'IND':
            CorrGT   = 0 # Force a 0 correlation, even if the CorrGT coefficient has another value
        elif nl.COMB == 'DEP':
            CorrGT = 1

        NormDT1 = np.random.normal(0, 1, N)
        # Build NormDT as a combination of NormD (the distribution of GMST) and an independent
        # normal distribution.
        if nl.CorrM == 'Pearson':
            rhoP  = CorrGT
        elif nl.CorrM == 'Spearman':
        # Convert correlation coefficient from Spearman to Pearson
            rhoP  = 2 * np.sin( np.pi / 6 * CorrGT)

        NormDT = NormD*rhoP + NormDT1*np.sqrt(1 - rhoP**2)

        if nl.LOC:
            if nl.ODYN == 'KNMI':
                X_Of = odyn.odyn_loc(SCE, MOD, nb_y, nb_y2, DIR_O, lat_N, lat_S, lon_W, \
                      lon_E, start_date, ye, SSH_VAR, N, i_ys, nl.GAM, NormDT)
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
            np.histogram(X_Of[1,:,t], bins=nbin, range=(bin_min, bin_max), density=True)[0]
            X_O_A_pdf[t,:] = X_O_A_pdf[t,:] + \
            np.histogram(X_Of[2,:,t], bins=nbin, range=(bin_min, bin_max), density=True)[0]

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
                #!!! swapaxes or squeeze are work arround a peculiar Numpy behaviour, see Tests.ipynb
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
            np.histogram(X_gsmb[:,t], bins=nbin, range=(bin_min, bin_max), density=True)[0]

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

        if nl.ANT_DYN in ['IPCC', 'KNMI16', 'LEV14', 'SROCC']:
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
            np.histogram(X_asmb[:,t], bins=nbin, range=(bin_min, bin_max), density=True)[0]
    
        # Update X_tot, the sum of all contributions

        if nl.COMB == 'DEP':
            # Reorder contribution in ascending order
            X_asmb = np.sort(X_asmb, 1)

        X_ant_tot  = np.zeros([N,nb_y2])
        if nl.NoU_A:
            for t in range(0, nb_y2): # Loop on the period
                X_tot[:,t] = X_tot[:,t] + X_asmb[:,t].mean()
                X_ant_tot[:,t] = X_ant_tot[:,t] + X_asmb[:,t].mean()

        else:
            X_tot     = X_tot + X_asmb
            X_ant_tot = X_ant_tot + X_asmb

        if nl.SaveAllSamples:
            X_all[comp,:,:] = X_asmb[:,ind_d]
            comp            = comp + 1

        #delete(X_asmb)
        
        
        
        END = True
    return




