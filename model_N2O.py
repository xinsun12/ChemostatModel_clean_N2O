# -*- coding: utf-8 -*-

"""
Created in March 2023, last update: June 2024

Purpose
-------
    A 0D chemostat model with microbes in marine OMZs,
    Modular denitrification included, yields of denitrifiers depend on Gibbs free energy.
    Organic matter pulses included
    
@authors: 
    Xin Sun, Emily Zakem, Pearse Buchanan
"""

### imports
import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
             pulse_Sd, pulse_O2, pulse_int, \
             K_o2_aer, K_o2_aoo, K_o2_noo, \
             K_n2o_Den, \
             mumax_Het, mumax_AOO, mumax_NOO, mumax_AOX, \
             VmaxS, K_s, \
             VmaxN_1Den, VmaxN_2Den, VmaxN_3Den, VmaxN_4Den, VmaxN_5Den, VmaxN_6Den, K_n_Den, \
             VmaxN_AOO, K_n_AOO, VmaxN_NOO, K_n_NOO, \
             VmaxNH4_AOX, K_nh4_AOX, VmaxNO2_AOX, K_no2_AOX, \
             y_oHet, y_oO2, \
             y_n1Den, y_n1NO3, y_n2Den, y_n2NO2, y_n3Den, y_n3NO3, y_n4Den, y_n4NO2, y_n5Den, y_n5N2O, y_n6Den, y_n6NO3, y_n7Den_NO3, y_n7NO3, e_n7Den_NO3, y_n7Den_N2O, y_n7N2O, e_n7Den_N2O,\
             y_nAOO, y_oAOO, y_nNOO, y_oNOO, y_nh4AOX, y_no2AOX, \
             e_n2Den, e_n3Den, e_no3AOX, e_n2AOX, e_n4Den, e_n5Den, e_n6Den, e_n1Den, \
             initialOM, initialNO2, in_Sd, in_O2, in_NO3, in_NO2, in_NH4, in_N2, in_N2O, \
             in_bHet, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX, in_b4Den, in_b5Den, in_b6Den, in_b7Den,\
             N2Oammonia):
    
    
    # transfer initial inputs to model variables, intial inputs are defined in the "call_model_N2O_clean" file
    m_Sd = initialOM # concentration of organic matter
    m_O2 = in_O2 # concentration of oxygen
    m_NO3 = in_NO3 # concentration of nitrate
    m_NO2 = initialNO2 # concentration of nitrite
    m_NH4 = in_NH4 # concentration of ammmonia
    m_N2 = in_N2 # concentration of dinitrogen gas
    m_N2O = in_N2O # concentration of nitrous oxide
    m_bHet = in_bHet # biomass of aerobic heterotrophic microbes
    m_b1Den = in_b1Den # biomass of NO3-->NO2 denitrifiers
    m_b2Den = in_b2Den # biomass of NO2-->N2 denitrifiers
    m_b3Den = in_b3Den # biomass of NO3-->N2 (complete) denitrifiers
    m_b4Den = in_b4Den # biomass of NO2-->N2O denitrifiers
    m_b5Den = in_b5Den # biomass of N2O-->N2 denitrifiers
    m_b6Den = in_b6Den # biomass of NO3-->N2O denitrifiers
    m_b7Den = in_b7Den # biomass of "book-end" denitrifiers (NO3-->NO2 and N2O-->N2)
    m_bAOO = in_bAOO # biomass of ammonia-oxidizing organisms
    m_bNOO = in_bNOO # biomass of nitrite-oxidizing organisms
    m_bAOX = in_bAOX # biomass of anaerobic ammonia-oxidizing (anammox) bacteria 
    
    # set the output arrays
    out_Sd = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NO3 = np.ones((int(nn_output)+1)) * np.nan
    out_NO2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_N2 = np.ones((int(nn_output)+1)) * np.nan
    out_N2O = np.ones((int(nn_output)+1)) * np.nan
    out_bHet = np.ones((int(nn_output)+1)) * np.nan
    out_b1Den = np.ones((int(nn_output)+1)) * np.nan
    out_b2Den = np.ones((int(nn_output)+1)) * np.nan
    out_b3Den = np.ones((int(nn_output)+1)) * np.nan
    out_b4Den = np.ones((int(nn_output)+1)) * np.nan
    out_b5Den = np.ones((int(nn_output)+1)) * np.nan
    out_b6Den = np.ones((int(nn_output)+1)) * np.nan
    out_b7Den = np.ones((int(nn_output)+1)) * np.nan
    out_bAOO = np.ones((int(nn_output)+1)) * np.nan
    out_bNOO = np.ones((int(nn_output)+1)) * np.nan
    out_bAOX = np.ones((int(nn_output)+1)) * np.nan
    out_uHet = np.ones((int(nn_output)+1)) * np.nan
    out_u1Den = np.ones((int(nn_output)+1)) * np.nan
    out_u2Den = np.ones((int(nn_output)+1)) * np.nan
    out_u3Den = np.ones((int(nn_output)+1)) * np.nan
    out_u4Den = np.ones((int(nn_output)+1)) * np.nan
    out_u5Den = np.ones((int(nn_output)+1)) * np.nan
    out_u6Den = np.ones((int(nn_output)+1)) * np.nan
    out_u7Den = np.ones((int(nn_output)+1)) * np.nan      
    out_uAOO = np.ones((int(nn_output)+1)) * np.nan
    out_uNOO = np.ones((int(nn_output)+1)) * np.nan
    out_uAOX = np.ones((int(nn_output)+1)) * np.nan
    out_rHet = np.ones((int(nn_output)+1)) * np.nan
    out_rHetAer = np.ones((int(nn_output)+1)) * np.nan
    out_rO2C = np.ones((int(nn_output)+1)) * np.nan
    out_r1Den = np.ones((int(nn_output)+1)) * np.nan
    out_r2Den = np.ones((int(nn_output)+1)) * np.nan
    out_r3Den = np.ones((int(nn_output)+1)) * np.nan
    out_r4Den = np.ones((int(nn_output)+1)) * np.nan
    out_r5Den = np.ones((int(nn_output)+1)) * np.nan
    out_r6Den = np.ones((int(nn_output)+1)) * np.nan     
    out_rAOO = np.ones((int(nn_output)+1)) * np.nan
    out_rNOO = np.ones((int(nn_output)+1)) * np.nan
    out_rAOX = np.ones((int(nn_output)+1)) * np.nan
    out_rN2Oammonia = np.ones((int(nn_output)+1)) * np.nan
    
    # set the array for recording the average activity of microbes 
    interval = int((1/dt * out_at_day))
    
    # record the initial conditions
    i = 0
    out_Sd[i] = m_Sd 
    out_O2[i] = m_O2
    out_NO3[i] = m_NO3
    out_NO2[i] = m_NO2 
    out_NH4[i] = m_NH4 
    out_N2[i] = m_N2
    out_N2O[i] = m_N2O
    out_bHet[i] = m_bHet
    out_b1Den[i] = m_b1Den
    out_b2Den[i] = m_b2Den
    out_b3Den[i] = m_b3Den 
    out_b4Den[i] = m_b4Den
    out_b5Den[i] = m_b5Den
    out_b6Den[i] = m_b6Den
    out_b7Den[i] = m_b7Den
    out_bAOO[i] = m_bAOO
    out_bNOO[i] = m_bNOO
    out_bAOX[i] = m_bAOX
    
    # begin the loop
    for t in np.arange(1,timesteps+1,1):

        # 3.	‘MMµ’: Define Vmax by µmax and yield, like the other Vmax in the model, then add M-M equation to that.
        # Vmax is the maximum nutrient update rate, K is the half saturation constant, and mumax is the maximum growth rate of a certain microbial functional type. 
        # p is the uptake rate of a nutrient calculated from Vmax and K
        # u is the growth rate of a certain microbial functional type calculated from p and biomass yield, y.
        VmaxO2_aer = mumax_Het / y_oO2
        p_O2_aer = VmaxO2_aer * m_O2 / (K_o2_aer + m_O2)

        VmaxO2_AOO = mumax_AOO / y_oAOO  
        p_O2_aoo = VmaxO2_AOO * m_O2 / (K_o2_aoo + m_O2)

        VmaxO2_NOO = mumax_NOO / y_oNOO
        p_O2_noo = VmaxO2_NOO * m_O2 / (K_o2_noo + m_O2)

        p_N2O_den = VmaxN_5Den * m_N2O / (K_n2o_Den + m_N2O)
      
        p_Sd = VmaxS * m_Sd / (K_s + m_Sd)                               # mol Org / day
        p_1DenNO3 = VmaxN_1Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_2DenNO2 = VmaxN_2Den * m_NO2 / (K_n_Den + m_NO2)               # mol NO2 / day
        p_3DenNO3 = VmaxN_3Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_4DenNO2 = VmaxN_4Den * m_NO2 / (K_n_Den + m_NO2)               # mol NO2 / day
        p_6DenNO3 = VmaxN_6Den * m_NO3 / (K_n_Den + m_NO3)               # mol NO3 / day
        p_NH4_AOO = VmaxN_AOO * m_NH4 / (K_n_AOO + m_NH4)   # mol NH4 / day
        p_NO2_NOO = VmaxN_NOO * m_NO2 / (K_n_NOO + m_NO2)   # mol NO2 / day
        p_NH4_AOX = VmaxNH4_AOX * m_NH4 / (K_nh4_AOX + m_NH4)     # mol NH4 / day
        p_NO2_AOX = VmaxNO2_AOX * m_NO2 / (K_no2_AOX + m_NO2)     # mol NO2 / day
        
        
        # growth rates (u) in the unit of d^(-1), determined by the min rate of the substrate (e.g., OM, DIN, O2) uptake
        u_Het = np.fmax(0.0, np.fmin(p_Sd * y_oHet, p_O2_aer * y_oO2))          # mol Org / day * mol Biomass / mol Org || mol O2 / day * mol Biomass / mol O2
        u_1Den = np.fmax(0.0, np.fmin(p_Sd * y_n1Den, p_1DenNO3 * y_n1NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3
        u_2Den = np.fmax(0.0, np.fmin(p_Sd * y_n2Den, p_2DenNO2 * y_n2NO2))         # mol Org / day * mol Biomass / mol Org || mol NO2 / day * mol Biomass / mol NO2
        u_3Den = np.fmax(0.0, np.fmin(p_Sd * y_n3Den, p_3DenNO3 * y_n3NO3))         # mol Org / day * mol Biomass / mol Org || mol NO3 / day * mol Biomass / mol NO3       
        u_4Den = np.fmax(0.0, np.fmin(p_Sd * y_n4Den, p_4DenNO2 * y_n4NO2))         # (NO2-->N2O)
        u_5Den = np.fmax(0.0, np.fmin(p_Sd * y_n5Den, p_N2O_den * y_n5N2O))         # (N2O-->N2)
        u_6Den = np.fmax(0.0, np.fmin(p_Sd * y_n6Den, p_6DenNO3 * y_n6NO3))         # (NO2-->N2O)
        u_7Den = np.fmax(np.fmax(0.0, np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3)), np.fmax(0.0, np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O)))         # (bookend: NO3-->NO2, N2O-->N2)  

        u_AOO = np.fmax(0.0, np.fmin(p_NH4_AOO * y_nAOO, p_O2_aoo * y_oAOO))    # mol NH4 / day * mol Biomass / mol NH4 || mol O2 / day * mol Biomass / mol O2
        u_NOO = np.fmax(0.0, np.fmin(p_NO2_NOO * y_nNOO, p_O2_noo * y_oNOO))    # mol NO2 / day * mol Biomass / mol NO2 || mol O2 / day * mol Biomass / mol O2
        u_AOX = np.fmax(0.0, np.fmin(p_NO2_AOX * y_no2AOX, p_NH4_AOX * y_nh4AOX)) # mol NO2 / day * mol Biomass / mol NO2 || mol NH4 / day * mol Biomass / mol NH4


        
        ###
        ### Change in state variables per day (ddt)
        ###
                   
        ### rates
        # aerobic OM uptake rate
        aer_heterotrophy = u_Het * m_bHet / y_oHet
        # total OM uptake rate, add new functional types
        if np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3) >= np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O):      
            heterotrophy = u_Het * m_bHet / y_oHet      \
                           + u_1Den * m_b1Den / y_n1Den \
                           + u_2Den * m_b2Den / y_n2Den \
                           + u_3Den * m_b3Den / y_n3Den \
                           + u_4Den * m_b4Den / y_n4Den \
                           + u_5Den * m_b5Den / y_n5Den \
                           + u_6Den * m_b6Den / y_n6Den \
                           + u_7Den * m_b7Den / y_n7Den_NO3
        else:
            heterotrophy = u_Het * m_bHet / y_oHet      \
                           + u_1Den * m_b1Den / y_n1Den \
                           + u_2Den * m_b2Den / y_n2Den \
                           + u_3Den * m_b3Den / y_n3Den \
                           + u_4Den * m_b4Den / y_n4Den \
                           + u_5Den * m_b5Den / y_n5Den \
                           + u_6Den * m_b6Den / y_n6Den \
                           + u_7Den * m_b7Den / y_n7Den_N2O
        # Oxygen consumption rate
        oxy_consumption = u_Het * m_bHet / y_oO2    \
                          + u_AOO * m_bAOO / y_oAOO \
                          + u_NOO * m_bNOO / y_oNOO
        # DIN uptake rates
        ammonia_ox = u_AOO * m_bAOO / y_nAOO   # ammonia uptake rate by AOA
        nitrite_ox = u_NOO * m_bNOO / y_nNOO   # nitrite uptake rate by NOB
        anammox_nh4 = u_AOX * m_bAOX / y_nh4AOX  # ammonia uptake rate by anammox bacteria
        anammox_no2 = u_AOX * m_bAOX / y_no2AOX  # nitrite uptake rate by anammox bacteria
        anammox_no3 = u_AOX * m_bAOX * e_no3AOX  # nitrate production rate by anammox bacteria
        
        if np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3) >= np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O):
            den_nar = u_1Den * m_b1Den / y_n1NO3        \
                      + u_7Den * m_b7Den / y_n7NO3
            den_nir = u_2Den * m_b2Den / y_n2NO2  # nitrite uptake rate by NO2-->N2 
            den_full = u_3Den * m_b3Den / y_n3NO3 # nitrate uptake rate by NO3-->N2
            den_NitritetoN2O = u_4Den * m_b4Den / y_n4NO2 # nitrite uptake rate by NO2-->N2O                       
            den_N2OtoN2 = u_5Den * m_b5Den / y_n5N2O # N2O uptake rate by N2O-->N2
            den_NitratetoN2O = u_6Den * m_b6Den / y_n6NO3 # nitrate uptake rate by NO3-->N2O
        else:
            den_nar = u_1Den * m_b1Den / y_n1NO3                     
            den_nir = u_2Den * m_b2Den / y_n2NO2 # nitrite uptake rate by NO2-->N2 
            den_full = u_3Den * m_b3Den / y_n3NO3 # nitrate uptake rate by NO3-->N2
            den_NitritetoN2O = u_4Den * m_b4Den / y_n4NO2 # nitrite uptake rate by NO2-->N2O                       
            den_N2OtoN2 = u_5Den * m_b5Den / y_n5N2O \
                          + u_7Den * m_b7Den / y_n7N2O # N2O uptake rate by N2O-->N2
            den_NitratetoN2O = u_6Den * m_b6Den / y_n6NO3 # nitrate uptake rate by NO3-->N2O
               
        #### include N2O from ammonia in the model
        if m_O2 <= (1/6.022*1e-18): #0.0685:
            N2OammoniaRate = 1/(1+1/((0.2/(1/6.022*1e-18) + 0.08)/100)) * ammonia_ox #N2Oammonia
        else:
            N2OammoniaRate = 1/(1+1/((0.2/(m_O2) + 0.08)/100)) * ammonia_ox #N2Oammonia
            
        ## Below are the change of OM, DIN.. over time. For example, d[OM]/dt, d[NO3]/dt
        # Organic matter
        ddt_Sd = dil * (in_Sd - m_Sd) - heterotrophy
        
        # Dissolved oxygen (consumed by 1, 2, 6, 7)
        ddt_O2 = dil * (in_O2 - m_O2) - oxy_consumption
                 
        # Nitrate (consumed by 2&3, 5, 56) (produced by 7, 8) 
        ddt_NO3 = dil * (in_NO3 - m_NO3)        \
                 - den_nar                      \
                 - den_full                     \
                 - den_NitratetoN2O             \
                 + nitrite_ox                   \
                 + anammox_no3        
        
        # Nitrite (consumed by 4, 7, 8, 54) (produced by 2&3, 6)
        ddt_NO2 = dil * (in_NO2 - m_NO2)        \
                 - den_nir                      \
                 - den_NitritetoN2O             \
                 - nitrite_ox                   \
                 - anammox_no2                  \
                 - N2OammoniaRate               \
                 + den_nar                      \
                 + u_AOO * m_bAOO * (1./y_nAOO - 1)   # because it uses one mol of NH4 for biomass synthesis                 
        
        
        if np.fmin(p_Sd * y_n7Den_NO3, p_1DenNO3 * y_n7NO3) >= np.fmin(p_Sd * y_n7Den_N2O, p_N2O_den * y_n7N2O):
            # Ammonium (consumed by 6, 7, 8) (produced by 1, 2, 3, 4, 5, 54, 55, 56)
            ddt_NH4 = dil * (in_NH4 - m_NH4)        \
                     - ammonia_ox                   \
                     - anammox_nh4                  \
                     - u_NOO * m_bNOO               \
                     + u_Het * m_bHet * (1./y_oHet - 1)    \
                     + u_1Den * m_b1Den * (1./y_n1Den - 1) \
                     + u_2Den * m_b2Den * (1./y_n2Den - 1) \
                     + u_3Den * m_b3Den * (1./y_n3Den - 1) \
                     + u_4Den * m_b4Den * (1./y_n4Den - 1) \
                     + u_5Den * m_b5Den * (1./y_n5Den - 1) \
                     + u_6Den * m_b6Den * (1./y_n6Den - 1) \
                     + u_7Den * m_b7Den * (1./y_n7Den_NO3 - 1)
            # Dinitrogen gas (produced by 4, 5, 8, 55)
            ddt_N2 = dil * (in_N2-m_N2)             \
                     + u_2Den * m_b2Den * e_n2Den   \
                     + u_3Den * m_b3Den * e_n3Den   \
                     + u_5Den * m_b5Den * e_n5Den \
                     + u_AOX * m_bAOX * e_n2AOX

        else:
            # Ammonium (consumed by 6, 7, 8) (produced by 1, 2, 3, 4, 5, 54, 55, 56)
            ddt_NH4 = dil * (in_NH4 - m_NH4)        \
                     - ammonia_ox                   \
                     - anammox_nh4                  \
                     - u_NOO * m_bNOO               \
                     + u_Het * m_bHet * (1./y_oHet - 1)    \
                     + u_1Den * m_b1Den * (1./y_n1Den - 1) \
                     + u_2Den * m_b2Den * (1./y_n2Den - 1) \
                     + u_3Den * m_b3Den * (1./y_n3Den - 1) \
                     + u_4Den * m_b4Den * (1./y_n4Den - 1) \
                     + u_5Den * m_b5Den * (1./y_n5Den - 1) \
                     + u_6Den * m_b6Den * (1./y_n6Den - 1) \
                     + u_7Den * m_b7Den * (1./y_n7Den_N2O - 1)
            # Dinitrogen gas (produced by 4, 5, 8, 55)
            ddt_N2 = dil * (in_N2-m_N2)             \
                     + u_2Den * m_b2Den * e_n2Den   \
                     + u_3Den * m_b3Den * e_n3Den   \
                     + u_5Den * m_b5Den * e_n5Den \
                     + u_7Den * m_b7Den * e_n7Den_N2O \
                     + u_AOX * m_bAOX * e_n2AOX

        # N2O gas (produced by 54 and 56) (consumed by 55)      
        if N2Oammonia == 0:
            ddt_N2O = dil * (in_N2O-m_N2O)          \
                     + u_4Den * m_b4Den * e_n4Den   \
                     + u_6Den * m_b6Den * e_n6Den   \
                     - den_N2OtoN2
        else:
            if N2Oammonia == 1:
                ddt_N2O = dil * (in_N2O-m_N2O)          \
                         + u_4Den * m_b4Den * e_n4Den   \
                         + u_6Den * m_b6Den * e_n6Den   \
                         + N2OammoniaRate \
                         - den_N2OtoN2            
         
        # Biomass of aerobic heterotrophs
        ddt_bHet = dil * (-m_bHet)              \
                   + u_Het * m_bHet       
        # Biomass of nitrate denitrifiers
        ddt_b1Den = dil * (-m_b1Den)            \
                   + u_1Den * m_b1Den     
        # Biomass of nitrite denitrifiers
        ddt_b2Den = dil * (-m_b2Den)            \
                   + u_2Den * m_b2Den       
        # Biomass of full denitrifiers
        ddt_b3Den = dil * (-m_b3Den)            \
                   + u_3Den * m_b3Den 
        # Biomass of NO2-->N2O
        ddt_b4Den = dil * (-m_b4Den)            \
                   + u_4Den * m_b4Den 
        # Biomass of N2O-->N2
        ddt_b5Den = dil * (-m_b5Den)            \
                   + u_5Den * m_b5Den          
        # Biomass of NO3-->N2O
        ddt_b6Den = dil * (-m_b6Den)            \
                   + u_6Den * m_b6Den 
        # Biomass of bookend NO3-->NO2, N2O-->N2
        ddt_b7Den = dil * (-m_b7Den)            \
                   + u_7Den * m_b7Den       
        # Biomass of AOA
        ddt_bAOO = dil * (-m_bAOO)              \
                   + u_AOO * m_bAOO       
        # Biomass of NOB
        ddt_bNOO = dil * (-m_bNOO)              \
                   + u_NOO * m_bNOO       
        # Biomass of anammox bacteria
        ddt_bAOX = dil * (-m_bAOX)              \
                   + u_AOX * m_bAOX 
        
        
        # apply changes to state variables normalized by timestep (dt = timesteps per day)
        m_Sd = m_Sd + ddt_Sd * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NO3 = m_NO3 + ddt_NO3 * dt
        m_NO2 = m_NO2 + ddt_NO2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_N2 = m_N2 + ddt_N2 * dt
        m_N2O = m_N2O + ddt_N2O * dt
        m_bHet = m_bHet + ddt_bHet * dt
        m_b1Den = m_b1Den + ddt_b1Den * dt
        m_b2Den = m_b2Den + ddt_b2Den * dt
        m_b3Den = m_b3Den + ddt_b3Den * dt
        m_b4Den = m_b4Den + ddt_b4Den * dt
        m_b5Den = m_b5Den + ddt_b5Den * dt
        m_b6Den = m_b6Den + ddt_b6Den * dt
        m_b7Den = m_b7Den + ddt_b7Den * dt
        m_bAOO = m_bAOO + ddt_bAOO * dt
        m_bNOO = m_bNOO + ddt_bNOO * dt
        m_bAOX = m_bAOX + ddt_bAOX * dt
        
        # pulse OM and O2 into the chemostat
        if (t % int((1/dt * pulse_int))) == 0:
            m_Sd = m_Sd + pulse_Sd
            m_O2 = m_O2 + pulse_O2
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_Sd[i] = m_Sd 
            out_O2[i] = m_O2
            out_NO3[i] = m_NO3
            out_NO2[i] = m_NO2 
            out_NH4[i] = m_NH4 
            out_N2O[i] = m_N2O
            out_N2[i] = m_N2
            out_bHet[i] = m_bHet
            out_b1Den[i] = m_b1Den
            out_b2Den[i] = m_b2Den
            out_b3Den[i] = m_b3Den
            out_b4Den[i] = m_b4Den
            out_b5Den[i] = m_b5Den
            out_b6Den[i] = m_b6Den
            out_bAOO[i] = m_bAOO
            out_bNOO[i] = m_bNOO
            out_bAOX[i] = m_bAOX
            out_uHet[i] = u_Het
            out_u1Den[i] = u_1Den
            out_u2Den[i] = u_2Den
            out_u3Den[i] = u_3Den 
            out_u4Den[i] = u_4Den
            out_u5Den[i] = u_5Den
            out_u6Den[i] = u_6Den          
            out_uAOO[i] = u_AOO
            out_uNOO[i] = u_NOO
            out_uAOX[i] = u_AOX
            out_rHet[i] = heterotrophy
            out_rHetAer[i] = aer_heterotrophy
            out_rO2C[i] = oxy_consumption
            '''
            # Oxygen consumption rate
            oxy_consumption = u_Het * m_bHet / y_oO2    \
                              + u_AOO * m_bAOO / y_oAOO \
                              + u_NOO * m_bNOO / y_oNOO
             '''                 
            out_r1Den[i] = den_nar # nitrate uptake rate by nitrate reducers 
            out_r2Den[i] = den_nir  # nitrite uptake rate by NO2-->N2 
            out_r3Den[i] = den_full # nitrate uptake rate by NO3-->N2
            out_r4Den[i] = den_NitritetoN2O  # nitrite uptake rate by NO2-->N2O
            out_r5Den[i] = den_N2OtoN2      # N2O uptake rate by N2O-->N2 
            out_r6Den[i] = den_NitratetoN2O  # nitrate uptake rate by NO3-->N2O
            out_rAOO[i] = ammonia_ox - N2OammoniaRate
            out_rNOO[i] = nitrite_ox
            out_rAOX[i] = anammox_nh4
                                              
            out_b7Den[i] = m_b7Den
            out_u7Den[i] = u_7Den
            out_rN2Oammonia[i] = N2OammoniaRate
            
    return [out_Sd, out_O2, out_NO3, out_NO2, out_NH4, out_N2O, out_N2, \
            out_bHet, out_b1Den, out_b2Den, out_b3Den, out_b4Den, out_b5Den, out_b6Den,           
            out_bAOO, out_bNOO, out_bAOX, \
            out_uHet, out_u1Den, out_u2Den, out_u3Den, out_u4Den, out_u5Den, out_u6Den,
            out_uAOO, out_uNOO, out_uAOX, \
            out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_r4Den, out_r5Den, out_r6Den, out_rAOO, out_rNOO, out_rAOX, \
            out_b7Den, out_u7Den, \
            out_rN2Oammonia]
