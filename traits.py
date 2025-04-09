# -*- coding: utf-8 -*-
"""
Created in March 2023, last update: June 2024
    
@authors: 
    Xin Sun, Emily Zakem, Pearse Buchanan
"""

import numpy as np
from GibbsEnergyYield_Xin import calc_y
P = 0.16

### 1) max growth rates (per day)
mumax_Het = 0.3     # 0.5 based on Rappé et al., 2002; 0.1 based on Kirchman 2016 
mumax_AOO = 0.25    # Kitzinger et al., 2020
mumax_NOO = 1.0     # Kitzinger et al., 2020
mumax_AOX = 0.1     # Oshiki et al., 2016


### 2) diffusive oxygen and n2o requirements based on cell diameters and carbon contents
# cell volumes (um^3)
vol_aer = 0.05  # based on SAR11 (Giovannoni 2017 Ann. Rev. Marine Science)
vol_den = vol_aer
vol_aoo = np.pi * (0.2*0.5)**2 * 0.8    # [rods] N. maritimus SCM1 in Table S1 from Hatzenpichler 2012 App. Envir. Microbiology
vol_noo = np.pi * (0.3*0.5)**2 * 3      # [rods] based on average cell diameter and length of Nitrospina gracilis (Spieck et al. 2014 Sys. Appl. Microbiology)
vol_aox = 4/3 * np.pi * (0.8*0.5)**3    # [cocci] diameter of 0.8 um based on Candidatus Scalindua (Wu et al. 2020 Water Science and Tech.)

# cell sizes (um), where vol = 4/3 * pi * r^3
diam_aer = ( 3 * vol_aer / (4 * np.pi) )**(1./3)*2
diam_den = ( 3 * vol_den / (4 * np.pi) )**(1./3)*2
diam_aoo = ( 3 * vol_aoo / (4 * np.pi) )**(1./3)*2
diam_noo = ( 3 * vol_noo / (4 * np.pi) )**(1./3)*2
diam_aox = ( 3 * vol_aox / (4 * np.pi) )**(1./3)*2

# cellular C:N of microbes
CN_aer = 5.0   # Zimmerman et al. 2014
CN_den = 5.0
CN_aoo = 5.0    
CN_noo = 5.0    
CN_aox = 5.0    

# g carbon per cell (assumes 0.1 g DW/WW for all microbial types as per communication with Marc Strous)
Ccell_aer = 0.1 * (12*CN_aer / (12*CN_aer + 7 + 16*2 + 14)) / (1e12 / vol_aer)
Ccell_den = 0.1 * (12*CN_den / (12*CN_den + 7 + 16*2 + 14)) / (1e12 / vol_den)
Ccell_aoo = 0.1 * (12*CN_aoo / (12*CN_aoo + 7 + 16*2 + 14)) / (1e12 / vol_aoo)
Ccell_noo = 0.1 * (12*CN_noo / (12*CN_noo + 7 + 16*2 + 14)) / (1e12 / vol_noo)
Ccell_aox = 0.1 * (12*CN_aox / (12*CN_aox + 8.7 + 16*1.55 + 14)) / (1e12 / vol_aox)

# cell quotas (mol C / um^3)
Qc_aer = Ccell_aer / vol_aer / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_den = Ccell_den / vol_den / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_aoo = Ccell_aoo / vol_aoo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_noo = Ccell_noo / vol_noo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019
Qc_aox = Ccell_aox / vol_aox / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured for SAR11 (aerobic heterotrophs) by White et al 2019

# diffusive oxygen and n2o coefficient
dc = 1.5776 * 1e-5      # cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day
dc_n2o = dc * 1.0049    # m^2/day 


### 3) Yields (y), products (e) and maximum uptake rate (Vmax)
###     Vmax = ( mu_max * Quota ) / yield 
###     we remove the need for the Quota term by considering everything in mols N per mol N-biomass, such that
###     Vmax = mu_max / yield (units of mol N per mol N-biomass per day)

### Delta G depended organic matter yield for heterotrophs (aerobes and denitrifiers)
##### THERMODYNAMIC DEFINITIONS
##### authors: Xin Sun and Emily Zakem

## define parameters
T = 25 # temperature (degree C)
pH = 7.5 
R = 8.3145 #J/(mol*K) Ideal gas constant
T = T + 273.15 #Converting temperature above to Kelvin
H = 10**(-pH) #Converting pH to concentration in units of mol/L

## Free energy of formation for relevant compounds and phases all values
## from (Amend & Shock 2001) at 25 degrees C
## EJZ: at pH = 0, not 7 (diff than R+McC chart)
DGf_NO3 = -110.91 * 1e3 #J/mol (aq) [NO3-]
DGf_NO2 = -32.22  * 1e3 #J/mol (aq) [NO2-]
DGf_H2O = -237.18 * 1e3 #J/mol (l)
DGf_N2O =  113.38 * 1e3 #J/mol (aq)
DGf_H   =    0    * 1e3 #J/mol (aq) [H+]
DGf_NO  =  102.06 * 1e3 #J/mol (aq)
DGf_N2  =   18.18 * 1e3 #J/mol (aq)
DGf_O2  =   16.54 * 1e3 #J/mol (aq)

## Calculating the standard free energy for the half reactions
## 1, 2, 3, 4 is NO3-->NO2-->NO-->N2O-->N2
# unit of DGo below is J/mol
DGheto = 0.5*DGf_H2O - 1/4*DGf_O2 - DGf_H 
DG1o = 0.5*DGf_NO2 + 0.5*DGf_H2O - 0.5*DGf_NO3 - DGf_H 
DG4o = 0.5*DGf_N2 + 0.5*DGf_H2O - 0.5*DGf_N2O - DGf_H 
DG123o = 1/8*DGf_N2O + 5/8*DGf_H2O - 1/4*DGf_NO3 - 5/4*DGf_H 
DG1234o = 1/10*DGf_N2 + 3/5*DGf_H2O - 1/5*DGf_NO3 - 6/5*DGf_H 
DG23o = 1/4*DGf_N2O + 3/4*DGf_H2O - 1/2*DGf_NO2 - 3/2*DGf_H 
DG234o = 1/6*DGf_N2 + 2/3*DGf_H2O - 1/3*DGf_NO2 - 4/3*DGf_H 

#avg concentrations (to initiate -- but don't want this to determine answer):
NO3avg = 30 * 1e-6 #mol/L 
NO2avg = 1e-6 #mol/L 
N2Oavg = 1e-8 #mol/L 
N2avg = 1e-4 #mol/L 
O2avg = 1e-6 #mol/L 

#per e- (In Julia, default based of log is e)
DGhet = DGheto + R*T*np.log(1/O2avg**(1/4)/H)
DG1 = DG1o + R*T*np.log(NO2avg**0.5/NO3avg**0.5/H) 
DG4 = DG4o + R*T*np.log(N2avg**0.5/N2Oavg**0.5/H)
DG123  = DG123o  + R*T*np.log(N2Oavg**(1/8)/NO3avg**(1/4)/H**(5/4)) 
DG1234 = DG1234o + R*T*np.log(N2avg**(1/10)/NO3avg**(1/5)/H**(6/5)) 
DG23  = DG23o  + R*T*np.log(N2Oavg**(1/4)/NO2avg**(1/2)/H**(3/2)) 
DG234 = DG234o + R*T*np.log(N2avg**(1/6)/NO2avg**(1/3)/H**(4/3)) 

## Gibbs rxn energy for oxidation of organic matter
## Estimated for avg marine OM:
om_energy = 3.33*1e3 #J/g cells from R&McC (estimate)
Cd=6.6; Hd=10.9; Od=2.6; Nd=1 # Based on (Anderson 1995)
## denominator for decomposition to nh4:
dD=4*Cd+Hd-2*Od-3*Nd;
DGom = om_energy*(Cd*12+Hd+Od*16+Nd*14)/dD

## energy to form biomass b from pyruvate -- R&McC:
b_energy = 3.33*1e3 #J/g cells from R&McC (estimate)
Cb=5; Hb=7; Ob=2; Nb=1 # Based on (Zimmerman et al. 2014)
dB=4*Cb+Hb-2*Ob-3*Nb
DGpb = b_energy*(Cb*12+Hb+Ob*16+Nb*14)/dB #energy to synthesize cells from pyruvate

## efficiency of e transfer:
ep = 0.6

## total energy needed for cell synthesis from given OM, including conv to pyruvate 
DGp = 35.09*1e3 - DGom #e to convert C to pyruvate. CHECK this 35.09 -- units?
#if Gp>=0; n=1; end;if Gp<0; n=-1; end 
n=1 #assume for now DGp > 0 (typical)
DGs = DGp/ep**n + DGpb/ep # total energy needed for synthesis. add uptake cost to this? 
## total energy needed for cell synthesis from given OM:
## energy needed to convert OM to pyruvate + energy needed to convert pyruvate to biomass
##### Finish defining THERMODYNAMICs
DGs = 300*1e3 # reset DGs to get the yield of aerobic hetero close to obs (0.2~0.3)

dO = 29.1   # assumes constant stoichiometry of all sinking organic matter of C6.6-H10.9-O2.6-N using equation: d = 4C + H - 2O -3N
dB = 20.0   # assumes constant stoichiometry of all heterotrophic bacteria of C5-H7-O2-N based on Zimmerman et al., 2014

# solve for the yield of heterotrophic bacteria using approach of Sinsabaugh et al. 2013
Y_max = 0.6     # (will not need this with GibbsFreeEnergy defined yield) maximum possible growth efficiency measured in the field
B_CN = 5.0      # C:N of bacterial biomass based on Zimmerman et al. 2014
OM_CN = 6.6     # C:N of labile dissolved organic matter (Letscher et al. 2015)
K_CN = 0.5      # C:N half-saturation coefficient (Sinsabaugh & Follstad Shah 2012 - Ecoenzymatic Stoichiometry and Ecological Theory)
EEA_CN = 1.123  # relative rate of enzymatic processing of complex C and complex N molecules into simple precursors for biosynthesis (Sinsabaugh & Follstad Shah 2012)

# aerobic heterotrophy 
f_oHet = calc_y(DGhet, DGom, ep, DGs) # old: y_oHet * dB/dO  # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_oHet =  f_oHet * dO/dB 
y_oO2 = (f_oHet/dB) / ((1.0-f_oHet)/4.0)  # yield of biomass per unit oxygen reduced
VmaxS = mumax_Het / y_oHet  # max mol Org N consum. rate per (mol BioN) per day

# nitrate reduction to nitrite (NO3 --> NO2), denstep = 1
f_n1Den = calc_y(DG1, DGom, ep, DGs)       # fraction of electrons used for biomass synthesis
y_n1Den = f_n1Den * dO/dB
y_n1NO3 = (f_n1Den/dB) / ((1.0-f_n1Den)/2.0) # yield of biomass per unit nitrate consumed (full equation for NO3->NO2 functional type)
e_n1Den = 1.0 / y_n1NO3         # mols NO2 produced per mol biomass synthesised
VmaxN_1Den = 50.8        # max mol nitrate consum. rate / mol cell N / day at 20C. (Zakem et al., 2018)

# nitrite reduction to N2 (NO2 --> N2), denstep = 3 (2)
f_n2Den = calc_y(DG234, DGom, ep, DGs) * (1 - P)      # fraction of electrons used for biomass synthesis
y_n2Den = f_n2Den * dO/dB #* (1 - P)
y_n2NO2 = (f_n2Den/dB) / ((1.0-f_n2Den)/3.0)       # yield of biomass per unit nitrite consumed
e_n2Den = 1.0 / y_n2NO2         # moles N-N2 produced per mole biomass synthesised
VmaxN_2Den = 50.8         # max mole Nitrite consum rate / mol cell N / day at 20C

# Full denitrification (NO3 --> N2), denstep = 4 (3)
f_n3Den = calc_y(DG1234, DGom, ep, DGs) * (1 - P * 2)      # fraction of electrons used for biomass synthesis
y_n3Den = f_n3Den * dO/dB #* (1 - P * 2)
y_n3NO3 = (f_n3Den/dB) / ((1.0-f_n3Den)/5.0) # yield of biomass per unit nitrate consumed (Zakem et al., 2019 A14)
e_n3Den = 1.0 / y_n3NO3         # moles N-N2 produced per mole biomass synthesised
VmaxN_3Den = 50.8         # max mole nitrate consum. rate / mol cell N / day at 20C

# NEW nitrite reduction to N2O (NO2 --> N2O), denstep = 2 (1)
f_n4Den = calc_y(DG23, DGom, ep, DGs)     # fraction of electrons used for biomass synthesis
y_n4Den = f_n4Den * dO/dB
y_n4NO2 = (f_n4Den/dB) / ((1.0-f_n4Den)/2.0) # yield of biomass per unit nitrite consumed
e_n4Den = 1.0 / y_n4NO2         # moles N-N2O produced per mole biomass synthesised
VmaxN_4Den = 50.8         # max mole nitrite consum. rate / mol cell N / day at 20C

# NEW N2O reduction to N2 (N2O --> N2), denstep = 1
f_n5Den = calc_y(DG4, DGom, ep, DGs)     # fraction of electrons used for biomass synthesis
y_n5Den = f_n5Den * dO/dB
y_n5N2O = (f_n5Den/dB) / ((1.0-f_n5Den)/1.0) # yield of biomass per unit N-N2O consumed
e_n5Den = 1.0 / y_n5N2O         # moles N-N2 produced per mole biomass synthesised
# Below uses max growth rate to define max N2O uptake rate, the other option is to use N2O diffusion to define max N2O uptake rate.
VmaxN_5Den = 50.8         # max mole N-N2O consum. rate / mol cell N / day at 20C

# NEW nitrate reduction to N2O (NO3 --> N2O), denstep = 3 (2)
f_n6Den = calc_y(DG123, DGom, ep, DGs) * (1 - P)     # fraction of electrons used for biomass synthesis
y_n6Den = f_n6Den * dO/dB #* (1 - P)
y_n6NO3 = (f_n6Den/dB) / ((1.0-f_n6Den)/4.0) # yield of biomass per unit nitrate consumed
e_n6Den = 1.0 / y_n6NO3         # moles N-N2O produced per mole biomass synthesised
VmaxN_6Den = 50.8 


# Bookend den (NO3 --> NO2, N2O-->N2), denstep = 2
f_n7Den_NO3 = calc_y(DG1, DGom, ep, DGs) * (1 - P)       # fraction of electrons used for biomass synthesis
y_n7Den_NO3 = f_n7Den_NO3 * dO/dB
y_n7NO3 = (f_n7Den_NO3/dB) / ((1.0-f_n7Den_NO3)/2.0) # yield of biomass per unit nitrate consumed (full equation for NO3->NO2 functional type)
e_n7Den_NO3 = 1.0 / y_n7NO3         # mols NO2 produced per mol biomass synthesised

f_n7Den_N2O = calc_y(DG4, DGom, ep, DGs) * (1 - P)     # fraction of electrons used for biomass synthesis
y_n7Den_N2O = f_n7Den_N2O * dO/dB
y_n7N2O = (f_n7Den_N2O/dB) / ((1.0-f_n7Den_N2O)/1.0) # yield of biomass per unit N-N2O consumed
e_n7Den_N2O = 1.0 / y_n7N2O         # moles N-N2 produced per mole biomass synthesised
VmaxN_7Den = 50.8        # max mol nitrate consum. rate / mol cell N / day at 20C. (Zakem et al., 2018)

# Chemoautotrophic ammonia oxidation (NH3 --> NO2)
y_nAOO = 0.0245         # mol N biomass per mol NH4 (Bayer et al. 2022; Zakem et al. 2022)
d_AOO = 4*5 + 7 - 2*2 -3*1  # number of electrons produced
f_AOO = y_nAOO / (6*(1/d_AOO - y_nAOO/d_AOO))         # fraction of electrons going to biomass synthesis from electron donor (NH4) (Zakem et al. 2022)
y_oAOO = f_AOO/d_AOO / ((1-f_AOO)/4.0)                # mol N biomass per mol O2 !!not O-O2 (Bayer et al. 2022; Zakem et al. 2022)
VmaxN_AOO = 50.8 #artvalue * mumax_AOO / y_nAOO            # max uptake rate of mol NH4 / mol cell N / day

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_nNOO = 0.0126         # mol N biomass per mol NO2 (Bayer et al. 2022)
d_NOO = 4*5 + 7 - 2*2 - 3*1
f_NOO = (y_nNOO * d_NOO) /2          # fraction of electrons going to biomass synthesis from electron donor (NO2) (Zakem et al. 2022)
y_oNOO = 4*f_NOO*(1-f_NOO)/d_NOO         # mol N biomass per mol O2 (Bayer et al. 2022)
VmaxN_NOO = 50.8 # mumax_NOO / y_nNOO = 1/0.0126 = 79.4
   

# Chemoautotrophic anammox (NH4 + NO2 --> NO3 + N2)
y_nh4AOX = 1./75                  # mol N biomass per mol NH4 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
y_no2AOX = 1./89                  # mol N biomass per mol NO2 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
e_n2AOX = 150                     # mol N-N2 formed per mol biomass N synthesised ***Rounded to nearest whole number
e_no3AOX = 13                     # mol NO3 formed per mol biomass N synthesised ***Rounded to nearest whole number

VmaxNH4_AOX = 50.8 #artvalue * mumax_AOX / y_nh4AOX              # max uptake rate of mol NH4 / mol cell N / day
VmaxNO2_AOX = 50.8 #artvalue * mumax_AOX / y_no2AOX              # max uptake rate of mol NO2 / mol cell N / day


### 4) Half-saturation constants (since the input conc will be in µM = mmol/m3, K will also be in µM)
K_s = 0.1           # organic nitrogen (uncertain) uM

K_n_Den = 4.0 # 4 – 25 µM NO2 for denitrifiers (Almeida et al. 1995)
K_n_AOO = 0.1 #K_n_universal #0.1       # Martens-Habbena et al. 2009 Nature
K_n_NOO = 0.1 #K_n_universal #0.1       # Reported from OMZ (Sun et al. 2017) and oligotrophic conditions (Zhang et al. 2020)
K_nh4_AOX = 0.45 #K_n_universal #0.45    # Awata et al. 2013 for Scalindua
K_no2_AOX = 0.45 #K_n_universal #0.45    # Awata et al. 2013 for Scalindua actually finds a K_no2 of 3.0 uM, but this excludes anammox completely in our experiments

K_n2o_Den = 0.3*2 #1e-4 # 0.3*2   # *2 convert µM N2O into µM N-N2O based on (Sun et al., 2021) in ODZ k = 0.3 µM N2O, in oxic layer = 1.4~2.8 µM

K_o2_aer = 0.2   # (µM-O2) 10 to 200 nM at extremely low oxygen concentrations (Tiano et al., 2014); 200 nM (low affinity terminal oxidases) (Morris et al., 2013)
K_o2_aoo = 0.333  # (µM-O2) 333 nM at oxic-anoxic interface (Bristow et al., 2016)
K_o2_noo = 0.8   # (µM-O2) 778 nM at oxic-anoxic interface or 0.5 nM and 1750 nM breaking into two M-M curves (Bristow et al., 2016)

