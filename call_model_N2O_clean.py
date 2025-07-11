# -*- coding: utf-8 -*-
"""
Created in June 2024

Purpose
-------
    A 0D chemostat model with microbes in marine OMZs,
    Modular denitrification included, yields of denitrifiers depend on Gibbs free energy.
    Organic matter pulses included
"""

#%% imports
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import cmocean.cm as cmo
from cmocean.tools import lighten
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# numerical packages
from numba import jit

#%% Set initial conditions and incoming concentrations to chemostat experiment

### Organic matter (S)
## Constant supply (assuming 1 m3 box and flux into top of box)
Sd0_exp = 0.05 #µM-N m-3 day-1
## Pulse intensity 
xpulse_Sd = np.arange(0.1,20.1,0.2)

### Oxygen supply
O20_exp = np.arange(0,300.1,2)

N2Oammonia = 1
### model parameters for running experiments
### dil = 0 for checking N balance 
dil = 0.04  # dilution rate (1/day)
if dil == 0:
    days = 10  # number of days to run chemostat
    dt = 0.001  # timesteps per day (days)
    timesteps = days/dt     # number of timesteps
    out_at_day = 0.1       # output results this often (days)
    nn_output = days/out_at_day     # number of entries for output
    print("dilution = 0, check N balance")
else:
    days = 4e4  # number of days to run chemostat
    dt = 0.001  # timesteps length (days)
    timesteps = days/dt     # number of timesteps
    out_at_day = dt         # output results this often
    nn_output = days/out_at_day     # number of entries for output
    print("dilution > 0, run experiments")
    
nn_outputforaverage = int(2000/out_at_day) # finish value is the average of the last XX (number) of outputs
     
#%% Define variables  
outputd1 = xpulse_Sd 
outputd2 = O20_exp

#%% initialize arrays for output

# Nutrients
fin_O2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_Sd = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NO3 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NO2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_NH4 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_N2 = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_N2O = np.ones((len(outputd1), len(outputd2))) * np.nan 
# Biomasses
fin_bHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_b6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_b7Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_bAOX = np.ones((len(outputd1), len(outputd2))) * np.nan
# Growth rates
fin_uHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_u6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_u7Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_uAOX = np.ones((len(outputd1), len(outputd2))) * np.nan
# Rates 
fin_rHet = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rHetAer = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rO2C = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r1Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r2Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r3Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_r4Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_r5Den = np.ones((len(outputd1), len(outputd2))) * np.nan 
fin_r6Den = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rAOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rNOO = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rAOX = np.ones((len(outputd1), len(outputd2))) * np.nan
fin_rN2Oammonia = np.ones((len(outputd1), len(outputd2))) * np.nan
#%% set traits of the different biomasses
os.chdir("YourPath")
### set traits
from traits import * 
fname = 'OMpulse_' 

#%% calculate R*-stars
from O2_star_Xin import O2_star
from N2O_star_Xin import N2O_star 
from R_star_Xin import R_star

# O2 (nM-O2) 
O2_star_aer = R_star(dil, K_o2_aer, mumax_Het / y_oO2, y_oO2) * 1e3 
O2_star_aoo = R_star(dil, K_o2_aoo, mumax_AOO / y_oAOO, y_oAOO) * 1e3 
O2_star_noo = R_star(dil, K_o2_noo, mumax_NOO / y_oNOO, y_oNOO) * 1e3 
# N2O (nM-N)
N2O_star_den5 = R_star(dil, K_n2o_Den, VmaxN_5Den, y_n5N2O) * 1e3
# OM
OM_star_aer = R_star(dil, K_s, VmaxS, y_oHet)
OM_star_den1 = R_star(dil, K_s, VmaxS, y_n1Den)
OM_star_den2 = R_star(dil, K_s, VmaxS, y_n2Den)
OM_star_den3 = R_star(dil, K_s, VmaxS, y_n3Den)
OM_star_den4 = R_star(dil, K_s, VmaxS, y_n4Den)
OM_star_den5 = R_star(dil, K_s, VmaxS, y_n5Den)
OM_star_den6 = R_star(dil, K_s, VmaxS, y_n6Den)
# Ammonia
Amm_star_aoo = R_star(dil, K_n_AOO, VmaxN_AOO, y_nAOO)
Amm_star_aox = R_star(dil, K_nh4_AOX, VmaxNH4_AOX, y_nh4AOX)
# Nitrite
nitrite_star_den2 = R_star(dil, K_n_Den, VmaxN_2Den, y_n2NO2)
nitrite_star_den4 = R_star(dil, K_n_Den, VmaxN_4Den, y_n4NO2)
nitrite_star_noo = R_star(dil, K_n_NOO, VmaxN_NOO, y_nNOO)
nitrite_star_aox = R_star(dil, K_no2_AOX, VmaxNO2_AOX, y_no2AOX)
# Nitrate
nitrate_star_den1 = R_star(dil, K_n_Den, VmaxN_1Den, y_n1NO3)
nitrate_star_den3 = R_star(dil, K_n_Den, VmaxN_3Den, y_n3NO3)
nitrate_star_den6 = R_star(dil, K_n_Den, VmaxN_6Den, y_n6NO3)


#%% begin loop of experiments by running the model

from model_N2O import OMZredox

for k in np.arange(len(outputd1)):
    for m in np.arange(len(O20_exp)):
        print(k,m)
        
        # 1) Chemostat influxes (µM-N or µM O2)
        in_Sd = Sd0_exp
        in_O2 = O20_exp[m]
        in_NO3 = 30.0
        in_NO2 = 0.0
        in_NH4 = 0.0
        in_N2 = 0.0
        in_N2O = 0.0
        # initial conditions
        initialOM = in_Sd
        initialNO2 = 0
        
        # 2) Initial biomasses (set to 0.0 to exclude a microbial group, 0.1 as default) 
        in_bHet = 0.1
        in_b1Den = 0.1
        in_b4Den = 0.1
        in_b5Den = 0.1
        in_b2Den = 0.1
        in_b3Den = 0.1
        in_b6Den = 0.1
        in_b7Den = 0  
        in_bAOO = 0.1
        in_bNOO = 0.1
        in_bAOX = 0.1
        
        # pulse conditions        
        pulse_int = 50 #pulse interval
        pulse_Sd = xpulse_Sd[k]
        pulse_O2 = 0.0
       
       
        # 3) Call main model
        results = OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
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
                           in_bHet, in_b1Den, in_b2Den, in_b3Den, in_bAOO, in_bNOO, in_bAOX, in_b4Den, in_b5Den, in_b6Den, in_b7Den, \
                           N2Oammonia)
        
        out_Sd = results[0]
        out_O2 = results[1]
        out_NO3 = results[2]
        out_NO2 = results[3]
        out_NH4 = results[4]
        out_N2O = results[5] 
        out_N2 = results[6]
        out_bHet = results[7]
        out_b1Den = results[8]
        out_b2Den = results[9]
        out_b3Den = results[10]
        out_b4Den = results[11]
        out_b5Den = results[12]
        out_b6Den = results[13]
        out_bAOO = results[14]
        out_bNOO = results[15]
        out_bAOX = results[16]
        out_uHet = results[17]
        out_u1Den = results[18]
        out_u2Den = results[19]
        out_u3Den = results[20]
        out_u4Den = results[21]
        out_u5Den = results[22]
        out_u6Den = results[23]
        out_uAOO = results[24]
        out_uNOO = results[25]
        out_uAOX = results[26]
        out_rHet = results[27]
        out_rHetAer = results[28]
        out_rO2C = results[29]
        out_r1Den = results[30]
        out_r2Den = results[31]
        out_r3Den = results[32]
        out_r4Den = results[33]
        out_r5Den = results[34]
        out_r6Den = results[35]
        out_rAOO = results[36]
        out_rNOO = results[37]
        out_rAOX = results[38]     
        out_b7Den = results[39]
        out_u7Den = results[40]
        out_rN2Oammonia = results[41]
        
        # 4) Record solutions
        fin_O2[k,m] = np.nanmean(out_O2[-nn_outputforaverage::])
        fin_Sd[k,m] = np.nanmean(out_Sd[-nn_outputforaverage::])
        fin_NO3[k,m] = np.nanmean(out_NO3[-nn_outputforaverage::])
        fin_NO2[k,m] = np.nanmean(out_NO2[-nn_outputforaverage::])
        fin_NH4[k,m] = np.nanmean(out_NH4[-nn_outputforaverage::])
        fin_N2[k,m] = np.nanmean(out_N2[-nn_outputforaverage::])
        fin_N2O[k,m] = np.nanmean(out_N2O[-nn_outputforaverage::]) 
        fin_bHet[k,m] = np.nanmean(out_bHet[-nn_outputforaverage::])
        fin_b1Den[k,m] = np.nanmean(out_b1Den[-nn_outputforaverage::])
        fin_b2Den[k,m] = np.nanmean(out_b2Den[-nn_outputforaverage::])
        fin_b3Den[k,m] = np.nanmean(out_b3Den[-nn_outputforaverage::])
        fin_b4Den[k,m] = np.nanmean(out_b4Den[-nn_outputforaverage::]) 
        fin_b5Den[k,m] = np.nanmean(out_b5Den[-nn_outputforaverage::]) 
        fin_b6Den[k,m] = np.nanmean(out_b6Den[-nn_outputforaverage::])
        fin_b7Den[k,m] = np.nanmean(out_b7Den[-nn_outputforaverage::]) 
        fin_bAOO[k,m] = np.nanmean(out_bAOO[-nn_outputforaverage::])
        fin_bNOO[k,m] = np.nanmean(out_bNOO[-nn_outputforaverage::])
        fin_bAOX[k,m] = np.nanmean(out_bAOX[-nn_outputforaverage::])
        fin_uHet[k,m] = np.nanmean(out_uHet[-nn_outputforaverage::])
        fin_u1Den[k,m] = np.nanmean(out_u1Den[-nn_outputforaverage::])
        fin_u2Den[k,m] = np.nanmean(out_u2Den[-nn_outputforaverage::])
        fin_u3Den[k,m] = np.nanmean(out_u3Den[-nn_outputforaverage::])
        fin_u4Den[k,m] = np.nanmean(out_u4Den[-nn_outputforaverage::]) 
        fin_u5Den[k,m] = np.nanmean(out_u5Den[-nn_outputforaverage::]) 
        fin_u6Den[k,m] = np.nanmean(out_u6Den[-nn_outputforaverage::]) 
        fin_u7Den[k,m] = np.nanmean(out_u7Den[-nn_outputforaverage::])
        fin_uAOO[k,m] = np.nanmean(out_uAOO[-nn_outputforaverage::])
        fin_uNOO[k,m] = np.nanmean(out_uNOO[-nn_outputforaverage::])
        fin_uAOX[k,m] = np.nanmean(out_uAOX[-nn_outputforaverage::])
        fin_rHet[k,m] = np.nanmean(out_rHet[-nn_outputforaverage::])
        fin_rHetAer[k,m] = np.nanmean(out_rHetAer[-nn_outputforaverage::])
        fin_rO2C[k,m] = np.nanmean(out_rO2C[-nn_outputforaverage::])
        fin_r1Den[k,m] = np.nanmean(out_r1Den[-nn_outputforaverage::])
        fin_r2Den[k,m] = np.nanmean(out_r2Den[-nn_outputforaverage::])
        fin_r3Den[k,m] = np.nanmean(out_r3Den[-nn_outputforaverage::])
        fin_r4Den[k,m] = np.nanmean(out_r4Den[-nn_outputforaverage::]) 
        fin_r5Den[k,m] = np.nanmean(out_r5Den[-nn_outputforaverage::]) 
        fin_r6Den[k,m] = np.nanmean(out_r6Den[-nn_outputforaverage::]) 
        fin_rAOO[k,m] = np.nanmean(out_rAOO[-nn_outputforaverage::])
        fin_rNOO[k,m] = np.nanmean(out_rNOO[-nn_outputforaverage::])
        fin_rAOX[k,m] = np.nanmean(out_rAOX[-nn_outputforaverage::])
        fin_rN2Oammonia[k,m] = np.nanmean(out_rN2Oammonia[-nn_outputforaverage::])
            
# delete results called "out_" and only save the averaged output "fin_" to save space
del results
del out_Sd, out_O2, out_NO3, out_NO2, out_NH4, out_N2, out_N2O
del out_bHet, out_b1Den, out_b2Den, out_b3Den, out_b4Den, out_b5Den, out_b6Den, out_b7Den, out_bAOO, out_bNOO, out_bAOX
del out_uHet, out_u1Den, out_u2Den, out_u3Den, out_u4Den, out_u5Den, out_u6Den, out_u7Den, out_uAOO, out_uNOO, out_uAOX
del out_rHet, out_rHetAer, out_rO2C, out_r1Den, out_r2Den, out_r3Den, out_r4Den, out_r5Den, out_r6Den, out_rAOO, out_rNOO, out_rAOX


#%% round up function, this will be used in figures below
import math
def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

#%% Plots
xpulse_int = 50
pulse_int = 50



#%% N2O Fig.3 a-f Pcolormesh_Contour plot_% biomasses 
fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(11,6))
gs = GridSpec(2, 3)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])
# set x and y axes
contourX = outputd2 * dil
contourY = xpulse_Sd/xpulse_int + Sd0_exp * dil 

# set colorbar range to be the same

fin_ballanaerobes = fin_b1Den+fin_b2Den+fin_b3Den+fin_b4Den+fin_b5Den+fin_b6Den+fin_bAOX
fin_ballaerobes = fin_bHet+fin_bAOO+fin_bNOO
fin_ballbio = fin_ballanaerobes+fin_ballaerobes
fin_allNs = fin_Sd+fin_NO3+fin_NO2+fin_NH4+fin_N2O+fin_N2
nh4_n2_AOX = (e_n2AOX*0.5*y_nh4AOX)



ax1.set_title('N$_2$O (nM)', fontsize=fslab)
p1 = ax1.pcolormesh(contourX, contourY, fin_N2O*1e3*0.5, cmap=colmap)

ax2.set_title('NO$_3$$^-$→N$_2$O (µM-N/d)', fontsize=fslab)
p2 = ax2.pcolormesh(contourX, contourY, fin_r6Den, cmap=colmap)

ax3.set_title('% Aer Auto', fontsize=fslab)
p3 = ax3.pcolormesh(contourX, contourY, (fin_bAOO+fin_bNOO)/fin_ballbio * 100, 
                     vmin=0, vmax=100, cmap=colmap)

ax4.set_title('O$_2$ (µM)', fontsize=fslab)
p4 = ax4.pcolormesh(contourX, contourY, fin_O2, 
                      cmap=colmap) #0.063

ax5.set_title('% NO$_3$$^-$→N$_2$O in N$_2$O prod', fontsize=fslab)
p5 = ax5.pcolormesh(contourX, contourY, (fin_r6Den)/(fin_rN2Oammonia + fin_r6Den) * 100,
                      vmin=0, vmax=100, cmap=colmap)
                   
ax6.set_title('% NH$_4$$^+$→N$_2$O in NH$_4$$^+$ oxi', fontsize=fslab)
p6 = ax6.pcolormesh(contourX, contourY, (fin_rN2Oammonia)/(fin_rAOO+fin_rN2Oammonia) * 100, 
                      vmin=0.08, vmax=3.24, cmap=colmap) #0.08



## delete axis number of some subplots
ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelleft=False, labelbottom=False)
ax3.tick_params(labelsize=fstic, labelleft=False, labelbottom=False)
ax1.tick_params(labelsize=fstic)
ax5.tick_params(labelsize=fstic, labelleft=False)
ax6.tick_params(labelsize=fstic, labelleft=False)
## add axis title to some subplots
contourYlabel = 'OM supply (µM-N d$^-$$^1$)'
contourXlabel = 'O$_2$ supply (µM d$^-$$^1$)'

ax1.set_ylabel(contourYlabel, fontsize=fslab)
ax4.set_ylabel(contourYlabel, fontsize=fslab)

ax4.set_xlabel(contourXlabel, fontsize=fslab)
ax5.set_xlabel(contourXlabel, fontsize=fslab)
ax6.set_xlabel(contourXlabel, fontsize=fslab)


xlowerlimit = 0
xupperlimit = max(O20_exp) * dil + 0.01
xtickdiv = 2
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))

cbar1 = fig.colorbar(p1, ax=ax1)
cbar2 = fig.colorbar(p2, ax=ax2)
cbar3 = fig.colorbar(p3, ax=ax3)
cbar4 = fig.colorbar(p4, ax=ax4)
cbar5 = fig.colorbar(p5, ax=ax5)
cbar6 = fig.colorbar(p6, ax=ax6)
plt.tight_layout()

#%% Save the plot
os.chdir("YourPath")
fig.savefig('Fig3_a-f.png', dpi=300) 



#%% N2O Fig.3 g,h,i
## plot the outcomes of the experiments against input O2 concentrations
fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(12,4)) # set figure size
gs = GridSpec(1, 3) # set row and column numbers

contourX = O20_exp * dil

OMsupply = xpulse_Sd/xpulse_int + Sd0_exp * dil
OMsupply_broadcasted = np.broadcast_to(OMsupply[:, np.newaxis], fin_O2.shape)
# Normalize the `fin_Sd` values to use with the colormap
norm = Normalize(vmin=np.min(OMsupply_broadcasted), vmax=np.max(OMsupply_broadcasted))
cmap = cm.Greens

# Subplot 1: N2O
ax1 = plt.subplot(gs[0, 0])
ax1.set_title('', fontsize=fslab)
scatter1 = ax1.scatter(fin_O2, fin_N2O * 1e3 * 0.5, c=OMsupply_broadcasted, cmap=cmap, norm=norm)

# Subplot 2: NO3- -> N2O
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('', fontsize=fslab)
scatter2 = ax2.scatter(fin_O2, fin_r6Den * 1e3, c=OMsupply_broadcasted, cmap=cmap, norm=norm)

# Subplot 3: NH4+ -> N2O
ax3 = plt.subplot(gs[0, 2])
ax3.set_title('', fontsize=fslab)
scatter3 = ax3.scatter(fin_O2, fin_rN2Oammonia * 1e3, c=OMsupply_broadcasted, cmap=cmap, norm=norm)

# ax1.set_xscale('log') # set axis at log scale
# ax2.set_xscale('log') # set axis at log scale
# ax3.set_xscale('log') # set axis at log scale

# add axis title to some subplots
ax1.set_xlabel('O$_2$ (µM)', fontsize=fslab)
ax2.set_xlabel('O$_2$ (µM)', fontsize=fslab)
ax3.set_xlabel('O$_2$ (µM)', fontsize=fslab)
ax1.set_ylabel('N$_2$O (nM)', fontsize=fslab)
ax2.set_ylabel('NO$_3$$^-$$\\rightarrow$N$_2$O (nM-N d$^-$$^1$)', fontsize=fslab)
ax3.set_ylabel('NH$_4$$^+$$\\rightarrow$N$_2$O (nM-N d$^-$$^1$)', fontsize=fslab)
plt.tight_layout()

#%% Save the plot
os.chdir("YourPath")
fig.savefig('Fig3-lines.png', dpi=300)


#%% N2O Fig.3 g,h,i insert
## plot the outcomes of the experiments against input O2 concentrations
fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(12,4))
gs = GridSpec(1, 3)

contourX = O20_exp * dil


OMsupply = xpulse_Sd/xpulse_int + Sd0_exp * dil
OMsupply_broadcasted = np.broadcast_to(OMsupply[:, np.newaxis], fin_O2.shape)
# Normalize the `fin_Sd` values to use with the colormap
norm = Normalize(vmin=np.min(OMsupply_broadcasted), vmax=np.max(OMsupply_broadcasted))
cmap = cm.Greens


# Subplot 1: N2O
ax1 = plt.subplot(gs[0, 0])
ax1.set_title('', fontsize=fslab)
scatter1 = ax1.scatter(fin_O2, fin_N2O * 1e3 * 0.5, c=OMsupply_broadcasted, cmap=cmap, norm=norm)
plt.ylim([0,10])
# Subplot 2: NO3- -> N2O
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('', fontsize=fslab)
scatter2 = ax2.scatter(fin_O2, fin_r6Den * 1e3, c=OMsupply_broadcasted, cmap=cmap, norm=norm)
plt.ylim([0,0.5])
# Subplot 3: NH4+ -> N2O
ax3 = plt.subplot(gs[0, 2])
ax3.set_title('', fontsize=fslab)
scatter3 = ax3.scatter(fin_O2, fin_rN2Oammonia * 1e3, c=OMsupply_broadcasted, cmap=cmap, norm=norm)
plt.ylim([0,0.5])

insertfont = 19
ax1.tick_params(labelsize=insertfont)
ax2.tick_params(labelsize=insertfont)
ax3.tick_params(labelsize=insertfont)
# add axis title to some subplots
ax1.set_xlabel('O$_2$ (µM)', fontsize=insertfont)
ax2.set_xlabel('O$_2$ (µM)', fontsize=insertfont)
ax3.set_xlabel('O$_2$ (µM)', fontsize=insertfont)

plt.tight_layout()


#%% Save the plot
os.chdir("YourPath")
fig.savefig('Fig3-lines-insert.png', dpi=300)









#%% Pcolormesh_Contour plot_all nuts and biomasses
fstic = 13
fslab = 15
colmap = lighten(cmo.haline, 0.8)

fig = plt.figure(figsize=(10,28))
gs = GridSpec(5, 3)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])
ax7 = plt.subplot(gs[2,0])
ax8 = plt.subplot(gs[2,1])
ax9 = plt.subplot(gs[2,2])
ax10 = plt.subplot(gs[3,0])
ax11 = plt.subplot(gs[3,1])
ax12 = plt.subplot(gs[3,2])
ax13 = plt.subplot(gs[4,0])
ax14 = plt.subplot(gs[4,1])
ax15 = plt.subplot(gs[4,2])


# set x and y axes
contourX = outputd2 * dil
contourY = xpulse_Sd/xpulse_int + Sd0_exp * dil #outputd1

# Compute the meshgrid for contouring
Xmesh, Ymesh = np.meshgrid(contourX, contourY)
# Avoid division by zero
ratio = np.divide(Ymesh, Xmesh, out=np.full_like(Ymesh, np.nan), where=Xmesh != 0)


# set colorbar range to be the same
colormin = 0.0
colormax = round_up(np.max([fin_b1Den, fin_b2Den, fin_b3Den, fin_b4Den, fin_b5Den, fin_b6Den]), 1) 
colormax1 = round_up(np.max([fin_bHet, fin_bAOO, fin_bNOO]), 1) 


nh4_n2_AOX = (e_n2AOX*0.5*y_nh4AOX)

ax1.set_title('O$_2$ (µM)', fontsize=fslab)
ax2.set_title('NO$_3$$^-$ (µM)', fontsize=fslab)
ax3.set_title('NO$_2$$^-$ (µM)', fontsize=fslab)
ax4.set_title('OM (µM-N)', fontsize=fslab)
ax5.set_title('N$_2$O (nM)', fontsize=fslab)
p1 = ax1.pcolormesh(contourX, contourY, fin_O2, cmap=colmap)#vmin=0, vmax=80, 

p2 = ax2.pcolormesh(contourX, contourY, fin_NO3, cmap=colmap)

p3 = ax3.pcolormesh(contourX, contourY, fin_NO2, cmap=colmap)
 
p4 = ax4.pcolormesh(contourX, contourY, fin_Sd, cmap=colmap)

p5 = ax5.pcolormesh(contourX, contourY, fin_N2O*1e3*0.5, cmap=colmap)


ax6.set_title('NH$_4$$^+$→N$_2$O (µM-N/d)', fontsize=fslab)
p6 = ax6.pcolormesh(contourX, contourY, fin_rN2Oammonia, cmap=colmap)


ax7.set_title('Bio AerHet (µM-N)', fontsize=fslab)
p7 = ax7.pcolormesh(contourX, contourY, fin_bHet, vmin=colormin, vmax=colormax1, cmap=colmap)

ax8.set_title('Bio AOA (µM-N)', fontsize=fslab)
p8 = ax8.pcolormesh(contourX, contourY, fin_bAOO, vmin=colormin, vmax=colormax1, cmap=colmap)

ax9.set_title('Bio NOB (µM-N)', fontsize=fslab)
p9 = ax9.pcolormesh(contourX, contourY, fin_bNOO, vmin=colormin, vmax=colormax1, cmap=colmap)


ax10.set_title('Bio NO$_3$$^-$→NO$_2$$^-$ (µM-N)', fontsize=fslab)
p10 = ax10.pcolormesh(contourX, contourY, fin_b1Den, vmin=colormin, vmax=colormax, cmap=colmap) 

ax11.set_title('Bio NO$_3$$^-$→N$_2$O (µM-N)', fontsize=fslab)
p11 = ax11.pcolormesh(contourX, contourY, fin_b6Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax12.set_title('Bio NO$_3$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p12 = ax12.pcolormesh(contourX, contourY, fin_b3Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax13.set_title('Bio NO$_2$$^-$→N$_2$O (µM-N)', fontsize=fslab)
p13 = ax13.pcolormesh(contourX, contourY, fin_b4Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax14.set_title('Bio NO$_2$$^-$→N$_2$ (µM-N)', fontsize=fslab)
p14 = ax14.pcolormesh(contourX, contourY, fin_b2Den, vmin=colormin, vmax=colormax, cmap=colmap)

ax15.set_title('Bio N$_2$O→N$_2$ (µM-N)', fontsize=fslab)
p15 = ax15.pcolormesh(contourX, contourY, fin_b5Den, vmin=colormin, vmax=colormax, cmap=colmap)


for ax in [ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]:
    ax.contour(Xmesh, Ymesh, ratio, levels=[0.01, 0.1, 1], colors='white', linewidths=1)
    CS = ax.contour(Xmesh, Ymesh, ratio, levels=[0.01, 0.1, 1], colors='white', linewidths=1)
    ax.clabel(CS, inline=False, fontsize=10, fmt='%1.2f', colors='white')

## delete axis number of some subplots
ax1.tick_params(labelsize=fstic, labelbottom=False)
ax2.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax3.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax4.tick_params(labelsize=fstic, labelbottom=False)
ax5.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax6.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax7.tick_params(labelsize=fstic, labelbottom=False)
ax8.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax9.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax10.tick_params(labelsize=fstic, labelbottom=False)
ax11.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax12.tick_params(labelsize=fstic, labelbottom=False, labelleft=False)
ax13.tick_params(labelsize=fstic)
ax14.tick_params(labelsize=fstic, labelleft=False)
ax15.tick_params(labelsize=fstic, labelleft=False)


## add axis title to some subplots
contourYlabel = 'OM supply (µM-N/d)'
contourXlabel = 'O$_2$ supply (µM/d)'

ax7.set_ylabel(contourYlabel, fontsize=fslab)


ax13.set_xlabel(contourXlabel, fontsize=fslab)
ax14.set_xlabel(contourXlabel, fontsize=fslab)
ax15.set_xlabel(contourXlabel, fontsize=fslab)

xlowerlimit = 0
xupperlimit = max(O20_exp) *dil + 0.01
xtickdiv = 2

ylowerlimit = 0
yupperlimit = max(contourY) + 0.001
ytickdiv = 0.1

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]:
    ax.set_xlim([xlowerlimit, xupperlimit])
    ax.set_xticks(np.arange(xlowerlimit, xupperlimit, xtickdiv))
    ax.set_ylim([ylowerlimit, yupperlimit])
    ax.set_yticks(np.arange(ylowerlimit, yupperlimit, ytickdiv))
    
cbar1 = fig.colorbar(p1, ax=ax1)
cbar2 = fig.colorbar(p2, ax=ax2)
cbar3 = fig.colorbar(p3, ax=ax3)
cbar4 = fig.colorbar(p4, ax=ax4)
cbar5 = fig.colorbar(p5, ax=ax5)
cbar6 = fig.colorbar(p6, ax=ax6)
cbar7 = fig.colorbar(p7, ax=ax7)
cbar8 = fig.colorbar(p8, ax=ax8)
cbar9 = fig.colorbar(p9, ax=ax9)
cbar10 = fig.colorbar(p10, ax=ax10)
cbar11 = fig.colorbar(p11, ax=ax11)
cbar12 = fig.colorbar(p12, ax=ax12)
cbar13 = fig.colorbar(p13, ax=ax13)
cbar14 = fig.colorbar(p14, ax=ax14)
cbar15 = fig.colorbar(p15, ax=ax15)

plt.tight_layout()
#%% Save the plot
os.chdir("YourPath")
fig.savefig('FigS5_biomass.png', dpi=300) 




#%% save the output to data folder
os.chdir("YourPath/Output")

fname = 'OMpulse_'
np.savetxt(fname+'_pulse.txt', xpulse_Sd, delimiter='\t')
np.savetxt(fname+'_O2supply.txt', O20_exp, delimiter='\t')

np.savetxt(fname+'_O2.txt', fin_O2, delimiter='\t')
np.savetxt(fname+'_N2.txt', fin_N2, delimiter='\t')
np.savetxt(fname+'_N2O.txt', fin_N2O, delimiter='\t')
np.savetxt(fname+'_NO3.txt', fin_NO3, delimiter='\t')
np.savetxt(fname+'_NO2.txt', fin_NO2, delimiter='\t')
np.savetxt(fname+'_NH4.txt', fin_NH4, delimiter='\t')
np.savetxt(fname+'_OM.txt', fin_Sd, delimiter='\t')

np.savetxt(fname+'_bHet.txt', fin_bHet, delimiter='\t')
np.savetxt(fname+'_b1Den.txt', fin_b1Den, delimiter='\t')
np.savetxt(fname+'_b2Den.txt', fin_b2Den, delimiter='\t')
np.savetxt(fname+'_b3Den.txt', fin_b3Den, delimiter='\t')
np.savetxt(fname+'_b4Den.txt', fin_b4Den, delimiter='\t')
np.savetxt(fname+'_b5Den.txt', fin_b5Den, delimiter='\t')
np.savetxt(fname+'_b6Den.txt', fin_b6Den, delimiter='\t')
np.savetxt(fname+'_b7Den.txt', fin_b7Den, delimiter='\t')
np.savetxt(fname+'_bAOO.txt', fin_bAOO, delimiter='\t')
np.savetxt(fname+'_bNOO.txt', fin_bNOO, delimiter='\t')
np.savetxt(fname+'_bAOX.txt', fin_bAOX, delimiter='\t')

np.savetxt(fname+'_uHet.txt', fin_uHet, delimiter='\t')
np.savetxt(fname+'_u1Den.txt', fin_u1Den, delimiter='\t')
np.savetxt(fname+'_u2Den.txt', fin_u2Den, delimiter='\t')
np.savetxt(fname+'_u3Den.txt', fin_u3Den, delimiter='\t')
np.savetxt(fname+'_u4Den.txt', fin_u4Den, delimiter='\t')
np.savetxt(fname+'_u5Den.txt', fin_u5Den, delimiter='\t')
np.savetxt(fname+'_u6Den.txt', fin_u6Den, delimiter='\t')
np.savetxt(fname+'_u7Den.txt', fin_u7Den, delimiter='\t')
np.savetxt(fname+'_uAOO.txt', fin_uAOO, delimiter='\t')
np.savetxt(fname+'_uNOO.txt', fin_uNOO, delimiter='\t')
np.savetxt(fname+'_uAOX.txt', fin_uAOX, delimiter='\t')

np.savetxt(fname+'_rHet.txt', fin_rHet, delimiter='\t')
np.savetxt(fname+'_rHetAer.txt', fin_rHetAer, delimiter='\t')
np.savetxt(fname+'_r1Den.txt', fin_r1Den, delimiter='\t')
np.savetxt(fname+'_r2Den.txt', fin_r2Den, delimiter='\t')
np.savetxt(fname+'_r3Den.txt', fin_r3Den, delimiter='\t')
np.savetxt(fname+'_r4Den.txt', fin_r4Den, delimiter='\t')
np.savetxt(fname+'_r5Den.txt', fin_r5Den, delimiter='\t')
np.savetxt(fname+'_r6Den.txt', fin_r6Den, delimiter='\t')
np.savetxt(fname+'_rAOO.txt', fin_rAOO, delimiter='\t')
np.savetxt(fname+'_rNOO.txt', fin_rNOO, delimiter='\t')
np.savetxt(fname+'_rAOX.txt', fin_rAOX, delimiter='\t')
np.savetxt(fname+'_rO2C.txt', fin_rO2C, delimiter='\t')

np.savetxt(fname+'_rN2Oammonia.txt', fin_rN2Oammonia, delimiter='\t')
