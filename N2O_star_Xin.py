# -*- coding: utf-8 -*-
"""
Created on Wed July 2022

Purpose
-------
    Calculate N2O* (subsistence O2 of microbial population)
    in the unit of nM-N-N2O
    
@author: Xin Sun
"""

import numpy as np

def N2O_star(loss, Qc, diam, dc_n2o, y_n5N2O_C, K):
    '''
    Calculates the subsistence O2 concentration required to sustain biomass at equilibrium

    Parameters
    ----------
    loss : Float
        rate at which biomass is lost (1 / day)
    Qc : TYPE
        carbon quota of a single cell (mol C / um^3)
    diam : Float
        equivalent spherical diatmeter of the microbe (um)
    dc_n2o : Float
        rate of diffusive N2O supply (m^2 / day)
    y_n5N2O_C = y_n5N2O * CN_den: 
        yield of biomass per unit N-N2O consumed (mol C-biomass/ mol N-N2O)
    K: 
        half saturation constant of N2O in the unit of µM-N-N2O (K_n2o_Den)

    Returns
    -------
    Float
        Subsistence N2O concentration (in nM? since * 1e-9 is uM = mmol/m3)

    '''
    Rstar_N2O_linear = (loss * Qc * (diam/2)**2) / (3 * dc_n2o * 1e-12 * y_n5N2O_C) # only consider diffusion
    Rstar_N2O_MM = (Rstar_N2O_linear + (Rstar_N2O_linear**2 + 4 * Rstar_N2O_linear * K * 1e3)**0.5 ) / 2 # K * 1e3 make it nM N-N2O
    return [Rstar_N2O_MM, Rstar_N2O_linear]
