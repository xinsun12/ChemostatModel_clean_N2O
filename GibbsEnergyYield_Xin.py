# -*- coding: utf-8 -*-
"""
Created in Aug 2022

Purpose
-------
    Calculate f (the fraction of e used in biomass synthesis) based on Gibbs free energy
@author: Xin Sun
"""
import numpy as np

def calc_y(DGe, DGom, ep, DGs):
    '''
    
    Parameters
    ----------
    DGe :
    electron acceptor half rxn energy

    Returns
    -------
    f

    '''
    
    # DGr = DGe - DGom #catabolic rxn: energy released for e eq of donor oxidized  
    # A = -DGs/ep/DGr #energy balance
    # f = 1./(1+A)
    
    # f = 1/(1-DGs/ep/(DGe - DGom))
    return 1/(1-DGs/ep/(DGe - DGom))

