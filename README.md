## An idealized OMZ ecosystem model for the major N2O production pathway in marine oxygen minimum zones
This repository contains a 0-D virtual chemostat model simulating nitrogen cycling processes within marine oxygen minimum zones (OMZs). The model is designed to examine the production and consumption of nitrogen species, especially nitrous oxide (N₂O). 

### 1. model_N2O.py 
The model. Equations determining the changes in nutrient concentration and microbial biomass are included in this file.

### 2. traits.py 
Contains model parameters for traits of different microbial functional groups and their references. For example, this file contains nutrient uptake kinetics such as the maximum uptake rate and the half-saturation constant, and biomass yield for each microbial functional type.

### 3. call_model_N2O_clean.py
This is the main file used to run model simulations and visualize results. It also contains the initialization of parameters and initial conditions. 

### 4. files ended with star_Xin.py
Contain functions to calculate different R* that are used in the model_N2O.py

### plot_data.ipynb
Codes used to plot the N2O and O2 map for the Eastern Tropical North Pacific Oxygen Minimum zone.

### !ExperimentalData_110425.xlsx
This file contains all new experimental data obtained in this study from the Eastern Tropical North Pacific Oxygen Minimum zone.
