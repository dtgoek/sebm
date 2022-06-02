# High-resolution debris cover mapping using UAV-derived thermal imagery

The model uses high resolution land surface temperature maps (LST) obtained from a uncrewed aerial vehicle (UAV) to solve a surface energy balance model (SEBM) for spatially distributed debris thicknesses at various times of a day. The diurnal variability allows for consideration of the rate of change of heat storage.
The model follows the principle that all energy fluxes (W/m-2) at the earth surface must sum to zero:

S + L(LST) + H(LST) + dS(LST, d) + G(LST, d) = 0

S = net shortwave radiation \
L = net longwave radiation \
H = sensible heat fluxes \
dS = rate of change of heat storage \
G = ground heat flux \
LST = land surface temperature \
d = debris thickness

DATA INPUT
1) LST maps
2) DEM
3) Meteorological data (air temperature, wind speed and relative humidity)
4) Other parameters

Additional information about the dataset in data/dataset_information.txt


RUN THE MODEL

To run the model and produce debris thickness maps run both notebooks in following order

1) __01_warming_rate.ipynb__

warming_rate.ipynb is a wrapper for the functions stored in the file sinosoidal_regression.py. It performs a least squares regression of a sine function in each pixel to describe the diurnal temperature variation within a layer of debris, assuming that the mean debris temperature equals the arithmetic mean of the LST and the assumed debris ice interface temperature of 0°C. The warming/cooling rate is then generated by the first derivative with respect to time. The notebook wrapper creates a warmingrate folder in the directory containing the warming/cooling rate maps as *.geotiff files for each time of flight.

2) __02_SEBM_debristhickness.ipynb__

Once the warming/cooling rate maps have been created, run SEBM_debristhickness.ipynb. The notebook is a wrapper for the functions stored in energy_balance_model.py. It calculates the individual energy balance components and solves a quadratic equation for debris thickness in each pixel. Resulting maps are saved as *.geotiff.
