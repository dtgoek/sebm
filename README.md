# surface energy balance model" 

The model uses high resolution land surface tempertaure maps (LST) to solve a surface energy balance model (sebm) for debris thickness.

The model follows the princible that all energy fluxes at the earth surface must sum to zero:

S+L(LST)+H(LST)+dS(LST, d)+G(LST, d)=0

S = shortwave radiation
L = longwave radiation
H = sensible heat fluxes
dS = rate of change of heat storage
G = ground heat flux
LST = land surface temperature
d = debris thickness

RUN THE MODEL

The model requires LST maps, a DEM and meteorological input (air temperature, wind speed and relative humidity)


The functions to calculate the debris thickness 

1) warming_rate.ipynb
2) SEBM_debristhickness.ipynb

