## Dataset information:

_ERA5_Land_aoi.csv_

ERA5-Land hourly reanalysis data on 30.08.2019, 7am - 11pm (UTM timezone) interpolated at 46.01, 7.46, Tsijiore-Noeve Glacier (TNG), Switzerland
Muñoz Sabater, J. (2019). ERA5-Land hourly data from 1981 to present. Climate Data Store. (Accessed on 28-Dec-2020), doi: 10.24381/cds.e2161bac

_Land Surface Temperature, LST (*.GeoTiFF)_

File naming: YYYYMMDD_HHMM_LST.tif
Data measured and processed by Deniz Gök

Orthomosaic maps of land surface temperature 30.08.2019 between 9am and 22pm (local time).
Radiometric thermal image data were measured from an unpiloted aerial vehicle (UAV), type:  DJI Mavic Pro using a FLIR Tau2 uncooled microbolometer (thermal sensor).
Overlapping thermal images have been previously processed into orthomosaic maps using Agisoft Metashape Professional 1.6.4

_20190830_class (*.GeoTiFF)_

File naming: YYYYMMDD_class.tif \
Data processed by Deniz Gök

Classification layer, based on the 15 h LST file. Supervised classification using a random forest algorithm with manually created training data.
Classes: (1) Debris surface and (2) ice surface. The classification map is used to assign emissivity and albedo values to the debris or ice surfaces during the processing.

_20190830_debristhickness_TNG.csv_

Subset of debris thickness measurements by Leif Anderson
Full dataset: https://doi.org/10.5281/zenodo.4317470

_CastShadowSurroundingTerrain (*.GeoTiFF)_

File naming : YYYYMMDD_HHMM_castshadow.tif
©swisstopo - Bundesamt für Landestopografie swisstopo

Cast shadows layers from surrounding terrain based on 5 m resolution swissALTI3D DEM. Cast shadows have been computed using the functions in energy_balance_model.py
Cast shadows have been resampled and clipped to match extent and resolution (0.16 m) of the LST data using QGIS Desktop 3.16.3

_20190830_DEM.tif_

Data measured and processed by Deniz Gök

DEM generated from overlapping RGB image data measured from an unpiloted aerial vehicle (UAV), type:  DJI Mavic Pro
RGB images have been processed using Agisoft Metashape Professional 1.6.4.
