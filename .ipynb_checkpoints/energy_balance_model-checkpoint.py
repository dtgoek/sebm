import numpy as np
import sys
import os
import pandas as pd
from osgeo import gdal
from datetime import datetime, timedelta
from scipy import interpolate


'''
Surface energy balance functions

implemented by Deniz Gök
d_goek@gfz-potsdam.de
'''

#SHORTWAVE RADIATION AND SOLAR GEOMETRY
'''
Python implementation of the R Package Insol from Javier G. Corrripio
https://doi.org/10.1080/713811744
https://github.com/cran/insol/blob/master/man/insol-package.Rd
'''


def JD(d):
    """Convert a datetime object to a julian date.
    A Julian date is the decimal number of days since January 1, 4713 BCE."""
    seconds_per_day = 86400
    return d.timestamp() / seconds_per_day + 2440587.5
    
def eqtime(jd):
    """Calculate the equation of time.
    See https://en.wikipedia.org/wiki/Equation_of_time.
    """
    jdc = (jd - 2451545.0) / 36525.0
    sec = 21.448 - jdc * (46.8150 + jdc * (0.00059 - jdc * 0.001813))
    e0 = 23.0 + (26.0 + (sec / 60.0)) / 60.0
    oblcorr = e0 + 0.00256 * np.cos(np.deg2rad(125.04 - 1934.136 * jdc))
    l0 = 280.46646 + jdc * (36000.76983 + jdc * 0.0003032)
    l0 = (l0 - 360 * (l0 // 360)) % 360
    gmas = 357.52911 + jdc * (35999.05029 - 0.0001537 * jdc)
    gmas = np.deg2rad(gmas)

    ecc = 0.016708634 - jdc * (0.000042037 + 0.0000001267 * jdc)
    y = (np.tan(np.deg2rad(oblcorr) / 2)) ** 2
    rl0 = np.deg2rad(l0)
    EqTime = y * np.sin(2 * rl0) \
        - 2.0 * ecc * np.sin(gmas) \
        + 4.0 * ecc * y * np.sin(gmas) * np.cos(2 * rl0)\
        - 0.5 * y * y * np.sin(4 * rl0) \
        - 1.25 * ecc * ecc * np.sin(2 * gmas)
    return np.rad2deg(EqTime) * 4
	 
def hourangle(jd, longitude, timezone):
    """Function for solar position calculation."""
    hour = ((jd-np.floor(jd))*24+12) % 24
    time_offset=eqtime(jd)
    standard_meridian=timezone * 15
    delta_longitude_time=(longitude-standard_meridian)*24.0/360.0
    omega_r = np.pi * (
        ((hour + delta_longitude_time + time_offset / 60) / 12.0) - 1.0)
    return omega_r
  
def sunr(jd):
    # Julian Centuries (Meeus, Astronomical Algorithms 1999. (24.1))
    T = (jd - 2451545)/36525.0
    # mean obliquity of the ecliptic (21.2)
    epsilon = (23+26/60.0+21.448/3600.0) - (46.8150/3600.0)*T - (0.00059/3600.0)*T**2 + (0.001813/3600.0)*T**3
    # mean anomaly of the Sun (24.3)
    M = 357.52910 + 35999.05030*T - 0.0001559*T**2 - 0.00000048*T**3
    # eccentricity of the Earth's orbit (24.4)
    e = 0.016708617 - 0.000042037*T - 0.0000001236*T**2
    # Sun's equation of center
    C = (1.914600 - 0.004817*T - 0.000014*T**2)*np.sin(np.radians(M)) + (0.019993 - 0.000101*T)*np.sin(2*np.radians(M)) +0.000290*np.sin(3*np.radians(M))
    # Sun's true anomaly
    v = M + C
    # Sun's Radius Vector (24.5)
    R = (1.000001018*(1-e**2))/(1 + e*np.cos(np.radians(v)))
    return(R)

def declination(jd):
    """Compute the declination of the sun on a given day."""
    jdc = (jd - 2451545.0) / 36525.0
    sec = 21.448 - jdc * (46.8150 + jdc * (0.00059 - jdc * .001813))
    e0 = 23.0 + (26.0 + (sec / 60.0)) / 60.0
    oblcorr = e0 + 0.00256 * np.cos(np.deg2rad(125.04 - 1934.136 * jdc))
    l0 = 280.46646 + jdc * (36000.76983 + jdc * 0.0003032)
    l0 = (l0 - 360 * (l0 // 360)) % 360
    gmas = 357.52911 + jdc * (35999.05029 - 0.0001537 * jdc)
    gmas = np.deg2rad(gmas)
    seqcent = np.sin(gmas) * (1.914602 - jdc * (0.004817 + 0.000014 * jdc)) + \
        np.sin(2 * gmas) * (0.019993 - 0.000101 * jdc) + np.sin(3 * gmas) * 0.000289

    suntl = l0 + seqcent
    sal = suntl - 0.00569 - 0.00478 * np.sin(np.deg2rad(125.04 - 1934.136 * jdc))
    delta = np.arcsin(np.sin(np.deg2rad(oblcorr)) * np.sin(np.deg2rad(sal)))
    return np.rad2deg(delta)
    
def sunvector(jd,latitude,longitude,timezone):
    # https://github.com/cran/insol/blob/master/R/sunvector.R
    omega=hourangle(jd,longitude,timezone)
    delta = np.radians(declination(jd))
    lat_rad = np.radians(latitude)
    svx = -np.sin(omega)*np.cos(delta)
    svy = np.sin(lat_rad)*np.cos(omega)*np.cos(delta)-np.cos(lat_rad)*np.sin(delta)
    svz = np.cos(lat_rad)*np.cos(omega)*np.cos(delta)+np.sin(lat_rad)*np.sin(delta)
    return(svx,svy,svz)
    
def sunpos(sunv):
    azimuth = np.degrees(np.pi - np.arctan2(sunv[0],sunv[1]))
    zenith = np.degrees(np.arccos(sunv[2]))
    return (azimuth,zenith)    

#DEM CALCULATIONS - SHADOW

def gradient(grid, length_x, length_y=None):
    """
    Computes a unit vector normal to every grid cell in a digital elevation model.
    
    Calculate the numerical gradient of a matrix in X, Y and Z directions.
    :param grid: Matrix or np.array (DEM)
    :param length_x: Length between two columns
    :param length_y: Length between two rows
    :return:
    
    https://github.com/tomderuijter/python-dem-shadows/blob/master/python_dem_shadows/gradient.py
    """
    if length_y is None:
        length_y = length_x

    assert len(grid.shape) == 2, "Grid should be a matrix."

    grad = np.empty((*grid.shape, 3))
    grad[:] = np.nan
    grad[:-1, :-1, 0] = 0.5 * length_y * (
        grid[:-1, :-1] - grid[:-1, 1:] + grid[1:, :-1] - grid[1:, 1:]
    )
    grad[:-1, :-1, 1] = 0.5 * length_x * (
        grid[:-1, :-1] + grid[:-1, 1:] - grid[1:, :-1] - grid[1:, 1:]
    )
    grad[:-1, :-1, 2] = length_x * length_y

    # Copy last row and column
    grad[-1, :, :] = grad[-2, :, :]
    grad[:, -1, :] = grad[:, -2, :]

    area = np.sqrt(
        grad[:, :, 0] ** 2 +
        grad[:, :, 1] ** 2 +
        grad[:, :, 2] ** 2
    )
    for i in range(3):
        grad[:, :, i] /= area
    return grad

def check_gradient(grad):
    assert len(grad.shape) == 3 and grad.shape[2] == 3, \
        "Gradient should be a tensor with 3 layers."

def hill_shade(grad, sun_vector):
    """
    Compute the intensity of illumination on a surface given the sun position.
    :param grad:
    :param sun_vector:
    :return:
    """
    check_gradient(grad)

    hsh = (
        grad[:, :, 0] * sun_vector[0] +
        grad[:, :, 1] * sun_vector[1] +
        grad[:, :, 2] * sun_vector[2]
    )
    # Remove negative incidence angles - indicators for self-shading.
    hsh = (hsh + abs(hsh)) / 2.

    return hsh

def _cast_shadow(row, col, rows, cols, dl, in_sun, inverse_sun_vector,
                 normal_sun_vector, z):
    n = 0
    z_previous = -sys.float_info.max
    """
    Uitleg Hans
    ------------
    Inverse sun vector staat je toe de projectie op projectievlak terug te rekenen
    naar de originele matrix.
    De dx en dy representeren tot hoever de schaduw rijkt vanaf de huidige cel.
    Normal sun vector is het vlak loodrecht op de richting van de zon.
    """

    while True:
        # Calculate projection offset
        dx = inverse_sun_vector[0] * n
        dy = inverse_sun_vector[1] * n
        col_dx = int(round(col + dx))
        row_dy = int(round(row + dy))
        if (col_dx < 0) or (col_dx >= cols) or (row_dy < 0) or (row_dy >= rows):
            break

        vector_to_origin = np.zeros(3)
        vector_to_origin[0] = dx * dl
        vector_to_origin[1] = dy * dl
        vector_to_origin[2] = z[col_dx, row_dy]
        z_projection = np.dot(vector_to_origin, normal_sun_vector)

        if z_projection < z_previous:
            in_sun[col_dx, row_dy] = 0
        else:
            z_previous = z_projection
        n += 1
 
def _normalize_sun_vector(sun_vector):
    normal_sun_vector = np.zeros(3)
    normal_sun_vector[2] = np.sqrt(sun_vector[0] ** 2 + sun_vector[1] ** 2)
    normal_sun_vector[0] = -sun_vector[0] * sun_vector[2] / normal_sun_vector[2]
    normal_sun_vector[1] = -sun_vector[1] * sun_vector[2] / normal_sun_vector[2]
    return normal_sun_vector

def _invert_sun_vector(sun_vector):
    return -sun_vector / max(abs(sun_vector[:2])) 
                
def project_shadows(dem, sun_vector, dx, dy=None):
    """Cast shadows on the DEM from a given sun position."""

    if dy is None:
        dy = dx

    inverse_sun_vector = _invert_sun_vector(sun_vector)
    normal_sun_vector = _normalize_sun_vector(sun_vector)

    rows, cols = dem.shape
    z = dem.T

    # Determine sun direction.
    if sun_vector[0] < 0:
        # The sun shines from the West.
        start_col = 1
    else:
        # THe sun shines from the East.
        start_col = cols - 1

    if sun_vector[1] < 0:
        # The sun shines from the North.
        start_row = 1
    else:
        # The sun shines from the South.
        start_row = rows - 1

    in_sun = np.ones_like(z)
    # Project West-East
    row = start_row
    for col in range(cols):
        _cast_shadow(row, col, rows, cols, dx, in_sun, inverse_sun_vector,
                     normal_sun_vector, z)

    # Project North-South
    col = start_col
    for row in range(rows):
        _cast_shadow(row, col, rows, cols, dy, in_sun, inverse_sun_vector,
                     normal_sun_vector, z)
    return in_sun.T

#METEO
def relative_humidity(TaK, TdK):
    '''
    Tak: air temperature  [K]
    TdK: dew point temperature [K]
    
    https://earthscience.stackexchange.com/questions/16570/how-to-calculate-relative-humidity-from-temperature-dew-point-and-pressure

    https://bmcnoldy.rsmas.miami.edu/Humidity.html
    Alduchov, O. A., and R. E. Eskridge, 1996: Improved Magnus' form approximation of saturation vapor pressure. J. Appl. Meteor., 35, 601–609.
    August, E. F., 1828: Ueber die Berechnung der Expansivkraft des Wasserdunstes. Ann. Phys. Chem., 13, 122–137.
    Magnus, G., 1844: Versuche über die Spannkräfte des Wasserdampfs. Ann. Phys. Chem., 61, 225–247. 
    '''
    
    c  = 243.04
    b  = 17.625
    rh = 100 * np.e**( (c * b * (TdK - TaK)) 
    / ((c + TaK) * (c + TdK)) )
    
    return rh

def z2p(z):
    P0=101325
    T0=288.15

    #elevato to pressure
    Earth_G = 9.80665# Acceleration due to gravity (m s-2)
    EarthR = 6.3756766E6 # Average earths radius (m)
    Md = 28.966 # Molecular weight of dry air
    R_star = 8.3145 # Universal gas constant J/molK
    stlapse = -0.0065 # standard lapse rate K/m
    H1 = (EarthR * z) /(EarthR + z)
    HB = 0.0
    zp = P0*(T0/(T0+stlapse*(H1-HB)))**((Earth_G*Md)/(R_star*stlapse*1000))
    zp = zp/100.0  
    return(zp)

def SatPresWatVap(Ta):
    '''
    I: air temperature [K]
    O: saturation pressure of water vapour in air [hPa]
    
    copied from Corripio, 2013 github:
    https://github.com/cran/insol/blob/master/R/wvapsat.R
    
    based on:
        Lowe, P. R.: 1977, An approximating polynomial for the computation of saturation vapor pressure,
        Journal of Applied Meteorology, 16, 100-103.      
    '''

    tempcl = Ta
    a0     = 6984.505294	
    a1     = -188.9039310
    a2     = 2.133357675	
    a3     = -1.288580973e-2	
    a4     = 4.393587233e-5
    a5     = -8.023923082e-8
    a6     = 6.136820929e-11
    
    return( a0+tempcl*(a1+tempcl*(a2+tempcl*(a3+tempcl*(a4+tempcl*(a5+tempcl*a6))))) )    

#ENERGY FLUXES AND DEBRIS THICKNESS

def insolation(zenith,jd,height,RH,tempK):
    
    '''
    based on:
    Iqbal, M. (1983) An Introduction to Solar Radiation
    Bird, R. E. and Hulstrom, R. L. (1981b) A simplified clear sky model for direct and diffuse insolation on horizontal surfaces
    '''
    O3 = 0.3
    visibility = 30
    alphag = 0.2 
    
    print('zenith:' , zenith)
    Isc = 1361.8   # solar constant (Wm^(-2)) (1)
    theta = np.radians(zenith)
    ssctalb = 0.9  # single scattering albedo (aerosols)(Iqbal, 1983)
    Fc = 0.84      # ratio of forward to total energy scattered (Iqbal, 1983)
    Pz = z2p(height)
    Mr = 1.0/(np.cos(theta)+0.15*((93.885-zenith)**(-1.253)))
    Ma = Mr*Pz/1013.25
    wvap_s =  SatPresWatVap(tempK) #** Use Lowe(1977) Lowes polynomials for vapor pressure
    Wprec = 46.5*(RH/100.0)*wvap_s/tempK  #Prata 1996
    rho2 = (1/sunr(jd))**2
    #rho2 =eccentricity(jd,tz)
    #rho2=1 + 0.033*np.cos((2*np.pi*242)/365)
    TauR = np.exp((-.09030*(Ma**0.84) )*(1.0+Ma-(Ma**1.01)) )
    TauO = 1.0-( ( 0.1611*(O3*Mr)*(1.0+139.48*(O3*Mr))**(-0.3035) )-0.002715*(O3*Mr)*( 1.0+0.044*(O3*Mr)+0.0003*(O3*Mr)**2 )**(-1))
    TauG = np.exp(-0.0127*(Ma**0.26))
    TauW = 1.0-2.4959*(Wprec*Mr)*( (1.0+79.034*(Wprec*Mr))**0.6828 + 6.385*(Wprec*Mr) )**(-1)
    TauA = ( 0.97-1.265*(visibility**(-0.66)) )**(Ma**0.9)   #Machler, 1983
    TauTotal = TauR*TauO*TauG*TauW*TauA     
    
    In = 0.9751*rho2*Isc*TauTotal
    
    tauaa = 1.0-(1.0-ssctalb)*(1.0-Ma+Ma**1.06)*(1.0-TauA)
    Idr = 0.79*rho2*Isc*np.cos(theta)*TauO*TauG*TauW*tauaa*0.5*(1.0-TauR)/(1.0-Ma+Ma**(1.02))
    tauas = (TauA)/tauaa
    Ida = 0.79*rho2*Isc*np.cos(theta)*TauO*TauG*TauW*tauaa*Fc*(1.0-tauas)/(1.0-Ma+Ma**1.02)
    alpha_atmos = 0.0685+(1.0-Fc)*(1.0-tauas)
    Idm = (In*np.cos(theta)+Idr+Ida)*alphag*alpha_atmos/(1.0-alphag*alpha_atmos)
    Id = Idr+Ida+Idm
   

    return (In, Id)

def calc_sin(dt_object,latitude,longitude,timezone, dem, res, relhum, TairK, shadow_surrounding=None):
    
    '''
    zenith, jd, elevation, vis, relhum, TairK, O3, albedo_surrounding, shadow_surrounding (cast shadow)
    '''
    jd = JD(dt_object)
    

    sv = sunvector(jd,latitude,longitude,timezone)
    zenith = sunpos(sv)[1]
    
    
    mean_elevation = np.nanmean(dem) #np.mean(dem[dem>0])
    #res = dem.geotransform[1] # richdem required
    
    cgr = gradient(dem, res, length_y=None)
    
    #Computes the intensity of illumination over a surface (DEM), according to the sun

    hsh = hill_shade(cgr,sv)
    #https://meteoexploration.com/blog/index.php/2022/01/09/hands-on-solar-radiation/
    hsh = (hsh+np.absolute(hsh))/2
    sh = project_shadows(dem, np.array(sv), res, dy=None)
    shade = hsh*sh
    
    if shadow_surrounding is not None:
        shade= shade*shadow_surrounding
    
    insol = insolation(zenith, jd, mean_elevation, relhum, TairK)
    
    if zenith<90:
        direct = insol[0]
        diffuse = insol[1]
    else:
        direct = 0 
        diffuse = 0
    
    print('Id: ', direct)
    print('In: ', diffuse)
    
    Iglobal = np.zeros(dem.shape)
    Iglobal = Iglobal + ( direct * shade + diffuse)
    
    return  Iglobal

def calc_snet(albedo, insolation):
    snet = (1-albedo)*insolation
    return snet

def calc_lout(emissivity, lst):
    
    if np.nanmean(lst)<200:
        
        lst = lst+273.15
    
    stefanbolzman_cont=5.67e-08
  
    # emissivity correction was applied while trad to tkin conversion
    lout = stefanbolzman_cont * emissivity * (lst**4)
    return lout
    
def calc_lnet(lin, lout):
    return lin-lout  

def calc_shf(TairK, lst, elevation, windspeed, zt, zu, roughness):

    '''
    TairK [K]: air temperature
    lst [K]: land surface temperature
    elevation [m]: mean DEM elevation
    windspeed [m/s]: from ERA5
    zt [m]: air temperature measurement height  (2m for ERA5)
    zu [m]: wind speed measurement height (10m for ERA5)
    roughness [-]: surface roughness length  (0.016)
    
    
    '''
    karmans_const = 0.41
    c_air = 1010 # specific heat capacity air
    
    P0 = 101325 # air pressure at sealevel in Pa
    rho_sl = 1.29 # air density sealeavel [kgm–3] Nicholson and Benn, 2006 
    
    P = z2p(elevation)*100 # air pressure at mean DEM elevation in Pa 
    #print('pressure local:' , P0)   
    
    rho_local = rho_sl *  (P/P0) #[kgm–3] air density local
    #print('air density local: ', rho_local)
    
    A = (karmans_const**2) / (np.log(zu / roughness)*np.log(zt / roughness)) # dimensionless bulk transfer coefficient

    H = rho_local * c_air * windspeed * (TairK - lst) * A  
    #print('mean H:', np.nanmean(H))
    return H
    
def quadraticEq(a, b, c):
    
    D = b**2 - (4*a*c)
    
    x0 = (-b + np.sqrt(D)) / (2*a)
    x1 = (-b - np.sqrt(D)) / (2*a)
    
    return x0, x1
    
def debristhickness_prediction(sw, lw, H, Ts, Ta, w_rate, params):
    '''
    Function to estimate debris thickness by solving quadratic equation. If warming rate is close to 
    threshold, debris thickness is estimated using a linear model
    
    Ts and Ta in °C
    '''
    
    
    density_debris = params['density_d']
    heat_capacity_debris = params['c_d']
    thermal_conductivity = -params['k_eff']
    
    a = (density_debris * heat_capacity_debris * w_rate) * -1
    b = sw + lw + H
    c = thermal_conductivity * Ts
    
    x0, x1 = quadraticEq(a, b, c)
    
    return x0 #, (a, b, c)
    
def hillshade(array,azimuth,angle_altitude):
    azimuth = 360.0 - azimuth 
    
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.
 
    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    
    return 255*(shaded + 1)/2
    