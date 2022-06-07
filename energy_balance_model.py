import numpy as np
import sys
import os
import pandas as pd
from osgeo import gdal
from datetime import datetime, timedelta
from scipy import interpolate


'''
Functions to caclulate surface energy balance componentents and solve for debris thickness

Deniz Gök, German Research Centre for Geosciences, Potsdam
d_goek@gfz-potsdam.de


SHORTWAVE RADIATION AND SOLAR GEOMETRY
Python implementation of the R Package Insol from Javier G. Corrripio (2003)
https://doi.org/10.1080/713811744
'''


def JD(d):
    """
    Convert a datetime object to a julian date.

    Parameters:
    d (datetime object)

    Returns:
    JulianDay (float)
    """
    seconds_per_day = 86400
    JulianDay = d.timestamp() / seconds_per_day + 2440587.5

    return JulianDay

def eqtime(jd):
    """
    Calculate the equation of time.

    Parameters:
    jd (float)

    Returns:
    EqTime_deg (float)

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
    EqTime_deg = np.rad2deg(EqTime) * 4

    return EqTime_deg

def hourangle(jd, longitude, timezone):
    """
    Function for solar position calculation.

    Parameters:
    jd (float)
    longitude (float)
    timezone (int)

    Returns:
    omega_r (float)

    """
    hour = ((jd-np.floor(jd))*24+12) % 24
    time_offset=eqtime(jd)
    standard_meridian=timezone * 15
    delta_longitude_time=(longitude-standard_meridian)*24.0/360.0
    omega_r = np.pi * (
        ((hour + delta_longitude_time + time_offset / 60) / 12.0) - 1.0)
    return omega_r

def sunr(jd):
    """
    Earth-Sun distance in unit AU.

    Parameters:
    jd (float)

    Returns:
    R (float)

    """
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

    return R

def declination(jd):
    """
    Compute the declination of the sun on a given day.

    Parameters:
    jd (float)

    Returns:
    R (float)

    """
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

    delta_deg = np.rad2deg(delta)
    return delta_deg

def sunvector(jd,latitude,longitude,timezone):
    """
    Calulate unit vector in direction of sun from observer point

    Parameters:
    jd (float)
    latitude (float)
    longitude (float)
    timezone (int)

    Returns:
    sv (tuple)
    """
    omega=hourangle(jd,longitude,timezone)
    delta = np.radians(declination(jd))
    lat_rad = np.radians(latitude)
    svx = -np.sin(omega)*np.cos(delta)
    svy = np.sin(lat_rad)*np.cos(omega)*np.cos(delta)-np.cos(lat_rad)*np.sin(delta)
    svz = np.cos(lat_rad)*np.cos(omega)*np.cos(delta)+np.sin(lat_rad)*np.sin(delta)
    sv = (svx,svy,svz)
    return sv

def sunpos(sunv):
    """
    Azimith and zenith angles from unit vector to the sun from observer postion, Corripio (2003)

    Parameters:
    sv (tuble)

    Returns:
    sp (tuple)
    """

    azimuth = np.degrees(np.pi - np.arctan2(sunv[0],sunv[1]))
    zenith = np.degrees(np.arccos(sunv[2]))
    sp = (azimuth,zenith)
    return sp

#DEM CALCULATIONS - SHADOW

def gradient(grid, length_x, length_y=None):
    """
    Computes a unit vector normal to every grid cell in a digital elevation model (DEM).

    Parameters:
    grid (2darray) #DEM

    Returns:
    grad (numpy.ndarray) #tensor
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

    Parameters:
    grad (numpy.ndarray) #tensor
    sun_vector (tuple)

    Returns:
    hsh (2darray)
    """
    check_gradient(grad)

    hsh = (
        grad[:, :, 0] * sun_vector[0] +
        grad[:, :, 1] * sun_vector[1] +
        grad[:, :, 2] * sun_vector[2]
    )

    hsh = (hsh + abs(hsh)) / 2.
    return hsh

def _cast_shadow(row, col, rows, cols, dl, in_sun, inverse_sun_vector, normal_sun_vector, z):
    n = 0
    z_previous = -sys.float_info.max

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
def relative_humidity(Tair_K, Tdew_K):

    """
    Calucates the relativity humidity [%] based on the dewpoint and air temperature [K]
    https://bmcnoldy.rsmas.miami.edu/Humidity.html, acessed   (Alduchov et al. 1996)

    Parameters:
    Tair_K (float)
    Tdew_K (float)

    Returns:
    rh (float)
    """


    c  = 243.04
    b  = 17.625
    rh = 100 * np.e**( (c * b * (Tdew_K - Tair_K))
    / ((c + Tair_K) * (c + Tdew_K)) )

    return rh

def z2p(z):
    """
    Computes air pressure (hPa) for a given altitude [m] according to the standart atmosphere

    Parameters:
    z (float)

    Returns:
    zp (float)
    """
    P0=101325 #air pressure sea level
    T0=288.15 #standart temperature

    Earth_G = 9.80665 # acceleration due to gravity (m s-2)
    EarthR = 6.3756766E6 # average earths radius (m)
    Md = 28.966 # Molecular weight of dry air
    R_star = 8.3145 # Universal gas constant J/molK
    stlapse = -0.0065 # standard lapse rate K/m
    H1 = (EarthR * z) /(EarthR + z)
    HB = 0.0
    zp = P0*(T0/(T0+stlapse*(H1-HB)))**((Earth_G*Md)/(R_star*stlapse*1000))
    zp = zp/100.0
    return(zp)

def SatPresWatVap(Ta):
    """
    Computes saturation pressure of water vapour in air [hPa] for  given air temperature [K]
    Lowe (1977)

    Parameters:
    Ta (float)

    Returns:
    watervaporpressure (float)
    """

    tempcl = Ta
    a0     = 6984.505294
    a1     = -188.9039310
    a2     = 2.133357675
    a3     = -1.288580973e-2
    a4     = 4.393587233e-5
    a5     = -8.023923082e-8
    a6     = 6.136820929e-11

    watervaporpressure = a0+tempcl*(a1+tempcl*(a2+tempcl*(a3+tempcl*(a4+tempcl*(a5+tempcl*a6)))))

    return watervaporpressure

#ENERGY FLUXES AND DEBRIS THICKNESS

def insolation(zenith,jd,height,RH,tempK):
    """
    Computes direct and diffusive component of incoming shortwave radiation (Wm-2)
    without topographic effects

    Solar geometry is based von Iqbal (1983)
    Atmospheric transmissivity is based von Bird & Hulstrom (1981b)

    Parameters:
        zenith (float)
        jd (float)
        height (float)
        RH (float)
        tempK (float)

    Returns:
        radiation (tuple)
            In (float)
            Id (float)
    """

    O3 = 0.3
    visibility = 30
    alphag = 0.2

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
    radiation = (In, Id)

    return radiation

def calc_sin(dt_object,latitude,longitude,timezone, dem, res, relhum, TairK, shadow_surrounding=None):
    """
    Computes incident shortwave radiation for each pixel. Illumination intensity (hillshade)
    base on the DEM (or dem of surrounding terrain) determines how much radiation is received by
    each pixel.

    Parameters:
        dt_object (datetime object)
        latitude (float)
        longitude (float)
        timezone (int)
        dem (numpy 2darray)
        res (float)
        relhum (float)
        TairK (float)
        shadow_surrounding (None or 2darray)

    Returns:
        Iglobal (2darray)

    """

    jd = JD(dt_object)
    sv = sunvector(jd,latitude,longitude,timezone)
    zenith = sunpos(sv)[1]
    mean_elevation = np.nanmean(dem)
    cgr = gradient(dem, res, length_y=None)

    hsh = hill_shade(cgr,sv)
    hsh = (hsh+np.absolute(hsh))/2
    sh = project_shadows(dem, np.array(sv), res, dy=None)
    shade = hsh*sh

    #add shade from surrounding terrain
    if shadow_surrounding is not None:
        shade=shade*shadow_surrounding

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

def calc_snet(albedo, swin):
    """
    Computes net shortwave radiation (Wm-2) from albedo and incoming shortwave radiation (Wm-2)

    Parameters:
    albedo (float or 2darray)
    swin (float or 2darray)

    Returns:
    lnet (loat or 2darray)
    """

    snet = (1-albedo)*swin

    return snet

def calc_lout(emissivity, lst):
    """
    Computes outgoinf longwave radiation (Wm-2) based on emssivity and land surface temperature (K)

    Parameters:
    emissivity (float or 2darray)
    lst (float or 2darray)

    Returns:
    lout (loat or 2darray)
    """
    # convert C to K
    if np.nanmean(lst)<200:
        lst = lst+273.15

    Sboltzman_const=5.67e-08

    lout = Sboltzman_const * emissivity * (lst**4) # Stefan Boltzmann Law
    return lout

def calc_lnet(lin, lout):
    """
    Computes net longwave radiation (Wm-2)

    Parameters:
    lin (float or 2darray)
    lout (loat or 2darray)

    Returns:
    lnet (float or 2darray)
    """
    lnet = lin-lout
    return lnet

def calc_shf(TairK, lst, elevation, windspeed, zt, zu, roughness):
    """
    Computes sensible heat flux (Wm-2) based on land surface temperature (K), air temperature (K)
    wind speed in (ms-1) and surface roughness length (m), Nicholson and Benn (2006)

    Parameters:
        TairK (float or 2darray)
        lst (float or 2darray)
        elevation (float)
        windspeed (float)
        zt (int)
        zu (int)
        roughness (float)

    Returns:
        H (float or 2darray)
    """

    karmans_const = 0.41 # Kármáns constant
    c_air = 1010 # specific heat capacity air (J kg-1 K-1)
    P0 = 101325 # air pressure at sealevel in (Pa)
    rho_sl = 1.29 # air density sealeavel (kgm–3)
    P = z2p(elevation)*100 # air pressure (Pa)
    rho_local = rho_sl *  (P/P0) # air density local (kgm-3)

    A = (karmans_const**2) / (np.log(zu / roughness)*np.log(zt / roughness)) # dimensionless bulk transfer coefficient

    H = rho_local * c_air * windspeed * (TairK - lst) * A
    return H

def quadraticEq(a, b, c):
    """
    Quadratic formula, provides solution for quadratic equation
    """
    D = b**2 - (4*a*c)
    x0 = (-b + np.sqrt(D)) / (2*a)
    x1 = (-b - np.sqrt(D)) / (2*a)

    return x0, x1

def debristhickness_prediction(sw, lw, H, Ts, Ta, w_rate, params):
    '''
    Solves the surface energy balance for debris thickness. Net shortwave (Wm-2),
    net longwave (Wm-2), sensible heat flux (Wm-2), land surface temperature (K),
    warming rate (Ks-1) and air temperature (K) need to be determinded in advance.
    Additional parameters are stored in a dictionary.

    Assumes linear temperature gradient with the layer of debris.

    Parameters:
        sw (ndarray)
        lw (ndarray)
        H (ndarray)
        Ts (ndarray)
        Ta (ndarray)
        w_rate (ndarray)
        params (dictionary)

    Returns:
        x0 (ndarray)
    '''

    density_debris = params['density_d']
    heat_capacity_debris = params['c_d']
    thermal_conductivity = -params['k_eff']

    a = (density_debris * heat_capacity_debris * w_rate) * -1
    b = sw + lw + H
    c = thermal_conductivity * Ts

    x0, x1 = quadraticEq(a, b, c)

    return x0
