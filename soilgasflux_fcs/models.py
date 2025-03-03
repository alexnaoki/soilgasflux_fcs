
import numpy as np
def hm_model(t, cx, a, t0, c0):
    e = 2.71828
    return cx+(c0-cx)*e**(-a*(t-t0))

def hm_model_dcdt(t0, c0, a, cx, t):
    e = 2.71828
    dcdt = a*(cx - c0)*e**(-a*(t-t0))
    return dcdt

def linear_model(t, dcdt, c0):
    '''
    linear function

    y = dcdt * t + c0
    '''
    return dcdt*t + c0

def calculate_saturated_vapor_pressure(temperature):
    '''
    temperature: Celsius
    '''
    # Buck's equation
    e_s = 0.61121*np.exp((18.678 - temperature/234.5)*(temperature/(257.14 + temperature))) # kPa
    return e_s

def mole_fraction_water_vapor(temperature, humidity, pressure):
    e_s = calculate_saturated_vapor_pressure(temperature) #saturation vapor pressure in kPa
    e = e_s * (humidity/100) # vapor pressure in kPa
    
    X_h2o = (e / pressure)*1000 # m mol/mol
    return X_h2o

def dcdt_from_soilgasflux(volume, area, p0, w0, t0, soilgasflux):
    '''
    volume [cm-3]
    area [cm-2]
    p0 [kPa]
    w0 [mmol mol-1]
    t0 [Celsius]
    R = 8.314 [J⋅K−1⋅mol−1]
    soilgasflux [umol m-2 s-1]

    dcdt: [ppm s-1]
    '''
    R = 8.31446261815324 #J⋅K−1⋅mol−1

    dcdt = soilgasflux * R * area * (t0+273.15)/(10*volume*p0*(1-w0/1000))
    return dcdt
