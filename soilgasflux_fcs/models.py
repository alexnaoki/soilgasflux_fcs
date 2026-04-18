import numpy as np

R_GAS = 8.31446261815324  # J K^-1 mol^-1


def hm_model(t, cx, a, t0, c0):
    return cx + (c0 - cx) * np.exp(-a * (t - t0))


def hm_model_dcdt(t0, c0, a, cx, t):
    return a * (cx - c0) * np.exp(-a * (t - t0))


def linear_model(t, dcdt, c0):
    return dcdt * t + c0


def calculate_saturated_vapor_pressure(temperature):
    # Buck's equation; temperature in Celsius, returns kPa
    return 0.61121 * np.exp((18.678 - temperature / 234.5) * (temperature / (257.14 + temperature)))


def mole_fraction_water_vapor(temperature, humidity, pressure):
    e_s = calculate_saturated_vapor_pressure(temperature)
    e = e_s * (humidity / 100)
    return (e / pressure) * 1000  # mmol/mol


def dcdt_from_soilgasflux(volume, area, p0, w0, t0, soilgasflux):
    '''
    volume [cm^3], area [cm^2], p0 [kPa], w0 [mmol/mol], t0 [Celsius]
    soilgasflux [umol m^-2 s^-1] -> dcdt [ppm s^-1]
    '''
    return soilgasflux * R_GAS * area * (t0 + 273.15) / (10 * volume * p0 * (1 - w0 / 1000))


def soilgasflux(volume, area, p0, w0, t0, dcdt):
    '''
    volume [cm^3], area [cm^2], p0 [kPa], w0 [mmol/mol], t0 [Celsius]
    dcdt [ppm s^-1] -> soilgasflux [umol m^-2 s^-1]
    '''
    return (10 * volume * p0 * (1 - w0 / 1000)) * dcdt / (R_GAS * area * (t0 + 273.15))
