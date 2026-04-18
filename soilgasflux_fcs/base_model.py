import numpy as np
from sklearn.linear_model import LinearRegression

from ._logging import get_logger
from .models import (
    calculate_saturated_vapor_pressure,
    mole_fraction_water_vapor,
    soilgasflux,
)

logger = get_logger(__name__)


class BaseChamberModel:
    '''
    Shared data loading, water-vapor, and flux-conversion logic for the HM
    and linear chamber models. Subclasses must implement `target_function`,
    `fit_target_function_cutoff`, and `calculate[/`calculate_MC`].
    '''

    def __init__(self, raw_data, metadata, using_rpi=True):
        self.area = metadata['area']
        self.volume = metadata['volume']
        self.using_rpi = using_rpi

        if using_rpi:
            self.timestamp = raw_data['timedelta']
            self.temperature = raw_data['si_temperature']
            self.humidity = raw_data['si_humidity']
            self.pressure = raw_data['bmp_pressure'] / 1000  # kPa
            self.co2 = raw_data['k30_co2']
        else:
            self.timestamp = raw_data['timestamp']
            self.temperature = raw_data['chamber_t']
            self.pressure = raw_data['chamber_p']
            self.co2 = raw_data['co2']
            self.X_h2o = raw_data['h2o']

    def calculate_saturated_vapor_pressure(self, temperature):
        return calculate_saturated_vapor_pressure(temperature)

    def mole_fraction_water_vapor(self, temperature, humidity, pressure):
        return mole_fraction_water_vapor(temperature, humidity, pressure)

    def _water_vapor_mmol(self):
        if self.using_rpi:
            return self.mole_fraction_water_vapor(self.temperature, self.humidity, self.pressure)
        return self.X_h2o

    def C_0_calculated(self, gas_concentration):
        '''
        Linear-regression estimate of the initial concentration from the first
        few samples (typically the first 10 points).
        '''
        n = np.shape(gas_concentration)[0]
        y = np.array(gas_concentration).reshape((n, 1))
        x = np.arange(1, n + 1).reshape((n, 1))
        regression = LinearRegression(fit_intercept=True).fit(x, y)
        return regression.intercept_

    def gas_eeflux_v2(self, volume, area, P0, W0, T0, dc_dt):
        return soilgasflux(volume=volume, area=area, p0=P0, w0=W0, t0=T0, dcdt=dc_dt)
