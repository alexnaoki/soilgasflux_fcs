import numpy as np
from lmfit import Model

from ._logging import get_logger
from .base_model import BaseChamberModel
from .models import linear_model

logger = get_logger(__name__)


class LINEAR_model(BaseChamberModel):
    def target_function(self, t, dcdt, c0):
        return linear_model(t, dcdt, c0)

    def fit_target_function_cutoff(self, t, gas_concentration, c_0, deadband, cutoff):
        fmodel = Model(self.target_function)
        params = fmodel.make_params(dcdt=1, c0=c_0)
        params['c0'].vary = False

        try:
            result = fmodel.fit(gas_concentration[deadband:cutoff], params, t=t[deadband:cutoff])
        except Exception as e:
            logger.warning('Linear target fit failed: %s', e)
            return None

        try:
            return {
                'parameters_best_fit': {
                    'dcdt': result.best_values['dcdt'],
                    'c0': result.best_values['c0'],
                },
                'uncertainty': {
                    'dcdt': result.params['dcdt'].stderr,
                    'c0': result.params['c0'].stderr,
                },
            }
        except Exception as e:
            logger.warning('Linear fit parameter extraction failed: %s', e)
            return None

    def _initial_conditions(self, deadband):
        if self.using_rpi:
            X_h2o = self.mole_fraction_water_vapor(self.temperature, self.humidity, self.pressure)[deadband]
        else:
            X_h2o = self.X_h2o
        return X_h2o, self.temperature[0], self.pressure[0]

    def calculate(self, deadband, cutoff):
        X_h2o, T0, P0 = self._initial_conditions(deadband)
        C_0 = self.co2.values[0]
        result = self.fit_target_function_cutoff(
            t=self.timestamp.values, gas_concentration=self.co2.values,
            c_0=C_0, deadband=deadband, cutoff=cutoff,
        )
        dcdt = result['parameters_best_fit']['dcdt']
        c0 = result['parameters_best_fit']['c0']
        soilgasflux_CO2 = self.gas_eeflux_v2(
            volume=self.volume, area=self.area,
            P0=P0, W0=X_h2o, T0=T0, dc_dt=dcdt,
        )
        return dcdt, c0, soilgasflux_CO2, deadband, cutoff

    def calculate_MC(self, deadband, cutoff, n=1000):
        X_h2o, T0, P0 = self._initial_conditions(deadband)
        C_0 = self.co2.values[0]
        result = self.fit_target_function_cutoff(
            t=self.timestamp.values, gas_concentration=self.co2.values,
            c_0=C_0, deadband=deadband, cutoff=cutoff,
        )
        dcdt = result['parameters_best_fit']['dcdt']
        c0 = result['parameters_best_fit']['c0']
        sigma_dcdt = result['uncertainty']['dcdt']
        sigma_c0 = result['uncertainty']['c0']

        dcdt_MC = dcdt + np.random.normal(0, sigma_dcdt, n)
        c0_MC = c0 + np.random.normal(0, sigma_c0, n)

        soilgasflux_CO2MC = self.gas_eeflux_v2(
            volume=self.volume, area=self.area,
            P0=P0, W0=X_h2o, T0=T0, dc_dt=dcdt_MC,
        )
        return dcdt_MC, c0_MC, soilgasflux_CO2MC, deadband, cutoff
