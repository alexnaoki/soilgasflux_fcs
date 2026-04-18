import warnings

import numpy as np
from lmfit import Model

from ._logging import get_logger
from .base_model import BaseChamberModel
from .mcmc import MCMC
from .models import hm_model, hm_model_dcdt

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

MCMC_WALKERS = 20
MEASUREMENT_SIGMA_PPM = 1.5  # sensor precision used as MCMC yerr


class HM_model(BaseChamberModel):
    def target_function(self, t, cx, a, t0, c0):
        return hm_model(t, cx, a, t0, c0)

    def dcdt(self, t_0, C_0, alpha, C_x, t):
        return hm_model_dcdt(t0=t_0, c0=C_0, a=alpha, cx=C_x, t=t)

    def gas_eeflux(self, P0, W0, T0, dc_dt):
        return self.gas_eeflux_v2(volume=self.volume, area=self.area,
                                  P0=P0, W0=W0, T0=T0, dc_dt=dc_dt)

    def fit_target_function_cutoff(self, t, gas_concentration, c_0, deadband, cutoff,
                                   display_results=True, pi=False):
        fmodel = Model(self.target_function)
        params = fmodel.make_params(cx=c_0, a=0.1, t0=0, c0=c_0)
        params['c0'].vary = False
        params['t0'].vary = True
        params['a'].min = 0
        params['t0'].max = cutoff
        params['t0'].min = 0

        try:
            result = fmodel.fit(gas_concentration[deadband:cutoff], params, t=t[deadband:cutoff])
        except Exception as e:
            logger.warning('HM target fit failed: %s', e)
            return None

        try:
            return {
                'parameters_best_fit': {
                    'cx': result.params['cx'].value,
                    'a': result.params['a'].value,
                    't0': result.params['t0'].value,
                    'c0': c_0,
                },
                'uncertainty': {
                    'cx': result.params['cx'].stderr,
                    'a': result.params['a'].stderr,
                    't0': result.params['t0'].stderr,
                },
            }
        except Exception as e:
            logger.warning('HM fit parameter extraction failed: %s', e)
            return None

    def _best_fit(self, deadband, cutoff):
        C_0 = self.C_0_calculated(self.co2.values[:10])
        result = self.fit_target_function_cutoff(
            self.timestamp.values, self.co2.values,
            np.float32(C_0)[0], deadband=deadband, cutoff=cutoff,
        )
        p = result['parameters_best_fit']
        return C_0, p['cx'], p['a'], p['t0'], result

    def calculate(self, deadband, cutoff):
        C_0, cx, a, t0, _ = self._best_fit(deadband, cutoff)
        X_h2o = self._water_vapor_mmol()
        t_window = self.timestamp[deadband:cutoff]
        dc_dt = self.dcdt(t0, C_0, a, cx, t_window).mean()
        soilgasflux_CO2 = self.gas_eeflux(
            P0=self.pressure.values[0],
            W0=X_h2o.values[0],
            T0=self.temperature.values[0],
            dc_dt=dc_dt,
        )
        return dc_dt, C_0, cx, a, t0, soilgasflux_CO2, deadband, cutoff

    def calculate_MC(self, deadband, cutoff, n, sensor_precision=None):
        '''
        sensor_precision: yerr (ppm) used in the MCMC likelihood. Defaults
        to MEASUREMENT_SIGMA_PPM. Pass a scalar to override globally or an
        array of length (cutoff - deadband) for a per-sample noise model.
        '''
        C_0, cx, a, t0, _ = self._best_fit(deadband, cutoff)

        sigma = MEASUREMENT_SIGMA_PPM if sensor_precision is None else sensor_precision
        yerr = np.asarray(sigma) if np.ndim(sigma) else np.ones(cutoff - deadband) * sigma

        mcmc = MCMC()
        _, flat_samples, logprob_samples = mcmc.run_mcmc(
            t=self.timestamp.values[deadband:cutoff],
            y=self.co2.values[deadband:cutoff],
            yerr=yerr,
            c0=C_0, cx_bf=cx, alpha_bf=a, t0_bf=t0,
            nwalkers=MCMC_WALKERS, nsteps=n,
        )

        t_median = np.median(self.timestamp.values[deadband:cutoff])
        dcdt_mcmc = self.dcdt(t_0=flat_samples[:, 2], C_0=C_0,
                              alpha=flat_samples[:, 0], C_x=flat_samples[:, 1],
                              t=t_median)

        random_index = np.random.choice(len(dcdt_mcmc), size=n)
        dc_dtMC = dcdt_mcmc[random_index]
        aMC = flat_samples[random_index, 0]
        cxMC = flat_samples[random_index, 1]
        t0MC = flat_samples[random_index, 2]
        logprob_selected = logprob_samples[random_index]

        X_h2o = self._water_vapor_mmol()
        soilgasflux_CO2MC = self.gas_eeflux(
            P0=self.pressure.head(1)[0],
            W0=X_h2o.head(1)[0],
            T0=self.temperature.head(1)[0],
            dc_dt=dc_dtMC,
        )

        return dc_dtMC, C_0, cxMC, aMC, t0MC, soilgasflux_CO2MC, deadband, cutoff, logprob_selected
