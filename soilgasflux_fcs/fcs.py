import numpy as np

from ._logging import get_logger
from .hm_model import HM_model
from .linear_model import LINEAR_model
from .metrics import calculate_AIC, rmse, r2, normalized_rmse
from .models import hm_model, linear_model

logger = get_logger(__name__)


class FCS:
    def __init__(self, df_data, chamber_id):
        self.max_time = df_data['timedelta'].max()
        self.df_data = df_data
        self.chamber_id = chamber_id

    def settings(self, moving_window=True, window_walk=10, min_window_size=30,
                 min_deadband=0, max_deadband=60, max_cutoff=None):
        '''
        max_cutoff: optional upper bound on cutoff grid. Defaults to the
        measurement's own max time; pass an explicit value to share a grid
        across measurements (needed for batch processing where per-measurement
        lengths differ).
        '''
        self.min_window_size = min_window_size
        if not moving_window:
            return
        upper = self.max_time if max_cutoff is None else max_cutoff
        self.deadband_options = np.arange(min_deadband, max_deadband, window_walk, dtype=int)
        self.cutoff_options = np.arange(min_deadband + min_window_size, int(upper), window_walk, dtype=int)

    def run_metrics(self, y_raw, y_model):
        return {
            'aic': calculate_AIC(y=y_raw, yhat=y_model, p=5),
            'rmse': rmse(y=y_raw, yhat=y_model),
            'r2': r2(y=y_raw, yhat=y_model),
            'nrmse': normalized_rmse(y=y_raw, yhat=y_model),
        }

    def _init_results_2d(self, n):
        XX, YY = np.meshgrid(self.deadband_options, self.cutoff_options)
        zeros = np.full_like(XX, np.nan, dtype=float)
        return {f'{n}': {
            'deadband': self.deadband_options,
            'cutoff': self.cutoff_options,
            'dcdt(HM)': zeros.copy(), 'dcdt(linear)': zeros.copy(),
            'AIC(HM)': zeros.copy(), 'AIC(linear)': zeros.copy(),
            'RMSE(HM)': zeros.copy(), 'RMSE(linear)': zeros.copy(),
            'R2(HM)': zeros.copy(), 'R2(linear)': zeros.copy(),
            'nRMSE(HM)': zeros.copy(), 'nRMSE(linear)': zeros.copy(),
        }}

    def _init_results_3d(self, n, n_MC):
        shape_3d = (len(self.cutoff_options), len(self.deadband_options), n_MC)
        return {f'{n}': {
            'deadband': self.deadband_options,
            'cutoff': self.cutoff_options,
            'MC': np.arange(n_MC),
            'dcdt(HM)': np.full(shape_3d, np.nan),
            'AIC(HM)': np.full(shape_3d, np.nan),
            'RMSE(HM)': np.full(shape_3d, np.nan),
            'R2(HM)': np.full(shape_3d, np.nan),
            'nRMSE(HM)': np.full(shape_3d, np.nan),
            'logprob(HM)': np.full(shape_3d, np.nan),
        }}

    def run(self, n, metadata={'area': 314, 'volume': 6283}):
        results = self._init_results_2d(n)
        r = results[f'{n}']
        y_raw_full = self.df_data['k30_co2'].values

        for n_deadband, deadband in enumerate(self.deadband_options):
            for n_cutoff, cutoff in enumerate(self.cutoff_options):
                if cutoff - deadband < self.min_window_size:
                    continue

                try:
                    hm = HM_model(raw_data=self.df_data, metadata=metadata)
                    dc_dt, C_0, cx, a, t0, _, _, _ = hm.calculate(deadband=deadband, cutoff=cutoff)
                    t = np.arange(deadband, cutoff, 1)
                    hm_co2 = hm_model(t=t, cx=cx, a=a, t0=t0, c0=C_0)
                    m = self.run_metrics(y_raw=y_raw_full[deadband:cutoff], y_model=hm_co2)
                    r['dcdt(HM)'][n_cutoff, n_deadband] = dc_dt
                    r['AIC(HM)'][n_cutoff, n_deadband] = m['aic']
                    r['RMSE(HM)'][n_cutoff, n_deadband] = m['rmse']
                    r['R2(HM)'][n_cutoff, n_deadband] = m['r2']
                    r['nRMSE(HM)'][n_cutoff, n_deadband] = m['nrmse']
                except Exception as e:
                    logger.warning('HM fit failed at deadband=%s cutoff=%s: %s', deadband, cutoff, e)

                try:
                    linear = LINEAR_model(raw_data=self.df_data, metadata=metadata)
                    dc_dt, C_0, _, _, _ = linear.calculate(deadband=deadband, cutoff=cutoff)
                    t = np.arange(deadband, cutoff, 1)
                    linear_co2 = linear_model(t=t, dcdt=dc_dt, c0=C_0)
                    m = self.run_metrics(y_raw=y_raw_full[deadband:cutoff], y_model=linear_co2)
                    r['dcdt(linear)'][n_cutoff, n_deadband] = dc_dt
                    r['AIC(linear)'][n_cutoff, n_deadband] = m['aic']
                    r['RMSE(linear)'][n_cutoff, n_deadband] = m['rmse']
                    r['R2(linear)'][n_cutoff, n_deadband] = m['r2']
                    r['nRMSE(linear)'][n_cutoff, n_deadband] = m['nrmse']
                except Exception as e:
                    logger.warning('Linear fit failed at deadband=%s cutoff=%s: %s', deadband, cutoff, e)

        return results

    @staticmethod
    def _broadcast_mc(value, n_MC):
        if isinstance(value, np.ndarray) and len(value) == n_MC:
            return value
        return np.full(n_MC, value)

    def _store_hm_slice(self, r, idx, dc_dt, logprob, metrics, n_MC):
        n_cutoff, n_deadband = idx
        r['dcdt(HM)'][n_cutoff, n_deadband, :] = dc_dt
        r['logprob(HM)'][n_cutoff, n_deadband, :] = logprob
        for key_in, key_out in [('aic', 'AIC(HM)'), ('rmse', 'RMSE(HM)'),
                                ('r2', 'R2(HM)'), ('nrmse', 'nRMSE(HM)')]:
            r[key_out][n_cutoff, n_deadband, :] = self._broadcast_mc(metrics[key_in], n_MC)

    def run_MC(self, n, n_MC, metadata={'area': 314, 'volume': 6283},
               sensor_precision=None):
        '''
        sensor_precision: yerr (ppm) used in the MCMC likelihood. If None,
        HM_model's default (MEASUREMENT_SIGMA_PPM) is used.
        '''
        results = self._init_results_3d(n, n_MC)
        r = results[f'{n}']
        y_raw_full = self.df_data['k30_co2'].values

        for n_deadband, deadband in enumerate(self.deadband_options):
            for n_cutoff, cutoff in enumerate(self.cutoff_options):
                if cutoff - deadband < self.min_window_size:
                    continue

                try:
                    hm = HM_model(raw_data=self.df_data, metadata=metadata)
                    dc_dt, C_0, cx, a, t0, _, _, _, logprob = hm.calculate_MC(
                        deadband=deadband, cutoff=cutoff, n=n_MC,
                        sensor_precision=sensor_precision,
                    )
                    t = np.arange(deadband, cutoff, 1)

                    cx = np.asarray(cx).reshape(n_MC, 1) if np.ndim(cx) <= 1 else cx
                    a = np.asarray(a).reshape(n_MC, 1) if np.ndim(a) <= 1 else a
                    t0 = np.asarray(t0).reshape(n_MC, 1) if np.ndim(t0) <= 1 else t0

                    TT, _ = np.meshgrid(t, np.arange(n_MC))
                    hm_co2_MC = hm_model(t=TT, cx=cx, a=a, t0=t0, c0=C_0)
                    m = self.run_metrics(y_raw=y_raw_full[deadband:cutoff], y_model=hm_co2_MC)
                    self._store_hm_slice(r, (n_cutoff, n_deadband), dc_dt, logprob, m, n_MC)
                except Exception as e:
                    logger.warning('HM MC fit failed at deadband=%s cutoff=%s: %s', deadband, cutoff, e)

        return results
