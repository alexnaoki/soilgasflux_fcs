import datetime as dt
import multiprocessing as mp

import numpy as np
import xarray as xr

from ._logging import get_logger
from .fcs import FCS
from .pareto import Pareto

logger = get_logger(__name__)

DEFAULT_METADATA = {'area': 314, 'volume': 6283}
DEFAULT_WINDOW_WALK = 10
DEFAULT_MIN_WINDOW_SIZE = 60
DEFAULT_MIN_DEADBAND = 0
DEFAULT_MAX_DEADBAND = 60
DEFAULT_N_MC = 8000


def _convert_keys_to_datetime(data):
    return {dt.datetime.strptime(k, '%Y-%m-%d_%H-%M-%S'): v for k, v in data.items()}


def _process_id(df, id, *, max_cutoff, mc, n_MC, metadata, sensor_precision=None):
    logger.info('Processing ID %s', id)
    df_id = df[df['id'] == id]
    fcs = FCS(df_data=df_id, chamber_id='test')
    fcs.settings(
        moving_window=True, window_walk=DEFAULT_WINDOW_WALK,
        min_window_size=DEFAULT_MIN_WINDOW_SIZE,
        min_deadband=DEFAULT_MIN_DEADBAND, max_deadband=DEFAULT_MAX_DEADBAND,
        max_cutoff=max_cutoff,
    )
    if mc:
        return fcs.run_MC(n=id, n_MC=n_MC, metadata=metadata,
                          sensor_precision=sensor_precision)
    return fcs.run(n=id, metadata=metadata)


class Multiprocessor:
    def __init__(self):
        pass

    # Retained for backwards compatibility with notebooks that reference these.
    def convert_keys_to_datetime(self, data):
        return _convert_keys_to_datetime(data)

    def process_id(self, df, id):
        return _process_id(df, id, max_cutoff=df['timedelta'].max(),
                           mc=False, n_MC=None, metadata=DEFAULT_METADATA)

    def process_id_MC(self, df, id, sensor_precision=None):
        return _process_id(df, id, max_cutoff=df['timedelta'].max(),
                           mc=True, n_MC=DEFAULT_N_MC, metadata=DEFAULT_METADATA,
                           sensor_precision=sensor_precision)

    def _run_day(self, df_1day, *, mc, n_MC, metadata, pool, sensor_precision=None):
        max_cutoff = df_1day['timedelta'].max()
        args = [(df_1day, n, max_cutoff, mc, n_MC, metadata, sensor_precision)
                for n in df_1day['id'].unique()]
        results = pool.starmap(_process_id_wrapper, args)
        combined = {k: v for result in results for k, v in result.items()}
        return _convert_keys_to_datetime(combined)

    def _build_dataset_2d(self, converted_data):
        times = list(converted_data.keys())
        first = converted_data[times[0]]
        cutoff = list(first['cutoff'])
        deadband = list(first['deadband'])

        keys = ['dcdt(HM)', 'dcdt(linear)', 'AIC(HM)', 'AIC(linear)',
                'RMSE(HM)', 'RMSE(linear)', 'R2(HM)', 'R2(linear)',
                'nRMSE(HM)', 'nRMSE(linear)']
        data_vars = {
            k: (['time', 'cutoff', 'deadband'],
                np.array([converted_data[t][k] for t in times]))
            for k in keys
        }
        return xr.Dataset(data_vars, coords={'time': times, 'deadband': deadband, 'cutoff': cutoff})

    def _build_dataset_3d(self, converted_data):
        times = list(converted_data.keys())
        first = converted_data[times[0]]
        cutoff = list(first['cutoff'])
        deadband = list(first['deadband'])
        n_MC = list(first['MC'])

        keys = ['dcdt(HM)', 'AIC(HM)', 'RMSE(HM)', 'R2(HM)', 'nRMSE(HM)', 'logprob(HM)']
        data_vars = {
            k: (['time', 'cutoff', 'deadband', 'MC'],
                np.array([converted_data[t][k] for t in times], dtype=np.float32))
            for k in keys
        }
        return xr.Dataset(data_vars, coords={'time': times, 'deadband': deadband,
                                             'cutoff': cutoff, 'MC': n_MC})

    def run(self, df, chamber_id, output_folder='./output'):
        logger.info('Multiprocessing started on %d CPUs', mp.cpu_count())
        with mp.Pool(mp.cpu_count()) as pool:
            ds = None
            for date in df['datetime'].dt.date.unique():
                logger.info('Processing date %s', date)
                df_1day = df[df['datetime'].dt.date == date]
                converted = self._run_day(df_1day, mc=False, n_MC=None,
                                          metadata=DEFAULT_METADATA, pool=pool)
                ds = self._build_dataset_2d(converted)
                ds.to_netcdf(f'{output_folder}/{chamber_id}_{date}.nc')
                logger.info('Saved NetCDF for %s', date)
        return ds

    def run_MC(self, df, chamber_id, output_folder='./output', save_netcdf=False,
               sensor_precision=None):
        '''
        sensor_precision: yerr (ppm) used in the MCMC likelihood. If None, the
        HM_model default (MEASUREMENT_SIGMA_PPM) is used. Scalar only in the
        multiprocessing path.
        '''
        logger.info('Multiprocessing (MC) started on %d CPUs', mp.cpu_count())
        with mp.Pool(mp.cpu_count()) as pool:
            ds = None
            for date in df['datetime'].dt.date.unique():
                logger.info('Processing date %s', date)
                df_1day = df[df['datetime'].dt.date == date]
                converted = self._run_day(df_1day, mc=True, n_MC=DEFAULT_N_MC,
                                          metadata=DEFAULT_METADATA, pool=pool,
                                          sensor_precision=sensor_precision)
                ds = self._build_dataset_3d(converted)
                if save_netcdf:
                    ds.to_netcdf(f'{output_folder}/{chamber_id}_{date}.nc')
                self.select_bestPareto(ds=ds, chamber_id=chamber_id, date=date,
                                       output_folder=output_folder)
        return ds

    def select_bestPareto(self, ds, chamber_id, date, output_folder=None):
        '''
        Return an xr.Dataset with the Pareto-optimal (deadband, cutoff) per time.
        '''
        logger.info('Selecting best Pareto front')
        deadband = ds.coords['deadband']
        cutoff = ds.coords['cutoff']
        n_MC = ds.coords['MC']

        times, deadbands, cutoffs, best_dcdt = [], [], [], []
        for time in ds.coords['time'].values:
            pa = Pareto(dsMC=ds.sel(time=time))
            norm_u, norm_l, flat_u, flat_l = pa.prepare_metrics()
            pareto_indices = pa.find_pareto_front(x=flat_u, y=flat_l,
                                                  maximize_x=False, maximize_y=False)
            try:
                best_x, best_y = pa.get_best_from_pareto(
                    pareto_indices=pareto_indices, metric_x=norm_u, metric_y=norm_l,
                )
            except (ValueError, IndexError) as e:
                logger.warning('Pareto front not found at time %s: %s', time, e)
                continue

            hist_dcdt = ds.sel(time=time,
                               deadband=deadband[best_y].values,
                               cutoff=cutoff[best_x].values)['dcdt(HM)']

            times.append(time)
            deadbands.append(deadband[best_y].values)
            cutoffs.append(cutoff[best_x].values)
            best_dcdt.append(hist_dcdt.values)

        ds_best = xr.Dataset(
            {
                'dcdt(HM)': (['time', 'MC'], np.array(best_dcdt, dtype=np.float32)),
                'best_deadband': (['time'], np.array(deadbands, dtype=int)),
                'best_cutoff': (['time'], np.array(cutoffs, dtype=int)),
            },
            coords={'time': times, 'MC': n_MC},
        )

        if output_folder is not None:
            ds_best.to_netcdf(f'{output_folder}/{chamber_id}_{date}_bestPareto.nc')
            logger.info('Saved best-Pareto NetCDF to %s', output_folder)
        return ds_best


def _process_id_wrapper(df, id, max_cutoff, mc, n_MC, metadata, sensor_precision=None):
    # Top-level wrapper so starmap can pickle it.
    return _process_id(df, id, max_cutoff=max_cutoff, mc=mc, n_MC=n_MC,
                       metadata=metadata, sensor_precision=sensor_precision)
