import datetime as dt
import logging
import multiprocessing as mp
import os
import tempfile
from contextlib import contextmanager
from functools import partial
from uuid import uuid4

import numpy as np
import xarray as xr

from ._logging import get_logger
from .fcs import FCS
from .pareto import Pareto

logger = get_logger(__name__)


def _init_worker_logging(log_path):
    # Runs once per worker process. Attaches a FileHandler pointing at the shared
    # log so worker records (including tracebacks from _process_id_wrapper) land
    # in the log file, not the parent's stderr (which Jupyter surfaces in the
    # notebook). Defensively strips any StreamHandler on the package logger in
    # case an application configured one.
    pkg_logger = logging.getLogger('soilgasflux_fcs')
    for h in list(pkg_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            pkg_logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s pid=%(process)d %(name)s: %(message)s'))
    pkg_logger.addHandler(fh)
    pkg_logger.setLevel(logging.INFO)
    pkg_logger.propagate = False

DEFAULT_METADATA = {'area': 314, 'volume': 6283}
DEFAULT_WINDOW_WALK = 10
DEFAULT_MIN_WINDOW_SIZE = 60
DEFAULT_MIN_DEADBAND = 0
DEFAULT_MAX_DEADBAND = 60
DEFAULT_N_MC = 8000


def _convert_keys_to_datetime(data, id_to_datetime=None):
    id_to_datetime = id_to_datetime or {}
    converted = {}
    for k, v in data.items():
        try:
            timestamp = dt.datetime.strptime(k, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            if k not in id_to_datetime:
                raise ValueError(
                    f"Measurement id {k!r} is not timestamp-formatted and "
                    "no datetime mapping was provided."
                )
            timestamp = id_to_datetime[k]
        converted[timestamp] = v
    return converted


def _id_to_datetime(df_1day):
    return (
        df_1day.groupby('id')['datetime'].min()
        .apply(lambda value: value.to_pydatetime())
        .to_dict()
    )


def _expected_times(df_1day):
    # The time coordinate each measurement of the day gets in the output.
    ids = dict.fromkeys(df_1day['id'].unique())
    return list(_convert_keys_to_datetime(ids, id_to_datetime=_id_to_datetime(df_1day)))


def _time_set(values):
    # NumPy datetime hashes depend on the unit even for equal timestamps.
    return set(np.asarray(values, dtype='datetime64[ns]'))


def _output_covers(path, times, mc):
    '''
    True when the NetCDF at path opens, matches the run type (with or without
    an MC dimension) and its time coordinate holds every time in times. A
    missing, unreadable or partial file returns False so the day is redone.
    '''
    if not os.path.exists(path):
        return False
    try:
        with xr.open_dataset(path) as existing:
            if ('MC' in existing.dims) != mc:
                return False
            stored = _time_set(existing['time'].values)
    except (OSError, ValueError, KeyError):
        return False
    return _time_set(times) <= stored


def _mc_pair_covers(path, best_path, times, pareto_only):
    try:
        with xr.open_dataset(path) as saved, xr.open_dataset(best_path) as best:
            if 'MC' not in saved.dims or 'MC' not in best.dims or 'time' not in best.coords:
                return False
            if saved.attrs.get('pareto_run_id') != best.attrs.get('pareto_run_id'):
                return False
            if pareto_only:
                if _time_set(saved['window_time'].values) != _time_set(best['time'].values):
                    return False
            return _time_set(times) <= _time_set(saved['time'].values)
    except (OSError, ValueError, KeyError):
        return False


def _write_netcdf(ds, path):
    # Publish a complete file so a concurrent reader or a retry after an
    # interrupted write cannot mistake a partial day's coordinates for success.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path)) as tmp:
        temporary_path = os.path.join(tmp, 'output.nc')
        ds.to_netcdf(temporary_path)
        os.replace(temporary_path, path)


def _select_windows(ds, windows, dim):
    """Index paired (time, cutoff, deadband) positions along one output dimension."""
    times, cutoffs, deadbands = np.asarray(windows, dtype=int).reshape(-1, 3).T
    return ds.isel(time=(dim, times), cutoff=(dim, cutoffs), deadband=(dim, deadbands))


def _process_id(df_id, id, *, max_cutoff, mc, n_MC, metadata, sensor_precision=None):
    logger.info('Processing ID %s', id)
    # Model windows and initial-condition lookups use offsets within a closure.
    fcs = FCS(df_data=df_id.reset_index(drop=True), chamber_id='test')
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
    """Fit batches locally, or on the Dask scheduler at scheduler_address."""

    def __init__(self, scheduler_address=None):
        self.scheduler_address = scheduler_address.strip() if scheduler_address else None
        # Dates the last run/run_MC call left alone because skip_existing found
        # their outputs complete.
        self.skipped_dates = []

    # Retained for backwards compatibility with notebooks that reference these.
    def convert_keys_to_datetime(self, data):
        return _convert_keys_to_datetime(data)

    def process_id(self, df, id):
        return _process_id(df[df['id'] == id], id, max_cutoff=df['timedelta'].max(),
                           mc=False, n_MC=None, metadata=DEFAULT_METADATA)

    def process_id_MC(self, df, id, sensor_precision=None):
        return _process_id(df[df['id'] == id], id, max_cutoff=df['timedelta'].max(),
                           mc=True, n_MC=DEFAULT_N_MC, metadata=DEFAULT_METADATA,
                           sensor_precision=sensor_precision)

    @contextmanager
    def _measurement_mapper(self, log_path):
        if self.scheduler_address:
            from ._dask import dask_mapper
            with dask_mapper(self.scheduler_address, log_path) as map_measurements:
                yield map_measurements
        else:
            logger.info('Multiprocessing started on %d CPUs', mp.cpu_count())
            options = {}
            if log_path is not None:
                options = {'initializer': _init_worker_logging, 'initargs': (str(log_path),)}
            with mp.Pool(mp.cpu_count(), **options) as pool:
                yield pool.map

    def _run_day(self, df_1day, *, mc, n_MC, metadata, map_measurements, sensor_precision=None):
        id_to_datetime = _id_to_datetime(df_1day)
        groups = df_1day.groupby('id', sort=False, observed=True)
        fit = partial(_process_id_wrapper, max_cutoff=df_1day['timedelta'].max(),
                      mc=mc, n_MC=n_MC, metadata=metadata, sensor_precision=sensor_precision)
        # First-seen measurement order also defines the output time coordinate.
        measurements = [(key, groups.get_group(key)) for key in df_1day['id'].unique()]
        results = map_measurements(fit, measurements)
        combined = {k: v for result in results for k, v in result.items()}
        return _convert_keys_to_datetime(combined, id_to_datetime=id_to_datetime)

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

    def run(self, df, chamber_id, output_folder='./output', metadata=None,
            log_path=None, skip_existing=False):
        '''
        skip_existing: if True, a date is not processed again when
        {chamber_id}_{date}.nc already holds a non-MC result for every
        measurement of that date. Dates with new or missing measurements are
        processed and overwritten as usual. Skipped dates are listed in
        self.skipped_dates; the return value is the last processed dataset,
        or None when every date was skipped.
        '''
        metadata = DEFAULT_METADATA if metadata is None else metadata
        self.skipped_dates = []
        with self._measurement_mapper(log_path) as map_measurements:
            ds = None
            for date in df['datetime'].dt.date.unique():
                df_1day = df[df['datetime'].dt.date == date]
                path = f'{output_folder}/{chamber_id}_{date}.nc'
                if skip_existing and _output_covers(path, _expected_times(df_1day), mc=False):
                    logger.info('Skipping date %s: %s is complete', date, path)
                    self.skipped_dates.append(date)
                    continue
                logger.info('Processing date %s', date)
                converted = self._run_day(df_1day, mc=False, n_MC=None,
                                          metadata=metadata, map_measurements=map_measurements)
                ds = self._build_dataset_2d(converted)
                ds.to_netcdf(path)
                logger.info('Saved NetCDF for %s', date)
        return ds

    def run_MC(self, df, chamber_id, output_folder='./output', save_netcdf=False,
               sensor_precision=None, n_MC=None, metadata=None, log_path=None,
               skip_existing=False, save_pareto_only=False):
        '''
        sensor_precision: yerr (ppm) used in the MCMC likelihood. If None, the
        HM_model default (MEASUREMENT_SIGMA_PPM) is used. Scalar only in the
        multiprocessing path.
        n_MC: number of MCMC samples per fit. If None, uses DEFAULT_N_MC.
        metadata: chamber geometry dictionary with area [cm^2] and volume
        [cm^3]. If None, uses DEFAULT_METADATA.
        log_path: append worker diagnostics to this file on the calling computer.
        save_pareto_only: write {chamber_id}_{date}_paretoFront.nc with every
        non-dominated window's MC samples and all fit metrics, plus the usual
        _bestPareto.nc. This takes precedence over save_netcdf. The returned
        dataset still contains the full grid.
        skip_existing: if True, a date is not processed again when its outputs
        are complete: with save_netcdf, {chamber_id}_{date}.nc holds an MC
        result for every measurement of that date and the _bestPareto.nc file
        exists; without it, the _bestPareto.nc file holds every measurement
        (so a date where a Pareto front was not found is always redone).
        In Pareto-only mode the front records all processed times, including
        measurements with zero retained windows. Both Pareto files are checked.
        Skipped dates are listed in self.skipped_dates; the return value is
        the last processed dataset, or None when every date was skipped.
        '''
        n_MC = DEFAULT_N_MC if n_MC is None else n_MC
        metadata = DEFAULT_METADATA if metadata is None else metadata
        self.skipped_dates = []
        with self._measurement_mapper(log_path) as map_measurements:
            ds = None
            for date in df['datetime'].dt.date.unique():
                df_1day = df[df['datetime'].dt.date == date]
                path = f'{output_folder}/{chamber_id}_{date}.nc'
                front_path = f'{output_folder}/{chamber_id}_{date}_paretoFront.nc'
                best_path = f'{output_folder}/{chamber_id}_{date}_bestPareto.nc'
                if skip_existing:
                    times = _expected_times(df_1day)
                    if save_pareto_only or save_netcdf:
                        done = _mc_pair_covers(front_path if save_pareto_only else path,
                                               best_path, times, save_pareto_only)
                    else:
                        done = _output_covers(best_path, times, mc=True)
                    if done:
                        logger.info('Skipping date %s: outputs in %s are complete',
                                    date, output_folder)
                        self.skipped_dates.append(date)
                        continue
                logger.info('Processing date %s', date)
                converted = self._run_day(df_1day, mc=True, n_MC=n_MC,
                                          metadata=metadata, map_measurements=map_measurements,
                                          sensor_precision=sensor_precision)
                ds = self._build_dataset_3d(converted)
                if save_netcdf and not save_pareto_only:
                    _write_netcdf(ds, path)
                self.select_bestPareto(ds=ds, chamber_id=chamber_id, date=date,
                                       output_folder=output_folder,
                                       front_path=front_path if save_pareto_only else None)
        return ds

    def select_bestPareto(self, ds, chamber_id, date, output_folder=None, *, front_path=None):
        '''
        Return an xr.Dataset with the Pareto-optimal (deadband, cutoff) per time.
        If front_path is supplied, also save all non-dominated windows there.
        '''
        logger.info('Selecting best Pareto front')
        best_windows, front_windows = [], []
        front_uncertainty, front_loss = [], []
        for i, time in enumerate(ds.time.values):
            if not ds.sizes['cutoff'] or not ds.sizes['deadband']:
                continue
            pa = Pareto(dsMC=ds.isel(time=i))
            norm_u, norm_l, flat_u, flat_l = pa.prepare_metrics()
            pareto_indices = pa.find_pareto_front(x=flat_u, y=flat_l)
            if not pareto_indices.size:
                logger.warning('Pareto front not found at time %s', time)
                continue
            best_cutoff, best_deadband = pa.get_best_from_pareto(pareto_indices, norm_u, norm_l)
            best_windows.append((i, best_cutoff, best_deadband))
            if front_path is not None:
                cutoffs, deadbands = pa.get_coords_pareto(pareto_indices)
                front_windows.extend((i, c, d) for c, d in zip(cutoffs, deadbands))
                front_uncertainty.extend(flat_u[pareto_indices])
                front_loss.extend(flat_l[pareto_indices])

        ds_best = _select_windows(ds[['dcdt(HM)']], best_windows, 'time').astype(np.float32)
        ds_best = ds_best.rename({'cutoff': 'best_cutoff', 'deadband': 'best_deadband'})
        ds_best = ds_best.reset_coords(['best_cutoff', 'best_deadband'])

        if front_path is not None:
            ds_front = _select_windows(ds, front_windows, 'window').rename({'time': 'window_time'})
            # Preserve normalization against all fitted windows for plots and
            # for reproducing the best compromise from the compact file alone.
            ds_front = ds_front.assign_coords(
                time=ds.time, normalized_uncertainty=('window', front_uncertainty),
                normalized_neg_logprob=('window', front_loss),
            )
            ds_front.attrs['pareto_run_id'] = ds_best.attrs['pareto_run_id'] = uuid4().hex
            _write_netcdf(ds_front, front_path)

        if output_folder is not None:
            _write_netcdf(ds_best, f'{output_folder}/{chamber_id}_{date}_bestPareto.nc')
            logger.info('Saved best-Pareto NetCDF to %s', output_folder)
        return ds_best


def _process_id_wrapper(measurement, *, max_cutoff, mc, n_MC, metadata, sensor_precision):
    id, df = measurement
    try:
        return _process_id(df, id, max_cutoff=max_cutoff, mc=mc, n_MC=n_MC,
                           metadata=metadata, sensor_precision=sensor_precision)
    except Exception:
        # log the full traceback attributed to this id, then re-raise so the
        # parent knows the task failed.
        logger.exception('worker failed on id=%s', id)
        raise
