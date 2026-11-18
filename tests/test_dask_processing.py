import importlib.util
import logging
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from soilgasflux_fcs import Multiprocessor


def measurement(key, start, length):
    elapsed = np.arange(length)
    return pd.DataFrame({
        'id': key, 'datetime': pd.Timestamp(start) + pd.to_timedelta(elapsed, unit='s'),
        'timedelta': elapsed, 'k30_co2': 620 - 200 * np.exp(-0.018 * elapsed),
        'si_temperature': 20., 'si_humidity': 60., 'bmp_pressure': 101325.,
    })


class OptionalDependencyTests(unittest.TestCase):
    def test_local_and_missing_extra(self):
        frame = measurement('one', '2025-06-01', 71)
        with patch.dict(sys.modules, {'distributed': None}), tempfile.TemporaryDirectory() as folder:
            with patch('soilgasflux_fcs.multiprocess_raw_data.mp.cpu_count', return_value=1):
                result = Multiprocessor('  ').run(frame, 'local', folder)
            self.assertTrue(np.isfinite(result['dcdt(linear)'][0, 0, 0]))
            with self.assertRaisesRegex(ImportError, r'\.\[dask\]'):
                Multiprocessor('tcp://127.0.0.1:8786').run(frame, 'remote', folder)


@unittest.skipUnless(importlib.util.find_spec('distributed'), 'requires the optional Dask extra')
class DaskProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from distributed import LocalCluster
        cls.cluster = LocalCluster(host='127.0.0.1', n_workers=2, threads_per_worker=1,
                                   processes=True, dashboard_address='127.0.0.1:0', memory_limit=0,
                                   silence_logs=logging.CRITICAL)
        cls.addClassCleanup(cls.cluster.close)

    def setUp(self):
        self.remote = Multiprocessor(self.cluster.scheduler_address)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.times = pd.to_datetime(['2025-06-01 12:00', '2025-06-01 11:00'])
        self.frame = pd.concat([measurement('later', self.times[0], 72),
                                measurement('earlier', self.times[1], 82)], ignore_index=True)

    def test_standard_fits_match_local_and_resume(self):
        from distributed import Client, get_task_stream
        log_path = self.folder / 'workers.log'
        with patch('soilgasflux_fcs.multiprocess_raw_data.mp.cpu_count', return_value=2):
            local = Multiprocessor().run(self.frame, 'local', self.folder)
        with Client(self.cluster.scheduler_address, set_as_default=False) as observer:
            with get_task_stream(client=observer) as stream:
                remote = self.remote.run(self.frame, 'remote', self.folder, log_path=log_path)
        self.assertEqual(len({task['worker'] for task in stream.data}), 2)
        xr.testing.assert_allclose(remote, local)
        np.testing.assert_array_equal(remote.time, self.times)
        np.testing.assert_array_equal(remote.cutoff, [60, 70, 80])
        self.assertTrue(np.isfinite(remote['dcdt(linear)'][:, 0, 0]).all())
        self.assertTrue(np.isfinite(remote['dcdt(HM)'][:, 0, 0]).all())
        with xr.open_dataset(self.folder / 'remote_2025-06-01.nc') as saved:
            xr.testing.assert_identical(saved, remote)
        log = log_path.read_text()
        for key in ('later', 'earlier'):
            self.assertIn(f'Processing ID {key}', log)
        self.assertIn('cutoff=80', log)  # The shared grid exceeds the shorter measurement.

        another_day = measurement('new', '2025-06-02', 71)
        resumed = self.remote.run(pd.concat([self.frame, another_day]), 'remote', self.folder,
                                  skip_existing=True)
        self.assertEqual(self.remote.skipped_dates, [self.times[0].date()])
        np.testing.assert_array_equal(resumed.time, another_day.datetime.iloc[:1])

    def test_mcmc_samples_pareto_and_skip(self):
        # Keep all MCMC windows within the shorter closure.
        frame = pd.concat([measurement('later', self.times[0], 81),
                           measurement('earlier', self.times[1], 82)], ignore_index=True)

        def run(skip=False):
            return self.remote.run_MC(frame, 'mc', self.folder, n_MC=100, save_netcdf=True,
                                      sensor_precision=1.5, skip_existing=skip)

        result = run()
        self.assertEqual(result.sizes['MC'], 100)
        with xr.open_dataset(self.folder / 'mc_2025-06-01_bestPareto.nc') as best:
            np.testing.assert_array_equal(best.time, self.times)
            for t in best.time.values:
                selected = best.sel(time=t)
                window = result.sel(time=t, cutoff=selected.best_cutoff.item(),
                                    deadband=selected.best_deadband.item())
                np.testing.assert_array_equal(selected['dcdt(HM)'], window['dcdt(HM)'])
        self.assertIsNone(run(skip=True))
        self.assertEqual(self.remote.skipped_dates, [self.times[0].date()])
        second = run()['dcdt(HM)'][:, 0, 0]
        first = result['dcdt(HM)'][:, 0, 0]
        self.assertTrue(np.isfinite(first).all() and np.isfinite(second).all())
        self.assertFalse(np.array_equal(first, second))

    def test_failure_reaches_local_log(self):
        log = self.folder / 'failed.log'
        with self.assertRaises(KeyError):
            self.remote.run(self.frame.drop(columns='k30_co2'), 'bad', self.folder, log_path=log)
        diagnostics = log.read_text()
        for key in ('later', 'earlier'):
            self.assertIn(f'worker failed on id={key}', diagnostics)
        self.assertIn('k30_co2', diagnostics)
        self.assertFalse((self.folder / 'bad_2025-06-01.nc').exists())

    @patch('soilgasflux_fcs._dask.CONNECTION_TIMEOUT', 0.5)
    def test_unreachable_scheduler_and_empty_cluster(self):
        from distributed import LocalCluster
        # Reserve a local port without listening on it.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            address = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
            with self.assertRaisesRegex(ConnectionError, 'scheduler address'):
                Multiprocessor(address).run(self.frame, 'bad', self.folder)
        with LocalCluster(host='127.0.0.1', n_workers=0, processes=True,
                          dashboard_address='127.0.0.1:0') as empty:
            with self.assertRaisesRegex(RuntimeError, 'dask worker'):
                Multiprocessor(empty.scheduler_address).run_MC(self.frame, 'bad', self.folder)
