import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from soilgasflux_fcs import Multiprocessor


class ParetoOutputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.times = pd.date_range('2026-10-21', periods=3, freq='h')
        self.frame = pd.DataFrame({'id': ['one', 'two', 'three'], 'datetime': self.times})
        self.processor = Multiprocessor()
        # Two distinct trade-offs, including two windows tied on both objectives.
        # The second measurement has a single valid fit, the third has none.
        widths = np.array([[[1, 2, 3], [1, 2, 4]], [[np.nan, 1, np.nan]] * 2,
                           [[np.nan] * 3] * 2])
        losses = np.array([[[3, 1, 4], [3, 2, 3]], [[np.nan, 1, np.nan],
                          [np.nan, np.inf, np.nan]], [[np.nan] * 3] * 2])
        samples = np.linspace(-1, 1, 2048)
        dims = ('time', 'cutoff', 'deadband', 'MC')
        dcdt = (widths[..., None] * samples).astype('float32')
        self.grid = xr.Dataset({
            'dcdt(HM)': (dims, dcdt),
            'logprob(HM)': (dims, np.broadcast_to(-losses[..., None], dcdt.shape).astype('float32')),
            **{k: (dims, dcdt + i) for i, k in enumerate(['AIC(HM)', 'RMSE(HM)', 'R2(HM)', 'nRMSE(HM)'])},
        }, coords={'time': self.times, 'cutoff': [60, 80], 'deadband': [0, 10, 20],
                   'MC': np.arange(len(samples))})
        self.front = self.folder / 'chamber_2026-10-21_paretoFront.nc'
        self.best = self.folder / 'chamber_2026-10-21_bestPareto.nc'
        self.full = self.folder / 'chamber_2026-10-21.nc'

    def run_mc(self, **kwargs):
        with patch.object(self.processor, '_measurement_mapper'), \
                patch.object(self.processor, '_run_day'), \
                patch.object(self.processor, '_build_dataset_3d', return_value=self.grid):
            return self.processor.run_MC(self.frame, 'chamber', self.folder, **kwargs)

    def test_front_round_trip_and_disk_savings(self):
        self.assertIs(self.run_mc(save_netcdf=True, save_pareto_only=True), self.grid)
        self.assertEqual(set(self.folder.iterdir()), {self.front, self.best})
        front = xr.load_dataset(self.front)
        np.testing.assert_array_equal(front.time, self.times)
        expected = [(0, 60, 0), (0, 60, 10), (0, 80, 0), (1, 60, 10)]
        self.assertEqual(front.sizes['window'], len(expected))
        for i, (t, cutoff, deadband) in enumerate(expected):
            window = front.isel(window=i)
            self.assertEqual(window.window_time.values, self.times[t].to_datetime64())
            self.assertEqual((window.cutoff.item(), window.deadband.item()), (cutoff, deadband))
            for name in self.grid.data_vars:
                self.assertEqual(front[name].dims, ('window', 'MC'))
                np.testing.assert_array_equal(window[name], self.grid[name].sel(
                    time=self.times[t], cutoff=cutoff, deadband=deadband))
        best = xr.load_dataset(self.best)
        np.testing.assert_array_equal(best.time, self.times[:2])
        np.testing.assert_array_equal(best.best_deadband, [10, 10])
        np.testing.assert_array_equal(best.best_cutoff, [60, 60])
        np.testing.assert_allclose(front.normalized_uncertainty, [0, 1/3, 0, 0])
        np.testing.assert_allclose(front.normalized_neg_logprob, [2/3, 0, 2/3, 0])
        for i in range(2):
            np.testing.assert_array_equal(best['dcdt(HM)'][i], self.grid['dcdt(HM)'].sel(
                time=self.times[i], cutoff=best.best_cutoff[i], deadband=best.best_deadband[i]))
        self.assertIsNone(self.run_mc(skip_existing=True, save_pareto_only=True))
        best.isel(time=slice(1)).to_netcdf(self.best)
        self.assertIs(self.run_mc(skip_existing=True, save_pareto_only=True), self.grid)
        self.run_mc(save_netcdf=True, skip_existing=True)
        self.assertLess(self.front.stat().st_size, self.full.stat().st_size)
        # A best file from another fit cannot complete the older compact pair.
        self.assertIs(self.run_mc(skip_existing=True, save_pareto_only=True), self.grid)

    def test_empty_grid_and_resume(self):
        self.run_mc()
        self.assertEqual(list(self.folder.iterdir()), [self.best])
        self.grid = self.grid.isel(cutoff=slice(0, 0))
        self.assertIs(self.run_mc(save_pareto_only=True, skip_existing=True), self.grid)
        front = xr.load_dataset(self.front)
        np.testing.assert_array_equal(front.time, self.times)
        self.assertEqual(front.sizes['window'], 0)
        self.assertEqual(xr.load_dataset(self.best).sizes['time'], 0)
        self.assertIsNone(self.run_mc(save_pareto_only=True, skip_existing=True))
        self.front.write_bytes(b'incomplete')
        self.assertIs(self.run_mc(save_pareto_only=True, skip_existing=True), self.grid)


if __name__ == '__main__':
    unittest.main()
