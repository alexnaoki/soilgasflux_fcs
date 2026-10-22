import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from soilgasflux_fcs.multiprocess_raw_data import (
    Multiprocessor, _expected_times, _output_covers,
)


class SkipExistingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.times = [dt.datetime(2026, 10, 21, 12), dt.datetime(2026, 10, 21, 13)]
        self.frame = pd.DataFrame({'id': ['first', 'second'], 'datetime': self.times})

    def write_output(self, name, times=None, mc=False):
        coords = {'time': self.times if times is None else times}
        if mc:
            coords['MC'] = [0, 1]
        path = self.folder / name
        xr.Dataset(coords=coords).to_netcdf(path)
        return path

    def test_coverage_rejects_missing_corrupt_partial_and_wrong_run_type(self):
        path = self.folder / 'output.nc'
        self.assertFalse(_output_covers(path, self.times, False))
        path.write_text('not a NetCDF file')
        self.assertFalse(_output_covers(path, self.times, False))
        self.write_output(path.name, self.times[:1])
        self.assertFalse(_output_covers(path, self.times, False))
        self.write_output(path.name)
        self.assertTrue(_output_covers(path, self.times, False))
        self.assertFalse(_output_covers(path, self.times, True))

    def test_coverage_normalizes_datetime_resolution(self):
        stored = MagicMock()
        stored.__enter__.return_value = stored
        stored.dims = {'time': len(self.times)}
        stored['time'].values = np.array(self.times, dtype='datetime64[us]')
        with patch('soilgasflux_fcs.multiprocess_raw_data.os.path.exists', return_value=True), \
                patch('soilgasflux_fcs.multiprocess_raw_data.xr.open_dataset', return_value=stored):
            self.assertTrue(_output_covers('output.nc', self.times, False))

    def test_expected_times_use_timestamp_ids_and_fallback_mapping(self):
        self.frame.loc[0, 'id'] = '2026-10-21_11-00-00'
        self.assertEqual(_expected_times(self.frame), [dt.datetime(2026, 10, 21, 11), self.times[1]])

    @patch('soilgasflux_fcs.multiprocess_raw_data.mp.Pool')
    def test_run_skips_complete_dates_and_resets_report(self, pool):
        self.write_output('chamber_2026-10-21.nc')
        processor = Multiprocessor()
        with patch.object(processor, '_run_day') as run_day:
            result = processor.run(self.frame, 'chamber', self.folder, skip_existing=True)
            self.assertIsNone(result)
            run_day.assert_not_called()
        self.assertEqual(processor.skipped_dates, [self.times[0].date()])
        result_ds = xr.Dataset(coords={'time': self.times})
        with patch.object(processor, '_run_day'), \
                patch.object(processor, '_build_dataset_2d', return_value=result_ds):
            self.assertIs(processor.run(self.frame, 'chamber', self.folder), result_ds)
        self.assertEqual(processor.skipped_dates, [])

    @patch('soilgasflux_fcs.multiprocess_raw_data.mp.Pool')
    def test_mc_requires_best_output_and_reprocesses_new_measurements(self, pool):
        self.write_output('chamber_2026-10-21.nc', mc=True)
        processor = Multiprocessor()
        result_ds = xr.Dataset(coords={'time': self.times, 'MC': [0, 1]})
        with patch.object(processor, '_run_day') as run_day, \
                patch.object(processor, '_build_dataset_3d', return_value=result_ds), \
                patch.object(processor, 'select_bestPareto'):
            processor.run_MC(self.frame, 'chamber', self.folder, save_netcdf=True, skip_existing=True)
            run_day.assert_called_once()
            self.write_output('chamber_2026-10-21_bestPareto.nc', self.times[:1], mc=True)
            run_day.reset_mock()
            self.assertIsNone(processor.run_MC(
                self.frame, 'chamber', self.folder, save_netcdf=True, skip_existing=True))
            run_day.assert_not_called()
            processor.run_MC(self.frame, 'chamber', self.folder, skip_existing=True)
            run_day.assert_called_once()


if __name__ == '__main__':
    unittest.main()
