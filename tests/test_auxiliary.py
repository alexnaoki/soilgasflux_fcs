import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from soilgasflux_fcs.auxiliary import prepare_measurements


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.ranges = {
            'pressure_kpa': (80, 110), 'temperature_c': (-30, 60),
            'relative_humidity_percent': (0, 100),
        }
        # Interleaved measurements must stay whole when split between workers.
        self.frame = pd.DataFrame({
            'chamber_id': ['a', 'b'] * 3,
            'measurement_id': ['closure'] * 6,
            'timestamp': pd.to_datetime(['2026-10-21 12:00:00'] * 2
                                       + ['2026-10-21 12:00:01'] * 2
                                       + ['2026-10-21 12:00:02'] * 2),
            'elapsed_seconds': [0, 0, 1, 1, 2, 2],
            'pressure_source': [100, np.nan, np.nan, np.nan, 102, np.nan],
            'temperature_source': [20] * 6,
            'relative_humidity_source': [50] * 6,
            'chamber_area_cm2': [314] * 6,
            'chamber_volume_cm3': [6283] * 6,
            'area_default_used': [True] * 6,
            'volume_default_used': [False] * 6,
        })
        self.frame.attrs = {'source_files': ['source.csv']}

    def test_interpolation_flags_invalid_measurements_and_attrs(self):
        progress = []
        pieces = prepare_measurements(self.frame, self.ranges, processes=1,
                                      progress_callback=lambda *args: progress.append(args))
        result = pd.concat(pieces, ignore_index=True)
        first = result[result.chamber_id == 'a']
        second = result[result.chamber_id == 'b']
        np.testing.assert_allclose(first.pressure_kpa, [100, 101, 102])
        self.assertEqual(first.pressure_quality_flag.tolist(), [0, 1, 0])
        self.assertTrue(first.measurement_quality_flag.eq(1).all())
        self.assertTrue(np.isfinite(first.water_vapor_mmol_mol).all())
        self.assertTrue(second.measurement_quality_flag.eq(2).all())
        self.assertTrue(second.water_vapor_mmol_mol.isna().all())
        self.assertTrue(all(not piece.attrs for piece in pieces))
        self.assertEqual(self.frame.attrs, {'source_files': ['source.csv']})
        self.assertEqual(progress[-1][:2], (2, 2))

    def test_spawned_workers_match_serial_results(self):
        serial = pd.concat(prepare_measurements(self.frame, self.ranges, processes=1),
                           ignore_index=True)
        with patch('soilgasflux_fcs.auxiliary.MIN_PARALLEL_MEASUREMENTS', 1):
            parallel = pd.concat(prepare_measurements(self.frame, self.ranges, processes=2),
                                 ignore_index=True)
        pd.testing.assert_frame_equal(serial, parallel)


if __name__ == '__main__':
    unittest.main()
