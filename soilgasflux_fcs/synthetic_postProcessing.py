import pathlib
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
from .models import hm_model, linear_model, hm_model_dcdt
from soilgasflux_fcs import json_reader

class Synthetic:
    def __init__(self, processed_data, raw_dataFolder):
        self.processed_data = pathlib.Path(processed_data)
        self.raw_dataFolder = pathlib.Path(raw_dataFolder)

        print(self.processed_data.name)

        print('Synthetic initialized')
        self.ds = xr.open_dataset(self.processed_data)
        # print(self.ds.time)
        # print(self.ds)

        json = json_reader.Initializer(self.raw_dataFolder)
        self.df = json.prepare_rawdata()
        # self.df['id_datetime'] = pd.strftime(self.df['id'], '%Y-%m-%d_%H-%M-%S')
        self.df['id_datetime'] = pd.to_datetime(self.df['id'], format='%Y-%m-%d_%H-%M-%S')
        # print(self.df.dtypes)

        expected_results = {}
        for t in self.ds.time.values:
            expected_results[t] = {'dcdt(HM)': None}
            expected_results[t]['deadband'] = None
            expected_results[t]['d_intensity'] = None
            expected_results[t]['d_startpoint'] = None
            expected_results[t]['add_noise'] = None
            expected_results[t]['c0'] = None
            expected_results[t]['alpha'] = None
            expected_results[t]['cs'] = None
            expected_results[t]['pressure'] = None
            expected_results[t]['temperature'] = None
            expected_results[t]['humidity'] = None
            expected_results[t]['area'] = np.pi*20**2/4
            expected_results[t]['volume'] = np.pi*20**2*20/4
            expected_results[t]['curvature'] = None
            

            # print('time',t)
            alpha, cs, c0, deadband, d_intensity, d_startpoint, add_noise, pressure, temperature, humidity, curvature = self.find_rawData_config(datetime=t)
            # print(alpha, cs, c0, deadband, d_intensity, d_startpoint, add_noise)
            # print(self.ds.cutoff.values)

            hm_dcdt = []
            # deadband = []
            # d_intensity = []
            # d_startpoint = []
            # add_noise = []

            for c in self.ds.cutoff.values:
                # print('c', c)
                hm_dcdt.append(hm_model_dcdt(t0=0, c0=c0, a=alpha, cx=cs, t=c))


            # print(hm_model_dcdt(t0=0, c0=c0, a=alpha, cx=cs, t=t.values))
            # print(hm_dcdt)
            expected_results[t]['dcdt(HM)'] = hm_dcdt
            expected_results[t]['deadband'] = deadband
            expected_results[t]['d_intensity'] = d_intensity
            expected_results[t]['d_startpoint'] = d_startpoint
            expected_results[t]['add_noise'] = add_noise
            expected_results[t]['c0'] = c0
            expected_results[t]['alpha'] = alpha
            expected_results[t]['cs'] = cs
            expected_results[t]['pressure'] = pressure
            expected_results[t]['temperature'] = temperature
            expected_results[t]['humidity'] = humidity
            expected_results[t]['curvature'] = curvature

            # print()

        self.new_ds = xr.Dataset(
            {
                'dcdt(HM)': (['time',  'cutoff'], np.array([expected_results[t]['dcdt(HM)'] for t in self.ds.time.values])),
                'deadband': (['time'], np.array([expected_results[t]['deadband'] for t in self.ds.time.values])),
                'd_intensity': (['time'], np.array([expected_results[t]['d_intensity'] for t in self.ds.time.values])),
                'd_startpoint': (['time'], np.array([expected_results[t]['d_startpoint'] for t in self.ds.time.values])),
                'add_noise': (['time'], np.array([expected_results[t]['add_noise'] for t in self.ds.time.values])),
                'c0': (['time'], np.array([expected_results[t]['c0'] for t in self.ds.time.values])),
                'alpha': (['time'], np.array([expected_results[t]['alpha'] for t in self.ds.time.values])),
                'cs': (['time'], np.array([expected_results[t]['cs'] for t in self.ds.time.values])),
                'pressure': (['time'], np.array([expected_results[t]['pressure'] for t in self.ds.time.values])),
                'temperature': (['time'], np.array([expected_results[t]['temperature'] for t in self.ds.time.values])),
                'humidity': (['time'], np.array([expected_results[t]['humidity'] for t in self.ds.time.values])),
                'area': (['time'], np.array([expected_results[t]['area'] for t in self.ds.time.values])),
                'volume': (['time'], np.array([expected_results[t]['volume'] for t in self.ds.time.values])),
                'curvature': (['time'], np.array([expected_results[t]['curvature'] for t in self.ds.time.values])),	
            },
            coords={
                'time': self.ds.time,
                'cutoff': self.ds.cutoff
            }
        )
        # print(new_ds)
        # return new_ds



    def get_expectedResults(self):
        return self.new_ds

    def save_expectedResults(self, path):
        self.new_ds.to_netcdf(path)
        print('Expected results saved in', path)

    def find_rawData_config(self, datetime):
        # print('Finding raw data', datetime)
        # print(self.df.loc[self.df['datetime'] == datetime])
        
        rowFirst = self.df.loc[self.df['id_datetime']==datetime].head(1)
        print(rowFirst)

        return rowFirst['alpha'].values[0], rowFirst['c_s'].values[0], rowFirst['c_c0'].values[0], rowFirst['deadband'].values[0], rowFirst['disturbance_intensity'].values[0], rowFirst['disturbance_starting_point'].values[0], rowFirst['add_noise'].values[0],rowFirst['bmp_pressure'].values[0], rowFirst['bmp_temperature'].values[0], rowFirst['si_humidity'].values[0], rowFirst['curvature'].values[0]



