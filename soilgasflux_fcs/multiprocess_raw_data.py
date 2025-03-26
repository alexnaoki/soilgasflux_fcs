import numpy as np
import pandas as pd
import datetime as dt
import json
import pathlib
import xarray as xr
import os
import multiprocessing as mp
from .fcs import FCS

class Multiprocessor:
    def __init__(self):
        pass

    def convert_keys_to_datetime(self,data):
        new_data = {}
        for key, value in data.items():
            new_key = dt.datetime.strptime(key, '%Y-%m-%d_%H-%M-%S')
            new_data[new_key] = value
        return new_data

    def process_id(self, df, id):
        print('Processing ID:', id)
        df_id = df[df['id'] == id]
        a_fcs = FCS(df_data=df_id, chamber_id='test')
        a_fcs.settings(moving_window=True, window_walk=10, min_window_size=20, 
                       min_deadband=0, max_deadband=60)
        
        results = a_fcs.run(n=id,metadata={'area':314, 'volume':6283})
        print('Results:', results.keys())
        return results
    
    def process_id_MC(self, df, id):
        print('Processing ID:', id)
        df_id = df[df['id'] == id]
        a_fcs = FCS(df_data=df_id, chamber_id='test')
        a_fcs.settings(moving_window=True, window_walk=10, min_window_size=20, 
                       min_deadband=0, max_deadband=60)
        
        results = a_fcs.run_MC(n=id, n_MC=500,metadata={'area':314, 'volume':6283})
        print('Results:', results.keys())
        return results

    def run(self, df, chamber_id, output_folder='./output'):
        print('Multiprocessing started')
        print('CPU core count:', mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        # print(os.listdir())

        all_results = []

        for date in df['datetime'].dt.date.unique():
            print('Date:', date)
            df_1day = df[df['datetime'].dt.date == date]

            results = pool.starmap(self.process_id, [(df_1day, n) for n in df_1day['id'].unique()])

            all_results.extend(results)

        # print('All results:', all_results)
            combined_results = {k: v for result in all_results for k, v in result.items()}
            # with open(f'{output_folder}/{chamber_id}_{date}.json', 'w') as f:
            #     json.dump(combined_results,f)

            converted_data = self.convert_keys_to_datetime(combined_results)
            times = list(converted_data.keys())
            cutoff = list(converted_data[times[0]]['cutoff'])
            deadband = list(converted_data[times[0]]['deadband'])
            print(deadband, cutoff)

            ds = xr.Dataset(
                {
                    'dcdt(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['dcdt(HM)'] for t in times])),
                    'dcdt(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['dcdt(linear)'] for t in times])),
                    'AIC(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['AIC(HM)'] for t in times])),
                    'AIC(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['AIC(linear)'] for t in times])),
                    'RMSE(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['RMSE(HM)'] for t in times])),
                    'RMSE(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['RMSE(linear)'] for t in times]))
                },
                coords={
                    'time': times,
                    'deadband': deadband,
                    'cutoff': cutoff
                }
            )

            ds.to_netcdf(f'{output_folder}/{chamber_id}_{date}.nc')
            print('NetCDF file saved')
        return ds
    
    def run_MC(self, df, chamber_id, output_folder='./output'):
        print('Multiprocessing started')
        print('CPU core count:', mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        # print(os.listdir())

        all_results = []

        for date in df['datetime'].dt.date.unique():
            print('Date:', date)
            df_1day = df[df['datetime'].dt.date == date]

            results = pool.starmap(self.process_id_MC, [(df_1day, n) for n in df_1day['id'].unique()])

            all_results.extend(results)

        # print('All results:', all_results)
            combined_results = {k: v for result in all_results for k, v in result.items()}
            # with open(f'{output_folder}/{chamber_id}_{date}.json', 'w') as f:
            #     json.dump(combined_results,f)

            converted_data = self.convert_keys_to_datetime(combined_results)
            times = list(converted_data.keys())
            cutoff = list(converted_data[times[0]]['cutoff'])
            deadband = list(converted_data[times[0]]['deadband'])
            n_MC = list(converted_data[times[0]]['MC'])
            print(deadband, cutoff)
            # print(converted_data[times[0]]['dcdt(linear)'])

            ds = xr.Dataset(
                {
                    'dcdt(HM)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['dcdt(HM)'] for t in times], dtype=np.float32)),
                    'dcdt(linear)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['dcdt(linear)'] for t in times], dtype=np.float32)),
                    'AIC(HM)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['AIC(HM)'] for t in times], dtype=np.float32)),
                    'AIC(linear)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['AIC(linear)'] for t in times], dtype=np.float32)),
                    'RMSE(HM)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['RMSE(HM)'] for t in times], dtype=np.float32)),
                    'RMSE(linear)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['RMSE(linear)'] for t in times], dtype=np.float32))
                },
                coords={
                    'time': times,
                    'deadband': deadband,
                    'cutoff': cutoff,
                    'MC': n_MC
                }
            )

            ds.to_netcdf(f'{output_folder}/{chamber_id}_{date}.nc')
            print('NetCDF file saved')
        return ds
