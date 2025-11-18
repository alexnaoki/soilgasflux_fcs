import numpy as np
import pandas as pd
import datetime as dt
import json
import pathlib
import xarray as xr
import os
import multiprocessing as mp
from .fcs import FCS
from .pareto import Pareto

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
        a_fcs.settings(moving_window=True, window_walk=10, min_window_size=60, 
                       min_deadband=0, max_deadband=60)
        
        results = a_fcs.run(n=id,metadata={'area':314, 'volume':6283})
        print('Results:', results.keys())
        return results
    
    def process_id_MC(self, df, id):
        print('Processing ID:', id)
        df_id = df[df['id'] == id]
        a_fcs = FCS(df_data=df_id, chamber_id='test')
        a_fcs.settings(moving_window=True, window_walk=10, min_window_size=60, 
                       min_deadband=0, max_deadband=60)
        
        results = a_fcs.run_MC(n=id, n_MC=8000,metadata={'area':314, 'volume':6283})
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
            # print(converted_data)
            print('test')
            print(times)
            print(np.shape(np.array(converted_data[times[0]]['dcdt(HM)'])))
            # print(np.array([np.array([[0,0],[0,1]]) for t in times]))
            for t in times:
                if np.shape(np.array(converted_data[t]['dcdt(HM)'])) != (len(cutoff), len(deadband)):
                    print('Shape mismatch at time:', t)
                    print('Expected shape:', (len(cutoff), len(deadband)))
                    print('Actual shape:', np.shape(np.array(converted_data[t]['dcdt(HM)'])))
                else:
                    pass
            print(np.array([converted_data[t]['dcdt(HM)'] for t in times]))

            print('test0')

            ds = xr.Dataset(
                {
                    'dcdt(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['dcdt(HM)'] for t in times])),
                    'dcdt(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['dcdt(linear)'] for t in times])),
                    'AIC(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['AIC(HM)'] for t in times])),
                    'AIC(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['AIC(linear)'] for t in times])),
                    'RMSE(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['RMSE(HM)'] for t in times])),
                    'RMSE(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['RMSE(linear)'] for t in times])),
                    'R2(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['R2(HM)'] for t in times])),
                    'R2(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['R2(linear)'] for t in times])),
                    'nRMSE(HM)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['nRMSE(HM)'] for t in times])),
                    'nRMSE(linear)': (['time',  'cutoff','deadband'], np.array([converted_data[t]['nRMSE(linear)'] for t in times])),

                },
                coords={
                    'time': times,
                    'deadband': deadband,
                    'cutoff': cutoff
                }
            )

            ds.to_netcdf(f'{output_folder}/{chamber_id}_{date}.nc')
            print('NetCDF file saved')
        try:
            return ds
        except Exception as e:
            print('Error returning dataset:', e)
            return None
    
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
                    'RMSE(linear)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['RMSE(linear)'] for t in times], dtype=np.float32)),
                    'R2(HM)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['R2(HM)'] for t in times], dtype=np.float32)),
                    'R2(linear)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['R2(linear)'] for t in times], dtype=np.float32)),
                    'nRMSE(HM)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['nRMSE(HM)'] for t in times], dtype=np.float32)),
                    'nRMSE(linear)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['nRMSE(linear)'] for t in times], dtype=np.float32)),
                    'logprob(HM)': (['time',  'cutoff','deadband', 'MC'], np.array([converted_data[t]['logprob(HM)'] for t in times], dtype=np.float32)),
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

    def select_bestPareto(self, ds, chamber_id,output_folder=None):
        '''
        
        output: dataset with MCMC results from  the best pareto front
        '''
        print('Selecting best pareto front from dataset')
        deadband = ds.coords['deadband']
        cutoff = ds.coords['cutoff']
        n_MC = ds.coords['MC']


        time_list = []
        deadband_list = []
        cutoff_list = []
        best_dcdt_list = []
        for time in ds.coords['time'].values:

            pa = Pareto(dsMC=ds.sel(time=time))
            Norm_uncertaintyRange, Norm_logprob, flatNorm_uncertaintyRange, flatNorm_logprob = pa.prepare_metrics()
            pareto_uncertaintyRange_logprob = pa.find_pareto_front(x=flatNorm_uncertaintyRange,
                                                                            y=flatNorm_logprob, 
                                                                            maximize_x=False, maximize_y=False)
            try:
                coords_pareto_uncertaintyRange_logprob = pa.get_coords_pareto(pareto_indices=pareto_uncertaintyRange_logprob)

                best_pareto_x, best_pareto_y = pa.get_best_from_pareto(pareto_indices=pareto_uncertaintyRange_logprob, 
                                                                        metric_x=Norm_uncertaintyRange,
                                                                        metric_y=Norm_logprob)
            except:
                print('Pareto front not found for time:', time)
                continue
            uncertaintyRange_hm = pa.uncertaintyRange
            logprob_hm = pa.logprob

            hist_dcdt = ds.sel(time=time, 
                               deadband=deadband[best_pareto_y].values, 
                               cutoff=cutoff[best_pareto_x].values)['dcdt(HM)']
            

            # print('Best pareto point (uncertaintyRange, logprob):', best_pareto_x, best_pareto_y)
            # print('Deadband:', deadband[best_pareto_y].values)
            # print('Cutoff:', cutoff[best_pareto_x].values)

            time_list.append(time)
            deadband_list.append(deadband[best_pareto_y].values)
            cutoff_list.append(cutoff[best_pareto_x].values)
            best_dcdt_list.append(hist_dcdt.values)

            
        print('Best dcdt shape:', np.shape(np.array(best_dcdt_list)))

        ds_best = xr.Dataset(
            {
                'dcdt(HM)': (['time', 'MC'], np.array(best_dcdt_list, dtype=np.float32)),
                'best_deadband': (['time'], np.array(deadband_list, dtype=int)),
                'best_cutoff': (['time'], np.array(cutoff_list, dtype=int)),
            },
            coords={
                'time': time_list,
                'MC': n_MC
            }
        )

        if output_folder is not None:
            ds_best.to_netcdf(f'{output_folder}/{chamber_id}_bestPareto.nc')
            print('Best pareto dataset saved to:', output_folder)

        return ds_best