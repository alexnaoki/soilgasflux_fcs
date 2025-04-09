import pandas as pd
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from .hm_model import HM_model
from .linear_model import LINEAR_model
from .metrics import calculate_AIC, rmse, r2, normalized_rmse
from .models import hm_model, linear_model, hm_model_dcdt

class FCS:
    def __init__(self, df_data, chamber_id):
        self.max_time = df_data['timedelta'].max()
        self.df_data = df_data

    def settings(self, 
                 moving_window=True, window_walk=10, min_window_size=30,
                 min_deadband=0, max_deadband=60):
        self.min_window_size = min_window_size
        if moving_window:
            self.deadband_options = np.arange(min_deadband, max_deadband, window_walk)
            self.cutoff_options = np.arange(min_deadband+min_window_size, self.max_time, window_walk)
        else:
            pass

    def run_metrics(self, y_raw, y_model):
        metrics = {'aic': None, 'rmse': None}
        aic = calculate_AIC(y=y_raw, yhat=y_model, p=5)
        rmse_ = rmse(y=y_raw, yhat=y_model)
        r2_ = r2(y=y_raw, yhat=y_model)
        nrmse = normalized_rmse(y=y_raw, yhat=y_model)
        metrics['aic'] = aic
        metrics['rmse'] = rmse_
        metrics['r2'] = r2_
        metrics['nrmse'] = nrmse
        return metrics


    def run(self, n,metadata={'area':314, 'volume':6283}):
        x_deadband = self.deadband_options
        y_cutoff = self.cutoff_options
        XX, YY = np.meshgrid(x_deadband, y_cutoff)
        zeros_like = np.zeros_like(XX)*np.nan
        results = {f'{n}':{}}
        results[f'{n}'] = {
            'deadband': x_deadband, 'cutoff': y_cutoff,
            'dcdt(HM)': zeros_like.copy(), 'dcdt(linear)': zeros_like.copy(),
            'AIC(HM)': zeros_like.copy(), 'AIC(linear)': zeros_like.copy(),
            'RMSE(HM)': zeros_like.copy(), 'RMSE(linear)': zeros_like.copy(),
            'R2(HM)': zeros_like.copy(), 'R2(linear)': zeros_like.copy(),
            'nRMSE(HM)': zeros_like.copy(), 'nRMSE(linear)': zeros_like.copy()
                   }
        

        for n_deadband, deadband in enumerate(self.deadband_options):
            for n_cutoff, cutoff in enumerate(self.cutoff_options):
                if cutoff - deadband < self.min_window_size:
                    continue
                # print('Deadband:', deadband, 'Cutoff:', cutoff)
                try:
                    # print(n_deadband, n_cutoff)
                    # print(zeros_like.shape)
                    hm = HM_model(raw_data=self.df_data,
                                  metadata=metadata)
                    hm_results = hm.calculate(deadband=deadband, cutoff=cutoff)
                    dc_dt, C_0, cx, a, t0, soilgasflux_CO2, deadband, cutoff = hm_results
                    t = np.arange(deadband, cutoff, 1) #TODO some points are contains gaps (e.g. 1,3,4,5,6,..)
                    hm_co2 = hm_model(t=t, 
                                    cx=cx, a=a, t0=t0, c0=C_0)
                    # hm_dcdt = hm_model_dcdt(t0=t0, c0=C_0, a=a, cx=cx, t=t)

                    metrics = self.run_metrics(y_raw=self.df_data['k30_co2'].values[deadband:cutoff],
                                               y_model=hm_co2)
                    
                    # print('dcdt (H-M):\t',dc_dt, '| AIC:\t', metrics['aic'], '| RMSE:\t', metrics['rmse'])
                    results[f'{n}']['dcdt(HM)'][n_cutoff,n_deadband] = dc_dt
                    results[f'{n}']['AIC(HM)'][n_cutoff,n_deadband] = metrics['aic']
                    results[f'{n}']['RMSE(HM)'][n_cutoff,n_deadband] = metrics['rmse']
                    results[f'{n}']['R2(HM)'][n_cutoff,n_deadband] = metrics['r2']
                    results[f'{n}']['nRMSE(HM)'][n_cutoff,n_deadband] = metrics['nrmse']
                    # print(metrics)
                except Exception as e:
                    print('ERROR HM ####')
                    print(e)

                try:
                    linear = LINEAR_model(raw_data=self.df_data,
                                          metadata=metadata)
                    linear_results = linear.calculate(deadband=deadband, cutoff=cutoff)
                    dc_dt, C_0, soilgasflux_CO2, deadband, cutoff = linear_results
                    t = np.arange(deadband, cutoff, 1)

                    linear_co2 = linear_model(t=t, dcdt=dc_dt, c0=C_0)
                    metrics = self.run_metrics(y_raw=self.df_data['k30_co2'].values[deadband:cutoff],
                                               y_model=linear_co2)
                    # print('dcdt (linear):\t',dc_dt, '| AIC:\t', metrics['aic'], '| RMSE:\t', metrics['rmse'])
                    results[f'{n}']['dcdt(linear)'][n_cutoff,n_deadband] = dc_dt
                    results[f'{n}']['AIC(linear)'][n_cutoff,n_deadband] = metrics['aic']
                    results[f'{n}']['RMSE(linear)'][n_cutoff,n_deadband] = metrics['rmse']
                    results[f'{n}']['R2(linear)'][n_cutoff,n_deadband] = metrics['r2']
                    results[f'{n}']['nRMSE(linear)'][n_cutoff,n_deadband] = metrics['nrmse']
                except Exception as e:
                    print('ERROR LINEAR ####')
                    print(e)
        return results


    def run_MC(self, n, n_MC, metadata={'area':314, 'volume':6283}):
        x_deadband = self.deadband_options
        y_cutoff = self.cutoff_options
        
        # Create proper 3D arrays
        shape_3d = (len(y_cutoff), len(x_deadband), n_MC)
        
        results = {f'{n}': {
            'deadband': x_deadband, 
            'cutoff': y_cutoff, 
            'MC': np.arange(n_MC),
            'dcdt(HM)': np.full(shape_3d, np.nan),
            'dcdt(linear)': np.full(shape_3d, np.nan),
            'AIC(HM)': np.full(shape_3d, np.nan),
            'AIC(linear)': np.full(shape_3d, np.nan),
            'RMSE(HM)': np.full(shape_3d, np.nan),
            'RMSE(linear)': np.full(shape_3d, np.nan),
            'R2(HM)': np.full(shape_3d, np.nan),
            'R2(linear)': np.full(shape_3d, np.nan),
            'nRMSE(HM)': np.full(shape_3d, np.nan),
            'nRMSE(linear)': np.full(shape_3d, np.nan)
        }}
        
        for n_deadband, deadband in enumerate(self.deadband_options):
            for n_cutoff, cutoff in enumerate(self.cutoff_options):
                if cutoff - deadband < self.min_window_size:
                    continue
                    
                try:
                    hm = HM_model(raw_data=self.df_data, metadata=metadata)
                    hm_results = hm.calculate_MC(deadband=deadband, cutoff=cutoff, n=n_MC)
                    dc_dt, C_0, cx, a, t0, soilgasflux_CO2, deadband, cutoff = hm_results
                    
                    # Ensure arrays have proper shape
                    t = np.arange(deadband, cutoff, 1)
                    
                    # Reshape parameter arrays if needed
                    if not isinstance(cx, np.ndarray) or cx.ndim == 1:
                        cx = np.array(cx).reshape(n_MC, 1)
                    if not isinstance(a, np.ndarray) or a.ndim == 1:
                        a = np.array(a).reshape(n_MC, 1)
                    if not isinstance(t0, np.ndarray) or t0.ndim == 1:
                        t0 = np.array(t0).reshape(n_MC, 1)
                    
                    TT, NN = np.meshgrid(t, np.arange(n_MC))
                    
                    hm_co2_MC = hm_model(t=TT, cx=cx, a=a, t0=t0, c0=C_0)
                    
                    metrics = self.run_metrics(y_raw=self.df_data['k30_co2'].values[deadband:cutoff],
                                              y_model=hm_co2_MC)
                    
                    # Store results in the 3D array
                    # For dc_dt, store all MC values in the appropriate slice
                    # print('hm')
                    # print(dc_dt.shape)

                    results[f'{n}']['dcdt(HM)'][n_cutoff, n_deadband, :] = dc_dt
                    
                    # For metrics, may need to handle differently depending on what run_metrics returns
                    if isinstance(metrics['aic'], np.ndarray) and len(metrics['aic']) == n_MC:
                        results[f'{n}']['AIC(HM)'][n_cutoff, n_deadband, :] = metrics['aic']
                    else:
                        results[f'{n}']['AIC(HM)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['aic'])
                        
                    if isinstance(metrics['rmse'], np.ndarray) and len(metrics['rmse']) == n_MC:
                        results[f'{n}']['RMSE(HM)'][n_cutoff, n_deadband, :] = metrics['rmse']
                    else:
                        results[f'{n}']['RMSE(HM)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['rmse'])

                    if isinstance(metrics['r2'], np.ndarray) and len(metrics['r2']) == n_MC:
                        results[f'{n}']['R2(HM)'][n_cutoff, n_deadband, :] = metrics['r2']
                    else:
                        results[f'{n}']['R2(HM)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['r2'])

                    if isinstance(metrics['nrmse'], np.ndarray) and len(metrics['nrmse']) == n_MC:
                        results[f'{n}']['nRMSE(HM)'][n_cutoff, n_deadband, :] = metrics['nrmse']
                    else:
                        results[f'{n}']['nRMSE(HM)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['nrmse'])

                except Exception as e:
                    print('ERROR HM ####')
                    print(e)
                    print(f'Deadband: {deadband}, Cutoff: {cutoff}')

                try:
                    linear = LINEAR_model(raw_data=self.df_data, metadata=metadata)
                    linear_results = linear.calculate_MC(deadband=deadband, cutoff=cutoff, n=n_MC)
                    dc_dt, C_0, soilgasflux_CO2, deadband, cutoff = linear_results
                    t = np.arange(deadband, cutoff, 1)
                    
                    if not isinstance(dc_dt, np.ndarray) or dc_dt.ndim == 1:
                        dc_dt = np.array(dc_dt).reshape(n_MC, 1)
                    if not isinstance(C_0, np.ndarray) or C_0.ndim == 1:
                        C_0 = np.array(C_0).reshape(n_MC, 1)
                    
                    TT, NN = np.meshgrid(t, np.arange(n_MC))

                    linear_co2_MC = linear_model(t=TT, dcdt=dc_dt, c0=C_0)
                    metrics = self.run_metrics(y_raw=self.df_data['k30_co2'].values[deadband:cutoff],
                                               y_model=linear_co2_MC)

                    dc_dt = np.squeeze(dc_dt)


                    results[f'{n}']['dcdt(linear)'][n_cutoff, n_deadband, :] = dc_dt

                    if isinstance(metrics['aic'], np.ndarray) and len(metrics['aic']) == n_MC:
                        results[f'{n}']['AIC(linear)'][n_cutoff, n_deadband, :] = metrics['aic']
                    else:
                        results[f'{n}']['AIC(linear)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['aic'])
                    if isinstance(metrics['rmse'], np.ndarray) and len(metrics['rmse']) == n_MC:
                        results[f'{n}']['RMSE(linear)'][n_cutoff, n_deadband, :] = metrics['rmse']
                    else:
                        results[f'{n}']['RMSE(linear)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['rmse'])
                    if isinstance(metrics['r2'], np.ndarray) and len(metrics['r2']) == n_MC:
                        results[f'{n}']['R2(linear)'][n_cutoff, n_deadband, :] = metrics['r2']
                    else:
                        results[f'{n}']['R2(linear)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['r2'])
                    if isinstance(metrics['nrmse'], np.ndarray) and len(metrics['nrmse']) == n_MC:
                        results[f'{n}']['nRMSE(linear)'][n_cutoff, n_deadband, :] = metrics['nrmse']
                    else:
                        results[f'{n}']['nRMSE(linear)'][n_cutoff, n_deadband, :] = np.full(n_MC, metrics['nrmse'])
                    
                
                except Exception as e:
                    print('ERROR LINEAR ####')
                    print(e)
                    print(f'Deadband: {deadband}, Cutoff: {cutoff}')

                    
                
        
        # Now results has proper 3D arrays that can be easily manipulated
        # print(f"Shape of dcdt(HM): {results[f'{n}']['dcdt(HM)'].shape}")
        return results

