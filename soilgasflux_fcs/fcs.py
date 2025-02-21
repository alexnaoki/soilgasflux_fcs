import pandas as pd
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from .hm_model import HM_model
from .linear_model import LINEAR_model
from .metrics import calculate_AIC, rmse
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
        metrics['aic'] = aic
        metrics['rmse'] = rmse_
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
            'RMSE(HM)': zeros_like.copy(), 'RMSE(linear)': zeros_like.copy()
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
                except Exception as e:
                    print('ERROR LINEAR ####')
                    print(e)

                    # aic = uncertainty.calculate_AIC(y=df_measurement['k30_co2'].values[deadband:c],
                    #                                 yhat=v[4], p=5)

                # try:
                #     return results
                # except Exception as e:
                #     print('ERROR ####')
                #     return None

        # print(results['dcdt(HM)'])
        # print(results['dcdt(linear)'])
        # fig, ax = plt.subplots(2,3, figsize=(10,6))
        # g1=ax[0,0].pcolormesh(XX, YY, results['dcdt(HM)'])
        # fig.colorbar(g1)
        # ax[0,0].set_title('dcdt HM')
        # ax[0,0].set_xlabel('Deadband')
        # ax[0,0].set_ylabel('Cutoff')

        # g4=ax[1,0].pcolormesh(XX, YY, results['AIC(HM)'])
        # fig.colorbar(g4)

        # g2=ax[0,1].pcolormesh(XX, YY, results['dcdt(linear)'])
        # fig.colorbar(g2)
        # ax[0,1].set_title('dcdt Linear')
        # ax[0,1].set_xlabel('Deadband')
        # ax[0,1].set_ylabel('Cutoff')

        # g5=ax[1,1].pcolormesh(XX, YY, results['AIC(linear)'])
        # fig.colorbar(g5)

        # g3=ax[0,2].pcolormesh(XX, YY, results['dcdt(HM)'] - results['dcdt(linear)'])
        # fig.colorbar(g3)
        # ax[0,2].set_title('dcdt HM - Linear')
        # ax[0,2].set_xlabel('Deadband')
        # ax[0,2].set_ylabel('Cutoff')

        # fig.tight_layout()

        return results


