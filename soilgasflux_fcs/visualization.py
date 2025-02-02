import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import hm_model, linear_model, hm_model_dcdt

class Visualizer:
    def __init__(self):
        print('Visualizer initialized')
        self.fig, self.ax = plt.subplots(1,2, figsize=(10,5))
        plot_HM = False
        plot_linear = False
        # pass

    def plot_hm_results(self, HM_results):
        '''
        HM_results: 
            [dc_dt, C_0, cx, a, t0, soilgasflux_CO2,deadband,cutoff]
        '''
        print('Plotting HM results')
        dc_dt, C_0, cx, a, t0, soilgasflux_CO2, deadband, cutoff = HM_results
        t = np.arange(deadband, cutoff, 1)

        hm_co2 = hm_model(t=t, 
                          cx=cx, a=a, t0=t0, c0=C_0)
        
        hm_dcdt = hm_model_dcdt(t0=t0, c0=C_0, a=a, cx=cx, t=t)
        
        # fig, ax = plt.subplots(1,1, figsize=(5,5))
        self.ax[0].plot(t, hm_co2, label='HM model')

        self.ax[1].plot(t, hm_dcdt, label='HM model')

        plot_HM = True
        self.ax[0].legend()
        self.fig.show()


    def plot_linear_results(self, LINEAR_results):
        '''
        LINEAR_results: 
            [dc_dt, C_0, soilgasflux_CO2, deadband,cutoff]
        '''
        print('Plotting Linear results')
        dc_dt, C_0, soilgasflux_CO2, deadband, cutoff = LINEAR_results
        t = np.arange(deadband, cutoff, 1)

        linear_co2 = linear_model(t=t, dcdt=dc_dt, c0=C_0)

        self.ax[0].plot(t, linear_co2, label='Linear model')

        plot_linear = True
        self.ax[0].legend()
        self.fig.show()

    def plot_raw_data(self, dataframe):
        '''
        dataframe:
            ['timedelta', 'k30_co2']
        '''

        self.ax[0].scatter(dataframe['timedelta'], dataframe['k30_co2'], label='Raw data', color='black', s=2, zorder=0)
        self.ax[0].legend()
        self.fig.show()
