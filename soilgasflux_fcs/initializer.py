import pathlib, json, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit import Parameters
from sklearn.metrics import mean_squared_error
import seaborn as sns

class Initializer:
    def __init__(self, raw_data, metadata, using_rpi=True):
        
        self.area = metadata['area']
        self.volume = metadata['volume']
        self.using_rpi = using_rpi
        if self.using_rpi:
            # Metadata

            try:
                # Raw data
                self.timestamp = raw_data['timedelta']
            
                # print('No timedelta')
                self.temperature = raw_data['si_temperature']
                self.humidity = raw_data['si_humidity']
                
                self.pressure = raw_data['bmp_pressure']/1000 #kPa
                self.co2 = raw_data['k30_co2']
            except Exception as e:
                # print(e)
                pass
        else:
            self.timestamp = raw_data['timestamp']
            self.temperature = raw_data['chamber_t']
            self.pressure = raw_data['chamber_p']
            self.co2 = raw_data['co2']
            self.X_h2o = raw_data['h2o']
            
        self._C_0 = None
        self._a = None
        self._t_0 = None
        self._C_x = None