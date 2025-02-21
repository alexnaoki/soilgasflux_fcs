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

class LINEAR_model:
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

    def calculate_saturated_vapor_pressure(self, temperature):
        '''
        temperature: Celsius
        '''
        # Buck's equation
        e_s = 0.61121*np.exp((18.678 - temperature/234.5)*(temperature/(257.14 + temperature))) # kPa
        return e_s
    
    def mole_fraction_water_vapor(self, temperature, humidity, pressure):
        e_s = self.calculate_saturated_vapor_pressure(temperature) #saturation vapor pressure in kPa
        e = e_s * (humidity/100) # vapor pressure in kPa
        
        X_h2o = (e / pressure)*1000 # m mol/mol
        return X_h2o
    
    def target_function(self, t, dcdt, c0):
        '''
        linear function

        y = dcdt * t + c0
        '''
        return dcdt*t + c0

    def gas_eeflux_v2(self, volume, area, P0, W0, T0, dc_dt):
        R = 8.31446261815324 #J⋅K−1⋅mol−1
        F_o = (10*volume*P0*(1-W0/1000))*dc_dt/(R*area*(T0+273.15))
        return F_o
    
    def fit_target_function_cutoff(self, t, gas_concentration, c_0, deadband, cutoff):
        fmodel = Model(self.target_function)
        params = fmodel.make_params(dcdt=1, c0=c_0)
        params['c0'].vary = False

    
        try:
            result = fmodel.fit(gas_concentration[deadband:cutoff], params, t=t[deadband:cutoff])
            # print(result)
        except Exception as e:
            print(e)
            result = None

        try:
            return {'parameters_best_fit':{'dcdt':result.best_values['dcdt'], 
                                           'c0':result.best_values['c0']}, 
                                        #    'result':result
                                           }
        except Exception as e:
            print(e)
            return None
        
    def calculate(self, deadband, cutoff):
        '''
        
        '''

        if self.using_rpi:
            X_h2o = self.mole_fraction_water_vapor(self.temperature, self.humidity, self.pressure)[deadband]
        else:
            X_h2o = self.X_h2o

        C_0 = self.co2.values[0]

        result_fit = self.fit_target_function_cutoff(t=self.timestamp.values,
                                                     gas_concentration=self.co2.values,
                                                    #  c_0=np.float32(C_0)[0],
                                                    c_0=C_0,
                                                     deadband=deadband, cutoff=cutoff)
        
        # print(result_fit)

        dcdt = result_fit['parameters_best_fit']['dcdt']
        c0 = result_fit['parameters_best_fit']['c0']

        temperature_start = self.temperature[0]#, self.temperature[deadband],self.temperature[cutoff])
        pressure_start = self.pressure[0]#, self.pressure[deadband],self.pressure[cutoff])
        humidity_start = self.humidity[0]#, self.humidity[deadband],self.humidity[cutoff])

        soilgasflux_CO2 = self.gas_eeflux_v2(volume=self.volume,
                                             area=self.area,
                                             P0=pressure_start,
                                             W0=X_h2o,
                                             T0=temperature_start,
                                             dc_dt=dcdt)
        
        # print(X_h2o)

        return dcdt, c0, soilgasflux_CO2, deadband, cutoff
