import pathlib, json, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit import Parameters
from lmfit import conf_interval2d, report_ci
from sklearn.metrics import mean_squared_error
import seaborn as sns
from .mcmc import MCMC
import warnings
warnings.filterwarnings("ignore")

class HM_model:
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
    
    def C_0_calculated(self, gas_concentration):
        '''
        gas_concentration: should be the around the first 10 points
        '''
        try:
            n = np.shape(gas_concentration)[0]
            gas_concentration = np.array(gas_concentration).reshape((n, 1))
            x = np.arange(1,n+1).reshape((n, 1))

            regression = LinearRegression(fit_intercept=True)
            regression.fit(x, gas_concentration)
            
            C_0 = regression.intercept_
            # regression2 = stats.linregress(x.reshape(n), co2_ppm.reshape(n))
        except Exception as e:
            print(e)
        
        return C_0

    
    def target_function(self, t, cx, a, t0, c0):
        e = 2.71828
        return cx+(c0-cx)*e**(-a*(t-t0))
    
    def dcdt(self, t_0, C_0, alpha, C_x, t):
        e = 2.71828
        dcdt = alpha*(C_x - C_0)*e**(-alpha*(t-t_0))
        return dcdt



    def gas_eeflux_v2(self, volume, area, P0, W0, T0, dc_dt):
        R = 8.31446261815324 #J⋅K−1⋅mol−1
        F_o = (10*volume*P0*(1-W0/1000))*dc_dt/(R*area*(T0+273.15))
        return F_o

    def fit_target_function_cutoff(self, t, gas_concentration, c_0, deadband, cutoff,display_results=True, pi=False):
        fmodel = Model(self.target_function)
        params = fmodel.make_params(cx=c_0, a=1, t0=0, c0=c_0)
        params['c0'].vary = False
        params['a'].min=0
        # params['cx'].max=10e5
        # params['cx'].min=c_0
        params['t0'].vary=False

        try:
            if not pi:
                result = fmodel.fit(gas_concentration[deadband:cutoff], params, t=t[deadband:cutoff])
            else:
                result = fmodel.fit(gas_concentration, params, t=t[deadband:cutoff])
        except Exception as e:
            print(e)
            return None
        

        
        try:
            return {'parameters_best_fit':{'cx':result.params['cx'].value,
                                           'a':result.params['a'].value,
                                           't0':result.params['t0'].value,
                                           'c0':c_0},
                    # 'uncertainty':result.eval_uncertainty(result.params,sigma=1),
                    'uncertainty': {'cx':result.params['cx'].stderr,
                                    'a':result.params['a'].stderr,
                                    't0':result.params['t0'].stderr},
                    
                    }
        except Exception as e:
            print(e)
            return None

    
    def calculate(self, deadband, cutoff):
        '''

        '''
        if self.using_rpi:
            X_h2o = self.mole_fraction_water_vapor(self.temperature, self.humidity, self.pressure)
        else:
            X_h2o = self.X_h2o

        C_0 = self.C_0_calculated(self.co2.values[:10])
        
        result_fit = self.fit_target_function_cutoff(self.timestamp.values, 
                                                     self.co2.values, 
                                                     np.float32(C_0)[0], 
                                                     deadband=deadband, cutoff=cutoff, display_results=True)
        
        
        cx = result_fit['parameters_best_fit']['cx']
        a = result_fit['parameters_best_fit']['a']
        t0 = result_fit['parameters_best_fit']['t0']


        temperature_start = self.temperature[0]#, self.temperature[deadband],self.temperature[cutoff])
        pressure_start = self.pressure[0]#, self.pressure[deadband],self.pressure[cutoff])
        humidity_start = self.humidity[0]#, self.humidity[deadband],self.humidity[cutoff])

        
        fitted_y = self.target_function(t=self.timestamp.values[deadband:cutoff],
                                        cx=cx,
                                        a=a,
                                        t0=t0,
                                        c0=C_0)
        fitted_x = self.timestamp.values[deadband:cutoff]
        
        
        dc_dt = (self.dcdt(t0, C_0, a, cx, self.timestamp[deadband:cutoff])).mean()
        soilgasflux_CO2 = self.gas_eeflux_v2(volume=self.volume, 
                                             area=self.area, 
                                             P0=self.pressure.head(1)[0], 
                                             W0=X_h2o.head(1)[0],
                                             T0=self.temperature.head(1)[0], 
                                             dc_dt=dc_dt)
        return dc_dt, C_0,cx, a, t0, soilgasflux_CO2, deadband, cutoff
        # return dc_dt,soilgasflux_CO2, uncertainty, fitted_x,fitted_y, fitted_y_lower, fitted_y_higher, dc_dt_lower, dc_dt_higher, lower_ci, higher_ci, cx, a, t0, temperature_start, pressure_start, humidity_start,C_0[0]


    def calculate_MC(self, deadband, cutoff, n=1000):
        '''
        
        '''
        if self.using_rpi:
            X_h2o = self.mole_fraction_water_vapor(self.temperature, self.humidity, self.pressure)
        else:
            X_h2o = self.X_h2o

        C_0 = self.C_0_calculated(self.co2.values[:10])
        
        result_fit = self.fit_target_function_cutoff(self.timestamp.values, 
                                                     self.co2.values, 
                                                     np.float32(C_0)[0], 
                                                     deadband=deadband, cutoff=cutoff, display_results=True)
        
        
        cx = result_fit['parameters_best_fit']['cx']
        a = result_fit['parameters_best_fit']['a']
        t0 = result_fit['parameters_best_fit']['t0']

        sigma_cx = result_fit['uncertainty']['cx']
        sigma_a = result_fit['uncertainty']['a']
        sigma_t0 = result_fit['uncertainty']['t0']

        cxMC = cx + sigma_cx*np.random.normal(size=n)
        aMC = a + sigma_a*np.random.normal(size=n)
        t0MC = t0 + sigma_t0*np.random.normal(size=n)

<<<<<<< HEAD
=======
        mcmc = MCMC()
        sampler, flat_samples = mcmc.run_mcmc(t=self.timestamp.values[deadband:cutoff], 
                                              y=self.co2.values[deadband:cutoff], 
                                              yerr=0.5, # measurement error 
                                              c0=C_0, 
                                              cx_bf=cx, 
                                              alpha_bf=a,
                                              nwalkers=100, nsteps=1000)
        
        dcdt_mcmc = self.dcdt(t_0=0, C_0=C_0, 
                              alpha=flat_samples[:,0], C_x=flat_samples[:,1], 
                              t=self.timestamp[deadband:cutoff].mean())

        dcdt_quantiles = np.quantile(dcdt_mcmc, [0.16, 0.5, 0.84])
        dcdt_mcmc_filter = dcdt_mcmc[(dcdt_mcmc > dcdt_quantiles[0]) & (dcdt_mcmc < dcdt_quantiles[2])]

        # print(len(dcdt_mcmc_filter))
        random_index = np.random.choice(len(dcdt_mcmc_filter), n)
        dcdt_samples = dcdt_mcmc_filter[random_index]

        dc_dtMC = dcdt_samples

>>>>>>> dabb4f2e65f5f933919eb08d8b4669a111784418
        temperature_start = self.temperature[0]#, self.temperature[deadband],self.temperature[cutoff])
        pressure_start = self.pressure[0]#, self.pressure[deadband],self.pressure[cutoff])
        humidity_start = self.humidity[0]#, self.humidity[deadband],self.humidity[cutoff])

        
        # fitted_y = self.target_function(t=self.timestamp.values[deadband:cutoff],
        #                                 cx=cx,
        #                                 a=a,
        #                                 t0=t0,
        #                                 c0=C_0)
        # fitted_x = self.timestamp.values[deadband:cutoff]
        
        
        # dc_dt = (self.dcdt(t0, C_0, a, cx, self.timestamp)).mean()
        # dc_dtMC = self.dcdt(t0MC, C_0, aMC, cxMC, self.timestamp[deadband:cutoff].mean())
        
        soilgasflux_CO2MC = self.gas_eeflux_v2(volume=self.volume, 
                                             area=self.area, 
                                             P0=self.pressure.head(1)[0], 
                                             W0=X_h2o.head(1)[0],
                                             T0=self.temperature.head(1)[0], 
                                             dc_dt=dc_dtMC)
        

        return dc_dtMC, C_0,cxMC, aMC, t0MC, soilgasflux_CO2MC, deadband, cutoff