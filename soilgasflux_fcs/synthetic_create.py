import numpy as np
import pandas as pd
import json
import pathlib
import matplotlib.pyplot as plt
import datetime as dt
from soilgasflux_fcs import models
from matplotlib.colors import LogNorm

class Generator:
    def __init__(self, total_time, c0):
        self.total_time = total_time
        self.c0 = c0

        self.datetime_creation = dt.datetime.now()
        self.selected = {
            'alpha': [], 'cs': [], 't0': [], 'c0': []
        }

    def find_nearest(self, array, value):
        array = np.asarray(array)
        # print(array)
        # idx = (np.abs(array - np.array(value))).argmin()
        print(np.abs(array-value).argmin())
        print(array.flat[np.abs(array-value).argmin()])
        # return array[idx]

    def find_nearest_n(self, array, value, n=100):
        """
        Find the n closest values to the target value in a 2D array
        
        Parameters:
        -----------
        array : ndarray
            2D array to search in
        value : float
            Target value to find closest elements to
        n : int, default=100
            Number of closest values to return
        
        Returns:
        --------
        values : ndarray
            Array of the n closest values
        indices : ndarray
            Array of (row, col) indices for each value
        """
        # Calculate absolute difference between each element and the target
        diff = np.abs(array - value)
        
        # Flatten the array for argpartition
        flat_diff = diff.flatten()
        
        # Get indices of n smallest differences
        flat_indices = np.argpartition(flat_diff, n)[:n]
        
        # Sort indices by increasing difference
        flat_indices = flat_indices[np.argsort(flat_diff[flat_indices])]
        
        # Convert back to 2D indices
        rows, cols = np.unravel_index(flat_indices, array.shape)
        
        # Get the actual values
        values = array[rows, cols]
        
        # Create array of (row, col) indices
        indices = np.column_stack((rows, cols))
        
        return values, indices

    def alpha_cs_plot(self, alpha_min, alpha_max, cs_min, cs_max, n=100):
        alpha_start = np.log10(alpha_min)
        alpha_stop = np.log10(alpha_max)

        cs_start = np.log10(cs_min)
        cs_stop = np.log10(cs_max)

        self.alpha = np.logspace(alpha_start, alpha_stop, n, base=10)
        self.cs = np.logspace(cs_start, cs_stop, n, base=10)

        self.aa, self.cc = np.meshgrid(self.alpha, self.cs)

        self.c_cx = models.hm_model(t=self.total_time, cx=self.cc, a=self.aa, t0=0, c0=self.c0)
        self.dcdt = models.hm_model_dcdt(t0=0, c0=self.c0, a=self.aa, cx=self.cc, t=np.arange(self.total_time).mean())

        self.dcdt_tInitial = models.hm_model_dcdt(t0=0, c0=self.c0, a=self.aa, cx=self.cc, t=1)
        self.dcdt_tEnd = models.hm_model_dcdt(t0=0, c0=self.c0, a=self.aa, cx=self.cc, t=self.total_time)
        # diff_dcdt = dcdt_tEnd - dcdt_tInitial
        self.diff_dcdt = self.dcdt_tInitial - self.dcdt_tEnd

        # print(diff_dcdt)
        
        levels = np.arange(430, 820, 20)
        levels_dcdt = np.arange(0, 2, 0.1)

        # levels_dcdt_diff = np.arange(-1,1,0.01)
        levels_dcdt_diff = np.logspace(-5, 0, 100)

        # levels_rel_dcdt_diff = np.logspace(-5, 0, 20)
        levels_rel_dcdt_diff = np.arange(0, 1, 0.01)
        
        
        fig, ax = plt.subplots(2,2, figsize=(8,7))
        c_cx_plot = ax[0,0].contourf(self.aa, self.cc, self.c_cx, 
                                   levels=levels,
                                   cmap='viridis', vmin=430, vmax=800,
                                   )
        
        print(np.where((self.c_cx > 430) & (self.c_cx <= 800))[0].shape)
        dcdt_masked = np.ma.masked_where((self.c_cx < 430) | (self.c_cx > 800), self.dcdt)
        dcdt_diff_masked = np.ma.masked_where((self.c_cx < 430) | (self.c_cx > 800), self.diff_dcdt)

        rel_dcdt_diff = dcdt_diff_masked/dcdt_masked


        dcdt_plot = ax[0,1].contourf(self.aa, self.cc, dcdt_masked,
                                   cmap='viridis',
                                   levels=levels_dcdt,
                                #    vmin=0, vmax=2
                                   )

        dcdt_diff_plot = ax[1,0].contourf(self.aa, self.cc, dcdt_diff_masked,
                                      cmap='viridis',
                                      levels=levels_dcdt_diff,
                                      norm='log'
                                    #   norm=LogNorm(vmin=0.01, vmax=1)
                                  #    vmin=0, vmax=2
                                      )
        
        rel_dcdt_diff_plot = ax[1,1].contourf(self.aa, self.cc, rel_dcdt_diff,
                                              cmap='viridis',
                                                levels=levels_rel_dcdt_diff,
                                                norm='log'
                                                )
        

        cbar = fig.colorbar(c_cx_plot, ax=ax[0,0])
        cbar.set_label(f'Concentration @ t={self.total_time}')
        ax[0,0].set_xscale('log')
        ax[0,0].set_yscale('log')
        ax[0,0].set_xlabel('alpha')
        ax[0,0].set_ylabel('cs')
        ax[0,0].set_title('Concentration vs alpha and cs')

        cbar2 = fig.colorbar(dcdt_plot, ax=ax[0,1])
        cbar2.set_label(f'dcdt @ t={self.total_time}')
        ax[0,1].set_xscale('log')
        ax[0,1].set_yscale('log')
        ax[0,1].set_xlabel('alpha')
        ax[0,1].set_ylabel('cs')
        ax[0,1].set_title('dcdt vs alpha and cs')

        cbar3 = fig.colorbar(dcdt_diff_plot, ax=ax[1,0])
        cbar3.set_label(f'dcdt diff')
        ax[1,0].set_xscale('log')
        ax[1,0].set_yscale('log')
        ax[1,0].set_xlabel('alpha')
        ax[1,0].set_ylabel('cs')
        ax[1,0].set_title('dcdt diff vs alpha and cs')

        cbar4 = fig.colorbar(rel_dcdt_diff_plot, ax=ax[1,1])
        cbar4.set_label(f'rel dcdt diff')
        ax[1,1].set_xscale('log')
        ax[1,1].set_yscale('log')
        ax[1,1].set_xlabel('alpha')
        ax[1,1].set_ylabel('cs')
        ax[1,1].set_title('rel dcdt diff vs alpha and cs')
        
        fig.tight_layout()

        fig.show()
        
    def cc_curve_plot(self, selected_dcdt):

        target_dcdt = self.find_nearest_n(self.dcdt, selected_dcdt, n=100)
        # print(target_dcdt)

        # print((self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]]).argmax())

        biggest_curve = self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]].argmax()
        most_straight_curve = self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]].argmin()

        n = 5
        top_n_biggest_curve = self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]].argsort()[-n:][::-1]
        top_n_most_straight_curve = self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]].argsort()[:n]
        top_n_inbetween_curve = self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]].argsort()[int(100/2):int(100/2)+n]


        # closest_values = target_dcdt

        fig, ax = plt.subplots(1,3, figsize=(10,3))
        levels_dcdt = np.arange(0, 2, 0.1)
        ax[2].contourf(self.aa, self.cc, self.dcdt, cmap='viridis', levels=levels_dcdt)

        for dcdt, idx in zip(*target_dcdt):
            # print(idx)
            ax[0].plot(models.hm_model(t=np.arange(self.total_time), 
                                    cx=self.cs[idx[0]],
                                    a=self.alpha[idx[1]], 
                                    t0=0, 
                                    c0=self.c0), color='blue', alpha=0.02)
            ax[2].scatter(self.alpha[idx[1]], self.cs[idx[0]], color='blue', alpha=0.5, s=2)
            
            
        for big_curve, straight_curve, inbetween_curve in zip(top_n_biggest_curve, top_n_most_straight_curve, top_n_inbetween_curve):
            ax[0].plot(models.hm_model(t=np.arange(self.total_time),
                                    cx=self.cs[target_dcdt[1][big_curve][0]],
                                    a=self.alpha[target_dcdt[1][big_curve][1]],
                                    t0=0,
                                    c0=self.c0), color='red', linestyle='-.')
            ax[0].plot(models.hm_model(t=np.arange(self.total_time),
                                    cx=self.cs[target_dcdt[1][straight_curve][0]],
                                    a=self.alpha[target_dcdt[1][straight_curve][1]],
                                    t0=0,
                                    c0=self.c0), color='red')
            ax[0].plot(models.hm_model(t=np.arange(self.total_time),
                                    cx=self.cs[target_dcdt[1][inbetween_curve][0]],
                                    a=self.alpha[target_dcdt[1][inbetween_curve][1]],
                                    t0=0,
                                    c0=self.c0), color='red', linestyle='--')
            
            self.selected['alpha'].append(self.alpha[target_dcdt[1][big_curve][1]])
            self.selected['cs'].append(self.cs[target_dcdt[1][big_curve][0]])
            self.selected['t0'].append(0)
            self.selected['c0'].append(self.c0)

            self.selected['alpha'].append(self.alpha[target_dcdt[1][straight_curve][1]])
            self.selected['cs'].append(self.cs[target_dcdt[1][straight_curve][0]])
            self.selected['t0'].append(0)
            self.selected['c0'].append(self.c0)

            self.selected['alpha'].append(self.alpha[target_dcdt[1][inbetween_curve][1]])
            self.selected['cs'].append(self.cs[target_dcdt[1][inbetween_curve][0]])
            self.selected['t0'].append(0)
            self.selected['c0'].append(self.c0)
            

            
            ax[1].axvline(target_dcdt[0][big_curve], color='red', linestyle='-.')
            ax[1].axvline(target_dcdt[0][straight_curve], color='red')
            ax[1].axvline(target_dcdt[0][inbetween_curve], color='red', linestyle='--')
            ax[2].scatter(self.alpha[target_dcdt[1][big_curve][1]], self.cs[target_dcdt[1][big_curve][0]], color='red')
            ax[2].scatter(self.alpha[target_dcdt[1][straight_curve][1]], self.cs[target_dcdt[1][straight_curve][0]], color='red')
            ax[2].scatter(self.alpha[target_dcdt[1][inbetween_curve][1]], self.cs[target_dcdt[1][inbetween_curve][0]], color='red', alpha=0.5)
        
        ax[1].hist(target_dcdt[0][:], bins=20, alpha=0.5, color='blue')
        
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].set_xlabel('alpha')
        ax[2].set_ylabel('cs')
        ax[2].set_title('dcdt vs alpha and cs')

        fig.tight_layout()


    def generate_base(self, alpha, cs, c0, t0,total_time, deadband, background_band,
                      unmixed_phase, unmixed_disturbance_intensity, 
                      mixed_phase_disturbance, disturbance_intensity, disturbance_starting_point,
                      add_noise, noise_intensity):
        config  = {}
        time = np.arange(total_time)
        final_values = np.zeros(total_time)

        background_values = np.ones(total_time)*c0

        c_cx = models.hm_model(t=time, cx=cs, a=alpha, t0=t0, c0=c0)

        final_values += c_cx
        final_values[:deadband] = background_values[:deadband]

        if deadband > 0:
            diff = c_cx[deadband] - c0

            interpolated = np.exp(np.interp(np.arange(deadband), [0, deadband], [np.log(diff*0.1), np.log(diff)]))

            final_values[:deadband] = interpolated+c0
            final_values[0] = c0

        if add_noise:
            noise = np.random.normal(0, noise_intensity, total_time)
            final_values[:] += noise
        
        # plt.scatter(time, background_values, color='blue', label='background', s=2)
        # plt.scatter(time, c_cx, color='red', label='c_cx', s=2)
        plt.scatter(time, final_values, color='green', label='final', s=2)

        config['alpha'] = alpha
        config['c_s'] = cs
        config['c_c0'] = c0
        config['t0'] = t0
        config['total_time'] = total_time
        config['deadband'] = deadband
        config['background_band'] = background_band
        config['unmixed_phase'] = unmixed_phase
        config['unmixed_disturbance_intensity'] = unmixed_disturbance_intensity
        config['mixed_phase_disturbance'] = mixed_phase_disturbance
        config['disturbance_intensity'] = disturbance_intensity
        config['disturbance_starting_point'] = disturbance_starting_point
        config['add_noise'] = add_noise
        config['noise_intensity'] = noise_intensity
        config['final_value'] = final_values

        return config

    def write_file(self, config, save_path):
        data = {'raw_data': {}}
        
        k30 = config['final_value'].tolist()
        datetime = np.arange(len(k30))
        bmp_pressure = np.ones(len(k30))*99000
        bmp_temperature = np.ones(len(k30))*20
        si_humidity = np.ones(len(k30))*70
        si_temperature = np.ones(len(k30))*20

        print(self.datetime_creation, self.datetime_creation + dt.timedelta(seconds=len(datetime)))
        datetime_list = [self.datetime_creation + dt.timedelta(seconds=int(i)) for i in np.arange(len(k30))]
        datetime_utc = [i.strftime('%Y-%m-%d %H:%M:%S') for i in datetime_list]
        self.datetime_creation = self.datetime_creation + dt.timedelta(seconds=len(datetime))

        data['raw_data']['datetime'] = datetime.tolist()
        data['raw_data']['datetime_utc'] = datetime_utc
        data['raw_data']['k30_co2'] = k30
        data['raw_data']['bmp_pressure'] = bmp_pressure.tolist()
        data['raw_data']['bmp_temperature'] = bmp_temperature.tolist()
        data['raw_data']['si_humidity'] = si_humidity.tolist()
        data['raw_data']['si_temperature'] = si_temperature.tolist()

        config['final_value'] = None
        data['config'] = config

        filename_datetime = datetime_list[0]
        filename = f'{filename_datetime.year}-{filename_datetime.month}-{filename_datetime.day}_{filename_datetime.hour}-{filename_datetime.minute}-{filename_datetime.second}'
        with open(f'{save_path}/{filename}.json', 'w') as f:
            json.dump(data, f)
            print(f'File {filename}.json saved')
        
        return data

    def create_selected(self, save_path):
        for i in range(len(self.selected['alpha'])):
            # print(i)
            # print(self.selected['alpha'][i], self.selected['cs'][i], self.selected['t0'][i], self.selected['c0'][i])
            config = self.generate_base(alpha=self.selected['alpha'][i], 
                                        cs=self.selected['cs'][i], 
                                        c0=self.selected['c0'][i], 
                                        t0=self.selected['t0'][i], 
                                        total_time=self.total_time, 
                                        deadband=0, 
                                        background_band=0,
                                        unmixed_phase=0, 
                                        unmixed_disturbance_intensity=0, 
                                        mixed_phase_disturbance=0, 
                                        disturbance_intensity=0, 
                                        disturbance_starting_point=0,
                                        add_noise=False, 
                                        noise_intensity=0)
            # print(config)
            self.write_file(config, save_path=save_path)
            
