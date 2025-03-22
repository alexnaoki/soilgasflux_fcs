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

        print((self.diff_dcdt[target_dcdt[1][:,0], target_dcdt[1][:,1]]).argmax())

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
                                    c0=self.c0), color='red', linestyle='--')
            ax[0].plot(models.hm_model(t=np.arange(self.total_time),
                                    cx=self.cs[target_dcdt[1][straight_curve][0]],
                                    a=self.alpha[target_dcdt[1][straight_curve][1]],
                                    t0=0,
                                    c0=self.c0), color='red')
            ax[0].plot(models.hm_model(t=np.arange(self.total_time),
                                    cx=self.cs[target_dcdt[1][inbetween_curve][0]],
                                    a=self.alpha[target_dcdt[1][inbetween_curve][1]],
                                    t0=0,
                                    c0=self.c0), color='red', alpha=0.5)
            
            ax[1].axvline(target_dcdt[0][big_curve], color='red', linestyle='--')
            ax[1].axvline(target_dcdt[0][straight_curve], color='red')
            ax[1].axvline(target_dcdt[0][inbetween_curve], color='red', alpha=0.5)
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
