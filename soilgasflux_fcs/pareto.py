import numpy as np
import pandas as pd
import xarray as xr


class Pareto:
    def __init__(self, dsMC):
        self.dsMC = dsMC

        self.deadband_coords = dsMC.coords['deadband'].values
        self.cutoff_cords = dsMC.coords['cutoff'].values

    def prepare_metrics(self):
        '''
        metric_x: Uncertainty range 
        metric_y: logprob from MCMC
        '''
        dsMC = self.dsMC
        # uncertaintyRange = dsMC.quantile(0.84, dim=['MC','time'])['dcdt(HM)'] - dsMC.quantile(0.84, dim=['MC','time'])['dcdt(HM)']
        uncertaintyRange = dsMC.quantile(0.84, dim=['MC'])['dcdt(HM)'] - dsMC.quantile(0.16, dim=['MC'])['dcdt(HM)']
        # logprob = -dsMC.median(dim=['MC','time'])['logprob(HM)']
        logprob = -dsMC.median(dim=['MC'])['logprob(HM)']
        logprob = logprob.where(logprob != np.inf, np.nan)

        self.logprob = logprob
        self.uncertaintyRange = uncertaintyRange

        # print('uncertaintyRange:', uncertaintyRange)

        norm_uncertaintyRange = (uncertaintyRange.values - np.nanmin(uncertaintyRange.values)) / (np.nanmax(uncertaintyRange.values) - np.nanmin(uncertaintyRange.values))
        norm_logprob = (logprob.values - np.nanmin(logprob.values)) / (np.nanmax(logprob.values) - np.nanmin(logprob.values))

        flatnorm_uncertaintyRange = (uncertaintyRange.values.flatten() - np.nanmin(uncertaintyRange.values.flatten())) / (np.nanmax(uncertaintyRange.values.flatten()) - np.nanmin(uncertaintyRange.values.flatten()))
        flatnorm_logprob = (logprob.values.flatten() - np.nanmin(logprob.values.flatten())) / (np.nanmax(logprob.values.flatten()) - np.nanmin(logprob.values.flatten()))

        return norm_uncertaintyRange, norm_logprob, flatnorm_uncertaintyRange, flatnorm_logprob

    def find_pareto_front(self, x, y, maximize_x=False, maximize_y=False):
        """
        Find the Pareto front for two objectives
        
        Parameters:
        -----------
        x, y : array-like
            Values of the two objectives
        maximize_x, maximize_y : bool
            Whether to maximize (True) or minimize (False) each objective
        
        Returns:
        --------
        pareto_indices : ndarray
            Indices of points on the Pareto front
        """
        
        # Copy arrays to avoid modifying originals
        x_values = np.copy(x)
        y_values = np.copy(y)
        
        # Convert maximization to minimization
        if maximize_x:
            x_values = -x_values
        if maximize_y:
            y_values = -y_values
        
        points = np.column_stack((x_values, y_values))
        pareto_indices = []
        
        for i, point in enumerate(points):
            if np.isnan(point).any():
                continue
                
            dominated = False
            for j, other_point in enumerate(points):
                if i != j and not np.isnan(other_point).any():
                    # Check if other_point dominates point (smaller is better)
                    if (all(other_point <= point) and any(other_point < point)):
                        dominated = True
                        break
            
            if not dominated:
                pareto_indices.append(i)
        
        return np.array(pareto_indices)

    def get_coords_pareto(self, pareto_indices):
        """
        Get the coordinates of the Pareto front points in the original dataset
        
        Parameters:
        -----------
        pareto_indices : ndarray
            Indices of points on the Pareto front
        
        Returns:
        --------
        coords_pareto : tuple
            Coordinates of the Pareto front points in the original dataset
        """
        # coords_pareto = np.unravel_index(pareto_indices, self.dsMC.shape)
        coords_pareto = np.unravel_index(pareto_indices, self.dsMC.median(dim=['MC'])['dcdt(HM)'].shape, order='C')
        return coords_pareto

    def get_best_from_pareto(self, pareto_indices, metric_x, metric_y):
        
        # coords_pareto = np.unravel_index(pareto_indices, self.dsMC['dcdt(HM)'].shape)
        coords_pareto = np.unravel_index(pareto_indices, self.dsMC.median(dim=['MC'])['dcdt(HM)'].shape, order='C')
        # print('shape:', self.dsMC.median(dim=['MC'])['dcdt(HM)'].shape)
        # print('coords_pareto:', coords_pareto)
        distance_pareto = np.sqrt(
            (metric_x[coords_pareto]**2) + (metric_y[coords_pareto]**2)
        )
        self.argmin_distance = np.nanargmin(distance_pareto)
        # print('argmin_distance:', argmin_distance)

        best_x = coords_pareto[0][self.argmin_distance]
        best_y = coords_pareto[1][self.argmin_distance]
        # print('best_x:', best_x, 'best_y:', best_y)

        return best_x, best_y
    
    def logprob(self):
        """
        Get the log probability values from the dataset
        """
        return self.logprob
    def uncertaintyRange(self):
        """
        Get the uncertainty range values from the dataset
        """
        return self.uncertaintyRange
    def argmin_distance(self):
        """
        Get the index of the point with the minimum distance from the Pareto front
        """
        return self.argmin_distance