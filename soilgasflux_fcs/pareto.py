import numpy as np


class Pareto:
    def __init__(self, dsMC):
        self.dsMC = dsMC
        self.deadband_coords = dsMC.coords['deadband'].values
        self.cutoff_cords = dsMC.coords['cutoff'].values

    def prepare_metrics(self):
        '''
        Builds the two-objective space:
            metric_x: 68% uncertainty range of dcdt(HM)
            metric_y: -median logprob(HM) (so lower is better)

        Returns both the 2D (cutoff, deadband) and flattened normalized arrays.
        '''
        dsMC = self.dsMC
        uncertaintyRange = dsMC.quantile(0.84, dim=['MC'])['dcdt(HM)'] - dsMC.quantile(0.16, dim=['MC'])['dcdt(HM)']
        logprob = -dsMC.median(dim=['MC'])['logprob(HM)']
        logprob = logprob.where(logprob != np.inf, np.nan)

        self.logprob = logprob
        self.uncertaintyRange = uncertaintyRange

        u = uncertaintyRange.values
        l = logprob.values
        norm_u = (u - np.nanmin(u)) / (np.nanmax(u) - np.nanmin(u))
        norm_l = (l - np.nanmin(l)) / (np.nanmax(l) - np.nanmin(l))

        u_flat = u.flatten()
        l_flat = l.flatten()
        flat_norm_u = (u_flat - np.nanmin(u_flat)) / (np.nanmax(u_flat) - np.nanmin(u_flat))
        flat_norm_l = (l_flat - np.nanmin(l_flat)) / (np.nanmax(l_flat) - np.nanmin(l_flat))

        return norm_u, norm_l, flat_norm_u, flat_norm_l

    def find_pareto_front(self, x, y, maximize_x=False, maximize_y=False):
        '''
        Return indices of non-dominated points given two objectives.
        '''
        x_values = -np.copy(x) if maximize_x else np.copy(x)
        y_values = -np.copy(y) if maximize_y else np.copy(y)

        points = np.column_stack((x_values, y_values))
        pareto_indices = []

        for i, point in enumerate(points):
            if np.isnan(point).any():
                continue
            dominated = False
            for j, other in enumerate(points):
                if i == j or np.isnan(other).any():
                    continue
                if np.all(other <= point) and np.any(other < point):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        return np.array(pareto_indices)

    def get_coords_pareto(self, pareto_indices):
        shape = self.dsMC.median(dim=['MC'])['dcdt(HM)'].shape
        return np.unravel_index(pareto_indices, shape, order='C')

    def get_best_from_pareto(self, pareto_indices, metric_x, metric_y):
        coords_pareto = self.get_coords_pareto(pareto_indices)
        distance_pareto = np.sqrt(metric_x[coords_pareto] ** 2 + metric_y[coords_pareto] ** 2)
        self.argmin_distance = np.nanargmin(distance_pareto)

        best_x = coords_pareto[0][self.argmin_distance]
        best_y = coords_pareto[1][self.argmin_distance]
        return best_x, best_y
