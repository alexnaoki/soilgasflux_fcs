import numpy as np


def _normalize(values):
    finite = np.isfinite(values)
    scaled = np.full(values.shape, np.nan)
    if finite.any():
        valid = values[finite].astype(float, copy=False)
        span = valid.max() - valid.min()
        # A constant objective contributes no distance, including a one-window fit.
        scaled[finite] = (valid - valid.min()) / span if span else 0.0
    return scaled


class Pareto:
    def __init__(self, dsMC):
        self.dsMC = dsMC
        self.deadband_coords = dsMC.coords['deadband'].values
        self.cutoff_cords = dsMC.coords['cutoff'].values

    def prepare_metrics(self, normalize=True):
        '''
        Builds the two-objective space:
            metric_x: 68% uncertainty range of dcdt(HM)
            metric_y: -median logprob(HM) (so lower is better)

        Returns the 2D (cutoff, deadband) and flattened arrays. If
        ``normalize`` is True (default), values are min-max scaled to [0, 1];
        otherwise raw values are returned.
        '''
        dsMC = self.dsMC
        quantiles = dsMC['dcdt(HM)'].quantile([0.16, 0.84], dim='MC')
        uncertaintyRange = quantiles.isel(quantile=1, drop=True) - quantiles.isel(quantile=0, drop=True)
        logprob = -dsMC['logprob(HM)'].median(dim='MC')
        logprob = logprob.where(np.isfinite(logprob))

        self.logprob = logprob
        self.uncertaintyRange = uncertaintyRange

        u = uncertaintyRange.values
        l = logprob.values

        if normalize:
            u, l = _normalize(u), _normalize(l)
        return u, l, u.flatten(), l.flatten()

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

        return np.array(pareto_indices, dtype=int)

    def get_coords_pareto(self, pareto_indices):
        shape = tuple(size for dim, size in self.dsMC['dcdt(HM)'].sizes.items() if dim != 'MC')
        return np.unravel_index(pareto_indices, shape, order='C')

    def get_best_from_pareto(self, pareto_indices, metric_x, metric_y):
        coords_pareto = self.get_coords_pareto(pareto_indices)
        distance_pareto = np.sqrt(metric_x[coords_pareto] ** 2 + metric_y[coords_pareto] ** 2)
        self.argmin_distance = np.nanargmin(distance_pareto)

        best_x = coords_pareto[0][self.argmin_distance]
        best_y = coords_pareto[1][self.argmin_distance]
        return best_x, best_y
