import numpy as np
import scipy.stats as stats


def _sse(resid):
    # Handle both 1-D (single fit) and 2-D (MC fits stacked on axis 0) residuals.
    return np.sum(resid ** 2, axis=-1) if resid.ndim == 2 else np.sum(resid ** 2)


def calculate_AIC(y, yhat, p):
    n = len(y)
    sse = _sse(y - yhat)
    return n * np.log(sse / n) + 2 * p


def calculate_BIC(y, yhat, p):
    n = len(y)
    sse = _sse(y - yhat)
    return n * np.log(sse / n) + p * np.log(n)


def rmse(y, yhat):
    n = len(y)
    return np.sqrt(_sse(y - yhat) / n)


def normalized_rmse(y, yhat):
    range_y = np.nanmax(y) - np.nanmin(y)
    return rmse(y, yhat) / range_y


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


def confidence_interval(x, xd, yd, yp, p, conf=0.95):
    alpha = 1 - conf
    n = len(yd)
    m = len(p)
    dof = n - m
    t = stats.t.ppf(1 - alpha / 2, dof)
    s_err = np.sqrt(np.sum((yd - yp) ** 2) / (n - m))
    return t * s_err * np.sqrt(1 / n + (x - np.mean(xd)) ** 2 / np.sum((xd - np.mean(xd)) ** 2))


def minimum_detectable_flux(Aa, tc, freq, V, A, P, T):
    '''
    Nickerson et al. (2016). Minimum detectable flux [nmol m-2 s-1].

    Aa: analytical accuracy [ppb]
    tc: chamber closure time [s]
    freq: measurement frequency [Hz]
    V: chamber volume [m^3]
    A: chamber surface area [m^2]
    P: atmospheric pressure [Pa]
    T: ambient temperature [K]
    '''
    R = 8.314
    return (Aa / (tc * (tc * freq) ** 0.5)) * (V * P / (A * R * T))
