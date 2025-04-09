import numpy as np
import pandas as pd
import scipy.stats as stats

def calculate_AIC(y, yhat, p):
    '''
    Inputs:
    y: observed values
    yhat: predicted values
    p: number of parameters
    '''
    n = len(y)
    resid = y - yhat
    try:
        sse = np.sum(resid**2, axis=1)
    except:
        sse = np.sum(resid**2)
    aic = n * np.log(sse/n) + 2*p
    return aic

def calculate_BIC(y, yhat, p):
    '''
    
    '''
    n = len(y)
    resid = y - yhat
    sse = np.sum(resid**2)
    bic = n * np.log(sse/n) + p*np.log(n)
    return bic

def rmse(y, yhat):
    '''
    Inputs:
    y: observed values
    yhat: predicted values
    '''
    n = len(y)
    # print(n)
    resid = y - yhat
    try:
        rmse = np.sqrt(np.sum(resid**2, axis=1)/n)
    except:
        rmse = np.sqrt(np.sum(resid**2)/n)
    return rmse

def normalized_rmse(y, yhat):
    '''
    Inputs:
    y: observed values
    yhat: predicted values
    '''
    n = len(y)
    resid = y - yhat
    try:
        rmse = np.sqrt(np.sum(resid**2, axis=1)/n)
    except:
        rmse = np.sqrt(np.sum(resid**2)/n)
    min_y = np.min(y)
    max_y = np.max(y)
    range_y = max_y - min_y
    try:
        nrmse = rmse / range_y
    except:
        nrmse = rmse / (max_y - min_y)
    return nrmse

def r2(y, yhat):
    '''
    Inputs:
    y: observed values
    yhat: predicted values
    '''
    n = len(y)
    resid = y - yhat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res/ss_tot)
    return r2


def confidence_interval(x, xd, yd, yp, p, conf=0.95):
    '''
    Inputs:
    x: requested points
    xd: x data
    yd: y data
    yp: predicted values
    p: parameters
    conf: confidence level
    '''
    alpha = 1 - conf
    n = len(yd)
    m = len(p)
    dof = n - m #degrees of freedom
    t = stats.t.ppf(1 - alpha/2, dof) #t-statistic
    s_err = np.sqrt(np.sum((yd - yp)**2) / (n - m))
    ci = t * s_err * np.sqrt(1/n + (x - np.mean(xd))**2 / np.sum((xd - np.mean(xd))**2))
    return ci

def minimum_detectable_flux(Aa, tc, freq, V, A, P, T):
    '''
    Developed by: Nickerson et al. (2016)
    Input
    Aa: analytical accuracy [ppb]
    tc: closure time of the chamber [s]
    freq: frequency of the measured data [Hz]
    V: chamber volume [m3]
    A: chamber surface area [m2]
    P: atmospheric pressure [Pa]
    T: ambient temperature [K]
    
    Output:
    mdf: Minimum detectable flux [nmol m-2 s-1]
    '''
    R = 8.314 #ideal gas constant [J mol-1 K-1]
    mdf = (Aa/(tc*(tc*freq)**(1/2)))*(V*P/(A*R*T))
    return mdf
    