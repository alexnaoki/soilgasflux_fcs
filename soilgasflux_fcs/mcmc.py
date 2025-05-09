import emcee
import numpy as np
import pandas as pd
from .models import hm_model

class MCMC:
    def __init__(self):
        pass
# def hm_model(t, cx, a, t0, c0):
#     e = 2.71828
#     return cx+(c0-cx)*e**(-a*(t-t0))

    def ln_likelihood(self, theta, t, y, yerr, c0):
        alpha, cx = theta
        model = hm_model(t, cx, alpha, 0, c0)
        yrange = np.nanmax(y) - np.nanmin(y)

        sigma2 = yerr ** 2

        return -0.5 * np.sum(((y - model)/yrange) ** 2 / sigma2 + np.log(2*np.pi*sigma2))
    
    def ln_prior(self, theta, cx_bf, alpha_bf):
        alpha, cx = theta

        if (cx_bf*10e-3 < cx < cx_bf*10e3) and (alpha_bf*10e-3 < alpha < alpha_bf*10e3):
            return 0.0
        return -np.inf
    
    def ln_probability(self, theta, t, y, yerr, c0, cx_bf, alpha_bf):
        lp = self.ln_prior(theta, cx_bf, alpha_bf)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, t, y, yerr, c0)
    
    def run_mcmc(self, t, y, yerr, c0, cx_bf, alpha_bf, nwalkers=100, nsteps=1000):
        ndim = 2

        # pos_alpha = np.random.uniform(low=alpha_bf*10e-2, high=alpha_bf*10e2, size=(nwalkers, 1))
        # pos_cx = np.random.uniform(low=cx_bf*10e-2, high=cx_bf*10e2, size=(nwalkers, 1))

        pos_alpha = np.exp(np.random.uniform(low=np.log(alpha_bf*10e-3), high=np.log(alpha_bf*10e3), size=(nwalkers, 1)))
        pos_cx = np.exp(np.random.uniform(low=np.log(cx_bf*10e-3), high=np.log(cx_bf*10e3), size=(nwalkers, 1)))

        pos = np.concatenate((pos_alpha, pos_cx), axis=1)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_probability, args=(t, y, yerr, c0, cx_bf, alpha_bf))
        sampler.run_mcmc(pos, nsteps)

        # samples = sampler.get_chain()
        flat_samples = sampler.get_chain(flat=True, discard=int(nsteps*0.5), thin=15) # ()
        log_prob = sampler.get_log_prob(flat=True, discard=int(nsteps*0.5), thin=15)

        return sampler, flat_samples, log_prob

