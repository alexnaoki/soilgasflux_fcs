import emcee
import numpy as np
import pandas as pd
from .models import hm_model

class MCMC:
    def __init__(self):
        pass

    def ln_prior(self, theta, cx_bf, alpha_bf, t0_bf):
        alpha, cx, t0 = theta
        mult = 1

        # if (cx_bf/(1e1*mult) < cx < cx_bf*(1e1*mult)) and (alpha_bf/(1e1*mult) < alpha < alpha_bf*(1e1*mult)):
        # if (cx_bf-(cx_bf*0.01) < cx < cx_bf+(cx_bf*.01)) and (alpha_bf-(alpha_bf*0.01) < alpha < alpha_bf+(alpha_bf*0.01)):
        # if 0 < cx < np.inf and 0 < alpha < np.inf:
        # if cx_bf*1e-2 < cx < cx_bf*1e2 and 0 < alpha < alpha_bf*1e2 and 0 < t0 < 10*t0_bf:
        
        # if cx_bf*1e-2 < cx < cx_bf*1e2 or 0 < alpha < alpha_bf*1e2:
        if cx_bf*1e-2 < cx < cx_bf*1e2 and alpha_bf*1e-2 < alpha < alpha_bf*1e2:
            return 0.0
        return -np.inf

    def ln_likelihood(self, theta, t, y, yerr, c0):
        alpha, cx, t0 = theta
        model = hm_model(t, cx, alpha, t0, c0)
        yrange = np.nanmax(y) - np.nanmin(y)

        sigma2 = yerr ** 2
        n = len(y)

        
        return -0.5 * np.sum(((y - model)) ** 2 / (sigma2) + np.log(sigma2))          # no
        # return -0.5 * np.sum(((y - model)/yrange) ** 2 / (sigma2) + np.log(2*np.pi*sigma2))   # yRange
        # return -0.5 * np.sum(((y - model)/n) ** 2 / (sigma2) + np.log(2*np.pi*sigma2))        # len

        # return -0.5*(n*np.log(2*np.pi*sigma2)) - 0.5*(np.sum(((y - model)/yrange) ** 2) / (sigma2)) #yRange + ln_v2
        # return -0.5*(np.log(np.sum(2*np.pi*sigma2))) - 0.5*(np.sum((y - model) ** 2) / (sigma2)) #no + ln_v2
    
    
    def ln_probability(self, theta, t, y, yerr, c0, cx_bf, alpha_bf, t0_bf):
        lp = self.ln_prior(theta, cx_bf, alpha_bf, t0_bf)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, t, y, yerr, c0)
    
    def run_mcmc(self, t, y, yerr, c0, cx_bf, alpha_bf, t0_bf,nwalkers, nsteps):
        ndim = 3

        # nwalkers = 50
        # nsteps=3000
        # pos_alpha = np.random.uniform(low=alpha_bf*10e-2, high=alpha_bf*10e2, size=(nwalkers, 1))
        # pos_cx = np.random.uniform(low=cx_bf*10e-2, high=cx_bf*10e2, size=(nwalkers, 1))
        mult = 1

        # pos_alpha = np.exp(np.random.uniform(low=np.log(alpha_bf/(1e1*mult)), high=np.log(alpha_bf*(1e1*mult)), size=(nwalkers, 1)))
        # pos_cx = np.exp(np.random.uniform(low=np.log(cx_bf/(1e1*mult)), high=np.log(cx_bf*(1e1*mult)), size=(nwalkers, 1)))
        
        # pos_alpha = np.exp(np.random.uniform(low=np.log(alpha_bf-(alpha_bf*0.1)), high=np.log(alpha_bf+(alpha_bf*0.1)), size=(nwalkers, 1)))
        # pos_cx = np.exp(np.random.uniform(low=np.log(cx_bf-(cx_bf*0.1)), high=np.log(cx_bf+(cx_bf*0.1)), size=(nwalkers, 1)))

        pos_alpha = np.exp(np.random.uniform(low=np.log(alpha_bf*1e-2), 
                                             high=np.log(alpha_bf*1e2), 
                                             size=(nwalkers,1)))
        pos_cx = np.exp(np.random.uniform(low=np.log(cx_bf*1e-2), 
                                          high=np.log(cx_bf*1e2), 
                                          size=(nwalkers,1)))
        pos_t0 = np.random.uniform(low=0, high=60, size=(nwalkers, 1)) # t0_bf

        #gaussian distribution
        # pos_alpha = np.ones((nwalkers, 1)) * alpha_bf + np.random.normal(0, alpha_bf*10, size=(nwalkers, 1))
        # pos_cx = np.ones((nwalkers, 1)) * cx_bf + np.random.normal(0, cx_bf*10, size=(nwalkers, 1))

        pos = np.concatenate((pos_alpha, pos_cx, pos_t0), axis=1)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_probability, 
                                        args=(t, y, yerr, c0, cx_bf, alpha_bf, t0_bf))
        sampler.run_mcmc(pos, nsteps)

        flat_samples = sampler.get_chain(flat=True, discard=int(nsteps*0.8), thin=15) # ()
        log_prob = sampler.get_log_prob(flat=True, discard=int(nsteps*0.8), thin=15)

        return sampler, flat_samples, log_prob

