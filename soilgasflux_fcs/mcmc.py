import emcee
import numpy as np

from .models import hm_model

# Prior bounds: alpha and cx are constrained to 2 decades around the best-fit
# (narrower/wider bounds were trialed and left the posterior degenerate or
# biased — see commit history for the exploration).
PRIOR_DECADES = 2
T0_UPPER_BOUND = 60  # s, covers the expected range of t0 for chamber measurements

# Ensemble sampler settings
DEFAULT_NDIM = 3
BURN_IN_FRACTION = 0.8
THIN = 15


class MCMC:
    def __init__(self):
        pass

    def ln_prior(self, theta, cx_bf, alpha_bf, t0_bf):
        alpha, cx, t0 = theta
        lo, hi = 10 ** (-PRIOR_DECADES), 10 ** PRIOR_DECADES
        if cx_bf * lo < cx < cx_bf * hi and alpha_bf * lo < alpha < alpha_bf * hi:
            return 0.0
        return -np.inf

    def ln_likelihood(self, theta, t, y, yerr, c0):
        alpha, cx, t0 = theta
        model = hm_model(t, cx, alpha, t0, c0)
        sigma2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    def ln_probability(self, theta, t, y, yerr, c0, cx_bf, alpha_bf, t0_bf):
        lp = self.ln_prior(theta, cx_bf, alpha_bf, t0_bf)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, t, y, yerr, c0)

    def run_mcmc(self, t, y, yerr, c0, cx_bf, alpha_bf, t0_bf, nwalkers, nsteps):
        lo, hi = 10 ** (-PRIOR_DECADES), 10 ** PRIOR_DECADES

        pos_alpha = np.exp(np.random.uniform(
            low=np.log(alpha_bf * lo), high=np.log(alpha_bf * hi),
            size=(nwalkers, 1),
        ))
        pos_cx = np.exp(np.random.uniform(
            low=np.log(cx_bf * lo), high=np.log(cx_bf * hi),
            size=(nwalkers, 1),
        ))
        pos_t0 = np.random.uniform(low=0, high=T0_UPPER_BOUND, size=(nwalkers, 1))
        pos = np.concatenate((pos_alpha, pos_cx, pos_t0), axis=1)

        sampler = emcee.EnsembleSampler(
            nwalkers, DEFAULT_NDIM, self.ln_probability,
            args=(t, y, yerr, c0, cx_bf, alpha_bf, t0_bf),
        )
        sampler.run_mcmc(pos, nsteps)

        discard = int(nsteps * BURN_IN_FRACTION)
        flat_samples = sampler.get_chain(flat=True, discard=discard, thin=THIN)
        log_prob = sampler.get_log_prob(flat=True, discard=discard, thin=THIN)

        return sampler, flat_samples, log_prob
