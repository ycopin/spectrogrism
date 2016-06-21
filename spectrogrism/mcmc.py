# -*- coding: utf-8 -*-
# Time-stamp: <2016-06-21 18:58:13 ycopin>

"""
mcmc
----

Monte-Carlo Markov Chain methods.

.. autosummary::

   run_mcmc
   plot_mcmc_chains
   plot_mcmc_corner
   mcmc_best_params
"""

from __future__ import division, print_function

import numpy as N
import matplotlib.pyplot as P

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"


def run_mcmc(spectro, positions, simcfg, optparams, lnprior, modes=None,
             nwalkers=10, nsteps=500, outfile='chains.dat'):
    """
    Monte-Carlo Markov Chain exploration of parameter space, using
    :pypi:`emcee`.

    .. Warning:: very preliminary implementation

    :param Spectrograph spectro: spectrograph
    :param DetectorPositions positions: target positions
    :param SimConfig simcfg: simulation configuration
    :param list optparams: optical parameters to be probed
    :param function lnprior: log-likelihood prior function
    :param list modes: adjusted observing modes (default: simulated modes)
    :param int nwalkers: the number of walkers will be `2*len(optparams)*nwalkers`
    :param int nsteps: number of MCMC-steps
    :param str outfile: incremental file output
    :return: Monte-Carlo Markov Chains array (nwalkers, nsteps, ndim)
    :rtype: :attr:`emcee.EnsembleSampler.chain`
    :raise KeyError: unknown optical parameter

    **Reference:** `Foreman-Mackey et al. 2012
    <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_
    """

    import emcee

    # Simulation parameters
    if modes is None:
        modes = simcfg.get('modes', [1])
    print("Adjusted modes:", modes)

    try:
        guessparams = N.array([ spectro.config[name] for name in optparams ])
    except KeyError:
        raise KeyError("Unknown optical parameter '{}'".format(name))

    print("Initial parameters:")
    for name, value in zip(optparams, guessparams):
        print("  {:20}: {}".format(name, value))

    # Initial guess simulation
    mpositions = spectro.predict_positions(simcfg)
    # Test compatibility with objective detector positions only once
    mpositions.check_alignment(positions)

    rmss = []
    for mode in modes:
        rms = positions.compute_rms(mpositions, mode=mode)
        print("Mode {} RMS: {} mm = {} px"
              .format(mode, rms / 1e-3, rms / spectro.detector.pxsize))
        rmss.append(rms)
    rms = (sum( rms ** 2 for rms in rmss ) / len(modes)) ** 0.5
    print("Total RMS:  {} mm = {} px"
          .format(rms / 1e-3, rms / spectro.detector.pxsize))

    def lnlike(params):
        """Log-likelihood function (without prior)."""

        # Update optical configuration
        spectro.update(**dict(zip(optparams, params)))
        # Simulate
        mpositions = spectro.predict_positions(simcfg)
        dtot = sum(
            ((mpositions[mode] - positions[mode]).abs()**2).values.mean()
            for mode in modes )

        return -0.5 * (dtot / len(modes))**0.5

    def lnprob(params):
        """Full log-likelihood function, including prior."""

        prior = lnprior(params)
        if N.isfinite(prior):
            prior += lnlike(params)

        return prior

    ndim = len(optparams)
    nwalkers *= 2 * ndim

    print("MCMC sampler: ndim={}, nwalkers={}, nsteps={}"
          .format(ndim, nwalkers, nsteps))

    # Emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Sampling
    initparams = N.array([ guessparams + 1e-4 * N.random.randn(ndim)
                           for i in range(nwalkers)])

    if outfile:
        f = open(outfile, "w")
        f.close()

    for i, (pos, lnp, state) in enumerate(sampler.sample(initparams,
                                                         iterations=nsteps)):
        if (i+1) % 10 == 0:
            print("{0:.1f}%".format(1e2 * i / nsteps))

        if outfile:
            f = open(outfile, "a")
            for k in range(pos.shape[0]):
                f.write("{:5d}  {:s}\n"
                        .format(k, " ".join(map(str, pos[k]))))
            f.close()

    return sampler.chain    # MCMC chains (nwalkers, nsteps, ndim)


def plot_mcmc_chains(chains, parameters):
    """Plot MCMC-chains (nwalkers, nsteps, ndim)."""

    assert chains.ndim == 3 and chains.shape[2] == len(parameters)

    fig, axs = P.subplots(len(parameters), 1, sharex=True)
    for i, (ax, name) in enumerate(zip(axs, parameters)):
        ax.plot(chains[:, :, i].T, color='k', alpha=0.4)
        ax.set_title(name)
    fig.tight_layout()

    return fig


def plot_mcmc_corner(chains, parameters, burnin=0):
    """
    :pypi:`corner` plot of MCMC parameters.

    **Reference:** `Foreman-Mackey 2016
    <http://joss.theoj.org/papers/10.21105/joss.00024>`_
    """

    import corner

    assert chains.ndim == 3 and chains.shape[2] == len(parameters)
    assert burnin < chains.shape[1]

    # Remove 1st burnin steps, and reshape (nwalkers*nsteps, ndim)
    samples = chains[:, burnin:, :].reshape((-1, len(parameters)))

    fig = corner.corner(samples, labels=parameters)

    return fig


def mcmc_best_params(chains, parameters, burnin=0):
    """
    Compute best parameter estimates and assymetric 1-sigma errors from
    marginalized distributions.  Return `{name: (median, low_err, high_err)}`.
    """

    assert chains.ndim == 3 and chains.shape[2] == len(parameters)
    assert burnin < chains.shape[1]

    # Remove 1st burnin steps, and reshape (nwalkers*nsteps, ndim)
    samples = chains[:, burnin:, :].reshape((-1, len(parameters)))

    # Compute percentiles (nperc, ndim)
    percentiles = N.percentile(samples, [15.9, 50, 84.1], axis=0)  # (3, ndim)

    bests = { name: (median, low, high)
              for name, median, low, high in zip(
                  parameters,
                  percentiles[1],  # Median
                  percentiles[1] - percentiles[0],   # Median - low
                  percentiles[2] - percentiles[1])}  # High - median

    print("MCMC best estimates:")
    for name in parameters:
        median, low, high = bests[name]
        print("  {:20s} = {:+g} + {:g} - {:g}"
              .format(name, median, low, high))

    return bests
