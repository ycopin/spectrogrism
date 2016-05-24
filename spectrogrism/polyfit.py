# -*- coding: utf-8 -*-
# Time-stamp: <2016-05-17 22:31 ycopin@lyonovae03.in2p3.fr>

from __future__ import division, print_function

import astropy.modeling as AM

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"


def fit_legendre2D(x, y, z, deg=(2, 2)):
    """
    Fit a 2D-polynomial to *z(x, y)*.
    """

    # Model: 2D Legendre polynomials
    p_init = AM.models.Legendre2D(*deg)

    # Least-square fit
    fitter = AM.fitting.LinearLSQFitter()
    p_fit = fitter(p_init, x, y, z)

    # Compute and store RMS
    p_fit.rms = (fitter.fit_info['residuals'] / z.size) ** 0.5

    return p_fit                # 2D-polynomial model
