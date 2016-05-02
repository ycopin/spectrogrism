# -*- coding: utf-8 -*-
# Time-stamp: <2016-05-02 14:53:05 ycopin>

from __future__ import division, print_function

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import astropy.modeling as AM


def fit_legendre2D(x, y, z, deg=(2, 2)):

    # Model: 2D Legendre polynomials
    p_init = AM.models.Legendre2D(*deg)

    # Least-square fit
    fitter = AM.fitting.LinearLSQFitter()
    p_fit = fitter(p_init, x, y, z)

    # Compute and store RMS
    p_fit.rms = (fitter.fit_info['residuals'] / z.size) ** 0.5

    return p_fit                # 2D-polynomial model
