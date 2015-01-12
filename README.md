# spectrogrism

This module implements the grism-based spectrograph optical model used for the *SuperNova Integral
Field Spectrograph* (Lantz et al. 2004) data-reduction pipeline, and presented in (Copin, 2000). It
provides a flexible chromatic mapping between the input focal plane and the output detector plane,
based on an arbitrarily complex semi-empirical ray-tracing model of the key optical elements
defining the spectrograph (collimator, prism, grating, camera), described by a restricted number of
physically-motivated and *ad-hoc* parameters.

