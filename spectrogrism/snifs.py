# -*- coding: utf-8 -*-
# Time-stamp: <2016-03-02 13:32 ycopin@lyonovae03.in2p3.fr>

"""
snifs
-----

SNIFS optical configuration and utilities.
"""

from __future__ import division, print_function, absolute_import

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings

import numpy as N

if __name__ == "__main__":
    # Cannot import explicitely local spectrogrism using relative import
    # in a script ("main"):
    # from . import spectrogrism as S
    # ValueError: Attempted relative import in non-package
    import spectrogrism as S    # Import *local* spectrogrism module
else:
    from . import spectrogrism as S

#: SNIFS optical configuration, R-channel
SNIFS_R = S.OptConfig([
    ('name', "SNIFS-R"),                # Configuration name
    ('wave_ref', 0.76e-6),              # Reference wavelength [m]
    ('wave_range', [0.5e-6, 1.02e-6]),  # Standard wavelength range [m]
    # Telescope
    ('telescope_flength', 22.5),        # Focal length [m]
    # Collimator
    ('collimator_flength', 169.549e-3),    # Focal length [m]
    ('collimator_distortion', +2.141),     # r² distortion coefficient
    ('collimator_lcolor_coeffs', [-4.39879e-6, 8.91241e-10, -1.82941e-13]),
    # Grism
    ('grism_prism_material', 'BK7'),           # Prism glass
    ('grism_prism_angle', 17.28 / S.RAD2DEG),  # Prism angle [rad]
    ('grism_grating_rho', 200.),   # Grating groove density [lines/mm]
    ('grism_dispersion', 2.86),    # Informative spectral dispersion [AA/px]
    ('grism_grating_material', 'EPR'),         # Grating resine
    ('grism_grating_blaze', 15. / S.RAD2DEG),  # Blaze angle [rad]
    # Camera
    ('camera_flength', 228.014e-3),          # Focal length [m]
    ('camera_distortion', -0.276),           # r² distortion coefficient
    ('camera_lcolor_coeffs', [+2.66486e-6, -5.52303e-10, 1.1365e-13]),
    # Detector
    ('detector_pxsize', 15e-6),          # Detector pixel size [m]
    ('detector_angle', 0. / S.RAD2DEG),  # Rotation of the detector (0=blue is up)
])

#: SNIFS simulation configuration
SNIFS_SIMU = S.SimConfig([
    ('name', u"standard"),                 # Configuration name
    ('wave_npx', 10),                      # Nb of pixels per spectrum
    ('modes', (1, 0, 2, -1)),              # Dispersion orders
    # Focal plane sampling
    ('input_coords', N.linspace(-1e-2, 1e-2, 5)),  # Focal plane grid [m]
    ('input_angle', -10. / S.RAD2DEG),             # Focal plane angle [rad]
])


# Simulations ==============================


def plot_SNIFS(optcfg=SNIFS_R, simcfg=SNIFS_SIMU,
               test=True, verbose=False):
    """
    Test-case w/ SNIFS-like configuration.
    """

    # Optical configuration
    print(optcfg)

    # Spectrograph
    spectro = S.Spectrograph(optcfg)
    print(spectro)

    # Simulation configuration
    print(simcfg)

    if test:
        print(" Spectrograph round-trip test ".center(S.LINEWIDTH, '-'))
        for mode in simcfg['modes']:
            try:
                spectro.test(simcfg, mode=mode, verbose=verbose)
            except AssertionError as err:
                warnings.warn("Order #{}: {}".format(mode, str(err)))
            else:
                print("Order #{:+d}: OK".format(mode))

    positions = spectro.predict_positions(simcfg)
    ax = positions.plot(modes=(-1, 0, 1, 2), blaze=True)
    ax.legend(loc='upper left', fontsize='small', frameon=True, framealpha=0.5)
    ax.set_aspect('auto')
    ax.axis(N.array([-2000, 2000, -4000, 4000]) *
            spectro.detector.pxsize / 1e-3)  # [mm]

    return ax


# Main ====================================================

if __name__ == '__main__':

    import matplotlib.pyplot as P
    try:
        import seaborn
        seaborn.set_style("whitegrid")
    except ImportError:
        pass

    ax = plot_SNIFS(test=True, verbose=False)

    embed_html = 'bokeh'
    if embed_html == 'mpld3':
        try:
            S.dump_mpld3(ax.figure, 'SNIFS-R_mpld3.html')
        except ImportError:
            warnings.warn("MPLD3 is not available, cannot export to HTML.")
    elif embed_html == 'bokeh':
        try:
            S.dump_bokeh(ax.figure, 'SNIFS-R_bokeh.html')
        except ImportError:
            warnings.warn("Bokeh is not available, cannot export to HTML.")
    elif embed_html:
        warnings.warn("Unknown HTML method '{}'".format(embed_html))

    P.show()
