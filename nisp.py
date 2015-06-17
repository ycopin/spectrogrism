#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nisp
----

.. autosummary::

   Zemax

.. Warning:: Questions to JZ:

   * input y-coordinates are offset by +0.85 deg
   * input y-coordinates are not centered: (dx, dy) = (0.689, -4.194)
   * input position (+0.4, +1.25) is missing the 1.85 µm wavelength
"""

from __future__ import division, print_function

import os
import warnings
import numpy as N
import matplotlib.pyplot as P

from spectrogrism import *

try:
    import seaborn
    seaborn.set_style("whitegrid",
                      # {'xtick.major.size': 6, 'xtick.minor.size': 3,
                      #  'ytick.major.size': 6, 'ytick.minor.size': 3}
    )
except ImportError:
    pass

#: NISP optical configuration, R-grism
#:
#: The detector plane is tiled with 4×4 detectors of 2k×2k pixels of 18 µm;
#: the spectrograph has a mean magnification (`NISPPlateScale`) of
#: 0.5 approximately.  Hence a focal plane of approximately 29×29 cm².
#:
#: Sources: *NISP Technical Description* `EUCL-LAM-OTH-7-001`
NISP_R = dict(
    name="NISP-R",
    wave_ref=1.5e-6,                  # Reference wavelength [m]
    wave_range=[1.25e-6, 1.85e-6],    # Wavelength range [m]
    # Grism
    grism_on=True,                    # Is prism on the way?
    grism_dispersion=9.8,             # Informative spectral dispersion [AA/px]
    grism_prism_material='FS',        # Prism glass
    grism_grating_material='FS',      # Grating resine
    grism_prism_angle=2.88/RAD2DEG,   # Prism angle [rad]
    # grism_grating_rho=19.29,         # Grating groove density [lines/mm]
    grism_grating_rho=13.72,          # Grating groove density [lines/mm]
    grism_grating_blaze=2.6/RAD2DEG,  # Blaze angle [rad]
    # Detector
    detector_pxsize=18e-6,            # Detector pixel size [m]
    # Telescope
    telescope_flength=24.9,           # Telescope focal length [m]
)

# Guessed values (not from official documents)
NISP_R.update(
    # Collimator
    collimator_flength=2000e-3,       # Focal length [m]
    # Camera
    camera_flength=1000e-3,           # Focal length [m]
    # Detector
    detector_dxdy=0.689e-3 - 4.194e-3j,  # Detector offset [m]
)

# # NISP simulation configuration (now directly read from simulation)
# ZMX_SIMU = dict(
#     name="Zemax run_190315",
#     wave_npx=13,                    # Nb of pixels per spectrum
#     wave_range=[1.20e-6, 1.80e-6],  # Wavelength range [m]
#     orders=[1],                     # Dispersion orders
#     # Sky sampling
#     input_coords=N.linspace(-0.4, +0.4, 17)/RAD2DEG,  # [rad]
# )


class Zemax(object):

    """
    Read results from JZ's Zemax simulations.
    """

    colnames = """
confNb wave xindeg yindeg ximgmm yimgmm pxsize xpuppx ypuppx
nximgpx nyimgpx ximgpx yimgpx ximgmm yimgmm xpsfmm ypsfmm
ee50mm ee80mm ee90mm ellpsf papsfdeg""".split()  #: Input column names

    def __init__(self, filename="Zemax/run_190315.dat"):
        """Initialize from *filename*."""

        self.filename = os.path.basename(filename)
        # Load dataset from filename
        self.data = self.load(filename)
        # Convert to DetectorPositions
        self.detector = self.detector_positions()
        # Load simulation configuration
        self.simcfg = self.get_simcfg()

    def __str__(self):

        waves = set(self.data['wave'])
        xin = set(self.data['xindeg'])
        yin = set(self.data['yindeg'])

        s = """\
{}: {} entries
  Wavelength: {} steps from {:.2f} to {:.2f} µm
  Coords: {} × {} sources\
""".format(self.filename, len(self.data),
           len(waves), min(waves), max(waves),
           len(xin), len(yin))

        return s

    def load(self, filename):
        """Load simulation from `filename`."""

        data = N.genfromtxt(filename, dtype=None, names=self.colnames)

        # Cleanup: some Xin are arbitrarily close to zero
        warnings.warn("Setting approximately null xindeg to 0")
        data['xindeg'][N.abs(data['xindeg']) < 1e-12] = 0

        # Cleanup: offset yindeg by 0.85 deg
        warnings.warn("Offsetting Yin by 0.85 deg")
        data['yindeg'] -= 0.85

        # Cleanup: upper-right position has no 1.85 µm wavelength
        warnings.warn("Discarding wavelengths > 1.81 µm")
        data = data[data['wave'] < 1.81]

        return data

    def get_simcfg(self, orders=[1]):
        """Generate a :class:`SimConfig` corresponding to simulation."""

        # Unique wavelengths [m]
        waves = N.unique(self.data['wave']) * 1e-6   # [m]
        # Unique input coordinates
        coords = N.unique(self.data['xindeg'] + 1j * self.data['yindeg'])
        # Convert back to [[x, y]]
        coords = N.vstack((coords.real, coords.imag)).T / RAD2DEG   # [rad]

        simcfg = dict(
            name=self.filename,
            wave_npx=len(waves),
            wave_range=[min(waves), max(waves)],
            orders=orders,
            input_coords=coords,
            )

        return SimConfig(simcfg)

    def detector_positions(self):
        """Convert simulation to :class:`DetectorPositions`."""

        waves = N.sort(N.unique(self.data['wave']))
        coords = N.unique(
            self.data['xindeg'] + 1j * self.data['yindeg'])  # [deg]
        # print("{}: {} spectra of {} px".format(
        #     self.filename, len(coords), len(waves)))

        detector = DetectorPositions(waves * 1e-6,   # Wavelengths in [m]
                                     name=self.filename)
        for xy in coords:                  # Loop over input positions [deg]
            select = (N.isclose(self.data['xindeg'], xy.real) &
                      N.isclose(self.data['yindeg'], xy.imag))
            subdata = self.data[select]                     # Data selection
            subdata = subdata[N.argsort(subdata['wave'])]   # Sort subdata
            # Sanity check
            assert N.allclose(subdata['wave'], waves)
            dpos = subdata['xpsfmm'] + 1j * subdata['ypsfmm']   # [mm]
            detector.add_spectrum(xy/RAD2DEG, dpos * 1e-3, order=1)

        return detector

    def plot_input(self, ax=None, **kwargs):
        """Plot input coordinates (degrees)."""

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x [deg]", ylabel="y [deg]",
                                 title="{} - Input".format(self.filename))

        coords = self.simcfg.get_coords() * RAD2DEG  # [deg]
        ax.scatter(coords.real, coords.imag, **kwargs)

        # ax.set_aspect('equal', adjustable='datalim')

        return ax

    def plot_output(self, ax=None, **kwargs):
        """Plot output (detector) coordinates (pixels)."""

        ax = self.detector.plot(ax=ax, orders=(1,), blaze=False, **kwargs)
        ax.set_title("{} - PSF centroid".format(self.filename))

        return ax


if __name__ == '__main__':

    filename = "Zemax/run_190315.dat"
    zmx = Zemax(filename)
    print(zmx)

    # Zemax simulation
    # ax = zmx.plot_input()
    ax = zmx.plot_output(marker='.', s=20, edgecolor='k')

    # Optical modeling
    optcfg = OptConfig(NISP_R)  # Optical configuration (default NISP)
    simcfg = zmx.get_simcfg()   # Simulation configuration

    spectro = Spectrograph(optcfg,
                           grism_on=optcfg.get('grism_on', True),
                           add_telescope=True)

    # Test
    try:
        spectro.test(simcfg, verbose=False)
    except AssertionError as err:
        warnings.warn(str(err))
    else:
        print("Spectrograph test: OK")

    # Simulation
    simu = spectro.simulate(simcfg)

    simu.plot(ax=ax, zorder=0)                      # Draw below Zemax
    ax.axis([-100, +100, -100, +100])               # [mm]

    ax.legend(fontsize='small', frameon=True, framealpha=0.5)
    P.show()
