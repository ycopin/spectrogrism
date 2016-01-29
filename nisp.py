#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nisp
----

.. autosummary::

   Zemax
"""

from __future__ import division, print_function

import os
import warnings
from collections import OrderedDict

import numpy as N
import matplotlib.pyplot as P

import spectrogrism as S

try:
    import seaborn
    seaborn.set_style("darkgrid",
                      # {'xtick.major.size': 6, 'xtick.minor.size': 3,
                      #  'ytick.major.size': 6, 'ytick.minor.size': 3}
                      )
except ImportError:
    pass

#: NISP effective optical configuration, R-grism
#:
#: .. Note:: The detector plane is tiled with 4×4 detectors of 2k×2k pixels of
#:    18 µm; the spectrograph has a mean magnification (`NISPPlateScale`) of
#:    0.5 approximately.  Hence a focal plane of approximately 29×29 cm².
NISP_R = dict(
    name="NISP-R",
    wave_ref=1.5e-6,                      # Reference wavelength [m]
    wave_range=[1.25e-6, 1.85e-6],        # Wavelength range [m]
    # Telescope
    telescope_flength=24.5,               # Telescope focal length [m]
    # Grism
    grism_on=True,                        # Is prism on the way?
    grism_dispersion=9.8,                 # Informative spectral dispersion [AA/px]
    grism_prism_material='FS',            # Prism glass
    grism_grating_material='FS',          # Grating resine
    grism_prism_angle=2.88 / S.RAD2DEG,   # Prism angle [rad]
    # grism_grating_rho=19.29,            # Grating groove density [lines/mm]
    grism_grating_rho=13.72,              # Grating groove density [lines/mm]
    grism_grating_blaze=2.6 / S.RAD2DEG,  # Blaze angle [rad]
    # Detector
    detector_pxsize=18e-6,                # Detector pixel size [m]
)

# Guessed values (not from official documents)
NISP_R.update(
    # Telescope
    telescope_flength=25.2,              # Telescope focal length [m]
    # Collimator
    collimator_flength=1946e-3,          # Focal length [m]
    collimator_distortion=2.8e-3,
    # Grism
    grism_prism_angle=2.70 / S.RAD2DEG,  # Prism angle [rad]
    grism_grating_rho=13.1,              # Grating groove density [lines/mm]
    # Camera
    camera_flength=957e-3,               # Focal length [m]
    camera_distortion=29.6e-3,
    # Detector (without input recentering)
    # detector_dx=+0.70e-3,              # Detector x-offset [m]
    # detector_dy=+179.7e-3,             # Detector y-offset [m]
    # Detector (with input offset of -0.85 deg)
    detector_dx=+0.70e-3,                # Detector x-offset [m]
    detector_dy=-4.20e-3,                # Detector y-offset [m]
)


class Zemax(object):

    """
    Read results from JZ's Zemax simulations.
    """

    colnames = """
confNb wave xindeg yindeg ximgmm yimgmm pxsize nxpup nypup nximg nyimg
ximgcpx yimgcpx ximgcmm yimgcmm xpsfcmm ypsfcmm
ee50mm ee80mm ee90mm ellpsf papsfdeg""".split()  #: Input column names

    def __init__(self, simulations):
        """Initialize from *simulations* = {simname: name, order: filename}."""

        self.filenames = simulations.copy()
        self.name = self.filenames.pop('simname')  # Simulation name
        # Orders (int) or undispersed bands (str)
        self.orders = self.filenames.keys()
        # Load datasets from filenames and minimally check consistency
        # {order: ndarray}
        self.data = self.load_multiorder_sims(self.filenames)
        # Convert to DetectorPositions (internally structured in orders)
        self.positions = self.detector_positions(orders=self.orders)
        # Load simulation configuration
        self.simcfg = self.get_simcfg(orders=self.orders)

    def __str__(self):

        s = "Simulations: {}".format(self.name)
        for order in self.orders:
            s += "\n  Order #{}: {}".format(order, self.filenames[order])

        # All orders are supposed to share the same input values
        ref_data = self.data[self.orders[0]]
        waves = set(ref_data['wave'])
        xin = set(ref_data['xindeg'])
        yin = set(ref_data['yindeg'])

        s += "\n  Wavelengths: {} steps from {:.2f} to {:.2f} µm".format(
            len(waves), min(waves), max(waves))
        s += "\n  Coords: {} × {} sources".format(
            len(xin), len(yin))

        return s

    def load_multiorder_sims(self, simulations):
        """Load multi-order simulations from
        *simulations* = {order: filename}."""

        data = { order: self.load_simulation(simulations[order])
                 for order in self.orders }

        # Consistency checks on configurations, wavelengths and input positions
        if len(self.orders) > 1:
            ref_order = self.orders[0]
            for order in self.orders[1:]:
                assert (data[order]['confNb'] ==
                        data[ref_order]['confNb']).all(), \
                    "ConfNb of orders {} and {} are incompatible".format(
                        order, ref_order)
                # Beware: all files are not sorted the same way
                for name in ('wave', 'xindeg', 'yindeg'):
                    vector = N.sort(N.unique(data[order][name]))
                    ref_vector = N.sort(N.unique(data[ref_order][name]))
                    assert N.allclose(vector, ref_vector), \
                    "'{}' of orders {} and {} are incompatible".format(
                        name, order, ref_order)

        return data

    def load_simulation(self, filename):
        """Load simulation from single *filename*."""

        data = N.genfromtxt(filename, dtype=None, names=self.colnames)

        # Cleanup: some Xin are arbitrarily close to zero
        warnings.warn("Setting approximately null xindeg to 0")
        data['xindeg'][N.abs(data['xindeg']) < 1e-12] = 0

        # Cleanup: offset yindeg by -0.85 deg
        warnings.warn("Offsetting Yin by -0.85 deg")
        data['yindeg'] += -0.85

        # Cleanup: upper-right position has no 1.85 µm wavelength
        warnings.warn("Discarding wavelengths > 1.81 µm")
        data = data[data['wave'] < 1.81]

        return data

    def get_simcfg(self, orders=None):
        """Generate a :class:`SimConfig` corresponding to simulation."""

        if orders is None:
            orders = self.orders

        ref_data = self.data[orders[0]]
        # Unique wavelengths [m]
        waves = N.unique(ref_data['wave']) * 1e-6   # [m]
        # Unique input coordinates
        coords = N.unique(ref_data['xindeg'] + 1j * ref_data['yindeg'])
        # Convert back to [[x, y]]
        coords = N.vstack((coords.real, coords.imag)).T / S.RAD2DEG   # [rad]

        simcfg = S.Configuration(dict(name=self.name,
                            wave_npx=len(waves),
                            wave_range=[min(waves), max(waves)],
                            orders=orders,
                            input_coords=coords))

        return S.SimConfig(simcfg)

    def detector_positions(self, orders=None, colname='psfcmm'):
        """Convert simulation to :class:`DetectorPositions` in mm."""

        if orders is None:
            orders = self.orders

        if 'mm' not in colname:  # Input coords are supposed to be in mm
            raise NotImplementedError()

        ref_order = orders[0]
        data = self.data[ref_order]
        waves = N.sort(N.unique(data['wave']))
        coords = N.unique(data['xindeg'] + 1j * data['yindeg'])  # [deg]

        positions = S.DetectorPositions(waves * 1e-6,   # Wavelengths in [m]
                                        name=self.name)
        for order in orders:
            data = self.data[order]
            for xy in coords:                  # Loop over input positions [deg]
                select = (N.isclose(data['xindeg'], xy.real) &
                          N.isclose(data['yindeg'], xy.imag))
                subdata = data[select]                         # Data selection
                subdata = subdata[N.argsort(subdata['wave'])]  # Sort subdata
                # Sanity check
                assert N.allclose(subdata['wave'], waves)
                dpos = subdata['x' + colname] + 1j * subdata['y' + colname]
                positions.add_spectrum(xy / S.RAD2DEG, dpos * 1e-3, order=order)

        return positions

    def plot_input(self, ax=None, **kwargs):
        """Plot input coordinates (degrees)."""

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x [deg]", ylabel="y [deg]",
                                 title="{} - Input".format(self.filename))

        coords = self.simcfg.get_coords() * S.RAD2DEG  # [deg]
        ax.scatter(coords.real, coords.imag, **kwargs)

        # ax.set_aspect('equal', adjustable='datalim')

        return ax

    def plot_output(self, ax=None, orders=None, subsampling=0,
                    **kwargs):
        """Plot output (detector) coordinates [mm]."""

        if orders is None:
            orders = self.orders

        ax = self.positions.plot(ax=ax, orders=orders, 
                                 subsampling=subsampling, **kwargs)

        title = "Simulation '{}'".format(self.name)
        if subsampling:
            title += u" (subsampled ×{})".format(subsampling)
        ax.set_title(title)

        return ax


if __name__ == '__main__':

    simulations = OrderedDict([("simname", "Zemax"),
                               (1, "Zemax/run_190315.dat"),
                               (0, "Zemax/run_011115_conf2_o0.dat"),
                               (2, "Zemax/run_161115_conf2_o2.dat"),
                               ])

    subsampling = 3             # Subsample output plot
    adjust = False              # Test optical parameter adjustment
    embed_html = False          # Generate MPLD3 figure

    # 1st-order Zemax simulation
    zmx = Zemax(simulations)
    print(zmx)

    # ax = zmx.plot_input()

    kwargs = dict(s=20, edgecolor='k', linewidths=1)  # Outlined symbols
    ax = zmx.plot_output(orders=zmx.orders, subsampling=subsampling, **kwargs)

    # Optical modeling
    optcfg = S.OptConfig(NISP_R)  # Optical configuration (default NISP)
    simcfg = zmx.get_simcfg()     # Simulation configuration

    spectro = S.Spectrograph(optcfg,
                             grism_on=optcfg.get('grism_on', True),
                             telescope=S.Telescope(optcfg))
    print(spectro)

    # Test
    try:
        spectro.test(simcfg, verbose=False)
    except AssertionError as err:
        warnings.warn("Spectrograph test:", str(err))
    else:
        print("Spectrograph test: OK")

    # Simulation
    simu = spectro.simulate(simcfg, orders=zmx.orders)

    # Compute RMS on 1st-order positions
    rms = zmx.positions.compute_rms(simu, order=1)
    print("1st-order RMS = {:.4f} mm = {:.2f} px".format(
        rms / 1e-3, rms / spectro.detector.pxsize))

    #kwargs = dict(edgecolor=None, facecolor='none', linewidths=1)  # Open symbols
    kwargs = {}                     # Default
    if not adjust:                  # Out-of-the-box optical model
        simu.plot(ax=ax, zorder=0,  # Draw below Zemax
                  orders=(1,),
                  subsampling=subsampling,
                  label="{} #1 (RMS={:.1f} px)".format(
                      simu.name, rms / spectro.detector.pxsize),
                  **kwargs)
        if 0 in zmx.orders:
            # Compute RMS on 0th-order positions
            rms = zmx.positions.compute_rms(simu, order=0)
            print("0th-order RMS = {:.4f} mm = {:.2f} px".format(
                rms / 1e-3, rms / spectro.detector.pxsize))
            simu.plot(ax=ax, zorder=0,  # Draw below Zemax
                      orders=(0,), blaze=True,
                      subsampling=subsampling,
                      label="{} #0 (RMS={:.1f} px)".format(
                          simu.name, rms / spectro.detector.pxsize),
                      **kwargs)
        if 2 in zmx.orders:
            # Compute RMS on 2nd-order positions
            rms = zmx.positions.compute_rms(simu, order=2)
            print("2nd-order RMS = {:.4f} mm = {:.2f} px".format(
                rms / 1e-3, rms / spectro.detector.pxsize))
            simu.plot(ax=ax, zorder=0,  # Draw below Zemax
                      orders=(2,), blaze=True,
                      subsampling=subsampling,
                      label="{} #2 (RMS={:.1f} px)".format(
                          simu.name, rms / spectro.detector.pxsize),
                      **kwargs)
    else:                           # Optical adjustment
        result = spectro.adjust(
            zmx.positions, simcfg, tol=1e-4,
            optparams=[
                'detector_dx', 'detector_dy',
                # 'telescope_flength',
                # 'collimator_flength', 'collimator_distortion',
                # 'camera_flength', 'camera_distortion',
            ])
        if result.success:          # Adjusted simulation
            simu2 = spectro.simulate(simcfg)
            simu2.plot(ax=ax, zorder=0,
                       subsampling=subsampling,
                       label="Adjusted {} (RMS={:.1f} px)".format(
                           simu.name,
                           result.rms / spectro.detector.pxsize))

    ax.axis([-100, +100, -100, +100])               # [mm]
    ax.set_aspect('equal', adjustable='datalim')
    #ax.set_axisbg('0.9')
    ax.legend(fontsize='small', frameon=True, framealpha=0.5)

    if embed_html:
        try:
            S.dump_mpld3(ax, zmx.filename.replace('.dat', '.html'))
        except ImportError:
            warnings.warn("MPLD3 is not available, cannot export to HTML.")

    P.show()
