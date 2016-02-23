#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nisp
----

NISP-specific tools, including a :class:`Zemax` simulation handler.

.. autosummary::

   Zemax
"""

from __future__ import division, print_function

import warnings
from collections import OrderedDict

import numpy as N
import matplotlib.pyplot as P

import spectrogrism as S

try:
    import seaborn
    seaborn.set_style("darkgrid",
                      # {'xtick.major.size': 6, 'xtick.minor.size': 3,
                      #  'ytick.major.size': 6, 'ytick.minor.size': 3},
                      )
except ImportError:
    pass

#: NISP effective optical configuration, R-grism
#:
#: .. Note:: The detector plane is tiled with 4×4 detectors of 2k×2k pixels of
#:    18 µm; the spectrograph has a mean magnification (`NISPPlateScale`) of
#:    0.5 approximately.  Hence a focal plane of approximately 29×29 cm².
NISP_R = OrderedDict([
    ('name', "NISP-R"),                   # Configuration name
    ('wave_ref', 1.5e-6),                 # Reference wavelength [m]
    ('wave_range', [1.25e-6, 1.85e-6]),   # Wavelength range [m]
    # Telescope
    ('telescope_flength', 24.5),          # Telescope focal length [m]
    # Grism
    ('grism_dispersion', 9.8),            # Rough spectral dispersion [AA/px]
    ('grism_prism_material', 'FS'),       # Prism glass
    ('grism_grating_material', 'FS'),     # Grating resine
    ('grism_prism_angle', 2.88 / S.RAD2DEG),  # Prism angle [rad]
    # ('grism_grating_rho', 19.29),        # Grating groove density [lines/mm]
    ('grism_grating_rho', 13.72),         # Grating groove density [lines/mm]
    ('grism_grating_blaze', 2.6 / S.RAD2DEG),  # Blaze angle [rad]
    # Detector
    ('detector_pxsize', 18e-6),           # Detector pixel size [m]
])

# Guessed values (not from official documents)
NISP_R.update([
    # Telescope
    ('telescope_flength', 25.2),         # Telescope focal length [m]
    # Collimator
    ('collimator_flength', 1946e-3),     # Focal length [m]
    ('collimator_distortion', 2.8e-3),
    # Grism
    ('grism_prism_angle', 2.70 / S.RAD2DEG),  # Prism angle [rad]
    ('grism_grating_rho', 13.1),   # Grating groove density [lines/mm]
    ('grism_prism_tiltx', 0),  # Prism x-tilt (around apex/groove axis) [rad]
    ('grism_prism_tilty', 0),  # Prism y-tilt [rad]
    ('grism_prism_tiltz', 0),  # Prism z-tilt (around optical axis) [rad]
    # Camera
    ('camera_flength', 957e-3),          # Focal length [m]
    ('camera_distortion', 29.6e-3),
    # Detector (without input recentering)
    # ('detector_dx', +0.70e-3),                 # Detector x-offset [m]
    # ('detector_dy', +179.7e-3),                # Detector y-offset [m]
    # Detector (with input offset of -0.85 deg)
    ('detector_dx', +0.70e-3),           # Detector x-offset [m]
    ('detector_dy', -4.20e-3),           # Detector y-offset [m]
])


class Zemax(object):

    """
    Zemax simulations, in spectroscopic or photometric mode.

    Zemax configuration modes:

    * 1: B-grism (NISP-S)
    * 2, 3, 4: R-grisms (NISP-S)
    * 5, 6, 7: Y, J, H (NISP-P)
    """

    colnames = """
confNb wave xindeg yindeg ximgmm yimgmm pxsize nxpup nypup nximg nyimg
ximgcpx yimgcpx ximgcmm yimgcmm xpsfcmm ypsfcmm
ee50mm ee80mm ee90mm ellpsf papsfdeg""".split()  #: Input column names

    def __init__(self, simulations):
        """
        Initialize from :class:`Configuration` `simulations` = {mode: filename}.
        """

        self.filenames = simulations
        self.name = simulations.name

        # Observing modes
        self.modes = [ mode
                       for mode in self.filenames.keys() if mode != 'name' ]
        # Dispersion orders (int modes)
        self.orders = [ mode
                        for mode in self.modes if isinstance(mode, int) ]
        # Undispersed photometric bands (str modes)
        self.bands = [ mode
                       for mode in self.modes if isinstance(mode, basestring) ]

        # Load datasets from filenames and minimally check consistency
        # {mode: ndarray}
        self.data = self.load_multimode_sims(self.filenames)

        # Convert to DetectorPositions (internally structured in modes)
        self.positions = self.detector_positions(modes=self.modes)
        # Load simulation configuration
        self.simcfg = self.get_simcfg(modes=self.modes)

    def __str__(self):

        s = "Simulations: {}, {} modes".format(self.name, len(self.modes))
        for order in self.orders:
            s += "\n  Order #{}: {}".format(order, self.filenames[order])
        for band in self.bands:
            s += "\n  Band   {}: {}".format(band, self.filenames[band])

        # All orders are supposed to share the same input values
        ref_data = self.data[self.modes[0]]
        waves = set(ref_data['wave'])
        xin = set(ref_data['xindeg'])
        yin = set(ref_data['yindeg'])

        s += "\n  Wavelengths: {} steps from {:.2f} to {:.2f} µm".format(
            len(waves), min(waves), max(waves))
        s += "\n  Coords: {} × {} sources".format(
            len(xin), len(yin))

        return s

    def load_multimode_sims(self, simulations):
        """
        Load multi-mode simulations from :class:`Configuration`
        `simulations` = {mode: filename}.
        """

        data = { mode: self.load_simulation(simulations[mode])
                 for mode in self.modes }

        # Consistency checks on configurations, wavelengths and input positions
        if len(self.modes) > 1:
            ref_mode = self.modes[0]
            for mode in self.modes[1:]:
                # Beware: all files are not sorted the same way
                for name in ('wave', 'xindeg', 'yindeg'):
                    vector = N.sort(N.unique(data[mode][name]))
                    ref_vector = N.sort(N.unique(data[ref_mode][name]))
                    assert N.allclose(vector, ref_vector), \
                        "'{}' of modes {} and {} are incompatible".format(
                            name, mode, ref_mode)

        return data

    def load_simulation(self, filename):
        """Load simulation from single `filename`."""

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

    def get_simcfg(self, modes=None):
        """Generate a :class:`SimConfig` corresponding to simulation."""

        if modes is None:
            modes = self.modes

        ref_data = self.data[modes[0]]
        # Unique wavelengths [m]
        waves = N.unique(ref_data['wave']) * 1e-6   # [m]
        # Unique input coordinates
        coords = N.unique(ref_data['xindeg'] + 1j * ref_data['yindeg'])
        # Convert back to [[x, y]]
        coords = N.vstack((coords.real, coords.imag)).T / S.RAD2DEG   # [rad]

        simcfg = S.Configuration([('name', self.name),
                                  ('wave_npx', len(waves)),
                                  ('wave_range', [min(waves), max(waves)]),
                                  ('modes', modes),
                                  ('input_coords', coords)
                                  ])

        return S.SimConfig(simcfg)

    def detector_positions(self, modes=None, colname='psfcmm'):
        """Convert simulation to :class:`DetectorPositions` in mm."""

        if modes is None:
            modes = self.modes

        if 'mm' not in colname:  # Input coords are supposed to be in mm
            raise NotImplementedError()

        ref_mode = modes[0]
        data = self.data[ref_mode]
        waves = N.sort(N.unique(data['wave']))
        coords = N.unique(data['xindeg'] + 1j * data['yindeg'])  # [deg]

        positions = S.DetectorPositions(waves * 1e-6,  # Wavelengths in [m]
                                        name=self.name)
        for mode in modes:
            data = self.data[mode]
            for xy in coords:                  # Loop over input positions [deg]
                select = (N.isclose(data['xindeg'], xy.real) &
                          N.isclose(data['yindeg'], xy.imag))
                subdata = data[select]                         # Data selection
                subdata = subdata[N.argsort(subdata['wave'])]  # Sort subdata
                # Sanity check
                assert N.allclose(subdata['wave'], waves)
                dpos = (subdata['x' + colname] +
                        1j * subdata['y' + colname])           # [mm]
                positions.add_spectrum(xy / S.RAD2DEG, dpos * 1e-3, mode=mode)

        return positions

    def plot_input(self, ax=None, **kwargs):
        """Plot input coordinates (degrees)."""

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(
                1, 1, 1,
                xlabel="x [deg]", ylabel="y [deg]",
                title="Simulation '{}'\nInput positions".format(self.name))

        coords = self.simcfg.get_coords() * S.RAD2DEG  # [deg]
        ax.scatter(coords.real, coords.imag, **kwargs)

        ax.set_aspect('equal', adjustable='datalim')

        return ax

    def plot_output(self, ax=None, modes=None, subsampling=0, **kwargs):
        """Plot output (detector) coordinates [mm]."""

        if modes is None:
            modes = self.modes

        ax = self.positions.plot(ax=ax, modes=modes,
                                 subsampling=subsampling, **kwargs)

        title = "Simulation '{}'".format(self.name)
        if subsampling:
            title += u" (subsampled ×{})".format(subsampling)
        ax.set_title(title)

        return ax

    def plot_offsets(self, other_positions, ax=None, mode=1, nwaves=3):
        """Plot output (detector) coordinate offsets [px]."""

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x [mm]", ylabel="y [mm]")
            title = "Simulation '{}', {}{}\nposition offsets [px]".format(
                self.name, "order #" if isinstance(mode, int) else "band {}",
                mode)
            ax.set_title(title)
            ax.set_aspect('equal', adjustable='datalim')
        else:
            fig = None          # Will serve as a flag

        # Wavelength-indexed DataFrames
        zpos = self.positions.spectra[mode] / 1e-3   # [mm]
        spos = other_positions.spectra[mode] / 1e-3        # [mm]
        dpos = spos - zpos                           # Position offset [mm]
        dpos /= other_positions.spectrograph.detector.pxsize / 1e-3  # [px]

        # Create wavelength vector
        waves = zpos.index
        iwaves = N.linspace(0, len(waves) - 1, nwaves).round().astype(int)
        waves = waves[iwaves]

        colorMap = P.matplotlib.cm.ScalarMappable(
            norm=P.matplotlib.colors.Normalize(vmin=waves[0], vmax=waves[-1]),
            cmap=P.get_cmap('Spectral_r'))   # (blue to red)

        qkey = None                       # Quiver legend (only once)

        for wave in waves:
            # Input (Zemax) position
            x, y = zpos.loc[wave].real, zpos.loc[wave].imag    # [mm]
            # Offset between modeled and input positions
            dx, dy = dpos.loc[wave].real, dpos.loc[wave].imag  # [px]

            col = colorMap.to_rgba(wave)                       # Get color
            q = ax.quiver(x, y, dx, dy, color=col, units='width')
            if qkey is None:              # Add quiver label
                qkey = ax.quiverkey(q, 0.9, 0.95, 10, "10 px",
                                    labelpos='E', coordinates='figure')

        ax.axis([-100, +100, -100, +100])               # [mm]

        return ax

if __name__ == '__main__':

    simulations = S.Configuration([
        ("name", "Zemax"),
        (1, "Zemax/run_190315.dat"),            # 1st-order dispersed simulation
        (0, "Zemax/run_011115_conf2_o0.dat"),   # 0th-order dispersed simulation
        (2, "Zemax/run_161115_conf2_o2.dat"),   # 2nd-order dispersed simulation
        ('J', "Zemax/run_071215_conf6_J.dat"),  # J-band undispersed simulation
    ])

    subsampling = 3             # Subsample output plot
    adjust = False              # Test optical parameter adjustment
    embed_html = False          # Generate MPLD3 figure
    plot_offset = False         # Offset plots

    # Zemax simulations
    zmx = Zemax(simulations)
    print(zmx)

    # Optical modeling
    optcfg = S.OptConfig(NISP_R)  # Optical configuration (default NISP)
    simcfg = zmx.get_simcfg()     # Simulation configuration

    spectro = S.Spectrograph(optcfg,
                             telescope=S.Telescope(optcfg))
    print(spectro)

    # Tests
    print(" Spectrograph round-trip test ".center(70, '-'))
    for mode in simcfg.get('modes', (1, 0, 2)):
        try:
            spectro.test(simcfg, mode=mode, verbose=False)
        except AssertionError as err:
            warnings.warn("Order #{}: {}".format(mode, str(err)))
        else:
            print("{}{}: OK".format(
                "Order #" if isinstance(mode, int) else "Band ", mode))

    # Spectroscopic modes ==============================

    spositions = spectro.predict_positions(simcfg, modes=zmx.orders)
    spositions.assert_compatibility(zmx.positions, modes=zmx.orders)

    # Plots
    # ax = zmx.plot_input()

    kwargs = dict(s=20, edgecolor='k', linewidths=1)  # Outlined symbols
    ax = zmx.plot_output(modes=zmx.orders, subsampling=subsampling, **kwargs)

    # kwargs = dict(edgecolor=None, facecolor='none', linewidths=1)  # Open symbols
    kwargs = {}                      # Default
    for order in zmx.orders:
        # Compute RMS on current order positions
        rms = zmx.positions.compute_rms(spositions, mode=order)
        print("Order #{} RMS = {:.4f} mm = {:.2f} px".format(
            order, rms / 1e-3, rms / spectro.detector.pxsize))
        spositions.plot(ax=ax, zorder=0,  # Draw below Zemax
                        modes=(order,), blaze=(order != 1),
                        subsampling=subsampling,
                        label="{} #{} (RMS={:.1f} px)".format(
                            spositions.name, order,
                            rms / spectro.detector.pxsize),
                        **kwargs)

    if adjust:                           # Optical adjustment
        result = spectro.adjust(
            zmx.positions, simcfg, tol=1e-4,
            optparams=[
                'detector_dy',  # 'detector_dx',
                'grism_prism_tiltz',
                # 'telescope_flength', 'collimator_flength', 'camera_flength',
                # 'collimator_distortion', 'camera_distortion',
            ])
        if result.success:          # Adjusted model
            spositions_fit = spectro.predict_positions(simcfg)
            spositions_fit.plot(ax=ax, zorder=0,
                                subsampling=subsampling,
                                label="Adjusted {} (RMS={:.1f} px)".format(
                                    spositions.name,
                                    result.rms / spectro.detector.pxsize))

    ax.axis([-100, +100, -100, +100])               # [mm]
    ax.set_aspect('equal', adjustable='datalim')
    # ax.set_axisbg('0.9')
    ax.legend(loc='upper left', fontsize='small', frameon=True, framealpha=0.5)

    if embed_html:
        figname = zmx.name + '_mpld3.html'
        try:
            S.dump_mpld3(ax, figname)
        except ImportError:
            warnings.warn("MPLD3 is not available, cannot export to HTML.")

    # Position offset quiver plots
    if plot_offset:
        for order in zmx.orders:
            ax = zmx.plot_offsets(spositions, mode=order)

    # Imagery modes ==============================

    ppositions = spectro.predict_positions(simcfg, modes=zmx.bands)
    ppositions.assert_compatibility(zmx.positions, modes=zmx.bands)

    kwargs = dict(s=20, edgecolor='k', linewidths=1)  # Outlined symbols
    ax = zmx.plot_output(modes=zmx.bands, **kwargs)

    kwargs = {}                 # Default
    for band in zmx.bands:
        # Compute RMS on positions
        rms = zmx.positions.compute_rms(ppositions, mode=band)
        print("Band {} RMS = {:.4f} mm = {:.2f} px".format(
            band, rms / 1e-3, rms / spectro.detector.pxsize))
        ppositions.plot(ax=ax, zorder=0,  # Draw below Zemax
                        modes=(band,),
                        label="{} {} (RMS={:.1f} px)".format(
                            ppositions.name, band,
                            rms / spectro.detector.pxsize),
                        **kwargs)

    ax.axis([-100, +100, -100, +100])               # [mm]
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', fontsize='small', frameon=True, framealpha=0.5)

    P.show()
