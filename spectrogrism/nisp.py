# -*- coding: utf-8 -*-
# Time-stamp: <2016-03-23 11:13:48 ycopin>

"""
nisp
----

NISP-specific tools, including a :class:`ZemaxPositions` handler.

.. autosummary::

   ZemaxPositions
"""

from __future__ import division, print_function, absolute_import

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings

import numpy as N
import matplotlib.pyplot as P
import pandas as PD

if __name__ == "__main__":
    # Cannot import explicitely local spectrogrism using relative import
    # in a script ("main"):
    # from . import spectrogrism as S
    # ValueError: Attempted relative import in non-package
    import spectrogrism as S    # Import *local* spectrogrism module
else:
    from . import spectrogrism as S

#: NISP effective optical configuration, R-grism
#:
#: .. Note:: The detector plane is tiled with 4×4 detectors of 2k×2k pixels of
#:    18 µm; the spectrograph has a mean magnification (`NISPPlateScale`) of
#:    0.5 approximately.  Hence a focal plane of approximately 29×29 cm².
NISP_R = S.OptConfig([
    ('name', "NISP-R"),                   # Configuration name
    ('wave_ref', 1.5e-6),                 # Reference wavelength [m]
    ('wave_range', [1.25e-6, 1.85e-6]),   # Wavelength range [m]
    # Telescope
    ('telescope_flength', 24.5),          # Telescope focal length [m]
    # Grism
    ('grism_dispersion', 9.8),            # Rough spectral dispersion [AA/px]
    ('grism_prism_material', 'FS'),       # Prism glass
    ('grism_grating_material', 'FS'),     # Grating resine
    ('grism_prism_angle', 2.88 / S.RAD2DEG),   # Prism angle [rad]
    # ('grism_grating_rho', 19.29),       # Grating groove density [lines/mm]
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
    ('grism_prism_tiltx', 0),   # Prism x-tilt (around apex/groove axis) [rad]
    ('grism_prism_tilty', 0),   # Prism y-tilt [rad]
    ('grism_prism_tiltz', 0),   # Prism z-tilt (around optical axis) [rad]
    # Camera
    ('camera_flength', 957e-3),    # Focal length [m]
    ('camera_distortion', 29.6e-3),
    # Detector (without input recentering)
    # ('detector_dx', +0.70e-3),                 # Detector x-offset [m]
    # ('detector_dy', +179.7e-3),                # Detector y-offset [m]
    # Detector (with input offset of -0.85 deg)
    ('detector_dx', +0.70e-3),           # Detector x-offset [m]
    ('detector_dy', -4.20e-3),           # Detector y-offset [m]
])


class ZemaxPositions(S.DetectorPositions):

    """
    Zemax simulated positions, in spectroscopic or photometric mode.

    Zemax configuration modes ('confNB'):

    * 1: B-grism (NISP-S)
    * 2, 3, 4: R-grisms (NISP-S)
    * 5, 6, 7: Y, J, H (NISP-P)
    """

    colnames = """
confNb wave xindeg yindeg ximgmm yimgmm pxsize nxpup nypup nximg nyimg
ximgcpx yimgcpx ximgcmm yimgcmm xpsfcmm ypsfcmm
ee50mm ee80mm ee90mm ellpsf papsfdeg""".split()  #: Input column names

    # List of non-standard dataframe attributes (see
    # http://pandas.pydata.org/pandas-docs/stable/internals.html#define-original-properties)
    _internal_names = S.DetectorPositions._internal_names + ['filenames']
    _internal_names_set = set(_internal_names)

    def __init__(self, simulations, colname='psfcmm'):
        """
        Initialize from `simulations` = {mode: filename}.
        """

        initialized = False

        # Extract simulation name from input dictionary
        name = simulations.get("name", "Anonymous")

        self.filenames = simulations  # {mode: filename}

        for mode, filename in self.filenames.iteritems():
            if mode == 'name':
                continue
            # Read dataframe (wavelengths as index, input coordinates in columns)
            df = self.read_simulation(filename=filename, colname=colname)
            if not initialized:
                # Initialize DetectorPositions from wavelenths and coordinates
                super(ZemaxPositions, self).__init__(
                    df.index.values,
                    df.columns.values.astype(complex),
                    name=name,
                )
                initialized = True
            # Add dataframe
            self.add_mode(mode, df)

    @property
    def orders(self):
        """Dispersion orders (int modes)."""

        return [ mode for mode in self.modes if S.is_spectred(mode) ]

    @property
    def bands(self):
        """Undispersed photometric bands (string modes)."""

        return [ mode for mode in self.modes if not S.is_spectred(mode) ]

    def __str__(self):

        s = "Simulations '{}': {} modes".format(self.name, len(self.modes))
        for order in self.orders:
            s += "\n  Order #{}: {}".format(order, self.filenames[order])
        for band in self.bands:
            s += "\n  Band   {}: {}".format(band, self.filenames[band])

        waves = self.wavelengths / 1e-6  # [µm]
        coords = self.coordinates
        s += "\n  Wavelengths: {} steps from {:.2f} to {:.2f} µm".format(
            len(waves), min(waves), max(waves))
        s += "\n  Coords: {} sources".format(len(coords))

        return s

    @classmethod
    def read_simulation(cls, filename, colname='psfcmm'):
        """
        Read single-mode simulation.

        :param str filename: input simulation filename
        :param str colname: column name suffix to be used as output positions
                            (in **mm**)
        :return: (wavelength, input coordinates) dataframe
        :rtype: :class:`pandas.DataFrame`

        .. Warning:: this function relies on :meth:`pandas.DataFrame.pivot`.
           Because of Pandas `issue #12666
           <https://github.com/pydata/pandas/issues/12666>`_ and related Numpy
           `issue 7535 <https://github.com/numpy/numpy/issues/7435>`_, this
           requires Numpy >= v1.11.
        """

        if 'mm' not in colname:  # Input coords are supposed to be in mm
            raise NotImplementedError("Only output columns in mm are supported.")

        data = N.genfromtxt(filename, dtype=None, names=cls.colnames)

        # Cleanup: some Xin are arbitrarily close to zero
        warnings.warn("Setting approximately null xindeg to 0")
        data['xindeg'][N.abs(data['xindeg']) < 1e-12] = 0

        # Cleanup: offset yindeg by -0.85 deg
        warnings.warn("Offsetting Yin by -0.85 deg")
        data['yindeg'] += -0.85

        # Cleanup: upper-right position has no 1.85 µm wavelength
        warnings.warn("Discarding wavelengths > 1.81 µm")
        data = data[data['wave'] < 1.81]

        # Convert to DataFrame and modify/add columns
        df = PD.DataFrame(data)
        # Wavelengths [m]
        df['wave'] *= 1e-6
        # Input complex coordinates [rad]
        df['xyinrad'] = (df['xindeg'] + 1j * df['yindeg']) / S.RAD2DEG
        # Output complex coordinates [m]
        df['xyoutm'] = (df['x' + colname] + 1j * df['y' + colname]) * 1e-3

        # Round wavelengths and input coordinates before pivoting
        df = df.round({'wave': S.DetectorPositions.digits,
                       'xyinrad': S.DetectorPositions.digits})

        # Convert column 'xyinrad' into columns
        pivot = df.pivot("wave", "xyinrad", "xyoutm")

        # Make sure wavelengths and input complex coordinates are sorted
        pivot.sort_index(inplace=True)
        pivot = pivot.reindex_axis(             # Complex coordinates
            N.sort(pivot.columns.values.astype(complex)), axis=1)

        return pivot

    def get_simcfg(self):
        """
        Generate a :class:`spectrogrism.spectrogrism.SimConfig`
        corresponding to current simulation.
        """

        # Wavelengths [m]
        waves = self.wavelengths
        # Input (complex) coordinates [rad]
        coords = self.coordinates
        # Convert back to [[x, y]]
        coords = N.vstack((coords.real, coords.imag)).T

        simcfg = S.Configuration([('name', self.name),
                                  ('wave_npx', len(waves)),
                                  ('wave_range', [min(waves), max(waves)]),
                                  ('modes', self.modes),
                                  ('input_coords', coords)
                                  ])

        return S.SimConfig(simcfg)

    def plot_input(self, ax=None, **kwargs):
        """Plot input coordinates [deg]."""

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(
                1, 1, 1,
                xlabel="x [deg]", ylabel="y [deg]",
                title="Simulation '{}'\nInput positions".format(self.name))

        coords = self.coordinates * S.RAD2DEG  # [deg]

        ax.scatter(coords.real, coords.imag, **kwargs)

        ax.set_aspect('equal', adjustable='datalim')

        return ax

    def plot_output(self, ax=None, modes=None, subsampling=0, **kwargs):
        """Plot output (detector-level) positions [mm]."""

        if modes is None:
            modes = self.modes

        ax = self.plot(ax=ax, modes=modes,
                       subsampling=subsampling, **kwargs)

        title = "Simulation '{}'".format(self.name)
        if subsampling:
            title += u" (subsampled ×{})".format(subsampling)
        ax.set_title(title)

        return ax

    def plot_offsets(self, other, ax=None, mode=1, nwaves=3):
        """Plot output (detector-level) coordinate offsets [px]."""

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x [mm]", ylabel="y [mm]")
            title = "Simulation '{}', {}\nposition offsets [px]".format(
                self.name, S.str_mode(mode))
            ax.set_title(title)
            ax.set_aspect('equal', adjustable='datalim')
        else:
            fig = None          # Will serve as a flag

        # Position offset
        zpos = self[mode] / 1e-3    # [mm]
        opos = other[mode] / 1e-3   # [mm]
        dpos = opos - zpos          # [mm]
        dpos /= other.spectrograph.detector.pxsize / 1e-3  # [px]

        # Create subsampled wavelength vector of length nwaves
        waves = self.wavelengths
        iwaves = N.linspace(0, len(waves) - 1, nwaves).round().astype(int)
        waves = waves[iwaves]
        # Generate colormap
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


# Main ====================================================

if __name__ == '__main__':

    try:
        import seaborn
        seaborn.set_style("darkgrid")
    except ImportError:
        pass

    simulations = S.Configuration([
        ("name", "Zemax"),
        (1, "data/run_190315.dat"),            # 1st-order dispersed simulation
        (0, "data/run_011115_conf2_o0.dat"),   # 0th-order dispersed simulation
        (2, "data/run_161115_conf2_o2.dat"),   # 2nd-order dispersed simulation
        ('J', "data/run_071215_conf6_J.dat"),  # J-band undispersed simulation
    ])

    subsampling = 3             # Subsample output plot
    adjust = False              # Test optical parameter adjustment
    embed_html = False          # Generate MPLD3 figure
    plot_offset = False         # Offset plots

    # Zemax simulations
    zmx_pos = ZemaxPositions(simulations)
    print(zmx_pos)

    # Optical modeling
    optcfg = NISP_R                 # Optical configuration (default NISP)
    simcfg = zmx_pos.get_simcfg()   # Simulation configuration

    spectro = S.Spectrograph(optcfg,
                             telescope=S.Telescope(optcfg))
    print(spectro)

    # Tests
    print(" Spectrograph round-trip test ".center(S.LINEWIDTH, '-'))
    for mode in zmx_pos.modes:                  # Loop over all observing modes
        if not spectro.test(
                waves=zmx_pos.wavelengths,      # Test on all input wavelengths
                coords=zmx_pos.coordinates[0],  # Test on 1st input coordinates
                mode=mode, verbose=False):
            warnings.warn("{}: backward modeling does not match."
                          .format(S.str_mode(mode)))
        else:
            print("{}: OK".format(S.str_mode(mode)))

    # Spectroscopic modes ==============================

    spec_pos = spectro.predict_positions(simcfg, modes=zmx_pos.orders)
    spec_pos.check_alignment(zmx_pos)

    # Plots
    # ax = zmx_pos.plot_input()

    kwargs = dict(s=20, edgecolor='k', linewidths=1)  # Outlined symbols
    ax = zmx_pos.plot_output(modes=zmx_pos.orders,
                             subsampling=subsampling, **kwargs)

    # kwargs = dict(edgecolor=None, facecolor='none', linewidths=1)  # Open symbols
    kwargs = {}                   # Default
    for order in zmx_pos.orders:  # Loop over spectroscopic dispersion orders
        # Compute RMS on current order positions
        rms = zmx_pos.compute_rms(spec_pos, mode=order)
        print("Order #{}: RMS = {:.4f} mm = {:.2f} px".format(
            order, rms / 1e-3, rms / spectro.detector.pxsize))
        spec_pos.plot(ax=ax, zorder=0,  # Draw below Zemax
                      modes=(order,), blaze=(order != 1),
                      subsampling=subsampling,
                      label="{} #{} (RMS={:.1f} px)".format(
                          spec_pos.name, order,
                          rms / spectro.detector.pxsize),
                      **kwargs)

    if adjust:                  # Optical adjustment
        result = spectro.adjust(
            zmx_pos, simcfg, tol=1e-4,
            optparams=[
                'detector_dy',  # 'detector_dx',
                'grism_prism_tiltz',
                # 'telescope_flength', 'collimator_flength', 'camera_flength',
                # 'collimator_distortion', 'camera_distortion',
            ])
        if result.success:          # Adjusted model
            spec_pos_fit = spectro.predict_positions(simcfg)
            spec_pos_fit.plot(ax=ax, zorder=0,
                              subsampling=subsampling,
                              label="Adjusted {} (RMS={:.1f} px)".format(
                                  spec_pos.name,
                                  result.rms / spectro.detector.pxsize))

    ax.axis([-100, +100, -100, +100])               # [mm]
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', fontsize='small', frameon=True, framealpha=0.5)

    if embed_html:
        figname = zmx_pos.name + '_mpld3.html'
        try:
            S.dump_mpld3(ax.figure, figname)
        except ImportError:
            warnings.warn("MPLD3 is not available, cannot export to HTML.")

    # Position offset quiver plots
    if plot_offset:
        for order in zmx_pos.orders:
            ax = zmx_pos.plot_offsets(spec_pos, mode=order)

    # Imagery modes ==============================

    phot_pos = spectro.predict_positions(simcfg, modes=zmx_pos.bands)
    phot_pos.check_alignment(zmx_pos)

    kwargs = dict(s=20, edgecolor='k', linewidths=1)  # Outlined symbols
    ax = zmx_pos.plot_output(modes=zmx_pos.bands, **kwargs)

    kwargs = {}                 # Default
    for band in zmx_pos.bands:  # Loop over photometric bands
        # Compute RMS on positions
        rms = zmx_pos.compute_rms(phot_pos, mode=band)
        print("Band   {}: RMS = {:.4f} mm = {:.2f} px".format(
            band, rms / 1e-3, rms / spectro.detector.pxsize))
        phot_pos.plot(ax=ax, zorder=0,  # Draw below Zemax
                      modes=(band,),
                      label="{} {} (RMS={:.1f} px)".format(
                          phot_pos.name, band,
                          rms / spectro.detector.pxsize),
                      **kwargs)

    ax.axis([-100, +100, -100, +100])               # [mm]
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', fontsize='small', frameon=True, framealpha=0.5)

    P.show()
