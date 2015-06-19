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
import numpy as N
import matplotlib.pyplot as P

import spectrogrism as S

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
#: .. Note:: The detector plane is tiled with 4×4 detectors of 2k×2k pixels of
#:    18 µm; the spectrograph has a mean magnification (`NISPPlateScale`) of
#:    0.5 approximately.  Hence a focal plane of approximately 29×29 cm².
#:
#: Sources: *NISP Technical Description* (EUCL-LAM-OTH-7-001)
NISP_R = dict(
    name="NISP-R",
    wave_ref=1.5e-6,                    # Reference wavelength [m]
    wave_range=[1.25e-6, 1.85e-6],      # Wavelength range [m]
    # Telescope
    telescope_flength=24.5,             # Telescope focal length [m]
    # Grism
    grism_on=True,                      # Is prism on the way?
    grism_dispersion=9.8,               # Informative spectral dispersion [AA/px]
    grism_prism_material='FS',          # Prism glass
    grism_grating_material='FS',        # Grating resine
    grism_prism_angle=2.88/S.RAD2DEG,   # Prism angle [rad]
    # grism_grating_rho=19.29,          # Grating groove density [lines/mm]
    grism_grating_rho=13.72,            # Grating groove density [lines/mm]
    grism_grating_blaze=2.6/S.RAD2DEG,  # Blaze angle [rad]
    # Detector
    detector_pxsize=18e-6,              # Detector pixel size [m]
)

# Guessed values (not from official documents)
NISP_R.update(
    # Telescope
    telescope_flength=25.2,            # Telescope focal length [m]
    # Collimator
    collimator_flength=1946e-3,        # Focal length [m]
    collimator_distortion=2.8e-3,
    # Grism
    grism_prism_angle=2.70/S.RAD2DEG,  # Prism angle [rad]
    grism_grating_rho=13.1,            # Grating groove density [lines/mm]
    # Camera
    camera_flength=957e-3,             # Focal length [m]
    camera_distortion=29.6e-3,
    # Detector (without input recentering)
    # detector_dx=+0.70e-3,            # Detector x-offset [m]
    # detector_dy=+179.7e-3,           # Detector y-offset [m]
    # Detector (with input offset of -0.85 deg)
    detector_dx=+0.70e-3,              # Detector x-offset [m]
    detector_dy=-4.20e-3,              # Detector y-offset [m]
)


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
        """Load simulation from *filename*."""

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

    def get_simcfg(self, orders=[1]):
        """Generate a :class:`SimConfig` corresponding to simulation."""

        # Unique wavelengths [m]
        waves = N.unique(self.data['wave']) * 1e-6   # [m]
        # Unique input coordinates
        coords = N.unique(self.data['xindeg'] + 1j * self.data['yindeg'])
        # Convert back to [[x, y]]
        coords = N.vstack((coords.real, coords.imag)).T / S.RAD2DEG   # [rad]

        simcfg = dict(
            name=self.filename,
            wave_npx=len(waves),
            wave_range=[min(waves), max(waves)],
            orders=orders,
            input_coords=coords,
            )

        return S.SimConfig(simcfg)

    def detector_positions(self):
        """Convert simulation to :class:`DetectorPositions`."""

        waves = N.sort(N.unique(self.data['wave']))
        coords = N.unique(
            self.data['xindeg'] + 1j * self.data['yindeg'])  # [deg]
        # print("{}: {} spectra of {} px".format(
        #     self.filename, len(coords), len(waves)))

        detector = S.DetectorPositions(waves * 1e-6,   # Wavelengths in [m]
                                       name=self.filename)
        for xy in coords:                  # Loop over input positions [deg]
            select = (N.isclose(self.data['xindeg'], xy.real) &
                      N.isclose(self.data['yindeg'], xy.imag))
            subdata = self.data[select]                     # Data selection
            subdata = subdata[N.argsort(subdata['wave'])]   # Sort subdata
            # Sanity check
            assert N.allclose(subdata['wave'], waves)
            dpos = subdata['xpsfmm'] + 1j * subdata['ypsfmm']   # [mm]
            detector.add_spectrum(xy/S.RAD2DEG, dpos * 1e-3, order=1)

        return detector

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

    def plot_output(self, ax=None, **kwargs):
        """Plot output (detector) coordinates (pixels)."""

        ax = self.detector.plot(ax=ax,
                                orders=kwargs.pop('orders', (1,)),
                                blaze=kwargs.pop('blaze', False),
                                subsampling=kwargs.pop('subsampling', 0),
                                **kwargs)
        title = "{} - PSF centroid".format(self.filename)
        if subsampling:
            title += u" (subsampled ×{})".format(subsampling)
        ax.set_title(title)

        return ax


if __name__ == '__main__':

    filename = "Zemax/run_190315.dat"
    subsampling = 3             # Subsample output plot
    adjust = False              # Test optical parameter adjustment
    embed_html = True           # Generate MPLD3 figure

    zmx = Zemax(filename)
    print(zmx)

    # Zemax simulation
    # ax = zmx.plot_input()
    ax = zmx.plot_output(subsampling=subsampling,
                         marker='.', s=20, edgecolor='k')

    # Optical modeling
    optcfg = S.OptConfig(NISP_R)  # Optical configuration (default NISP)
    simcfg = zmx.get_simcfg()     # Simulation configuration

    spectro = S.Spectrograph(optcfg,
                             grism_on=optcfg.get('grism_on', True),
                             telescope=S.Telescope(optcfg))

    # Test
    try:
        spectro.test(simcfg, verbose=False)
    except AssertionError as err:
        warnings.warn("Spectrograph test:", str(err))
    else:
        print("Spectrograph test: OK")

    # Simulation
    simu = spectro.simulate(simcfg, orders=(0, 1, 2))

    # Compute RMS on 1st-order positions
    zmx.detector.assert_compatibility(simu)
    # Dataframe of position offsets for 1st-order
    dpos = simu.spectra[1] - zmx.detector.spectra[1]
    rms = (dpos.abs()**2).values.mean()**0.5
    print("RMS = {:.4f} mm = {:.2f} px".format(
        rms / 1e-3, rms / spectro.detector.pxsize))

    if not adjust:                  # Out-of-the-box optical model
        simu.plot(ax=ax, zorder=0,  # Draw below Zemax
                  orders=(0,),
                  subsampling=subsampling,
                  label="{} #0".format(simu.name))
        simu.plot(ax=ax, zorder=0,  # Draw below Zemax
                  orders=(1,),
                  subsampling=subsampling,
                  label="{} #1 (RMS={:.1f} px)".format(
                      simu.name, rms / spectro.detector.pxsize))
        simu.plot(ax=ax, zorder=0,  # Draw below Zemax
                  orders=(2,), blaze=True,
                  subsampling=subsampling,
                  label="{} #2".format(simu.name))
    else:                           # Optical adjustment
        result = spectro.adjust(
            zmx.detector, simcfg, tol=1e-4,
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
    ax.legend(fontsize='small', frameon=True, framealpha=0.5)

    if embed_html:
        try:
            S.dump_mpl3d(ax, zmx.filename.replace('.dat', '.html'))
        except ImportError:
            warnings.warn("MPLD3 is not available, cannot export to HTML.")

    P.show()
