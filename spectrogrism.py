#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
spectrogrism
------------

.. autosummary::

   Configuration
   OptConfig
   SimConfig
   Spectrum
   PointSource
   DetectorPositions
   LateralColor
   Material
   CameraOrCollimator
   Collimator
   Camera
   Telescope
   Prism
   Grating
   Grism
   Spectrograph
"""

from __future__ import division, print_function

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"
__version__ = "0.1"
__docformat__ = 'restructuredtext en'


import warnings

import numpy as N
import pandas as PD
import matplotlib.pyplot as P
try:
    import seaborn
    seaborn.set_style("whitegrid",
                      # {'xtick.major.size': 6, 'xtick.minor.size': 3,
                      #  'ytick.major.size': 6, 'ytick.minor.size': 3},
    )
except ImportError:
    pass

# Options
N.set_printoptions(linewidth=100, threshold=10)
PD.set_option("display.float.format",
              # PD.core.format.EngFormatter(accuracy=2, use_eng_prefix=True)
              lambda x: '{:g}'.format(x)
)

# Constants ===============================================

RAD2DEG = 57.29577951308232     #: Convert from radians to degrees
RAD2MIN = RAD2DEG * 60          #: Convert from radians to arc minutes
RAD2SEC = RAD2MIN * 60          #: Convert from radians to arc seconds

#: SNIFS optical configuration, R-channel
SNIFS_R = dict(
    name="SNIFS-R",                   # Configuration name
    wave_ref=0.76e-6,                 # Reference wavelength [m]
    wave_range=[0.5e-6, 1.02e-6],     # Standard wavelength range [m]
    # Telescope
    telescope_flength=22.5,           # Focal length [m]
    # Collimator
    collimator_flength=169.549e-3,    # Focal length [m]
    collimator_distortion=+2.141,     # r² distortion coefficient
    collimator_lcolor_coeffs=[-4.39879e-6, 8.91241e-10, -1.82941e-13],
    # Grism
    grism_on=True,                    # Is prism on the way?
    grism_prism_material='BK7',       # Prism glass
    grism_prism_angle=17.28/RAD2DEG,  # Prism angle [rad]
    grism_grating_rho=200.,           # Grating groove density [lines/mm]
    grism_dispersion=2.86,            # Informative spectral dispersion [AA/px]
    grism_grating_material='EPR',     # Grating resine
    grism_grating_blaze=15./RAD2DEG,  # Blaze angle [rad]
    # Camera
    camera_flength=228.014e-3,        # Focal length [m]
    camera_distortion=-0.276,         # r² distortion coefficient
    camera_lcolor_coeffs=[+2.66486e-6, -5.52303e-10, 1.1365e-13],
    # Detector
    detector_pxsize=15e-6,            # Detector pixel size [m]
    detector_angle=0/RAD2DEG,         # Rotation of the detector (0=blue is up)
)

#: SNIFS simulation configuration
SNIFS_SIMU = dict(
    name="standard, order=(-1, 0, +1, +2)",      # Configuration name
    wave_npx=10,                      # Nb of pixels per spectrum
    orders=range(-1, 3),              # Dispersion orders
    # Focal plane sampling
    input_coords=N.linspace(-1e-2, 1e-2, 5),  # [m]
    input_angle=-10/RAD2DEG,          # Rotation of the focal plane
)

# Technical Classes ==========================================


class Configuration(dict):

    """
    A simple dict-like configuration.

    .. autosummary::

       override
       save
       load
    """

    conftype = 'Configuration'            #: Configuration type

    def __init__(self, adict={}):

        dict.__init__(self, adict)
        self.name = self.pop('name', 'default')

    def __str__(self):

        s = [" {} {!r} ".format(self.conftype, self.name).center(60, '-')]
        s += [ '  {:10s}: {}'.format(key, self[key])
               for key in sorted(self.keys()) ]

        return '\n'.join(s)

    def override(self, adict):
        """Override configuration from dictionary."""

        # warnings.warn(
        #     "Overriding configuration {!r} with test values {}".format(
        #         self.name, adict))

        self.name = adict.pop('name',
                              self.name + ' (overrided)'
                              if not self.name.endswith(' (overrided)')
                              else self.name)
        self.update(adict)

    def save(self, yamlname):
        """Save configuration to YAML file."""

        import yaml

        with open(yamlname, 'w') as yamlfile:
            yaml.dump(dict(self, name=self.name), yamlfile)

        print("Configuration {!r} saved in {!r}".format(self.name, yamlname))

    @classmethod
    def load(cls, yamlname):
        """Load configuration from YAML file."""

        import yaml

        with open(yamlname, 'r') as yamlfile:
            adict = yaml.load(yamlfile)

        self = cls(adict)
        print("Configuration {!r} loaded from {!r}".format(
            self.name, yamlname))

        return self


class OptConfig(Configuration):

    """
    Optical configuration.
    """

    conftype = "Optical configuration"

    @property
    def wref(self):

        return self.get('wave_ref', sum(self['wave_range'])/2.)


class SimConfig(Configuration):

    """
    Simulation configuration.

    .. autosummary::

       get_waves
       get_coords
    """

    conftype = "Simulation configuration"

    def get_waves(self, config):
        """Simulated wavelengthes."""

        if 'WAVES' in self:
            waves = N.atleast_1d(self['WAVES'])
        else:
            npx = self.get('wave_npx', 1)
            if npx > 1:
                wmin, wmax = self.get('wave_range', config['wave_range'])
                waves = N.linspace(wmin, wmax, npx)
            elif npx == 1:
                waves = config.wref
            else:
                raise ValueError("Invalid number of pixels: {}".format(npx))

        return waves

    def get_coords(self):
        """Simulated input coordinates `[ x + 1j*y ]`."""

        incoords = N.atleast_1d(self.get('input_coords', 0))
        if N.ndim(incoords) == 1:        # [x]: generate square sampling x × x
            x, y = N.meshgrid(incoords, incoords)
            coords = (x + 1j * y).ravel()
        elif N.ndim(incoords) == 2 and N.shape(incoords)[1] == 2:
            # [[x, y]]: arbitrary sampling
            coords = incoords[:, 0] + 1j * incoords[:, 1]
        else:
            raise NotImplementedError("Unsupported input coordinates")

        return coords


class Spectrum(object):

    """
    A list of fluxes at different wavelengths.

    .. autosummary::

       default
    """

    def __init__(self, wavelengths, fluxes, name='spectrum'):
        """
        Initialize from wavelength and flux arrays.

        :param numpy.ndarray wavelengths: input wavelengths [m]
        :param numpy.ndarray fluxes: input fluxes [arbitrary units]
        :param str name: optional spectrum name (e.g. "Grism transmission")
        """

        self.wavelengths = N.array(wavelengths)  #: Wavelengths (sorted) [m]
        self.fluxes = N.array(fluxes)            #: Fluxes [AU]
        self.name = name                         #: Spectrum name

        # Some tests
        assert len(self.wavelengths) == len(self.fluxes), \
            "Incompatible wavelength and flux arrays"
        assert N.all(N.diff(self.wavelengths) > 0), \
            "Wavelengths not strictly increasing"

    def __str__(self):

        return "{}{:d} px within {:.2f}-{:.2f} µm".format(
            self.name + ': ' if self.name else '',
            len(self.wavelengths),
            self.wavelengths[0] / 1e-6, self.wavelengths[-1] / 1e-6)

    @classmethod
    def default(cls, waves=1e-6, name='spectrum'):
        """
        A default constant-flux spectrum.

        :param waves: wavelength vector [m]
        :param str name: optional name
        :return: constant-flux spectrum
        :rtype: :class:`Spectrum`
        """

        wavelengths = N.atleast_1d(waves)
        fluxes = N.ones_like(wavelengths)

        return cls(wavelengths, fluxes, name=name)


class Coordinates2D(complex):

    """
    A 2D-coordinate system, for linear positions or angular directions.

    .. autosummary::

       to_polar
       from_polar
    """

    def __new__(cls, *args, **kwargs):

        return complex.__new__(cls, *args, **kwargs)

    def to_polar(self):

        return N.abs(self), N.angle(self)

    @classmethod
    def from_polar(cls, r, phi):

        return Coordinates2D(r * N.exp(1j*phi))


class Direction2D(Coordinates2D):

    """
    A 2D-angular direction.
    """

    # def __str__(self):

    #     tantheta, phi = self.to_polar()
    #     return "{:+.2f} x {:+.2f} arcmin".format(
    #         N.arctan(N.abs(self))*RAD2MIN, N.angle(self)*RAD2MIN)

    def __str__(self):

        z = self * RAD2MIN
        return "{:+.1f} × {:+.1f} arcmin".format(z.real, z.imag)


class Position2D(Coordinates2D):

    """
    A 2D-linear position.
    """

    def __str__(self):

        z = self / 1e-3                    # [mm]
        return "{:+.1f} × {:+.1f} mm".format(z.real, z.imag)


class PointSource(object):

    """
    A :class:`Spectrum` associated to a 2D-position or direction.
    """

    def __init__(self, coords, spectrum=None, **kwargs):
        """
        Initialize from position/direction and spectrum.

        :param complex coords: 2D-position [m] or direction [rad]
        :param spectrum: :class:`Spectrum` (default to standard spectrum)
        :param kwargs: propagated to :func:`Spectrum.default()` constructor
        """

        assert isinstance(coords, Coordinates2D)
        self.coords = coords                     #: Position/direction

        if spectrum is None:
            spectrum = Spectrum.default(**kwargs)
        else:
            assert isinstance(spectrum, Spectrum), \
                "spectrum should be a Spectrum, not {}".format(type(spectrum))
        self.spectrum = spectrum                 #: Spectrum

    def __str__(self):

        return "{}, {}".format(self.coords, self.spectrum)


class DetectorPositions(object):

    """
    A container for positions on the detector.

    A Pandas-based container for (complex) positions in the detector plane,
    namely an order-keyed dictionary of :class:`pandas.DataFrame` including
    complex detector positions, with wavelengths as `index` and coords as
    `columns`.

    .. Warning:: :class:`pandas.Panel` does not seem to support deffered
       construction, hence the usage of a order-keyed dict.

    .. Warning:: indexing by float is not really a good idea. Float indices
       (wavelengths and coordinates) are therefore rounded first with a
       sufficient precision (e.g. nm for wavelengths).

    .. autosummary::

       add_spectrum
       plot
    """

    markers = {0: '.', 1: 'o', 2: 's'}

    def __init__(self, wavelengths, spectrograph=None, name='default'):
        """
        Initialize from spectrograph and wavelength array.
        """

        if spectrograph is not None:
            assert isinstance(spectrograph, Spectrograph), \
                "spectrograph should be a Spectrograph"
        self.spectrograph = spectrograph        #: Associated spectrograph

        self.lbda = N.around(wavelengths, 12)   #: Rounded wavelengths [m]
        self.spectra = {}                       #: {order: dataframe}
        self.name = name                        #: Name

    def get_coords(self, order=1):

        return N.sort_complex(
            self.spectra[order].columns.values.astype(N.complex))

    def add_spectrum(self, coords, detector_positions, order=1):
        """
        Add `detector_positions` corresponding to source at `coords`
        and dispersion `order`.

        :param complex coords: input source coordinates
        :param numpy.ndarray detector_positions: (complex) positions
            in the detector plane
        :param int order: dispersion order
        """

        assert len(detector_positions) == len(self.lbda), \
            "incompatible detector_positions array"

        rcoords = N.around(coords, 12)          # Rounded coordinates
        df = self.spectra.setdefault(order,
                                     PD.DataFrame(index=self.lbda,
                                                  columns=(rcoords,)))
        df[rcoords] = detector_positions

    def plot(self, ax=None, coords=None, orders=None, blaze=False, **kwargs):
        """
        Plot spectra on detector plane.

        :param ax: pre-existing :class:`matplotlib.pyplot.Axes` instance if any
        :param tuple coords: selection of input coordinates to be plotted
        :param tuple orders: selection of dispersion orders to be plotted
        :param bool blaze: encode the blaze function in the marker size
        :return: :class:`matplotlib.pyplot.Axes`
        """

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x [mm]", ylabel="y [mm]",
                                 title=self.name)
            # ax.set_aspect('equal', adjustable='datalim')
        else:
            fig = None          # Will serve as a flag

        kwargs.setdefault('edgecolor', 'none')
        kwargs.setdefault('label', self.name)

        # Segmented colormap
        cmap = P.matplotlib.colors.LinearSegmentedColormap(
            'dummy', P.get_cmap('Spectral_r')._segmentdata, len(self.lbda))

        # Default blaze transmission
        bztrans = N.ones_like(self.lbda)

        if orders is None:                        # Plot all orders
            orders = sorted(self.spectra)

        for order in orders:
            try:
                df = self.spectra[order]
            except KeyError:
                warnings.warn("Order #{} not in '{}', skipped".format(
                    order, self.name))
                continue

            kwcopy = kwargs.copy()
            marker = kwcopy.pop('marker',
                                self.markers.get(abs(order), 'o'))

            if blaze and self.spectrograph:
                bztrans = self.spectrograph.grism.blaze_function(
                    self.lbda, order)
                s = kwcopy.pop('s', N.maximum(20 * N.sqrt(bztrans), 5))
            else:
                s = kwcopy.pop('s', 20)

            if coords is None:                    # Plot all spectra
                coords = self.get_coords(order=order)

            for xy in coords:                     # Loop over sources
                try:
                    positions = df[xy].values / 1e-3  # Complex positions [mm]
                except KeyError:
                    warnings.warn("Source {} is unknown, skipped".format(
                        str_position(xy)))
                    continue

                sc = ax.scatter(positions.real, positions.imag,
                                c=self.lbda / 1e-6,   # Wavelength [µm]
                                cmap=cmap, s=s, marker=marker, **kwcopy)

                kwcopy.pop('label', None)  # Label only for one source

            kwargs.pop('label', None)      # Label only for one order

        if fig:
            fig.colorbar(sc, label=u"Wavelength [µm]")

        return ax

    def assert_compatibility(self, other, order=1):

        assert isinstance(other, DetectorPositions)
        assert N.allclose(self.lbda, other.lbda), \
            "{!r} and {!r} have incompatible wavelengths".format(
                self.name, other.name)
        assert N.allclose(self.get_coords(order), other.get_coords(order)), \
            "{!r} and {!r} have incompatible input coordinates".format(
                self.name, other.name)


class LateralColor(object):

    """
    A description of lateral color chromatic distortion.

    The transverse chromatic aberration (so-called *lateral color*) occurs when
    different wavelengths are focused at different positions in the focal
    plane.

    **Reference:** `Chromatic aberration
    <https://en.wikipedia.org/wiki/Chromatic_aberration>`_

    .. autosummary::

       amplitude
    """

    def __init__(self, wref, coeffs):
        """
        Initialization from reference wavelength [m] and lateral color
        coefficients.

        :param float wref: reference wavelength [m]
        :param list coeffs: lateral color coefficients
        """

        self.wref = wref               #: Reference wavelength [m]
        self.coeffs = N.array(coeffs)  #: Lateral color coefficients

    def __str__(self):

        return "Lateral color: lref={:.2f} µm, coeffs={}".format(
            self.wref / 1e-6,
            ', '.join( '{:+g}'.format(coeff) for coeff in self.coeffs))

    def amplitude(self, wavelengths):
        """
        Amplitude of the lateral color chromatic distortion.

        :param numpy.ndarray wavelengths: wavelengths [mm]
        :return: lateral color amplitude
        """

        if len(self.coeffs):
            return N.sum([ c*(wavelengths - self.wref)**i
                           for i, c in enumerate(self.coeffs, start=1) ],
                           axis=0)
        else:
            return N.zeros_like(wavelengths)


class Material(object):

    """
    Optical material.

    The refractive index is described by its Sellmeier coefficients.

    **Reference:** `Sellmeier equation
    <https://en.wikipedia.org/wiki/Sellmeier_equation>`_

    .. autosummary::

       index
    """

    #: Sellmeier coefficients [B1, B2, B3, C1, C2, C3] of known materials.
    materials =  dict(
        # Glasses
        BK7=[ 1.03961212,    2.31792344e-1, 1.01046945,
              6.00069867e-3, 2.00179144e-2, 1.03560653e+2],
        UBK7=[1.01237433,    2.58985218e-1, 1.00021628,
              5.88328615e-3, 1.90239921e-2, 1.04079777e+2],
        SF4=[ 1.61957826,    3.39493189e-1, 1.02566931,
              1.25502104e-2, 5.33559822e-2, 1.1765222e+2],
        SK5=[ 0.99146382,    4.95982121e-1, 0.98739392,
              5.22730467e-3, 1.72733646e-2, 0.983594579e+2],
        F2=[  1.34533359,    2.09073176e-1, 0.93735716,
              9.97743871e-3, 4.70450767e-2, 1.11886764e+2],
        SF57=[1.81651371,    4.28893641e-1, 1.07186278,
              1.43704198e-2, 5.92801172e-2, 1.21419942e+2],
        # Fused silica
        FS=[  0.6961663,     4.079426e-1,   0.8974794,
              4.679148e-3,   1.351206e-2,  97.9340],
        # Epoxy
        EPR=[ 0.512479,      0.838483,     -0.388459,
             -0.0112765,     0.0263791,   557.682],
        EPB=[ 0.406836,      1.03517,      -0.140328,
             -0.0247382,     0.0261501,   798.366],
    )

    def __init__(self, name):
        """
        Initialize material from its name.

        :param str name: material name (should be in Material.materials)
        :raise KeyError: unknown material name
        """

        self.name = name  #: Name of the material
        try:
            #: Sellmeier coefficients `[B1, B2, B3, C1, C2, C3]`
            self.coeffs = self.materials[name]
        except KeyError:
            raise KeyError("Unknown material {}".format(name))

    def __str__(self):

        return u"Material: {}, n(1 µm)={:.3f}".format(
            self.name, self.index(1e-6))

    def index(self, wavelengths):
        r"""
        Compute refractive index from Sellmeier expansion.

        Sellmeier expansion for refractive index:

        .. math:: n(\lambda)^2 = 1 + \sum_{i}\frac{B_i\lambda^2}{\lambda^2-C_i}

        with :math:`\lambda` in microns.

        :param numpy.ndarray wavelengths: wavelengths [m]
        :return: refractive index
        """

        lmu2 = (wavelengths / 1e-6)**2          # (wavelength [µm])**2
        n2m1 = N.sum([ b / (1 - c / lmu2)       # n**2 - 1
                       for b, c in zip(self.coeffs[:3], self.coeffs[3:]) ],
                       axis=0)

        return N.sqrt(n2m1 + 1)


# Optical element classes =================================

class CameraOrCollimator(object):

    """
    An optical system converting to and fro directions and positions.

    .. autosummary::

       rect2pol
       pol2rect
       invert_camcoll
    """

    def __init__(self, flength, distortion, lcolor):
        """
        Initialize the optical component from optical parameters.

        :param float flength: focal length [m]
        :param float distortion: r² distortion coefficient
        :param lcolor: :class:`LateralColor`

        .. Note:: We restrict here the model to a quadratic radial geometric
           distortion, responsible for barrel and pincushion distortions.
           Higher order radial distortions or tangential distortions could be
           implemented if needed.

           **Reference:** `Distortion
           <https://en.wikipedia.org/wiki/Distortion_%28optics%29>`_
        """

        if lcolor is None:
            lcolor = LateralColor(0, [])   # Null lateral color
        else:
            assert isinstance(lcolor, LateralColor), \
                "lcolor should be a LateralColor"

        self.flength = flength        #: Focal length [m]
        self.distortion = distortion  #: r² distortion coefficient
        self.lcolor = lcolor          #: Lateral color

    def __str__(self):

        s = "f={:.1f} m, e={:+.3f}".format(self.flength, self.distortion)
        if self.lcolor is not None and len(self.lcolor.coeffs):
            s += '\n  {}'.format(self.lcolor)

        return s

    @staticmethod
    def rect2pol(position):
        r"""
        Convert position :math:`x + jy` into modulus :math:`r` and
        phase :math:`\phi`.

        :param complex position: 2D-position(s) :math:`x + jy`
        :return: (r, phi) [rad]
        """

        return N.absolute(position), N.angle(position)

    @staticmethod
    def pol2rect(r, phi):
        r"""
        Convert modulus :math:`r` and phase :math:`\phi` into position
        :math:`x + jy = r\exp(j\phi`)`.

        :param numpy.ndarray r: modulus
        :param numpy.ndarray phi: phase [rad]
        :return: 2D-position
        """

        return r * N.exp(1j*phi)

    @staticmethod
    def invert_camcoll(y, e, b):
        """
        Invert :math:`y = x(ex^2 + b)`.

        :func:`numpy.poly1d` solves polynomial equation :math:`e x^3 + 0 x^2 +
        b x - y = 0`.

        :return: real solution (or NaN if none)
        """

        # Trivial cases
        if y == 0:
            return 0.
        elif b == 0 and e == 0:
            return N.nan
        elif b == 0:
            return (y/e)**(1/3)
        elif e == 0:
            return y/b

        poly = N.poly1d([e, 0, b, -y])
        roots = poly.r

        # Pure real roots
        real_roots = [ root.real for root in roots if N.isclose(root.imag, 0) ]

        if len(real_roots) == 1:   # A single real root
            return real_roots[0]
        elif len(real_roots) > 1:  # Multiple real roots: take the smallest
            return real_roots[N.argmin(N.abs(real_roots))]
        else:                      # No real roots
            return N.nan


class Collimator(CameraOrCollimator):

    """
    Convert a 2D-position (in the focal plane) into a 2D-direction.

    .. autosummary::

       forward
       backward
    """

    def __init__(self, config):
        """
        Initialization from optical configuration dictionary.

        :param dict config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            flength = config['collimator_flength']
            distortion = config.get('collimator_distortion', 0)
            lcolor = LateralColor(config.wref,
                                  config.get('collimator_lcolor_coeffs', []))
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}".format(
                    err.args[0]))

        super(Collimator, self).__init__(flength, distortion, lcolor)

    def __str__(self):

        return "Collimator: {}".format(super(Collimator, self).__str__())

    def forward(self, position, wavelengths, gamma):
        r"""
        Forward light propagation through the collimator.

        The collimator equation is:

        .. math::

            \tan\theta = r/f \times (1 + e(r/f)^2 + a(\lambda)/\gamma)

        where:

        * *f* is the focal length
        * *e* is the :math:`r^2` radial distortion coefficient
        * :math:`a(\lambda)` is the lateral color
        * :math:`\gamma` is the spectrograph magnification

        :param complex position: 2D-position in the focal plane [m]
        :param numpy.ndarray wavelengths: wavelengths [m]
        :param float gamma: spectrograph magnification (fcam/fcoll)
        :return: 2D-direction [rad]
        :rtype: complex
        """

        r, phi = self.rect2pol(position)  # Modulus [m] and phase [rad]
        rr = r / self.flength             # Normalized radius
        tmp = (self.distortion * rr**2 +
               self.lcolor.amplitude(wavelengths) / gamma)
        tantheta = rr * (1 + tmp)

        return self.pol2rect(tantheta, phi + N.pi)  # Direction

    def backward(self, direction, wavelength, gamma):
        """
        Backward light propagation through the collimator.

        See :func:`Collimator.forward` for parameters.
        """

        tantheta, phi = self.rect2pol(direction)

        rovf = self.invert_camcoll(tantheta,
                                   self.distortion,
                                   1 + self.lcolor.amplitude(wavelength)/gamma)

        return self.pol2rect(rovf * self.flength, phi + N.pi)  # Position


class Camera(CameraOrCollimator):

    """
    Convert a 2D-direction into a 2D-position (in the detector plane).

    .. Note:: the detector coordinate axes are flipped, so that
       sources remain in the same quadrant in the focal *and* detector
       planes.

    .. autosummary::

       forward
       backward
    """

    def __init__(self, config):
        """
        Initialization from optical configuration dictionary.

        :param dict config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            flength = config['camera_flength']
            distortion = config.get('camera_distortion', 0)
            lcolor = LateralColor(config.wref,
                                  config.get('camera_lcolor_coeffs', []))
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}".format(
                    err.args[0]))

        super(Camera, self).__init__(flength, distortion, lcolor)

    def __str__(self):

        return "Camera:     {}".format(super(Camera, self).__str__())

    def forward(self, direction, wavelengths):
        r"""
        Forward light propagation through the camera.

        The camera equation is:

        .. math:: r/f = \tan\theta  \times (1 + e\tan^2\theta + a(\lambda))

        where:

        * *f* is the focal length
        * *e* is the :math:`r^2` radial distortion coefficient
        * :math:`a(\lambda)` is the lateral color

        :param complex direction: 2D-direction [rad]
        :param numpy.ndarray wavelengths: wavelengths [m]
        :return: 2D-position [m]
        """

        tantheta, phi = self.rect2pol(direction)
        rovf = (1 + self.distortion * tantheta**2 +
                self.lcolor.amplitude(wavelengths)) * tantheta

        return self.pol2rect(rovf*self.flength, phi + N.pi)  # Flipped position

    def backward(self, position, wavelength):
        """
        Backward light propagation through the camera.

        See :func:`Camera.forward` for parameters.
        """

        r, phi = self.rect2pol(position)  # Modulus [m] and phase [rad]

        tantheta = self.invert_camcoll(r/self.flength,
                                       self.distortion,
                                       1 + self.lcolor.amplitude(wavelength))

        return self.pol2rect(tantheta, phi + N.pi)  # Flipped direction


class Telescope(Camera):

    """
    Convert a 2D-direction in the sky into a 2D-position in the focal plane.
    """

    def __init__(self, config):
        """
        Initialization from optical configuration dictionary.

        :param dict config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            flength = config['telescope_flength']
            distortion = config.get('telescope_distortion', 0)
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}".format(
                    err.args[0]))

        # Initialize from Camera parent class
        super(Camera, self).__init__(flength, distortion, lcolor=None)

    def __str__(self):

        return "Telescope:  {}".format(super(Camera, self).__str__())


class Prism(object):

    """
    A triangular transmissive prism.

    .. Note::

       * The entry surface is roughly perpendicular (up to the tilt
         *angles) to the optical axis Oz*.
       * The apex (prism angle) is aligned with the *x*-axis

    .. autosummary::

       rotation
       rotation_x
       rotation_y
       rotation_z
       refraction
    """

    def __init__(self, angle, material, tilts=(0, 0, 0)):
        """
        Initialize grism from optical parameters.

        :param float angle: prism angle [rad]
        :param material: prism :class:`Material`
        :param 3-tuple tilts: prism tilts (x,y,z) [rad]
        """

        assert isinstance(material, Material), "material should be a Material"
        assert len(tilts) == 3, "tilts should be a 3-tuple"

        self.angle = angle                #: Prism angle [rad]
        self.material = material          #: Prism material
        self.tilts = tilts                #: Prism tilts (x,y,z) [rad]

    def __str__(self):

        # Present tilt angles in arcmin
        tilts = ','.join( "{:+.0f}'".format(t * RAD2MIN) for t in self.tilts )

        return "Prism [{}]: A={:.2f}°, tilts={}".format(
            self.material.name, self.angle*RAD2DEG, tilts)

    @staticmethod
    def rotation(x, y, theta):
        """
        2D-rotation of position around origin with direct angle theta [rad].
        """

        # Rotation in the complex plane
        p = (N.array(x) + 1j*N.array(y)) * N.exp(1j*theta)

        return (p.real, p.imag)

    @classmethod
    def rotation_x(cls, xyz, theta):
        """Rotation around x-axis."""

        x, y, z = xyz
        y, z = cls.rotation(y, z, theta)
        return N.vstack((x, y, z)).squeeze()

    @classmethod
    def rotation_y(cls, xyz, theta):
        """Rotation around y-axis."""

        x, y, z = xyz
        x, z = cls.rotation(x, z, theta)
        return N.vstack((x, y, z)).squeeze()

    @classmethod
    def rotation_z(cls, xyz, theta):
        """Rotation around z-axis."""

        x, y, z = xyz
        x, y = cls.rotation(x, y, theta)
        return N.vstack((x, y, z)).squeeze()

    @staticmethod
    def refraction(xyz, n1, n2):
        """
        Refraction law from medium 1 to medium 2, by plane interface
        :math:`(Oxy)`.

        :param 3-tuple xyz: input 3D-direction (from medium 1)
        :param float n1: Refractive index of medium 1
        :param float n2: Refractive index of medium 2
        :return: output 3D-direction (to medium 2)
        :rtype: 3-tuple
        """

        x1, y1, z1 = xyz
        x2 = x1 * n1/n2
        y2 = y1 * n1/n2
        z2 = N.sqrt(1 - (x2**2 + y2**2))

        return N.vstack((x2, y2, z2)).squeeze()


class Grating(object):

    """
    A transmissive grating.

    The grooves of the (transmission) grating are aligned along *x*.

    .. autosummary::

       forward
       backward
    """

    def __init__(self, rho, material, blaze=0):
        """
        Initialize grating from optical parameters.

        :param float rho: grating groove density [lines/mm]
        :param material: grating :class:`Material`
        :param float blaze: grating blaze angle [rad]
        """

        assert isinstance(material, Material), "material should be a Material"

        self.rho = rho            #: Grating groove density [lines/mm]
        self.material = material  #: Grating material
        self.blaze = blaze        #: Grating blaze angle [rad]

    def __str__(self):

        return "Grating [{}]: rho={:.1f} g/mm, blaze={:.2f}°".format(
            self.material.name, self.rho, self.blaze*RAD2DEG)

    def forward(self, xyz, wavelengths, order=1):
        """
        Forward light propagation through a grating.

        The propagation is done from material (*n*) to vacuum (*1*).

        :param 3-tuple xyz: input 3D-direction *(x, y, z)* [m]
        :param numpy.ndarray wavelengths: wavelengths [m]
        :param int order: dispersion order
        :return: output 3D-direction *(x', y', z')*
        :rtype: 3-tuple
        """

        n = self.material.index(wavelengths)  # Index of refraction

        x, y, z = xyz
        xp = x * n
        yp = y * n + order * wavelengths * self.rho / 1e-3
        zp = N.sqrt(1 - (xp**2 + yp**2))

        return N.vstack((xp, yp, zp)).squeeze()

    def backward(self, xyz, wavelength, order=1):
        """
        Backward light propagation through a grating.

        The propagation is done from vacuum (*1*) to material (*n*).

        :param 3-tuple xyz: output 3D-direction *(x', y', z')* [m]
        :param float wavelength: wavelength [m]
        :param int order: dispersion order
        :return: input 3D-direction *(x, y, z)*
        :rtype: 3-tuple
        """

        n = self.material.index(wavelength)  # Index of refraction

        xp, yp, zp = xyz
        x = xp / n
        y = (yp - order * wavelength * self.rho / 1e-3) / n
        z = N.sqrt(1 - (x**2 + y**2))

        return N.vstack((x, y, z)).squeeze()


class Grism(object):

    """
    A :class:`Prism` and a :class:`Grating` on the exit surface.

    .. autosummary::

       blaze_function
       direction2xyz
       xyz2direction
       forward
       backward
       null_deviation
    """

    def __init__(self, config):
        """
        Initialization from optical configuration dictionary.

        :param dict config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            angle = config['grism_prism_angle']
            prism_material = config['grism_prism_material']
            rho = config['grism_grating_rho']
            grating_material = config['grism_grating_material']
            blaze = config.get('grism_grating_blaze', 0)
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}".format(
                    err.args[0]))

        self.prism = Prism(angle, Material(prism_material))
        self.grating = Grating(rho, Material(grating_material), blaze)

    @property
    def tilts(self):
        """
        Expose prism tilts *(x, y, z)* [rad].
        """

        return self.prism.tilts

    @tilts.setter
    def tilts(self, tilts):

        self.prism.tilts = tilts

    def __str__(self):

        return """Grism:
  {0.prism}
  {0.grating}
  1st-order null-deviation wavelength: {1:.2f} µm""".format(
            self, self.null_deviation(order=1) / 1e-6)

    def blaze_function(self, wavelengths, order=1):
        r"""
        Return blaze function.

        In the normal configuration, the blaze function of a grism is given by

        .. math:: B = \frac{\sin^2\Theta}{\Theta^2}

        with:

        - :math:`\rho \lambda \Theta = \pi \cos\gamma \times (n_g\sin
          i - \sin r)`, :math:`\gamma` being the blaze angle and
          :math:`\rho` the line density of the grating;
        - :math:`i = \alpha' - \gamma` where :math:`n_g \sin\alpha' =
          n_p\sin A` with :math:`n_p` and :math:`n_g` the refraction
          index of prism glass and grating resine respectively and
          :math:`A` the grism angle
        - :math:`r = \beta - \gamma` where :math:`\sin\beta = n_p\sin
          A - m\rho\lambda` with :math:`m` the diffraction order.

        :param numpy.ndarray wavelengths: wavelengths [m]
        :param int order: dispersion order
        :return: blaze function (transmission)
        :rtype: :class:`numpy.ndarray`
        """

        np = self.prism.material.index(wavelengths)    # Prism index
        ng = self.grating.material.index(wavelengths)  # Grating index

        rholbda = self.grating.rho / 1e-3 * wavelengths  # g/m * m = unitless
        npsinA = np * N.sin(self.prism.angle)

        i = N.arcsin(npsinA / ng) - self.grating.blaze               # [rad]
        r = N.arcsin(npsinA - order * rholbda) - self.grating.blaze  # [rad]

        theta = (N.pi / rholbda * N.cos(self.grating.blaze) *
                 (ng*N.sin(i) - N.sin(r)))
        bf = (N.sin(theta)/theta)**2

        return bf

    @staticmethod
    def direction2xyz(direction):
        """
        Convert a 2D-direction into a 3D-direction (i.e. a unit vector).

        :param complex direction: 2D-direction
        :return: 3D-direction
        :type: 3-tuple
        """

        tantheta, phi = CameraOrCollimator.rect2pol(direction)
        tan2 = tantheta**2
        costheta = N.sqrt(1/(1 + tan2))
        sintheta = N.sqrt(tan2/(1 + tan2))

        return N.vstack((N.cos(phi)*sintheta,
                         N.sin(phi)*sintheta,
                         costheta)).squeeze()

    @staticmethod
    def xyz2direction(xyz):
        """
        Convert a 3D-direction (i.e. a unit vector) into a 2D-direction.

        :param 3-tuple xyz: 3D-direction
        :return: 2D-direction
        :rtype: complex
        """

        x, y, z = xyz
        tantheta = N.hypot(x, y) / z
        phi = N.arctan2(y, x)

        return CameraOrCollimator.pol2rect(tantheta, phi)

    def forward(self, direction, wavelengths, order=1):
        """
        Forward propagation through a grism (prism + grating).

        :param complex direction: 2D-direction [rad]
        :param numpy.ndarray wavelengths: wavelengths [m]
        :param int order: dispersion order
        :return: 2D-directions [rad]
        :rtype: :class:`numpy.ndarray`
        """

        # Convert input 2D-direction into a 3D-direction
        xyz = self.direction2xyz(direction)

        # Get aligned with prism
        xyz = self.prism.rotation_x(xyz, self.prism.tilts[0])
        xyz = self.prism.rotation_y(xyz, self.prism.tilts[1])
        xyz = self.prism.rotation_z(xyz, self.prism.tilts[2])

        # Vacuum/prism interface
        xyz = self.prism.refraction(xyz, 1,
                                    self.prism.material.index(wavelengths))

        # Prism angle
        xyz = self.prism.rotation_x(xyz, self.prism.angle)

        # Prism/grating interface
        xyz = self.prism.refraction(xyz,
                                    self.prism.material.index(wavelengths),
                                    self.grating.material.index(wavelengths))

        # Grating dispersion
        xyz = self.grating.forward(xyz, wavelengths, order=order)

        # Back to original orientation
        xyz = self.prism.rotation_x(xyz, -self.prism.angle)
        xyz = self.prism.rotation_z(xyz, -self.prism.tilts[2])
        xyz = self.prism.rotation_y(xyz, -self.prism.tilts[1])
        xyz = self.prism.rotation_x(xyz, -self.prism.tilts[0])

        # Convert output 3D-directions into 2D-directions
        direction = self.xyz2direction(xyz)

        return direction

    def backward(self, direction, wavelength, order=1):
        """
        Backward propagation through a grism (prism + grating).

        See :func:`Grism.forward` for parameters.
        """

        # Convert output 2D-direction into a 3D-direction
        xyz = self.direction2xyz(direction)

        # Get aligned with A-tilted grating
        xyz = self.prism.rotation_x(xyz, self.prism.tilts[0])
        xyz = self.prism.rotation_y(xyz, self.prism.tilts[1])
        xyz = self.prism.rotation_z(xyz, self.prism.tilts[2])
        xyz = self.prism.rotation_x(xyz, self.prism.angle)

        # Grating
        xyz = self.grating.backward(xyz, wavelength, order=order)

        # Grating/prism interface
        xyz = self.prism.refraction(xyz,
                                    self.grating.material.index(wavelength),
                                    self.prism.material.index(wavelength))

        # Prism
        xyz = self.prism.rotation_x(xyz, -self.prism.angle)

        # Prism/vacuum interface
        xyz = self.prism.refraction(xyz,
                                    self.prism.material.index(wavelength), 1)

        # Back to original orientation
        xyz = self.prism.rotation_z(xyz, -self.prism.tilts[2])
        xyz = self.prism.rotation_y(xyz, -self.prism.tilts[1])
        xyz = self.prism.rotation_x(xyz, -self.prism.tilts[0])

        # Convert intput 3D-direction into 2D-direction
        direction = self.xyz2direction(xyz)

        return direction

    def null_deviation(self, order=1):
        r"""
        Null-deviation wavelength (approximate) [m].

        This is the solution to:

        .. math:: m \rho \lambda = (n(\lambda) - 1) \sin(A)

        where:

        - *A*: grism angle [rad]
        - :math:`\rho`: groove density [line/mm]
        - :math:`n(\lambda)`: prism refractive index
        - *m*: dispersion order

        :param int order: dispersion order
        :return: Null deviation wavelength [m]
        :raise RuntimeError: if not converging
        """

        import scipy.optimize as SO

        k = N.sin(self.prism.angle)/(order * self.grating.rho / 1e-3)
        f = lambda l: l - k * (self.prism.material.index(l) - 1)

        lbda0 = SO.newton(f, 1e-6)  # Look around 1 µm

        return lbda0


class Spectrograph(object):

    """
    A :class:`Collimator`, a :class:`Grism` and a :class:`Camera`.

    .. autosummary::

       dispersion
       forward
       backward
       test
       simulate
    """

    def __init__(self, config, grism_on=True, add_telescope=False):
        """
        Initialize spectrograph from optical configuration.

        :param OptConfig config: optical configuration
        :param bool grism_on: dispersor presence
        :param bool add_telescope: add input telescope
        """

        self.config = config

        self.telescope = None
        if add_telescope:                 # Add a telescope
            self.telescope = Telescope(self.config)
        self.collimator = Collimator(self.config)
        self.grism = Grism(self.config)
        self.camera = Camera(self.config)

        self.grism_on = grism_on

    def set_grism(self, flag):

        self.grism_on = flag

    @property
    def gamma(self):
        r"""
        Spectrograph magnification :math:`f_{\mathrm{cam}}/f_{\mathrm{coll}}`.
        """

        return self.camera.flength / self.collimator.flength

    def __str__(self):

        s = [" Spectrograph ".center(60, '-')]
        if self.telescope:
            s.append(self.telescope.__str__())
        s.append(self.collimator.__str__())
        if self.grism_on:
            s.append(self.grism.__str__())
        else:
            s.append("Grism:      ***REMOVED***")
        s.append(self.camera.__str__())
        s.append("Spectrograph magnification: {0.gamma:.3f}".format(self))
        wref = self.config.wref
        s.append("Central dispersion: {:.2f} AA/px at {:.2f} µm".format(
            self.dispersion(wref) / 1e-10 * self.config['detector_pxsize'],
            wref / 1e-6))

        return '\n'.join(s)

    def dispersion(self, wavelength, order=1, eps=1e-6):
        r"""
        Spectral dispersion (approximate) [m/m].

        This is given by :math:`D(\lambda) = (\mathrm{d}y /
        \mathrm{d}\lambda)^{-1}` with

        .. math:: y = f_{\mathrm{cam}}\tan\beta

        and

        .. math:: \sin\beta = m \rho \lambda - n(\lambda) \sin(A).

        :param float wavelength: wavelength [m]
        :param int order: dispersion order
        :return: spectral dispersion [m/m]
        """

        from scipy.misc import derivative

        def yoverf(l):

            sinbeta = (order * self.grism.grating.rho / 1e-3 * l -
                       self.grism.prism.material.index(l) *
                       N.sin(self.grism.prism.angle))

            return N.tan(N.arcsin(sinbeta))

        dydl = (derivative(yoverf, wavelength, dx=wavelength*eps) *
                self.camera.flength)

        return 1/dydl

    def forward(self, source, order=1):
        """
        Forward light propagation from a focal-plane point source.

        :param source: input :class:`PointSource`
        :param int order: dispersion order
        :return: (complex) 2D-positions in detector plane
        :rtype: :class:`numpy.ndarray`
        """

        assert isinstance(source, PointSource), \
            "source should be a PointSource"

        wavelengths = source.spectrum.wavelengths

        if self.telescope:
            # Telescope
            positions = self.telescope.forward(source.coords, wavelengths)
        else:
            positions = source.coords
        # Collimator
        directions = self.collimator.forward(
            positions, wavelengths, self.gamma)
        if self.grism_on:
            # Grism
            directions = self.grism.forward(
                directions, wavelengths, order=order)
        # Camera
        positions = self.camera.forward(directions, wavelengths)

        return positions

    def backward(self, position, wavelength, order=1):
        """
        Backward light propagation from a detector-plane 2D-position
        and wavelength.

        :param complex position: 2D-position in the detector plane [m]
        :param float wavelength: wavelength [m]
        :param int order: dispersion order
        :return: 2D-position in the focal plane [m]
        :rtype: complex
        """

        # Camera
        direction = self.camera.backward(position, wavelength)
        if self.grism_on:
            # Grism
            direction = self.grism.backward(direction, wavelength, order=order)
        # Collimator
        position = self.collimator.backward(direction, wavelength, self.gamma)
        if self.telescope:
            # Telescope
            coords = self.telescope.backward(position, wavelength)
        else:
            coords = position

        return coords

    def test(self, simu,
             coords=(1e-3 + 2e-3j), order=1, verbose=False):
        """
        Test forward and backward propagation in spectrograph.

        :param SimConfig simu: simulation configuration
        :param complex position: tested 2D-position in the focal plane [m]
        :param int order: tested dispersion order
        :param bool verbose: verbose-mode
        """

        # Test source
        if self.telescope:
            input = Direction2D(coords)
        else:
            input = Position2D(coords)
        source = PointSource(input,
                             Spectrum.default(simu.get_waves(self.config)))
        wavelengths = source.spectrum.wavelengths

        if verbose:
            print(" SPECTROGRAPH TEST ".center(60, '='))
            print("Input source:", source)
            print("Wavelengths [µm]:", wavelengths / 1e-6)

        # Forward step-by-step
        if self.telescope:
            # Telescope
            fpositions = self.telescope.forward(source.coords, wavelengths)
            if verbose:
                print("Positions (tel, forward) [×1e6]:", fpositions*1e6)
        else:
            fpositions = source.coords
        # Collimator
        fdirections = self.collimator.forward(fpositions,
                                              wavelengths, self.gamma)
        if verbose:
            print("Directions (coll, forward) [×1e6]:", fdirections*1e6)
        if self.grism_on:
            # Grism
            fdirections = self.grism.forward(fdirections,
                                             wavelengths, order=order)
            if verbose:
                print("Directions (grism, forward) [×1e6]:", fdirections*1e6)
        # Camera
        dpositions = self.camera.forward(fdirections, wavelengths)
        if verbose:
            print("Positions (detector) [mm]:", dpositions / 1e-3)

        # Loop over positions in detector plane
        for lbda, dpos in zip(wavelengths, dpositions):
            # Backward step-by-step
            if verbose:
                print("Test position (detector) [mm]:", dpos / 1e-3,
                      "Wavelength [µm]:", lbda / 1e-6)
            # Camera
            bdirection = self.camera.backward(dpos, lbda)
            if verbose:
                print("Direction (camera, backward) [×1e6]:", bdirection*1e6)
            if self.grism_on:
                # Grism
                bdirection = self.grism.backward(bdirection, lbda, order=order)
                if verbose:
                    print("Direction (grism, backward) [×1e6]:",
                          bdirection*1e6)
            # Collimator
            fposition = self.collimator.backward(bdirection,
                                                 lbda, self.gamma)
            if self.telescope:
                # Telescope
                tdirection = self.telescope.backward(fposition, lbda)
                if verbose:
                    print("Position (coll, backward) [×1e6]:", fposition*1e6)
                    print("Direction (tel, backward) [×1e6]:", tdirection*1e6)
                    print("Input direction (reminder) [×1e6]:",
                          source.coords*1e6)

                assert N.isclose(source.coords, tdirection), \
                    "Backward modeling does not match"
            else:
                if verbose:
                    print("Focal-plane position (backward) [mm]:",
                          fposition / 1e-3)
                    print("Input position (reminder) [mm]:",
                          source.coords / 1e-3)

                assert N.isclose(source.coords, fposition), \
                    "Backward modeling does not match"

    def simulate(self, simcfg):
        """
        Simulate detector spectra.

        :param dict simcfg: input simulation
        :return: simulated spectra
        :rtype: :class:`DetectorPositions`
        """

        # Input coordinates
        coords = simcfg.get_coords()           # 1D complex array
        # Rotation in the input plane
        angle = simcfg.get('input_angle', 0)   # [rad]
        if angle:
            coords *= N.exp(1j * angle)

        # Input source (coordinates will be updated later on)
        waves = simcfg.get_waves(self.config)
        source = PointSource(Position2D(0), waves=waves)
        wavelengths = source.spectrum.wavelengths

        # Simulation parameters
        orders = simcfg.get('orders', [1])
        det_angle = self.config.get('detector_angle', 0)
        det_dxdy = self.config.get('detector_dxdy', 0)

        # Detector positions
        detector = DetectorPositions(
            wavelengths, spectrograph=self,
            name=self.config.name)

        # Simulate forward propagation for all focal-plane positions
        for xy in coords:
            source.coords = xy    # Update source position
            for order in orders:  # Loop over dispersion orders
                dpos = self.forward(source, order)  # Detector plane position
                if det_dxdy:      # Offset in the detector plane
                    dpos += det_dxdy
                if det_angle:     # Rotation in the detector plane
                    dpos *= N.exp(1j * det_angle)
                detector.add_spectrum(source.coords, dpos, order=order)

        return detector

# Utility functions =======================================


def str_position(position):
    """
    Pretty-printer of a complex position.

    .. Warning:: work on a single (complex) position.
    """

    z = position / 1e-3         # [mm]
    return "{:+.1f} × {:+.1f} mm".format(z.real, z.imag)


def str_direction(direction):
    """
    Pretty-printer of a complex direction.

    .. Warning:: work on a single (complex) direction.
    """

    # tantheta, phi = CameraOrCollimator.rect2pol(direction)
    # return "{:+.2f} x {:+.2f} arcmin".format(
    #     N.arctan(tantheta)*RAD2MIN, phi*RAD2MIN)

    z = direction * RAD2MIN     # [arcmin]
    return "{:+.1f} × {:+.1f} arcmin".format(z.real, z.imag)

# Simulations ==============================


def plot_SNIFS_R(optcfg=OptConfig(SNIFS_R),
                 simcfg=SimConfig(SNIFS_SIMU),
                 test=True):

    # Optical configuration
    print(optcfg)

    # Spectrograph
    spectro = Spectrograph(optcfg, grism_on=optcfg.get('grism_on', True))
    print(spectro)

    # Simulation configuration
    print(simcfg)

    if test:
        try:
            spectro.test(simcfg, verbose=False)
        except AssertionError as err:
            warnings.warn(str(err))
        else:
            print("Spectrograph test: OK")

    detector = spectro.simulate(simcfg)
    ax = detector.plot(orders=(-1, 0, 1, 2), blaze=True)
    ax.set_aspect('auto')
    ax.axis(N.array([-2000, 2000, -4000, 4000]) *
            optcfg['detector_pxsize'] / 1e-3)  # [mm]

    return ax


# Main ====================================================

if __name__ == '__main__':

    ax = plot_SNIFS_R(test=True)

    P.show()
