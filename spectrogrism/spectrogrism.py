# -*- coding: utf-8 -*-
# Time-stamp: <2016-06-21 18:48:18 ycopin>

"""
spectrogrism
------------

Generic utilities for modeling grism-based spectrograph.

.. autosummary::

   Configuration
   OptConfig
   SimConfig
   Spectrum
   PointSource
   DetectorPositions
   Material
   Lens
   CameraOrCollimator
   Collimator
   Camera
   Telescope
   Prism
   Grating
   Grism
   Detector
   Spectrograph

.. inheritance-diagram:: spectrogrism.spectrogrism
"""

from __future__ import division, print_function

import warnings
from collections import OrderedDict
from functools import partial
import cmath as C

import numpy as N
import pandas as PD

try:
    from . import distortion as D
except ValueError:              # Attempted relative import in non-package
    import distortion as D

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

# Print options
LINEWIDTH = 70
N.set_printoptions(linewidth=LINEWIDTH, threshold=10)
PD.set_option("display.float.format",
              # PD.core.format.EngFormatter(accuracy=2, use_eng_prefix=True)
              lambda x: '{:g}'.format(x))

# Constants ===============================================

RAD2DEG = 57.29577951308232     #: Convert from radians to degrees
RAD2MIN = RAD2DEG * 60          #: Convert from radians to arc minutes
RAD2SEC = RAD2MIN * 60          #: Convert from radians to arc seconds

# Technical Classes ==========================================


class Configuration(OrderedDict):

    """
    A :class:`OrderedDict`-derived configuration.

    .. autosummary::

       override
       save
       load
    """

    conftype = 'Configuration'            #: Configuration type

    def __init__(self, adict):
        """Initialize from dictionary `adict`."""

        OrderedDict.__init__(self, adict)
        self.name = self.get('name', 'default')

    def __str__(self):

        s = [" {} {!r} "
             .format(self.conftype, self.name).center(LINEWIDTH, '-')]
        s += [ '  {:20}: {}'.format(str(key), self[key])
               for key in self.keys() ]

        return '\n'.join(s)

    def override(self, adict):
        """Override configuration from dictionary `adict`."""

        # warnings.warn(
        #     "Overriding configuration {!r} with test values {}".format(
        #         self.name, adict))

        self.name = adict.pop('name',
                              self.name + ' (overrided)'
                              if not self.name.endswith(' (overrided)')
                              else self.name)
        self.update(adict)

    def save(self, yamlname):
        """Save configuration to YAML file `yamlname`."""

        import yaml

        with open(yamlname, 'w') as yamlfile:
            yaml.dump(self, yamlfile)

        print("Configuration {!r} saved in {!r}".format(self.name, yamlname))

    @classmethod
    def load(cls, yamlname):
        """Load configuration from YAML file `yamlname`."""

        import yaml

        with open(yamlname, 'r') as yamlfile:
            self = yaml.load(yamlfile)

        print("Configuration {!r} loaded from {!r}".format(self.name, yamlname))

        return self

    def _repr_html_(self):
        """Pretty-printing in ipython notebooks."""

        html = ["<table>"]
        html.append("<caption>{0} {1}</caption>"
                    .format(self.conftype, self.name))
        for key in self.keys():
            html.append("<tr><td><pre>{0}</pre></td><td>{1}</td></tr>"
                        .format(key, self[key]))
        html.append("</table>")
        return ''.join(html)

    def read_coeffs(self, name, start=0):
        """Read keys 'name{i}' and return list `[name0, name1, ...]`."""

        try:
            # List all keys starting with 'name', and get maximum index
            imax = max( int(key[len(name):])
                        for key in self if key.startswith(name) )
        except ValueError:      # No such key: return empty list
            # print("No '{}' parameters".format(name))
            return []
        else:                   # Construct list from start to imax
            print("Reading '{}' parameters for i={}...{}"
                  .format(name + '#i', start, imax))
            return [ self.get(name + str(i), 0)
                     for i in range(start, imax + 1) ]


class OptConfig(Configuration):

    """
    Optical configuration.
    """

    conftype = "Optical configuration"

    @property
    def wref(self):

        return self.get('wave_ref', 0.5 * sum(self['wave_range']))


class SimConfig(Configuration):

    """
    Simulation configuration.

    .. autosummary::

       get_wavelengths
       get_coordinates
    """

    conftype = "Simulation configuration"

    def get_wavelengths(self, optcfg):
        """Simulated wavelengths."""

        if 'WAVES' in self:
            waves = N.atleast_1d(self['WAVES'])
        else:
            npx = self.get('wave_npx', 1)
            if npx > 1:
                wmin, wmax = self.get('wave_range', optcfg['wave_range'])
                waves = N.linspace(wmin, wmax, npx)
            elif npx == 1:
                waves = optcfg.wref
            else:
                raise ValueError("Invalid number of pixels: {}".format(npx))

        return waves

    def get_coordinates(self):
        """Simulated input complex coordinates `[ x + 1j*y ]`."""

        incoords = N.atleast_1d(self.get('input_coords', 0))
        if N.ndim(incoords) == 1:
            # [x]: generate square sampling x × x
            x, y = N.meshgrid(incoords, incoords)
            coords = (x + 1j * y).ravel()
        elif N.ndim(incoords) == 2 and N.shape(incoords)[1] == 2:
            # [[x, y]]: arbitrary sampling
            coords = incoords[:, 0] + 1j * incoords[:, 1]
        else:
            raise NotImplementedError("Unsupported input coordinates.")

        # Apply rotation if any
        angle = self.get('input_angle', 0)   # [rad]
        if angle:
            coords *= C.exp(1j * angle)

        return coords


class Spectrum(PD.Series):

    """
    A wavelength-indexed :class:`pandas.Series`.
    """

    def __init__(self, wavelengths, fluxes, name='spectrum'):
        """
        Initialize from wavelength and flux arrays.

        :param list wavelengths: strictly increasing input wavelengths [m]
        :param list fluxes: input fluxes [arbitrary units]
        :param str name: optional spectrum name (e.g. "Grism transmission")
        """

        super(Spectrum, self).__init__(
            data=fluxes, index=wavelengths, name=name)

        assert self.index.is_monotonic_increasing, \
            "Wavelengths not strictly increasing."

    @property
    def wavelengths(self):
        """Wavelength array."""

        return self.index.values

    @property
    def fluxes(self):
        """Fluxes array."""

        return self.values

    def __str__(self):

        return "{}{:d} wavelengths in [{:.2f}, {:.2f}] µm".format(
            self.name + ': ' if self.name else '',
            len(self), self.index[0] / 1e-6, self.index[-1] / 1e-6)

    @classmethod
    def default(cls, wavelengths=[1e-6], name=''):
        """
        A default constant-flux spectrum.

        :param list wavelengths: wavelength vector [m]
        :param str name: optional name
        :return: constant-flux spectrum
        :rtype: :class:`Spectrum`
        """

        wavelengths = N.atleast_1d(wavelengths)
        fluxes = N.ones_like(wavelengths)

        return cls(wavelengths, fluxes, name=name)


class PointSource(object):

    """
    A :class:`Spectrum` associated to a complex 2D-position or direction.
    """

    def __init__(self, coords, spectrum=None, **kwargs):
        """
        Initialize from position/direction and spectrum.

        :param complex coords: single source (complex) coordinates
        :param Spectrum spectrum: source spectrum (default to standard spectrum)
        :param kwargs: propagated to :func:`Spectrum.default` constructor
        """

        self.coords = coords            #: 2D-position/direction

        if spectrum is None:
            spectrum = Spectrum.default(**kwargs)
        else:
            assert isinstance(spectrum, Spectrum), \
                "spectrum should be a Spectrum, not '{}'.".format(type(spectrum))
        self.spectrum = spectrum        #: :class:`Spectrum`

    def __str__(self):

        zmm = self.coords / 1e-3

        return "{0} at ({1.real:+.2f}, {1.imag:+.2f}) mm".format(
            self.spectrum, zmm)


class DetectorPositions(PD.Panel):

    """
    A container for complex 2D-positions on the detector.

    A :class:`pandas.Panel`-derived container for (complex) positions in the
    detector plane:

    * items are observational modes (dispersion orders or photometric bands)
    * major axis is wavelength (shared among all modes)
    * minor axis is input complex coordinates (shared among all modes)

    For each mode, the corresponding :class:`pandas.DataFrame` is therefore
    organized the following way::

                zin_1     zin_2  ...     zin_N
      lbda1  zout_1_1  zout_2_1  ...  zout_N_1
      lbda2  zout_1_2  zout_2_2  ...  zout_N_2
      ...
      lbdaM  zout_1_M  zout_2_M  ...  zout_N_M

    * `self.modes` returns the available observing modes;
    * `df = self.panel[mode]` is the dataframe corresponding to a given
      observing `mode`;
    * `self.wavelengths` returns wavelength array `[lbdas]`;
    * `self.coordinates` returns (complex) input coordinate array `[zins]`;
    * `df[zin]` returns a wavelength-indexed :class:`pandas.Series` of complex
      detector positions.

    .. WARNING:: Indexing by float is not precisely a good idea... Float
       indices (wavelengths and input coordinates) are therefore rounded first
       with a sufficient precision (e.g. nm for wavelengths).

    .. TODO:: do not use complex coordinates as column name (this is not
       natively supported in Pandas, nor a good idea). Store input coordinates
       in an associated :class:`pandas.Series`, and use series index as column
       names.

    .. autosummary::

       add_mode
       plot
       check_alignment
       compute_offset
       compute_rms
    """

    digits = 12
    markers = {-1: '.', 0: 'D', 1: 'o', 2: 's',
               'J': 'D', 'H': 'o', 'K': 's'}

    def __init__(self, wavelengths, coordinates,
                 spectrograph=None, name='default'):
        """
        Initialize container from spectrograph and wavelength array.

        :param list wavelengths: input wavelengths [m]
        :param list coordinates: input complex coordinates
        :param Spectrograph spectrograph: associated spectrograph (if any)
        :param str name: informative label
        """

        super(DetectorPositions, self).__init__(
            major_axis=N.around(wavelengths, self.digits),
            minor_axis=N.around(coordinates, self.digits))

        if spectrograph is not None:
            assert isinstance(spectrograph, Spectrograph), \
                "spectrograph should be a Spectrograph."
        self.spectrograph = spectrograph        #: Associated spectrograph

        self.name = name                        #: Name

    @property
    def modes(self):
        """Observational mode list."""

        return self.items.tolist()

    @property
    def coordinates(self):
        """Input source (complex) coordinate array."""

        return self.minor_axis.values.astype(complex)

    @property
    def wavelengths(self):
        """Wavelength array."""

        return self.major_axis.values

    def add_mode(self, mode, dataframe):
        """Add *dataframe* as observational *mode* to current panel."""

        # Coherence tests
        assert N.allclose(dataframe.index, self.major_axis), \
            "Dataframe wavelengths are incompatible with current panel."
        assert N.allclose(dataframe.columns.values.astype(complex),
                          self.minor_axis.values.astype(complex)), \
            "Dataframe coordinates are incompatible with current panel."

        # If coordinates are coherent (up to digits rounding), use
        # previous coordinates to avoid any comparison error later on
        # (complex are not treated natively in Pandas, and comparison
        # is performed at object level)
        dataframe.columns = self.minor_axis.values.astype(complex)

        self[mode] = dataframe

    def plot(self, ax=None, coords=None, modes=None, blaze=False,
             subsampling=1, **kwargs):
        """
        Plot spectra on detector plane.

        :param matplotlib.pyplot.Axes ax: pre-existing axes instance if any
        :param list coords: selection of input coordinates to be plotted
        :param list modes: selection of observing modes to be plotted
        :param bool blaze: encode the blaze function in the marker size
        :param int subsampling: sub-sample coordinates and wavelengths
        :param kwargs: options propagated to
            :func:`matplotlib.pyplot.Axes.scatter`
        :return: :class:`matplotlib.pyplot.Axes`
        """

        import matplotlib.pyplot as P

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

        # Input observational modes
        if modes is None:             # Plot all observational modes
            modes = self.modes

        # Input coordinates
        if coords is None:            # Plot all spectra
            xys = self.coordinates
        else:
            xys = coords              # Plot specific spectra

        if subsampling > 1:
            xys = xys[::subsampling]

        # Input wavelengths
        if subsampling > 1:
            lbda = self.wavelengths[::subsampling]
        else:
            lbda = self.wavelengths

        bztrans = N.ones_like(lbda)     # Default blaze transmission
        # Color has to have same shape as coordinates, (nlbda, ncoords)
        c = N.broadcast_to(lbda[:, N.newaxis], (len(lbda), len(xys)))

        # Segmented colormap (blue to red)
        cmap = P.matplotlib.colors.LinearSegmentedColormap(
            'dummy', P.get_cmap('Spectral_r')._segmentdata,
            len(self.wavelengths))

        for mode in modes:              # Loop over observational modes
            try:
                df = self[mode]
            except KeyError:
                warnings.warn("{} not in '{}', skipped"
                              .format(str_mode(mode), self.name))
                continue

            kwcopy = kwargs.copy()
            marker = kwcopy.pop('marker', self.markers.get(mode, 'o'))
            kwcopy['label'] += " {}{}".format(
                "#" if is_spectred(mode) else '', mode)

            if blaze and self.spectrograph:
                bztrans = self.spectrograph.grism.blaze_function(lbda, mode)
                s = kwcopy.pop('s', N.maximum(60 * N.sqrt(bztrans), 10))
                # Size has to have same shape as coordinates, (nlbda, ncoords)
                s = N.broadcast_to(s[:, N.newaxis], (len(lbda), len(xys)))
            else:
                s = kwcopy.pop('s', 40)

            # This might raise a KeyError if one of the coordinates is not in df
            positions = df[xys].values / 1e-3  # Complex positions [mm]
            if subsampling > 1:
                positions = positions[::subsampling]

            sc = ax.scatter(positions.real, positions.imag,
                            c=c / 1e-6,   # Wavelength [µm]
                            cmap=cmap, s=s, marker=marker, **kwcopy)

            # kwargs.pop('label', None)      # Label only for one mode

        if fig:                 # Newly created axes
            fig.colorbar(sc, label=u"Wavelength [µm]")

        return ax

    def check_alignment(self, other):
        """
        Check compatibility in wavelengths and positions with other instance.

        :param DetectorPositions other: other instance to be compared to
        :raise IndexError: incompatible instance
        """

        if not N.allclose(other.wavelengths, self.wavelengths):
            raise IndexError(
                "{!r} and {!r} have incompatible wavelengths."
                .format(self.name, other.name))

        if not N.allclose(other.coordinates, self.coordinates):
            raise IndexError(
                "{!r} and {!r} have incompatible coordinates."
                .format(self.name, other.name))

    def compute_offset(self, other, mode=1):
        """
        Compute (complex) position offsets to other instance.

        :param DetectorPositions other: other instance to be compared to
        :param mode: observing mode (dispersion order or photometric band)
        :return: (complex) position offsets [m]
        :rtype: :class:`pandas.DataFrame`
        :raise KeyError: requested mode cannot be found

        .. Warning:: `self` and `other` are supposed to be compatible
           (see :func:`test_compatibility`).
        """

        # Dataframe of (complex) position offsets for requested mode
        return other[mode] - self[mode]

    def compute_rms(self, other, mode=1):
        """
        Compute total RMS distance to other instance.

        :param DetectorPositions other: other instance to be tested
        :param mode: observing mode (dispersion order or photometric band)
        :return: RMS offset [m]
        :rtype: float
        :raise KeyError: requested mode cannot be found

        .. Warning:: `self` and `other` are supposed to be compatible
           (see :func:`test_compatibility`).
        """

        offsets = self.compute_offset(other, mode=mode).abs()

        return ((offsets.values ** 2).mean(axis=None) ** 0.5)


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
    materials = dict(
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
        # Null material (refractive index = 1)
        null=[0, 0, 0, 0, 0, 0],
    )

    def __init__(self, name):
        """
        Initialize material from its name.

        :param str name: material name (should be in :attr:`Material.materials`)
        :raise KeyError: unknown material name
        """

        try:
            #: Sellmeier coefficients `[B1, B2, B3, C1, C2, C3]`
            self.coeffs = self.materials[name]
        except KeyError:
            raise KeyError("Unknown material {}".format(name))
        self.name = name        #: Name of the material

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

        if self.name == 'null':
            return N.ones_like(wavelengths)

        lmu2 = (wavelengths / 1e-6) ** 2        # (wavelength [µm])**2
        n2m1 = N.sum([ b / (1 - c / lmu2)       # n**2 - 1
                       for b, c in zip(self.coeffs[:3], self.coeffs[3:]) ],
                     axis=0)

        return N.sqrt(n2m1 + 1)


# Optical element classes =================================

class Lens(object):

    """
    Convert a 2D-position into a 2D-position.

    .. autosummary:

       forward
       backward
    """

    def __init__(self, gamma, gdist=None, cdist=None):
        """
        Initialize the element from its optical parameters.

        :param float gamma: transverse 'grandissement'
        :param D.GeometricDistortion gdist: geometric distortion
        :param D.ChromaticDistortion cdist: chromatic distortion (lateral color)
        """

        self.gamma = float(gamma)              #: Transverse 'grandissement'

        if gdist is not None:
            assert isinstance(gdist, D.GeometricDistortion), \
                "gdist should be a GeometricDistortion."
        else:
            gdist = D.GeometricDistortion()        # Null geometric distortion

        if cdist is not None:
            assert isinstance(cdist, D.ChromaticDistortion), \
                "cdist should be a ChromaticDistortion."
        else:
            cdist = D.ChromaticDistortion()        # Null lateral color

        self.gdist = gdist              #: Geometric distortion
        self.cdist = cdist              #: Chromatic distortion (lateral color)

    def forward(self, positions):

        raise NotImplementedError()

    def backward(self, positions):

        raise NotImplementedError()


class CameraOrCollimator(object):

    """
    An optical element converting to and fro directions and positions.
    """

    def __init__(self, flength, gdist=None, cdist=None):
        """
        Initialize the element from its optical parameters.

        :param float flength: focal length [m]
        :param D.GeometricDistortion gdist: geometric distortion
        :param D.ChromaticDistortion cdist: chromatic distortion (lateral color)
        """

        self.flength = float(flength)              #: Focal length [m]

        if gdist is not None:
            assert isinstance(gdist, D.GeometricDistortion), \
                "gdist should be a GeometricDistortion."
        else:
            gdist = D.GeometricDistortion()        # Null geometric distortion

        if cdist is not None:
            assert isinstance(cdist, D.ChromaticDistortion), \
                "cdist should be a ChromaticDistortion."
        else:
            cdist = D.ChromaticDistortion()        # Null lateral color

        self.gdist = gdist              #: Geometric distortion
        self.cdist = cdist              #: Chromatic distortion (lateral color)

    def __str__(self):

        s = "{}: f={:.3f} m".format(self.__class__.__name__, self.flength)
        s += '\n  {}'.format(self.gdist)
        s += '\n  {}'.format(self.cdist)

        return s


class Collimator(CameraOrCollimator):

    """
    Convert a 2D-position (in the focal plane) into a 2D-direction.

    .. autosummary::

       forward
       backward
    """

    def __init__(self, config):
        """
        Initialize from optical configuration.

        :param OptConfig config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            flength = config['collimator_flength']
            center = (config.get('collimator_gdist_x0', 0) +
                      config.get('collimator_gdist_y0', 0) * 1j )
            Kcoeffs = config.read_coeffs('collimator_gdist_K', start=1)
            Pcoeffs = config.read_coeffs('collimator_gdist_P', start=1)
            lcoeffs = config.read_coeffs('collimator_cdist_C', start=1)
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}"
                .format(err.args[0]))
        else:
            gdist = D.GeometricDistortion(
                center, Kcoeffs=Kcoeffs, Pcoeffs=Pcoeffs)
            cdist = D.ChromaticDistortion(config.wref, lcoeffs)

        super(Collimator, self).__init__(flength, gdist, cdist)

    def forward(self, positions, wavelengths, gamma):
        r"""
        Forward light propagation through the collimator.

        .. math:

           z' = D(-z / f) - z/f \times L(\lambda) / \gamma

        where:

        * :math:`z = x + iy` is the complex position
        * *D* is the :class:`D.GeometricDistortion` geometric distortion
        * *L* is the :class:`D.ChromaticDistortion` lateral color
        * :math:`\gamma` is the spectrograph magnification (fcam/fcoll)

        The geometric distortion is therefore applied on the (flipped)
        positions.
        """

        zu = - positions / self.flength  # Undistorted (flipped) directions
        zd = self.gdist.forward(zu)      # Distorted directions
        if self.cdist:                   # Add lateral colors
            zd += zu * self.cdist.amplitude(wavelengths) / gamma

        return zd

    def backward(self, directions, wavelength, gamma):
        """
        Backward light propagation through the collimator.

        See :func:`Collimator.forward` for details.
        """

        if self.cdist:                   # Lateral colors
            lcol = self.cdist.amplitude(wavelength) / gamma  # scalar
        else:
            lcol = 0.

        return - self.flength * self.gdist.backward(directions, lcol=lcol)


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
        Initialize from optical configuration.

        :param OptConfig config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            flength = config['camera_flength']
            center = (config.get('camera_gdist_x0', 0) +
                      config.get('camera_gdist_y0', 0) * 1j )
            Kcoeffs = config.read_coeffs('camera_gdist_K', start=1)
            Pcoeffs = config.read_coeffs('camera_gdist_P', start=1)
            lcoeffs = config.read_coeffs('camera_cdist_C', start=1)
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}"
                .format(err.args[0]))
        else:
            gdist = D.GeometricDistortion(
                center, Kcoeffs=Kcoeffs, Pcoeffs=Pcoeffs)
            cdist = D.ChromaticDistortion(config.wref, lcoeffs)

        super(Camera, self).__init__(flength, gdist, cdist)

    def forward(self, directions, wavelengths):
        r"""
        Forward light propagation through the camera.

        .. math:

           z' = -[D(z) + z L(\lambda)] \times f

        where:

        * :math:`z = x + iy` is the complex position
        * *D* is the :class:`D.GeometricDistortion` geometric distortion
        * *L* is the :class:`D.ChromaticDistortion` lateral color

        The geometric distortion is therefore applied on the (input) directions.
        """

        zd = self.gdist.forward(directions)  # Distorted (unflipped) directions
        if self.cdist:
            zd += directions * self.cdist.amplitude(wavelengths)

        return - zd * self.flength           # Distorted (flipped) positions

    def backward(self, positions, wavelength):
        """
        Backward light propagation through the camera.

        See :func:`Camera.forward` for details.
        """

        if self.cdist:                   # Lateral colors
            lcol = self.cdist.amplitude(wavelength)  # scalar
        else:
            lcol = 0.

        return self.gdist.backward(- positions / self.flength, lcol=lcol)


class Telescope(Camera):

    """
    All-reflective telescope (no chromatic distortions).

    Convert a 2D-direction in the sky into a 2D-position in the focal plane.
    """

    def __init__(self, config):
        """
        Initialize from optical configuration.

        :param OptConfig config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            flength = config['telescope_flength']
            center = (config.get('telescope_gdist_x0', 0) +
                      config.get('telescope_gdist_y0', 0) * 1j )
            Kcoeffs = config.read_coeffs('telescope_gdist_K', start=1)
            Pcoeffs = config.read_coeffs('telescope_gdist_P', start=1)
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}"
                .format(err.args[0]))
        else:
            gdist = D.GeometricDistortion(
                center, Kcoeffs=Kcoeffs, Pcoeffs=Pcoeffs)

        # Initialize from CameraOrCollimator parent class (without
        # chromatic aberration)
        super(Camera, self).__init__(flength, gdist, cdist=None)

        # Remove explicit dependency to wavelengths in Camera.forward
        # and backward
        self.forward = partial(
            super(Telescope, self).forward, wavelengths=None)
        self.backward = partial(
            super(Telescope, self).backward, wavelength=None)


class Prism(object):

    """
    A triangular transmissive prism.

    .. Note::

       - The entry surface is roughly perpendicular (up to the tilt
         angles) to the optical axis Oz.
       - The apex (prism angle) is aligned with the *x*-axis

    .. autosummary::

       rotation
       rotation_x
       rotation_y
       rotation_z
       refraction
    """

    def __init__(self, angle, material, tilts=[0, 0, 0]):
        """
        Initialize grism from its optical parameters.

        :param float angle: prism angle [rad]
        :param Material material: prism material
        :param 3-list tilts: prism tilts (x, y, z) [rad]
        """

        assert isinstance(material, Material), "material should be a Material."
        assert len(tilts) == 3, "tilts should be a length-3 list."

        self.angle = angle                #: Prism angle [rad]
        self.material = material          #: Prism material
        self.tilts = tilts                #: Prism tilts (x, y, z) [rad]

    def __str__(self):

        # Present tilt angles in arcmin
        tilts = ','.join( "{:+.0f}'".format(t * RAD2MIN) for t in self.tilts )

        return "Prism [{}]: A={:.2f}°, tilts={}".format(
            self.material.name, self.angle * RAD2DEG, tilts)

    @property
    def tiltx(self):
        """
        Expose prism x-tilt [rad], rotation around the prism apex/grating
        grooves.
        """

        return self.tilts[0]

    @tiltx.setter
    def tiltx(self, xtilt):

        self.tilts[0] = xtilt

    @property
    def tilty(self):
        """
        Expose prism y-tilt [rad].
        """

        return self.tilts[1]

    @tilty.setter
    def tilty(self, ytilt):

        self.tilts[1] = ytilt

    @property
    def tiltz(self):
        """
        Expose prism z-tilt [rad], rotation around the optical axis.
        """

        return self.tilts[2]

    @tiltz.setter
    def tiltz(self, ztilt):

        self.tilts[2] = ztilt

    @staticmethod
    def rotation(x, y, theta):
        """
        2D-rotation of position around origin with direct angle `theta` [rad].
        """

        # Rotation in the complex plane
        p = (N.array(x) + 1j * N.array(y)) * C.exp(1j * theta)

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

        x1, y1, _ = xyz
        x2 = x1 * n1 / n2
        y2 = y1 * n1 / n2
        z2 = N.sqrt(1 - (x2 ** 2 + y2 ** 2))

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
        Initialize grating from its optical parameters.

        :param float rho: grating groove density [lines/mm]
        :param Material material: grating material
        :param float blaze: grating blaze angle [rad]
        """

        assert isinstance(material, Material), "material should be a Material."

        self.rho = rho            #: Grating groove density [lines/mm]
        self.material = material  #: Grating material
        self.blaze = blaze        #: Grating blaze angle [rad]

    def __str__(self):

        return "Grating [{}]: rho={:.1f} g/mm, blaze={:.2f}°".format(
            self.material.name, self.rho, self.blaze * RAD2DEG)

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

        x, y, _ = xyz
        xp = x * n
        yp = y * n + order * wavelengths * self.rho / 1e-3
        zp = N.sqrt(1 - (xp ** 2 + yp ** 2))

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

        xp, yp, _ = xyz
        x = xp / n
        y = (yp - order * wavelength * self.rho / 1e-3) / n
        z = N.sqrt(1 - (x ** 2 + y ** 2))

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
        Initialize from optical configuration.

        :param OptConfig config: optical configuration
        :raise KeyError: missing configuration key
        """

        try:
            angle = config['grism_prism_angle']
            prism_material = config['grism_prism_material']
            tilts = [ config.get('grism_prism_tilt' + ax, 0) for ax in 'xyz' ]
            rho = config['grism_grating_rho']
            grating_material = config['grism_grating_material']
            blaze = config.get('grism_grating_blaze', 0)
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}"
                .format(err.args[0]))

        self.prism = Prism(angle, Material(prism_material), tilts=tilts)
        self.grating = Grating(rho, Material(grating_material), blaze)

    def __str__(self):

        return """Grism:
  {0.prism}
  {0.grating}
  1st-order null-deviation wavelength: {1:.2f} µm""".format(
            self, self.null_deviation(order=1) / 1e-6)

    def blaze_function(self, wavelengths, order=1):
        r"""
        Blaze function.

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
        :return: blaze function (i.e. transmission at input wavelengths)
        :rtype: :class:`numpy.ndarray`
        """

        np = self.prism.material.index(wavelengths)    # Prism index
        ng = self.grating.material.index(wavelengths)  # Grating index

        rholbda = self.grating.rho / 1e-3 * wavelengths  # g/m * m = unitless
        npsinA = np * N.sin(self.prism.angle)

        i = N.arcsin(npsinA / ng) - self.grating.blaze               # [rad]
        r = N.arcsin(npsinA - order * rholbda) - self.grating.blaze  # [rad]

        theta = (N.pi / rholbda * N.cos(self.grating.blaze) *
                 (ng * N.sin(i) - N.sin(r)))
        bf = (N.sin(theta) / theta) ** 2

        return bf

    @staticmethod
    def direction2xyz(direction):
        """
        Convert a 2D-direction into a 3D-direction (a unit vector).

        :param complex direction: 2D-direction
        :return: 3D-direction
        :type: 3-tuple
        """

        tantheta, phi = rect2pol(direction)
        tan2 = tantheta ** 2
        costheta = N.sqrt(1 / (1 + tan2))
        sintheta = N.sqrt(tan2 / (1 + tan2))

        return N.vstack((N.cos(phi) * sintheta,
                         N.sin(phi) * sintheta,
                         costheta)).squeeze()

    @staticmethod
    def xyz2direction(xyz):
        """
        Convert a 3D-direction (a unit vector) into a 2D-direction.

        :param 3-tuple xyz: 3D-direction
        :return: 2D-direction
        :rtype: complex
        """

        x, y, z = xyz
        tantheta = N.hypot(x, y) / z
        phi = N.arctan2(y, x)

        return pol2rect(tantheta, phi)

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
        Null-deviation wavelength (approximated) [m].

        This is the solution to:

        .. math:: m \rho \lambda = (n(\lambda) - 1) \sin(A)

        where:

        - *A*: grism angle [rad]
        - :math:`\rho`: groove density [line/mm]
        - :math:`n(\lambda)`: prism refractive index
        - *m*: dispersion order

        :param int order: dispersion order
        :return: null deviation wavelength [m]
        :raise RuntimeError: if not converging
        """

        import scipy.optimize as SO

        k = N.sin(self.prism.angle) / (order * self.grating.rho / 1e-3)

        def objf(l):

            return l - k * (self.prism.material.index(l) - 1)

        lbda0 = SO.newton(objf, 1e-6)  # Look around 1 µm

        return lbda0


class Detector(object):

    """
    A simple translated and rotated detector.

    .. autosummary::

       forward
       backward
    """

    def __init__(self, config):
        """
        Initialize from optical configuration.

        :param OptConfig config: optical configuration
        :raise KeyError: missing configuration key
        """

        self.dx = config.get('detector_dx', 0)        #: X-offset [m]
        self.dy = config.get('detector_dy', 0)        #: Y-offset [m]
        self.angle = config.get('detector_angle', 0)  #: Rotation [rad]
        self.pxsize = config['detector_pxsize']       #: Pixel size [m]

    @property
    def dxdy(self):
        """
        Expose complex offset `dx + 1j*dy` [m].
        """

        return self.dx + 1j * self.dy     # Faster than complex(dx, dy)

    @dxdy.setter
    def dxdy(self, dxdy):

        self.dx, self.dy = dxdy.real, dxdy.imag

    def __str__(self):

        return """Detector: pxsize={:.0f} µm
  Offset=({:+.3f}, {:+.3f}) mm, angle={:.1f} deg""".format(
            self.pxsize / 1e-6,
            self.dx / 1e-3, self.dy / 1e-3, self.angle * RAD2DEG)

    def forward(self, positions):
        """Forward propagation to detector."""

        return (positions + self.dxdy) * C.exp(1j * self.angle)

    def backward(self, positions):
        """Backward propagation from detector."""

        return positions / C.exp(1j * self.angle) - self.dxdy


class Spectrograph(object):

    """
    A collimated spectrograph.

    A :class:`Collimator`, a :class:`Grism` in a collimated beam, a
    :class:`Camera` and a :class:`Detector`, plus an optional
    :class:`Telescope`.

    .. autosummary::

       dispersion
       forward
       backward
       test
       predict_positions
    """

    def __init__(self, config, telescope=None):
        """
        Initialize spectrograph from optical configuration.

        :param OptConfig config: optical configuration
        :param Telescope telescope: input telescope if any
        """

        self.config = config                       #: Optical configuration

        self.telescope = telescope                 #: Telescope
        self.collimator = Collimator(self.config)  #: Collimator
        self.grism = Grism(self.config)            #: Grism
        self.camera = Camera(self.config)          #: Camera
        self.detector = Detector(self.config)      #: Detector

    @property
    def gamma(self):
        r"""
        Spectrograph magnification :math:`f_{\mathrm{cam}}/f_{\mathrm{coll}}`.
        """

        return self.camera.flength / self.collimator.flength

    def __str__(self):

        s = [" Spectrograph ".center(LINEWIDTH, '-')]
        if self.telescope:
            s.append(self.telescope.__str__())
        s.append(self.collimator.__str__())
        s.append(self.grism.__str__())
        s.append(self.camera.__str__())
        s.append(self.detector.__str__())
        s.append("Spectrograph magnification: {0.gamma:.3f}".format(self))
        wref = self.config.wref
        s.append("Central dispersion: {:.2f} AA/px at {:.2f} µm"
                 .format(self.dispersion(wref) / 1e-10 * self.detector.pxsize,
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

        dydl = self.camera.flength * derivative(yoverf, wavelength,
                                                dx=wavelength * eps)

        return 1 / dydl

    def forward(self, source, mode=1):
        """
        Forward light propagation from a focal-plane point source.

        :param PointSource source: input source
        :param mode: observing mode (dispersion order or photometric band)
        :return: (complex) 2D-positions in detector plane
        :rtype: :class:`numpy.ndarray`
        """

        assert isinstance(source, PointSource), \
            "source should be a PointSource."

        wavelengths = source.spectrum.wavelengths

        # Telescope if any
        positions = (self.telescope.forward(source.coords)
                     if self.telescope else source.coords)
        # Collimator
        directions = self.collimator.forward(
            positions, wavelengths, self.gamma)
        if is_spectred(mode):   # Spectroscopic mode: grism
            directions = self.grism.forward(
                directions, wavelengths, order=mode)
        # Camera
        positions = self.camera.forward(directions, wavelengths)
        # Detector
        positions = self.detector.forward(positions)

        return positions

    def backward(self, position, wavelength, mode=1):
        """
        Backward light propagation from a detector-plane 2D-position
        and wavelength.

        :param complex position: 2D-position in the detector plane [m]
        :param float wavelength: wavelength [m]
        :param mode: observing mode (dispersion order or photometric band)
        :return: 2D-position in the focal plane [m]
        :rtype: complex
        """

        # Detector
        position = self.detector.backward(position)
        # Camera
        direction = self.camera.backward(position, wavelength)
        if is_spectred(mode):   # Spectroscopic mode
            # Grism
            direction = self.grism.backward(direction, wavelength, order=mode)
        # Collimator
        position = self.collimator.backward(direction, wavelength, self.gamma)
        # Telescope if any
        coords = (self.telescope.backward(position)
                  if self.telescope else position)

        return coords

    def test(self, waves=None, coords=(1e-3 + 2e-3j), mode=1, verbose=False):
        """
        Test forward and backward propagation in spectrograph.

        :param list waves: wavelength vector to be tested
        :param complex position: input (complex) coordinates to be tested
        :param mode: observing mode (dispersion order or photometric band)
        :param bool verbose: verbose-mode
        :return: boolean result of the forward/backward test
        """

        # Test source
        if waves is None:       # Single test wavelength
            waves = N.array([self.config.wref])
        source = PointSource(coords, Spectrum.default(waves))

        if verbose:
            print(" SPECTROGRAPH TEST - MODE {} "
                  .format(mode).center(LINEWIDTH, '='))
            print("| Input source:", source)
            print("| Wavelengths [µm]:", waves / 1e-6)

        spectred = is_spectred(mode)  # Spectroscopic mode

        # Forward step-by-step ------------------------------

        # Telescope
        if self.telescope:
            fpositions = self.telescope.forward(source.coords)
            if verbose:
                print("> Positions (tel.forward) [×1e6]:", fpositions * 1e6)
        else:
            fpositions = source.coords

        # Collimator
        fdirections = self.collimator.forward(fpositions, waves, self.gamma)
        if verbose:
            print("> Directions (coll.forward) [×1e6]:", fdirections * 1e6)

        if spectred:            # Spectroscopic mode
            # Grism
            fdirections = self.grism.forward(fdirections, waves, order=mode)
            if verbose:
                print("> Directions (grism.forward) [×1e6]:", fdirections * 1e6)

        # Camera
        dpositions = self.camera.forward(fdirections, waves)
        if verbose:
            print("> Positions (camera.forward) [mm]:", dpositions / 1e-3)

        # Detector
        dpositions = self.detector.forward(dpositions)
        if verbose:
            print("> Positions (detector) [mm]:", dpositions / 1e-3)

        # Backward step-by-step ------------------------------

        output = []             # Round-trip positions, to be compared to input

        # Loop over positions in detector plane
        for lbda, dpos in zip(waves, N.atleast_1d(dpositions)):
            if verbose:
                print("| Test position (detector) [mm]:", dpos / 1e-3,
                      "Wavelength [µm]:", lbda / 1e-6)

            # Detector
            dpos = self.detector.backward(dpos)
            if verbose:
                print("< Direction (detector.backward) [mm]:", dpos / 1e-3)

            # Camera
            bdirection = self.camera.backward(dpos, lbda)
            if verbose:
                print("< Direction (camera.backward) [×1e6]:", bdirection * 1e6)

            if spectred:        # Spectroscopic mode
                # Grism
                bdirection = self.grism.backward(bdirection, lbda, order=mode)
                if verbose:
                    print("< Direction (grism.backward) [×1e6]:",
                          bdirection * 1e6)

            # Collimator
            fposition = self.collimator.backward(bdirection,
                                                 lbda, self.gamma)

            # Telescope
            if self.telescope:
                tdirection = self.telescope.backward(fposition)
                if verbose:
                    print("< Position (coll.backward) [mm]:",
                          fposition / 1e-3)
                    print("< Direction (tel.backward) [×1e6]:",
                          tdirection * 1e6,
                          "---OK---"
                          if N.isclose(tdirection, source.coords)
                          else '***ERROR***')

                output.append(tdirection)
            else:
                if verbose:
                    print("< Focal-plane position (backward) [mm]:",
                          fposition / 1e-3,
                          "---OK---"
                          if N.isclose(fposition, source.coords)
                          else '***ERROR***')

                output.append(fposition)

        return N.allclose(source.coords, output)

    def predict_positions(self, simcfg, **kwargs):
        """
        Simulate detector spectra from optical model.

        :param SimConfig simcfg: input simulation configuration
        :param kwargs: configuration options (e.g. predicted `modes`)
        :return: predicted positions [m]
        :rtype: :class:`DetectorPositions`
        """

        # Update simulation configuration on-the-fly
        if kwargs:
            # Make a copy, to leave input simcfg intact
            simcfg = SimConfig(simcfg)
            simcfg.update(kwargs)

        # Input coordinates
        coords = simcfg.get_coordinates()      # 1D complex array

        # Input source (coordinates will be updated later on)
        waves = simcfg.get_wavelengths(self.config)

        # Initiate detector positions
        detector = DetectorPositions(waves, coords,
                                     spectrograph=self,
                                     name=self.config.name)
        # Use rounded detector wavelengths and coordinates
        waves = detector.wavelengths
        coords = detector.coordinates

        # Input spectrum
        spec = Spectrum.default(waves)

        # Simulated observing modes
        modes = simcfg.get('modes', [1])

        # Simulate forward propagation for all input coordinates
        for mode in modes:      # Loop over observing modes
            positions = N.transpose([
                self.forward(PointSource(xy, spec), mode=mode)
                for xy in coords ])  # ([nlbda,] npos)
            if positions.ndim == 1:  # No chromatic dimension
                # Expand along wavelength dimension
                positions = N.resize(positions, detector.shape[1:])
            df = PD.DataFrame(positions, index=waves, columns=coords)
            detector.add_mode(mode, df)

        return detector

    def update(self, **kwargs):
        """
        Update both optical configuration and structure parameters.
        """

        for name, value in kwargs.iteritems():
            # Test parameter validity
            if name not in self.config:
                # raise KeyError("Unknown optical parameter '{}'".format(name))
                warnings.warn("Unknown optical parameter '{}'".format(name))
            # Update optical configuration
            self.config[name] = value
            # Update structure
            attrs = name.split('_')  # [ attributes ]
            obj = self
            for attr in attrs[:-1]:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    raise KeyError("Unknown attribute '{}' in {}"
                                   .format(attr, obj.__class__.__name__))
            attr = attrs[-1]
            try:
                getattr(obj, attr)
            except AttributeError:
                raise KeyError("Unknown final attribute '{}' in {}"
                               .format(attr, obj.__class__.__name__))
            else:
                setattr(obj, attr, value)

    def optimize(self, positions, simcfg, modes=None,
                 optparams=['telescope_flength',
                            'collimator_flength',
                            'camera_flength'], tol=1e-6):
        """
        :mod:`scipy.optimize` optical parameters to match target detector
        positions, according to simulation configuration.

        :param DetectorPositions positions: target positions
        :param SimConfig simcfg: simulation configuration
        :param list modes: adjusted observing modes (default: simulated modes)
        :param list optparams: optical parameters to be adjusted
        :param float tol: optimization tolerance
        :return: result from the optimization
        :rtype: :class:`scipy.optimize.OptimizeResult`
        :raise KeyError: unknown optical parameter
        """

        import scipy.optimize as SO

        print(" SPECTROGRAPH ADJUSTMENT (scipy.optimize) ".center(LINEWIDTH, '='))

        # Simulation parameters
        if modes is None:
            modes = simcfg.get('modes', [1])
        print("Adjusted modes:", modes)

        try:
            guessparams = [ self.config[name] for name in optparams ]
        except KeyError:
            raise KeyError("Unknown optical parameter '{}'".format(name))

        print(" Initial parameters ".center(LINEWIDTH, '-'))
        for name, value in zip(optparams, guessparams):
            print("  {:20}: {}".format(name, value))

        # Initial guess simulation
        mpositions = self.predict_positions(simcfg)
        # Test compatibility with objective detector positions only once
        mpositions.check_alignment(positions)

        rmss = []
        for mode in modes:
            rms = positions.compute_rms(mpositions, mode=mode)
            print("Mode {} RMS: {} mm = {} px"
                  .format(mode, rms / 1e-3, rms / self.detector.pxsize))
            rmss.append(rms)
        rms = (sum( rms ** 2 for rms in rmss ) / len(modes)) ** 0.5
        print("Total RMS: {} mm = {} px"
              .format(rms / 1e-3, rms / self.detector.pxsize))

        def objfun(params, positions):
            """Sum of squared mean offset position."""

            # Update optical configuration
            self.update(**dict(zip(optparams, params)))
            # Simulate
            mpositions = self.predict_positions(simcfg)
            dtot = sum(
                ((mpositions[mode] - positions[mode]).abs()**2).values.mean()
                for mode in modes )

            return dtot

        result = SO.minimize(objfun,
                             guessparams,
                             args=(positions,),
                             method='tnc',
                             options={'disp': True, 'xtol': tol})
        print("Minimization {}: {}".format(
            'succeeded' if result.success else 'failed', result.message))

        if result.success:
            print(" Adjusted parameters ".center(LINEWIDTH, '-'))
            for name, value in zip(optparams, N.atleast_1d(result.x)):
                print("  {:20}: {}".format(name, value))
            # Compute final RMS
            result.rms = (result.fun / len(modes)) ** 0.5  # [m]
            print("  RMS: {} mm = {} px"
                  .format(result.rms / 1e-3, result.rms / self.detector.pxsize))

        return result

    def adjust(self, positions, simcfg, modes=None,
               optparams=['telescope_flength',
                          'collimator_flength',
                          'camera_flength'], **options):
        """
        :pypi:`iminuit` adjustement of optical parameters to match target detector
        positions, according to simulation configuration.

        :param DetectorPositions positions: target positions
        :param SimConfig simcfg: simulation configuration
        :param list modes: adjusted observing modes (default: simulated modes)
        :param list optparams: optical parameters to be adjusted
        :param dict options: iminuit additional options
        :return: result from the optimization
        :rtype: :class:`iminuit.Minuit`
        :raise KeyError: unknown optical parameter
        """

        from iminuit import Minuit

        print(" SPECTROGRAPH ADJUSTMENT (Minuit) ".center(LINEWIDTH, '='))

        # Simulation parameters
        if modes is None:
            modes = simcfg.get('modes', [1])
        print("Adjusted modes:", modes)

        try:
            guessparams = [ self.config[name] for name in optparams ]
        except KeyError:
            raise KeyError("Unknown optical parameter '{}'".format(name))

        kwargs = {'forced_parameters': optparams,  # objfun signature
                  'errordef': 1}                   # Least-square

        print(" Initial parameters ".center(LINEWIDTH, '-'))
        for name, value in zip(optparams, guessparams):
            print("  {:20}: {}".format(name, value))
            kwargs.update(((name, value),
                           ('error_' + name, abs(value) if value else 1e-3)))

        # Initial guess simulation
        mpositions = self.predict_positions(simcfg)
        # Test compatibility with objective detector positions only once
        mpositions.check_alignment(positions)

        rmss = []
        for mode in modes:
            rms = positions.compute_rms(mpositions, mode=mode)
            print("Mode {} RMS: {} mm = {} px"
                  .format(mode, rms / 1e-3, rms / self.detector.pxsize))
            rmss.append(rms)
        rms = (sum( rms ** 2 for rms in rmss ) / len(modes)) ** 0.5
        print("Total RMS:  {} mm = {} px"
              .format(rms / 1e-3, rms / self.detector.pxsize))

        def objfun(*args):
            """Sum of squared mean offset position."""

            # Update optical configuration
            self.update(**dict(zip(optparams, args)))
            # Simulate
            mpositions = self.predict_positions(simcfg)
            dtot = sum(
                ((mpositions[mode] - positions[mode]).abs()**2).values.mean()
                for mode in modes )

            return dtot

        kwargs.update(**options)

        # Minuit optimizer
        minuit = Minuit(objfun, **kwargs)
        if kwargs.get('print_level', 0) >= 1:
            minuit.print_param()

        # Optimization
        minuit.migrad()

        print("Minuit minimization: {}".format(
            'succeeded' if minuit.migrad_ok() else 'failed'))

        if minuit.migrad_ok():
            print(" Adjusted parameters ".center(LINEWIDTH, '-'))
            for name in optparams:
                print("  {:20}: {} +/- {}"
                      .format(name, minuit.values[name], minuit.errors[name]))
            # Update parameters to adjusted values
            objfun(*minuit.args)
            # Compute final RMS
            rms = (minuit.fval / len(modes)) ** 0.5  # [m]
            print("  RMS: {} mm = {} px"
                  .format(rms / 1e-3, rms / self.detector.pxsize))

        return minuit

# Utility functions =======================================


def is_spectred(mode):
    """
    Is observational *mode* a spectroscopic (int-like) or photometric
    (str-like) mode?
    """

    return not isinstance(mode, basestring)


def str_mode(mode):
    """String 'Order #X' or 'Band Y' from *mode*."""

    return "{}{}".format("Order #" if is_spectred(mode) else "Band ", mode)


def rect2pol(position):
    r"""
    Convert complex position(s) :math:`x + jy` into modulus :math:`r` and
    phase :math:`\phi`.

    :param complex position: 2D-position(s) :math:`x + jy`
    :return: (r, phi) [rad]
    """

    return N.absolute(position), N.angle(position)


def pol2rect(r, phi):
    r"""
    Convert modulus :math:`r` and phase :math:`\phi` into complex position(s)
    :math:`x + jy = r\exp(j\phi)`.

    :param float r: module(s)
    :param float phi: phase(s) [rad]
    :return: complex 2D-position(s)
    """

    return r * N.exp(1j * phi)


def dump_mpld3(fig, filename):
    """
    Dump figure to `mpld3 <http://mpld3.github.io/>`_ HTML-file.
    """

    import mpld3                # Raise ImportError if needed

    for ax in fig.axes:
        # Remove legend title if any
        # (https://github.com/jakevdp/mpld3/issues/275)
        leg = ax.get_legend()
        if leg is not None:
            txt = leg.get_title().get_text()
            if txt == 'None':
                leg.set_title("")

    # Add mouse position
    mpld3.plugins.connect(fig,
                          mpld3.plugins.MousePosition(fontsize='small'))
    # Save figure to HTML
    mpld3.save_html(fig, filename,
                    no_extras=False, template_type='simple')
    print("MPLD3 figure saved in", filename)


def dump_bokeh(fig, filename):
    """
    Dump figure to `bokeh <http://bokeh.pydata.org/>`_ HTML-file.

    .. WARNING:: Bokeh-0.11 does not yet convert properly the figure.
    """

    from bokeh import mpl       # Raise ImportError if needed
    from bokeh.plotting import output_file, show

    output_file(filename)
    show(mpl.to_bokeh(fig))
    print("Bokeh figure saved in", filename)
