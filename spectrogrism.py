#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
spectrogrism
------------

.. autosummary::

   Spectrum
   FocalPointSource
   DetectorPositions
   LateralColor
   Material
   CameraCollimator
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

from collections import OrderedDict  # Used for configuration
import warnings

import numpy as N
import matplotlib.pyplot as P
try:
    import seaborn
    seaborn.set_style("whitegrid",
        # {'xtick.major.size': 6, 'xtick.minor.size': 3,
        #  'ytick.major.size': 6, 'ytick.minor.size': 3}
    )
except ImportError:
    pass

N.set_printoptions(linewidth=100, threshold=10)

# Constants ===============================================

RAD2DEG = 57.29577951308232  #: Convert from radians to degrees
RAD2MIN = RAD2DEG * 60       #: Convert from radians to arc minutes
RAD2SEC = RAD2MIN * 60       #: Convert from radians to arc seconds

#: SNIFS optical configuration, R-channel
SNIFS_R = OrderedDict(
    name="SNIFS-R",            # Configuration name
    LREF=0.76e-6,              # Reference wavelength [m]
    LRANGE=(0.5e-6, 1.02e-6),  # Wavelength range [m]
    # Telescope
    TEL_F=22.5,                # Focal length [m]
    TEL_E=0.,                  # r² distortion coefficient
    # Collimator
    COLL_F=169.549e-3,         # Focal length [m]
    COLL_E=+2.141,             # r² distortion coefficient
    COLL_A=[-4.39879e-6, 8.91241e-10, -1.82941e-13],  # Lateral color coeffs
    # Grism
    GRISM_ON=True,             # Is prism on the way?
    GRISM_GLA='BK7',           # Prism glass
    GRISM_ANG=17.28/RAD2DEG,   # Prism angle [rad]
    GRISM_RHO=200.,            # Grating groove density [lines/mm]
    GRISM_D=2.86,              # Wavelength dispersion [AA/pix]
    GRISM_GRA='EPR',           # Grating resine
    GRISM_BLA=15./RAD2DEG,     # Blaze angle [rad]
    # Camera
    CAM_F=228.014e-3,          # Focal length [m]
    CAM_E=-0.276,              # r² distortion coefficient
    CAM_A=[+2.66486e-6, -5.52303e-10, 1.1365e-13],  # Lateral color coeffs
    # Detector
    DET_PXSIZE=15e-6,          # Detector pixel size [m]
)

#: EUCLID optical configuration, R-grism of NISP spectrograph
#:
#: The detector plane is tiled with 4×4 detectors of 2k×2k pixels of 18
#: µm; the spectrograph has a mean magnification (`NISPPlateScale`) of
#: 0.5 approximately.  Hence a focal plane of approximately 29×29 cm².
EUCLID_R = OrderedDict(
    name="EUCLID-R",
    LRANGE=(1.25e-6, 1.85e-6),  # Wavelength range [m]
    GRISM_D=9.8,                # Mean dispersion [AA/pix]
    # Detector
    DET_PXSIZE=18e-6           # Detector pixel size [m]
)

#: TEST optical configuration (mimic somehow EUCLID)
#:
#: .. WARNING:: NOT IMPLEMENTED YET
TEST = OrderedDict(
    name="Euclid-R like",       # Configuration name
    LREF=1.5e-6,                # Reference wavelength [m]
    LRANGE=(1.25e-6, 1.85e-6),  # Wavelength range [m]
    # Collimator
    COLL_F=300e-3,             # Focal length [m]
    COLL_E=+2.,                # r² distortion coefficient
    COLL_A=[-4e-6, +9e-10, -2e-13],  # Lateral color coefficients
    # Grism
    GRISM_ON=True,             # Is prism on the way?
    GRISM_GLA='BK7',           # Prism glass
    GRISM_ANG=20/RAD2DEG,      # Prism angle [rad]
    GRISM_RHO=100.,            # Grating groove density [lines/mm]
    GRISM_GRA='EPR',           # Grating resine
    GRISM_BLA=15./RAD2DEG,     # Blaze angle [rad]
    # Camera
    CAM_F=150e-3,              # Focal length [m]
    CAM_E=-0.3,                # r² distortion coefficient
    CAM_A=[+3e-6, -6e-10, +1e-13],  # Lateral color coefficients
    # Detector
    DET_PXSIZE=18e-6           # Detector pixel size [m]
)

#: Simulation configuration
SIMU = OrderedDict(
    NPX=10,                    # Nb of pixels per spectrum
    ORDERS=range(-1, 3),       # Dispersion orders
    # Focal plane
    FOC_SAMPLE=N.linspace(-1e-2, 1e-2, 5), # Sampling of the focal plane [m]
    FOC_ANGLE=-10/RAD2DEG,     # Rotation of the focal plane
    # Detector
    DET_ANGLE=0/RAD2DEG,       # Rotation of the detector (0 = 0th-order up)
)


# Helper Classes ==========================================

class Spectrum(object):

    """
    A list of fluxes at different wavelengths.
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
            self.wavelengths[0] * 1e6, self.wavelengths[-1] * 1e6)

    @classmethod
    def default(cls, lrange=(0.5e-6, 1.5e-6), npx=1, name='spectrum'):
        """
        A default spectrum, with linearly-sampled wavelengths and
        constant flux.

        :param 2-tuple lrange: wavelength range [m]
        :param int npx: number of pixels linearly sampling the spectral ramp
        :param str name: optional name
        :return: linearly-sampled constant spectrum
        :rtype: :class:`Spectrum`
        """

        lmin, lmax = lrange
        if npx > 1:
            wavelengths = N.linspace(lmin, lmax, npx)
        elif npx == 1:
            wavelengths = N.mean(lrange, keepdims=True) # Singleton 1D-array
        fluxes = N.ones_like(wavelengths)

        return cls(wavelengths, fluxes, name=name)


class FocalPointSource(object):

    """
    A :class:`Spectrum` associated to a 2D-position in the focal plane.
    """

    def __init__(self, position, spectrum=None, **kwargs):
        """
        Initialize from position and spectrum.

        :param complex position: 2D-position in the focal plane [m]
        :param spectrum: :class:`Spectrum` (default to standard spectrum)
        :param kwargs: propagated to :func:`Spectrum.default()` constructor
        """

        self.position = complex(position)  #: Position
        if spectrum is None:
            spectrum = Spectrum.default(**kwargs)
        else:
            assert isinstance(spectrum, Spectrum), \
                "spectrum should be a Spectrum"
        self.spectrum = spectrum  #: Spectrum

    def __str__(self):

        return "{}, {}".format(str_position(self.position), self.spectrum)


class DetectorPositions(object):

    """
    A container for positions on the detector.

    A dictionary-based container for (complex) positions in the
    detector plane, labeled by (complex) source position in the focal
    plane and dispersion orders.

    .. autosummary::

       add_spectrum
       plot_detector
    """

    markers = {0:'.', 1:'o', 2:'s'}

    def __init__(self, spectrograph, wavelengths):
        """
        Initialize from spectrograph and wavelength array.
        """

        assert isinstance(spectrograph, Spectrograph), \
            "spectrograph should be a Spectrograph"
        self.spectro = spectrograph #: Associated spectrograph

        self.lbda = N.array(wavelengths) #: Wavelengths [m]

        #: { f_position (complex): { order: [ d_positions (complexes) ] } }
        self.spectra = {}

    def add_spectrum(self, focal_position, detector_positions, order):
        """
        Add `detector_positions` corresponding to source at
        `focal_position` and dispersion `order`.

        :param complex focal_position: source position in the focal plane
        :param numpy.ndarray detector_positions: (complex) positions
            in the detector plane
        :param int order: dispersion order
        """

        assert len(detector_positions) == len(self.lbda), \
            "incompatible detector_positions array"

        self.spectra.setdefault(focal_position, {})[order] = detector_positions

    def plot_detector(self, ax=None,
                      focal_positions=None, orders=None, blaze=True):
        """
        Plot spectra on detector plane.

        :param ax: pre-existing :class:`matplotlib.pyplot.Axes` instance if any
        :param tuple focal_positions: selection of complex focal plane
            2D-positions to be plotted
        :param tuple orders: selection of dispersion orders to be plotted
        :param blaze: if `True`, encode the blaze function in the marker size
        :return: :class:`matplotlib.pyplot.Axes`
        """

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel=u"x [µm]", ylabel=u"y [µm]",
                                 title=self.spectro.config.get('name'))
            ax.set_aspect('equal', adjustable='datalim')
        else:
            fig = None  # Will serve as a flag

        # Segmented colormap
        cmap = P.get_cmap('Spectral_r')
        cmap = P.matplotlib.colors.LinearSegmentedColormap(
            'dummy', cmap._segmentdata, len(self.lbda))

        if focal_positions is None:
            focal_positions = self.spectra.keys()

        # Blaze function cache (they depend on wavelength and orders only)
        bz_functions = {}

        for fpos in focal_positions:  # Loop over sources
            try:
                source = self.spectra[fpos]
            except KeyError:
                warnings.warn(
                    "Source {} is unknown, skipped".format(str_position(fpos)))
                continue
            if orders is None:
                all_orders = source.keys()
            else:
                all_orders = orders
            for order in all_orders:
                try:
                    positions = source[order]
                except KeyError:  # This dispersion order has not been computed
                    warnings.warn(
                        "Source {} is not available for order {}".format(
                            str_position(fpos), order))
                    continue

                if blaze:
                    if order not in bz_functions:
                        bz_fn = self.spectro.grism.blaze_function(
                            self.lbda, order)
                        bz_functions[order] = bz_fn
                    else:
                        bz_fn = bz_functions[order]
                else:
                    bz_fn = N.ones_like(self.lbda)

                sc = ax.scatter(positions.real/1e-6, positions.imag/1e-6,
                                s=N.maximum(50 * N.sqrt(bz_fn), 5),
                                c=self.lbda/1e-6,
                                cmap=cmap,
                                marker=self.markers.get(abs(order), 'o'),
                                edgecolor='none')

        if fig:
            fig.colorbar(sc, label=u"Wavelength [µm]")

        return ax


class LateralColor(object):

    """
    A description of lateral color chromatic distortion.

    The transverse chromatic aberration (so-called *lateral color*) occurs when
    different wavelengths are focused at different positions in the focal
    plane.

    **Reference:** `Chromatic aberration
    <https://en.wikipedia.org/wiki/Chromatic_aberration>`_
    """

    def __init__(self, lref, coeffs):
        """
        Initialization from reference wavelength [m] and lateral color
        coefficients.

        :param float lref: reference wavelength [m]
        :param list coeffs: lateral color coefficients
        """

        self.lref = lref  #: Reference wavelength [m]
        self.coeffs = N.array(coeffs)  #: Lateral color coefficients

    def __str__(self):

        return "Lateral color: lref={:.2f} µm, coeffs={}".format(
            self.lref*1e6, self.coeffs)

    def amplitude(self, wavelengths):
        """
        Amplitude of the lateral color chromatic distortion.

        :param numpy.ndarray wavelengths: wavelengths [mm]
        :return: lateral color amplitude
        """

        return sum([ c*(wavelengths - self.lref)**i
                     for i,c in enumerate(self.coeffs, start=1) ])


class Material(object):

    """
    Optical material.

    The refractive index is described by its Sellmeier coefficients.

    **Reference:** `Sellmeier equation
    <https://en.wikipedia.org/wiki/Sellmeier_equation>`_
    """

    #: Sellmeier coefficients [B1, B2, B3, C1, C2, C3] of known materials.
    materials =  dict(
        # Glasses
        BK7=[ 1.03961212, 2.31792344e-1, 1.01046945, 6.00069867e-3, 2.00179144e-2, 103.560653],
        UBK7=[1.01237433, 2.58985218e-1, 1.00021628, 5.88328615e-3, 1.90239921e-2, 104.079777],
        SF4=[ 1.61957826, 3.39493189e-1, 1.02566931, 1.25502104e-2, 5.33559822e-2, 117.65222],
        SK5=[ 0.99146382, 4.95982121e-1, 0.98739392, 5.22730467e-3, 1.72733646e-2,  98.3594579],
        F2=[  1.34533359, 2.09073176e-1, 0.93735716, 9.97743871e-3, 4.70450767e-2, 111.886764],
        SF57=[1.81651371, 4.28893641e-1, 1.07186278, 1.43704198e-2, 5.92801172e-2, 121.419942],
        # Fused silica
        FS=[  0.6961663,  4.079426e-1,   0.8974794,  4.679148e-3,   1.351206e-2,    97.9340],
        # Epoxy
        EPR=[ 0.512479,   0.838483,     -0.388459,  -0.0112765,     0.0263791,     557.682],
        EPB=[  0.406836,  1.03517,      -0.140328,  -0.0247382,     0.0261501,     798.366],
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

        return "Material: {}, n(1 µm)={:.3f}".format(
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

        lmu2 = (wavelengths / 1e-6)**2    # (wavelength [µm])**2
        n2m1 = sum([ b / (1 - c / lmu2)   # n**2 - 1
                     for b,c in zip(self.coeffs[:3], self.coeffs[3:]) ])

        return N.sqrt(n2m1 + 1)


# Optical element classes =================================

class CameraCollimator(object):

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

        assert lcolor is None or isinstance(lcolor, LateralColor), \
            "lcolor should be a LateralColor"

        self.flength = flength        #: Focal length [m]
        self.distortion = distortion  #: r² distortion coefficient
        self.lcolor = lcolor          #: Lateral color

    def __str__(self):

        s = "f={:.1f} m, e={:+.3f}".format(self.flength, self.distortion)
        if self.lcolor is not None:
            s += '\n   {}'.format(self.lcolor)

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
        if y==0:
            return 0.
        elif b==0 and e==0:
            return N.nan
        elif b==0:
            return (y/e)**(1/3)
        elif e==0:
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


class Collimator(CameraCollimator):

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
        """

        try:
            flength = config['COLL_F']
            distortion = config['COLL_E']
            lcolor = LateralColor(config['LREF'], config['COLL_A'])
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
        rr = r / self.flength     # Normalized radius
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


class Camera(CameraCollimator):

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
        """

        try:
            flength = config['CAM_F']
            distortion = config['CAM_E']
            lcolor = LateralColor(config['LREF'], config['CAM_A'])
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
        """

        try:
            flength = config['TEL_F']
            distortion = config['TEL_E']
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

        self.angle = angle  #: Prism angle [rad]
        self.material = material  #: Prism material
        self.tilts = tilts  #: Prism tilts (x,y,z) [rad]

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

        self.rho = rho  #: Grating groove density [lines/mm]
        self.material = material  #: Grating material
        self.blaze = blaze  #: Grating blaze angle [rad]

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
        yp = y * n + order * wavelengths * self.rho * 1e3
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
        y = (yp - order * wavelength * self.rho * 1e3) / n
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
        """

        try:
            A = config['GRISM_ANG']
            prism_material = config['GRISM_GLA']
            rho = config['GRISM_RHO']
            grating_material = config['GRISM_GRA']
            blaze = config['GRISM_BLA']
        except KeyError as err:
            raise KeyError(
                "Invalid configuration file: missing key {!r}".format(
                    err.args[0]))

        self.prism = Prism(A, Material(prism_material))
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
            self, self.null_deviation(order=1)/1e-6)

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

        np = self.prism.material.index(wavelengths)  # Prism index
        ng = self.grating.material.index(wavelengths)  # Grating index

        rholbda = self.grating.rho * 1e3 * wavelengths  # g/m * m = unitless
        npsinA = np * N.sin(self.prism.angle)

        i = N.arcsin(npsinA / ng) - self.grating.blaze # [rad]
        r = N.arcsin(npsinA - order * rholbda) - self.grating.blaze # [rad]

        theta = (N.pi / rholbda * N.cos(self.grating.blaze) *
                 (ng*N.sin(i) - N.sin(r)))
        bf = (N.sin(theta)/theta)**2

        return bf

    @staticmethod
    def direction2xyz(direction):
        """
        Convert a 2D-direction into a 3D-direction.

        :param complex direction: 2D-direction
        :return: 3D-direction
        :type: 3-tuple
        """

        tantheta, phi = CameraCollimator.rect2pol(direction)
        tan2 = tantheta**2
        costheta = N.sqrt(1/(1 + tan2))
        sintheta = N.sqrt(tan2/(1 + tan2))

        return N.vstack((N.cos(phi)*sintheta,
                         N.sin(phi)*sintheta,
                         costheta)).squeeze()

    @staticmethod
    def xyz2direction(xyz):
        """
        Convert a 3D-direction into a 2D-direction.

        :param 3-tuple xyz: 3D-direction
        :return: 2D-direction
        :rtype: complex
        """

        x, y, z = xyz
        tantheta = N.hypot(x, y) / z
        phi = N.arctan2(y, x)

        return CameraCollimator.pol2rect(tantheta, phi)

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
        :raise: `RuntimeError` if not converging
        """

        import scipy.optimize as SO

        k = N.sin(self.prism.angle)/(order * self.grating.rho * 1e3)
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

    def __init__(self, config, grism_on=True):
        """
        Initialize spectrograph from optical configuration (dictionary).

        :param dict config: optical configuration
        """

        self.config = config

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

        s = " Spectrograph ".center(60, '-') + '\n'
        s += self.collimator.__str__() + '\n'
        if self.grism_on:
            s += self.grism.__str__() + '\n'
        else:
            s += "Grism:      ***REMOVED***\n"
        s += self.camera.__str__() + '\n'
        s += "Spectrograph magnification: {0.gamma:.3f}\n".format(self)
        try:
            lmean = N.mean(self.config['LRANGE'])
            pxsize = self.config['DET_PXSIZE']
            s += "Central dispersion: {:.2f} AA/px at {:.2f} µm\n".format(
                self.dispersion(lmean)/1e-10*pxsize, lmean*1e6)
        except KeyError:
            pass
        s += '-'*60

        return s

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

            sinbeta = (order * self.grism.grating.rho * 1e3 * l -
                       self.grism.prism.material.index(l) *
                       N.sin(self.grism.prism.angle))

            return N.tan(N.arcsin(sinbeta))

        dydl = (derivative(yoverf, wavelength, dx=wavelength*eps) *
                self.camera.flength)

        return 1/dydl


    def forward(self, source, order=1):
        """
        Forward light propagation from a focal-plane point source.

        :param source: input :class:`FocalPointSource`
        :param int order: dispersion order
        :return: (complex) 2D-positions in detector plane
        :rtype: :class:`numpy.ndarray`
        """

        assert isinstance(source, FocalPointSource), \
            "source should be a FocalPointSource"

        wavelengths = source.spectrum.wavelengths

        directions = self.collimator.forward(
            source.position, wavelengths, self.gamma)
        if self.grism_on:
            directions = self.grism.forward(
                directions, wavelengths, order=order)
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

        direction = self.camera.backward(position, wavelength)
        if self.grism_on:
            direction = self.grism.backward(direction, wavelength, order=order)
        position = self.collimator.backward(direction, wavelength, self.gamma)

        return position

    def test(self, position=(1e-3 + 2e-3j), npx=3, order=1, verbose=False):
        """
        Test forward and backward propagation in spectrograph.

        :param complex position: tested 2D-position in the focal plane [m]
        :param int npx: number of wavelengths to be tested within
            spectral domain
        :param int order: tested dispersion order
        """

        # Test source
        source = FocalPointSource(position,
                                  lrange=self.config['LRANGE'], npx=npx)
        wavelengths = source.spectrum.wavelengths
        if verbose:
            print(" SPECTROGRAPH TEST ".center(40, '='))
            print("Source (focal plane):", source)
            print("Wavelengths [µm]:", wavelengths*1e6)

        # Forward step-by-step
        fdirections = self.collimator.forward(source.position,
                                              wavelengths, self.gamma)
        if verbose:
            print("Directions (coll, forward) [x1e6]:", fdirections*1e6)

        if self.grism_on:
            fdirections = self.grism.forward(fdirections,
                                             wavelengths, order=order)
            if verbose:
                print("Directions (grism, forward) [x1e6]:", fdirections*1e6)

        dpositions = self.camera.forward(fdirections, wavelengths)
        if verbose:
            print("Positions (detector) [µm]:", dpositions*1e6)

        # Loop over positions in detector plane
        for lbda, dpos in zip(wavelengths, dpositions):
            # Backward step-by-step
            if verbose:
                print("Test position (detector) [µm]:", dpos*1e6,
                      "Wavelength [µm]:", lbda*1e6)
            bdirection = self.camera.backward(dpos, lbda)
            if verbose:
                print("Direction (camera, backward) [x1e6]:", bdirection*1e6)
            if self.grism_on:
                bdirection = self.grism.backward(bdirection, lbda, order=order)
                if verbose:
                    print("Direction (grism, backward) [x1e6]:", bdirection*1e6)
            fposition = self.collimator.backward(bdirection,
                                                 lbda, spectro.gamma)
            if verbose:
                print("Focal-plane position (backward) [µm]:", fposition*1e6)
                print("Source position (reminder) [µm]:", source.position*1e6)

            assert N.isclose(source.position, fposition), \
                "Backward modeling does not match"

    def simulate(self, simu):
        """
        Simulate detector spectra.

        :param dict simu: input simulation
        :return: simulated spectra
        :rtype: :class:`DetectorPositions`
        """

        import itertools

        # Simulation parameters
        npx = simu.get('NPX', 1)
        orders = simu.get('ORDERS', (1,))
        foc_sample = simu.get('FOC_SAMPLE', N.array([0]))
        foc_angle = simu.get('FOC_ANGLE')
        det_angle = simu.get('DET_ANGLE')

        source = FocalPointSource(0, lrange=self.config['LRANGE'], npx=npx)
        wavelengths = source.spectrum.wavelengths

        # Detector positions
        detector = DetectorPositions(self, wavelengths)

        # Focal-plane positions [m]
        fpos = N.array([ complex(x, y)
                         for x,y in itertools.product(foc_sample, foc_sample) ])
        # Rotation in the focal plane
        if foc_angle:
            fpos *= N.exp(1j*foc_angle)

        # Simulate forward propagation for all focal-plane positions
        for xy in fpos:
            # Update source position
            source.position = xy
            for order in orders:  # Loop over dispersion orders
                dpos = self.forward(source, order)  # Detector plane position
                if det_angle:     # Rotation in the detector plane
                    dpos *= N.exp(1j*det_angle)
                detector.add_spectrum(source.position, dpos, order=order)

        return detector

# Utility functions =======================================

def str_position(position):
    """
    Pretty-printer of a complex position.

    .. warning:: work on a single (complex) position.
    """

    return "{:+.1f} x {:+.1f} µm".format(position.real*1e6, position.imag*1e6)

def str_direction(direction):
    """
    Pretty-printer of a complex direction.

    .. warning:: work on a single (complex) direction.
    """

    tantheta, phi = CameraCollimator.rect2pol(direction)
    return "{:+.2f} x {:+.2f} arcmin".format(
        N.arctan(tantheta)*RAD2MIN, phi*RAD2MIN)

# Main ====================================================

if __name__ == '__main__':

    # Optical configuration
    config = SNIFS_R
    print("Configuration name: {}".format(config['name']))

    # Override test optical configuration
    OVERRIDE = dict(
        # GRISM_ON = False,
        # COLL_E = 0,
        # COLL_A = (0, 0, 0),
        # CAM_E = 0,
        # CAM_A = (0, 0, 0),
        )
    if OVERRIDE:
        warnings.warn(
            "overriding configuration with test values {}".format(OVERRIDE))
        OVERRIDE.setdefault( 'name', config['name'] + ' (overrided)')
        config.update(OVERRIDE)

    # Spectrograph
    spectro = Spectrograph(config, grism_on=config['GRISM_ON'])
    print(spectro)

    try:
        spectro.test(verbose=False)
    except AssertionError as err:
        warnings.warn(str(err))
    else:
        print("Spectrograph test: OK")

    # Simulation configuration
    simu = SIMU

    detector = spectro.simulate(simu)
    ax = detector.plot_detector(orders=(-1, 0, 1, 2))
    ax.set_aspect('auto')
    ax.axis(N.array([-2000, 2000, -4000, 4000])*config['DET_PXSIZE']/1e-6)
    P.show()
