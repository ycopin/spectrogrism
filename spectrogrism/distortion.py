# -*- coding: utf-8 -*-
# Time-stamp: <2016-04-06 22:07 ycopin@lyonovae03.in2p3.fr>

"""
distortion
-----------

Distortion utilities.

.. autosummary::

   StructuredGrid
   GeometricDistortion
   ChromaticDistortion
"""

from __future__ import division, print_function

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"

import warnings

import numpy as N


class StructuredGrid(object):

    """
    A structured 2D-grid, stored as a complex 2D-array.
    """

    def __init__(self, xy):

        self.xy = N.atleast_2d(xy).astype(complex)
        try:
            ny, nx = self.xy.shape
        except ValueError:
            raise ValueError("Invalid array shape {}".format(self.xy.shape))

    def __str__(self):

        try:
            sign = self.signature
        except ValueError:
            sign = "unsigned"

        s = "Structured grid: {} x {} = {} positions, {}".format(
            self.nx, self.ny, self.nx * self.ny, sign)

        return s

    @classmethod
    def create(cls, nx, ny, step='auto', offset=0, rotation=0.):
        u"""
        Create a regular rectangular grid.
        """

        assert (nx, ny) >= (2, 2), "Invalid grid size."
        if step == 'auto':     # Grid extent will be ~ ± 1
            step = 2 / (nx * ny)**0.5

        x = N.arange(nx) - (nx - 1)/2.
        y = N.arange(ny) - (ny - 1)/2.
        xx, yy = N.meshgrid(x, y)
        xy = xx + 1j * yy
        if step != 1:
            xy *= step
        if rotation:
            xy *= N.exp(1j*rotation)
        if offset:
            xy += offset

        return cls(xy)

    @property
    def nx(self):

        return self.xy.shape[1]

    @property
    def ny(self):

        return self.xy.shape[0]

    @property
    def x(self):
        """Flattened x-coordinate array."""

        return self.xy.real.ravel()

    @property
    def y(self):
        """Flattened y-coordinate array."""

        return self.xy.imag.ravel()

    @property
    def signature(self):
        """
        Return a 4-character signature string such as 'x+y-'.

        Standard order -- x increasing along 1st axis and y increasing
        along 0th axis -- corresponds to 'x+y+'. 'y-x+' corresponds to
        a transposed coordinate array, with y decreasing along the 1st
        axis, and x increasing along the 0th axis.

        Noting '+.' a mean finite difference where the real part is
        positive and larger (in absolute value) than the imaginary
        part, one has the following correspondance::

          fd[0] |  +.    -.    .+    .-
          fd[1] |
          ------+-----------------------
            +.  |             y+x+  y-x+
            -.  |             y+x-  y-x-
            .+  | x+y+  x-y+
            .-  | x+y-  x-y-
        """

        # d[0] is the mean finite difference along the columns
        # (axis=1), d[1] is the mean increment along the rows (axis=0)
        fd = [ N.diff(self.xy.mean(axis=axis)).mean() for axis in (0, 1) ]
        xgrad = [ abs(dd.real) > abs(dd.imag) for dd in fd ]
        if xgrad[0] and not xgrad[1]:    # ±. & .±
            axes = 'xy'
        elif not xgrad[0] and xgrad[1]:  # .± & ±.
            axes = 'yx'
        else:
            raise ValueError("Cannot identify grid signature (axes).")

        def sign_major(z):
            """Sign of major component of z."""

            if abs(z.real) > abs(z.imag):  # ±.
                return '+' if z.real > 0 else '-' if z.real < 0 else '0'
            else:                          # .±
                return '+' if z.imag > 0 else '-' if z.imag < 0 else '0'

        signs = [ sign_major(dd) for dd in fd ]  # [±, ±]
        if '0' in signs:
            raise ValueError("Cannot identify grid signature (sort).")

        return ''.join( a+s for a, s in zip(axes, signs) )  # '[xy]±[yx]±'

    def reorder(self, signature):
        """Manipulate self.xy to match signature."""

        assert signature in ('x+y+', 'x+y-', 'x-y+', 'x-y-',
                             'y+x+', 'y+x-', 'y-x+', 'y-x-'), \
            "Invalid signature '{}'.".format(signature)

        try:
            sig_self = self.signature
        except ValueError:
            raise ValueError("Cannot reorder unsigned grid.")

        # First, is it the same ordering?
        if sig_self[0] != signature[0]:  # No: transpose self.xy
            self.xy = N.transpose(self.xy)
            sig_self = sig_self[2:] + sig_self[:2]  # Update the signature

        # Order along axis 0
        if sig_self[1] != signature[1]:
            self.xy = N.fliplr(self.xy)  # No need to update the signature

        # Order along axis 1
        if sig_self[3] != signature[3]:
            self.xy = N.flipud(self.xy)  # No need to update the signature

        return self.xy

    def plot(self, ax=None, label=None, color='k'):
        """Plot structured grid."""

        import matplotlib.pyplot as P

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x", ylabel="y",
                                 title=self.__str__())
            ax.set_aspect('equal', adjustable='datalim')

        for xx, yy in zip(self.xy.real, self.xy.imag):  # Loop over rows
            ax.plot(xx, yy,
                    color=color, marker='.', label=label)
            if label:                 # Only one label
                label = None
        for xx, yy in zip(self.xy.real.T, self.xy.imag.T):  # Over columns
            ax.plot(xx, yy,
                    color=color, marker='.', label='_')

        return ax

    def estimate_parameters(self, rmin=0.1, frac=0.1, rescale=False, fig=None):
        """
        Estimate regular grid parameters from triangulation analysis.

        :param float rmin: minimal circle ratio (see
            :func:`matplotlib.tri.TriAnalyzer.get_flat_tri_mask`)
        :param float frac: fraction of edges used for distortion parameters
        :param bool rescale: rescale long edges
            (not trustworthy for significantly distorted grid)
        :param matplotlib.pyplot.Figure fig: produde a control plot if not None
        :return: (step, rotation [rad],
            complex offset, complex center of distortion)
        """

        from matplotlib.tri import (Triangulation, TriAnalyzer,
                                    LinearTriInterpolator)

        # Triangulation
        tri = Triangulation(self.x, self.y)
        # Optimization: mask border triangles which are too flat
        mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio=rmin)
        tri.set_mask(mask)

        # Edge lengths
        xy = N.vstack((self.x, self.y)).T  # npts × [x, y]
        edges = xy[tri.edges]              # nedges × [start, end] × [x, y]
        dxy = edges[:, 1] - edges[:, 0]    # nedges × [x, y]
        lengths = N.sum(dxy**2, axis=1)**0.5  # nedges

        # Diagonals (i.e. long edges)
        cut = N.percentile(lengths, 66)  # 2/3 are short edges, 1/3 are long
        medinf = N.median(lengths[lengths <= cut])  # Median of short edges
        medsup = N.median(lengths[lengths >= cut])  # Median of long edges
        med = (medinf + medsup) / 2                 # Separator short/long
        diags = lengths > med

        # Rescale diagonal lengths
        if rescale:
            lengths[diags] /= 2**0.5

        # Angles rewarp in ± pi/4 [rad]
        angles = N.arctan2(dxy[:, 1], dxy[:, 0]) + N.pi  # in [0, 2*pi]
        angles = (4 * angles - N.pi * N.around(4 * angles/N.pi)) / 4

        # Sort lengths by increasing sizes
        iedges = N.argsort(lengths)

        # Use the central fraction of less distorted points (shortest
        # edges) to estimate undistorted grid parameters
        ncen = int(len(edges) * frac)     # Nb of central edges
        shortests = edges[iedges[:ncen]]  # ncen × [start, end] × [x, y]
        step = N.median(lengths[iedges[:ncen]])     # Undistorted grid step
        angle = N.median(angles[iedges[:ncen]])     # Undistorted grid angle
        center = N.complex(
            *N.mean(shortests, axis=(0, 1)))  # Center of distortion (complex)

        # Reference grid
        ref = StructuredGrid.create(self.nx, self.ny,
                                    step=step, rotation=angle)
        ref.reorder(self.signature)
        offsets = (self.xy - ref.xy).ravel()

        # Interpolate offset at center of distortion
        dx = LinearTriInterpolator(tri, offsets.real)(center.real, center.imag)
        dy = LinearTriInterpolator(tri, offsets.imag)(center.real, center.imag)
        offset = complex(dx, dy)  # Offset (complex)
        ref.xy += offset

        if fig:                 # Create control plot
            import matplotlib.pyplot as P

            if fig is True:     # Create the figure
                fig = P.figure()
            else:               # Use the incoming figure
                assert isinstance(fig, P.Figure)

            axgrid = P.GridSpec(2, 3)
            ax = fig.add_subplot(
                axgrid.new_subplotspec((0, 0),
                                       rowspan=2, colspan=2),
                xlabel="x", ylabel="y",
                title=self.__str__())
            axl = fig.add_subplot(axgrid.new_subplotspec((0, 2)),
                                  xlabel="Side length")
            axa = fig.add_subplot(axgrid.new_subplotspec((1, 2)),
                                  xlabel="Angle [deg]")

            # Input grid
            ax.plot(self.x, self.y, 'o', label='Input')

            # Triangulation
            ax.triplot(tri)
            ax.plot(shortests[..., 0].T, shortests[..., 1].T, 'k-', lw=2)

            # Reference grid
            ax.plot(ref.x, ref.y, '.k', label='Reference')

            # Center of distortion
            ax.plot([center.real], [center.imag],
                    marker='*', ms=10, ls='none', label='Center of distortion')

            ax.legend(loc='best', numpoints=1, fontsize='small',
                      framealpha=0.5, title='')
            ax.set_aspect('equal', adjustable='datalim')

            # Length histogram
            axl.hist([lengths[~diags], lengths[diags]],
                     bins=30, stacked=True, histtype='stepfilled',
                     label=['Short', u'Long/√2'])
            axl.axvline(step, c='k', lw=2)

            # Angle histogram
            axa.hist([N.rad2deg(angles[~diags]),
                      N.rad2deg(angles[diags])],
                     bins=30, stacked=True, histtype='stepfilled',
                     label=['Short', u'Long - 45°'])
            axa.axvline(N.rad2deg(angle), c='k', lw=2)

        return step, angle, offset, center

    def rms(self, xy):
        """
        Compute RMS distance to 2D-array.

        .. Note:: intput grids are supposed to be compatible both in shape and
           signature. RMS cannot be computed or is meaningless otherwise.
        """

        return (N.abs(self.xy - xy)**2).mean()**0.5

    def adjust_distortion(self, other, gdist,
                          scale=False, rotation=False, offset=False,
                          **options):
        """
        Adjust geometric distortion to match other (compatible) grid.

        If *scale* (resp. *rotation* and *offset*), the grid scale
        (resp. rotation and offset) is adjusted simultaneously.

        Other *options* are transmitted to `Minuit` initialization.
        """

        from iminuit import Minuit

        assert self.signature == other.signature, \
            "Incompatible grid signatures."
        assert self.xy.shape == other.xy.shape, \
            "Incompatible grid shapes."

        nkcoeff = len(gdist.Kcoeffs)
        npcoeff = len(gdist.Pcoeffs)

        # Objective function: RMS**2 (see signature below)
        def objfun(*args):

            gdist.x0, gdist.y0 = args[:2]        # Center of distortion
            gdist.Kcoeffs = args[2:2 + nkcoeff]  # K-coeffs
            last = 2 + nkcoeff
            gdist.Pcoeffs = args[last:last + npcoeff]  # P-coeffs
            last += npcoeff
            if scale:
                _scale = args[last]
                last += 1
            else:
                _scale = 1.
            if rotation:
                _rotation = args[last]
                last += 1
            else:
                _rotation = 0.
            if offset:
                _offset = complex(*args[last:last + 2])
                last += 2
            else:
                _offset = 0.

            xy = self.xy * _scale * N.exp(1j*_rotation) + _offset

            return (N.abs(gdist.forward(xy) - other.xy)**2).mean()  # RMS**2

        # Objective function parameters
        parameters = ['x0', 'y0'] + \
                     [ 'K%d' % i for i in range(1, nkcoeff + 1) ] + \
                     [ 'P%d' % i for i in range(1, npcoeff + 1) ]

        kwargs = {'forced_parameters': parameters,  # objfun signature
                  'errordef': 1}  # Least-square

        # Estimate position step size from position RMS
        xystep = (N.abs(self.xy)**2).mean()**0.5 * 1e-1
        kwargs.update((('x0', gdist.x0), ('error_x0', xystep),
                       ('y0', gdist.y0), ('error_y0', xystep)))

        # Arbitrarly set distortion parameter step size to 1e-3
        kwargs.update( (key, 0)
                       for key in parameters if key[0] in 'KP' )
        kwargs.update( ('error_' + key, 1e-3)
                       for key in parameters if key[0] in 'KP' )

        if scale:
            parameters += ['scale']
            kwargs.update((('scale', 1),
                           ('error_scale', 1e-2)))
        if rotation:
            parameters += ['rotation']
            kwargs.update((('rotation', 0),
                           ('error_rotation', 1e-2),))
        if offset:
            parameters += ['dx', 'dy']
            # Estimate offset step size from offset RMS
            offstep = self.rms(other.xy)
            kwargs.update((('dx', 0), ('error_dx', offstep),
                           ('dy', 0), ('error_dy', offstep)))

        # Additional Minuit options
        kwargs.update(**options)

        minuit = Minuit(objfun, **kwargs)
        if kwargs.get('print_level', 0):
            minuit.print_param()

        minuit.migrad()

        return minuit

    def plot_offsets(self, other, ax=None, units=(1, '')):
        """
        Plot offset between current and other grid (used as reference).
        """

        import matplotlib.pyplot as P

        uscale, uname = units   # (float, str)

        rms = self.rms(other.xy) / uscale  # [unit]

        if ax is None:
            ustr = ' [{}]'.format(uname) if uname else ''
            fig = P.figure()
            ax = fig.add_subplot(1, 1, 1,
                                 xlabel="x{}".format(ustr),
                                 ylabel="y{}".format(ustr),
                                 title="RMS = {:.4g} {}".format(rms, uname))
            ax.set_aspect('equal', adjustable='datalim')

        dxy = (self.xy - other.xy) / uscale
        q = ax.quiver(other.x / uscale, other.y / uscale,
                      dxy.real, dxy.imag)
        scale = "{:.1g}".format(rms)
        ustr = ' {}'.format(uname) if uname else ''
        ax.quiverkey(q, 0.95, 0.95, float(scale), "{}{}".format(scale, ustr),
                     labelpos='W', coordinates='figure')

        return ax


class GeometricDistortion(object):

    r"""
    Brown-Conrady (achromatic) distortion model.

    .. math::

       x_d &= x_u \times (1 + K_1 r^2 + K_2 r^4 + \ldots) \\
       &+ \left(P_2(r^2 + 2x_u^2) + 2P_1 x_u y_u\right)
       (1 + P_3 r^2 + P_4 r^4 + \ldots) \\
       y_d &= y_u \times (1 + K_1r^2 + K_2r^4 + \ldots) \\
       &+ \left(P_1(r^2 + 2y_u^2) + 2P_2 x_u y_u\right)
       (1 + P_3 r^2 + P_4 r^4 + \ldots)

    where:

    - :math:`x_u + j y_u` is the undistorted complex position,
    - :math:`x_d + j y_d` is the distorted complex position,
    - :math:`r^2 = (x_u - x_0)^2 + (y_u - y_0)^2`,
    - :math:`x_0 + j y_0` is the complex center of distortion.

    The K-coefficients (resp. P-coefficients) model the *radial*
    (resp. *tangential*) distortion.

    **Reference:** `Optical distortion
    <https://en.wikipedia.org/wiki/Distortion_%28optics%29>`_

    .. autosummary::

       forward
       backward
    """

    def __init__(self, center=0, Kcoeffs=[], Pcoeffs=[]):
        """
        Initialize from center of distortion and K- and P-coefficients.

        :param complex center: complex position of center of distortion [m]
        :param list Kcoeffs: radial distortion coefficients
        :param list Pcoeffs: tangential distortion coefficients
            (empty or length >= 2)
        """

        self.center = complex(center)

        # Radial component
        self._polyk = N.polynomial.Polynomial([1] + list(Kcoeffs))
        # Tangential component
        if len(Pcoeffs):
            if not len(Pcoeffs) >= 2:
                raise ValueError("Pcoeffs should be empty or of length >= 2.")
            self.p1 = Pcoeffs[0]
            self.p2 = Pcoeffs[1]
            self._polyp = N.polynomial.Polynomial([1] + list(Pcoeffs)[2:])
        else:
            self._polyp = None  # Will be used as a flag

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Initialize from kwargs."""

        center = complex(kwargs.pop('x0', 0), kwargs.pop('y0', 0))
        try:
            kmax = max( int(key[1:]) for key in kwargs if key.startswith('K') )
        except ValueError:
            Kcoeffs = []
        else:
            Kcoeffs = [ kwargs.pop('K'+str(i), 0) for i in range(1, kmax + 1) ]
        try:
            pmax = max( int(key[1:]) for key in kwargs if key.startswith('P') )
        except ValueError:
            Pcoeffs = []
        else:
            Pcoeffs = [ kwargs.pop('P'+str(i), 0) for i in range(1, pmax + 1) ]

        return cls(center, Kcoeffs=Kcoeffs, Pcoeffs=Pcoeffs)

    @property
    def Kcoeffs(self):
        """Radial coefficients."""

        return self._polyk.coef[1:]       # ndarray

    @Kcoeffs.setter
    def Kcoeffs(self, coeffs):

        self._polyk.coef[1:] = coeffs

    @property
    def Pcoeffs(self):
        """Tangential coefficients."""

        if self._polyp:
            coeffs = [self.p1, self.p2] + self._polyp.coef[1:].tolist()  # list
        else:
            coeffs = []

        return N.array(coeffs)            # ndarray

    @Pcoeffs.setter
    def Pcoeffs(self, coeffs):

        if len(coeffs):         # Should be of length >= 2
            if not len(coeffs) >= 2:
                raise ValueError("coeffs should be empty or of length >= 2.")
            self.p1 = coeffs[0]
            self.p2 = coeffs[1]
            self._polyp.coef[1:] = coeffs[2:]

    def __nonzero__(self):

        return len(self.Kcoeffs) or len(self.Pcoeffs)

    def __str__(self):

        if self.__nonzero__():
            s = ("Geometric distortion: "
                 "center=({:+g}, {:+g}), K-coeffs={}, P-coeffs={}"
                 .format(self.x0, self.y0,
                         self.Kcoeffs, self.Pcoeffs))
        else:
            s = "Null geometric distortion"

        return s

    @property
    def x0(self):
        """x-coordinate of center of distortion."""

        return self.center.real

    @x0.setter
    def x0(self, x0):

        self.center = x0 + 1j * self.center.imag

    @property
    def y0(self):
        """y-coordinate of center of distortion."""

        return self.center.imag

    @y0.setter
    def y0(self, y0):

        self.center = self.center.real + 1j * y0

    def forward(self, xyu):
        """
        Apply distortion to undistorted complex positions.
        """

        xyr = xyu - self.center           # Relative complex positions
        r2 = N.abs(xyr) ** 2              # Undistorted radii squared
        xu, yu = xyr.real, xyr.imag       # Undistorted coordinates

        xd = N.copy(xu)                   # Distorted coordinates
        yd = N.copy(yu)

        if self.Kcoeffs.any():            # Radial distortion
            polyk_r2 = self._polyk(r2)    # Polynomial in r²
            xd *= polyk_r2
            yd *= polyk_r2

        if self.Pcoeffs.any():            # Tangential distortion
            polyp_r2 = self._polyp(r2)    # Polynomial in r²
            two_xuyu = 2 * xu * yu
            xd += (self.p2 * (r2 + 2 * xu ** 2) + self.p1 * two_xuyu) * polyp_r2
            yd += (self.p1 * (r2 + 2 * yu ** 2) + self.p2 * two_xuyu) * polyp_r2

        xyd = (xd + 1j * yd) + self.center

        return xyd                        # Distorted complex positions

    def backward(self, xyd):
        """
        Correct distortion from distorted complex positions.
        """

        import scipy.optimize as SO

        def fun(x_y):
            """
            SO.root works on real functions only: we convert N complex
            2D-positions to and fro 2N real vector.
            """

            x, y = N.array_split(x_y, 2)                  # (2n,) ℝ → 2×(n,) ℝ
            off = self.forward(x + 1j * y) - xyd.ravel()  # (n,) ℂ
            return N.concatenate((off.real, off.imag))    # (n,) ℂ → (2n,) ℝ

        xyd = N.atleast_1d(xyd)
        xd, yd = xyd.real.ravel(), xyd.imag.ravel()       # → 2×(n,) ℝ
        result = SO.root(fun, N.concatenate((xd, yd)))

        if not result.success:
            raise RuntimeError("GeometricDistortion model is not invertible.")

        xu, yu = N.array_split(result.x, 2)               # → 2×(n,) ℝ

        return (xu + 1j * yu).reshape(xyd.shape).squeeze()

    def plot(self, xy=None, ax=None):
        """
        Plot distortions for a 2D-grid of complex positions.
        """

        import matplotlib.pyplot as P

        if xy is None:
            x = N.linspace(-1.5, 1.5, 13)   # Default grid
            xx, yy = N.meshgrid(x, x)
            xy = xx + 1j * yy               # Undistorted positions
        else:
            assert N.ndim(xy) == 2, "Invalid input grid"

        xyd = self.forward(xy)              # Distorted positions

        # Test
        try:
            test = N.allclose(self.backward(xyd), xy)
        except RuntimeError as err:         # Could not invert distortion
            warnings.warn(str(err))
        else:
            if not test:
                warnings.warn("GeometricDistortion inversion is invalid.")

        if ax is None:
            fig = P.figure()
            title = ' '.join((
                "K-coeffs={} ".format(self.Kcoeffs)
                if self.Kcoeffs.any() else '',
                "P-coeffs={}".format(self.Pcoeffs)
                if self.Pcoeffs.any() else ''))
            ax = fig.add_subplot(1, 1, 1,
                                 title=title,
                                 xlabel="x", ylabel="y")

        def plot_grid(ax, xy, label=None, color='k'):
            for xx, yy in zip(xy.real, xy.imag):
                ax.plot(xx, yy,
                        color=color, marker='.', label=label)
                if label:                 # Only one label
                    label = None
            for xx, yy in zip(xy.T.real, xy.T.imag):
                ax.plot(xx, yy,
                        color=color, marker='.', label='_')

        # Undistorted positions
        plot_grid(ax, xy, label='Undistorted', color='0.8')
        # Distorted grid
        plot_grid(ax, xyd, label='Distorted', color='k')
        # Center of distortion
        ax.plot([self.center.real], [self.center.imag],
                ls='None', marker='*', ms=10, label='Center')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='lower left', fontsize='small',
                  framealpha=0.5, title='')

        return ax


class ChromaticDistortion(object):

    """
    A polynomial description of Transverse Chromatic Distortion.

    The Transverse Chromatic Aberration (so-called *lateral color*)
    occurs when different wavelengths are focused at different positions
    in the focal plane.

    **Reference:** `Chromatic aberration
    <https://en.wikipedia.org/wiki/Chromatic_aberration>`_

    **See also:** Klein, Brauers & Aach, 2010, for a more detailed modeling of
    Transversal Chromatic Aberrations.

    .. autosummary::

       coeffs
       amplitude
    """

    def __init__(self, wref=0, coeffs=[]):
        """
        Initialize from reference wavelength and lateral color coefficients.

        .. math::

           dr = \sum_{i=1}^N c_i (\lambda - \lambda_{\mathrm{ref}})^i

        Note that :math:`c_0 = 0`.

        :param float wref:
            reference wavelength :math:`\lambda_{\mathrm{ref}}` [m]
        :param list coeffs:
            lateral color coefficients :math:`[c_{i=1, \ldots N}]`
        """

        self.wref = wref               #: Reference wavelength [m]
        self._poly = N.polynomial.Polynomial([0] + list(coeffs))

    @property
    def coeffs(self):
        """Expose non-null coefficients :math:`[c_{i \geq 1}]`."""

        return self._poly.coef[1:]  # ndarray

    def __nonzero__(self):

        return self.coeffs.any()

    def __str__(self):

        if self.__nonzero__():
            s = ("Chromatic distortion: lref={:.2f} µm, coeffs=[{}]"
                 .format(self.wref / 1e-6,
                         ', '.join( '{:+g}'.format(coeff)
                                    for coeff in self.coeffs )))
        else:
            s = "Null chromatic distortion"

        return s

    def amplitude(self, wavelengths):
        """
        Compute amplitude of the (radial) chromatic distortion.

        :param numpy.ndarray wavelengths: wavelengths [m]
        :return: lateral color radial amplitude
        """

        return self._poly(wavelengths - self.wref)


if __name__ == '__main__':

    import matplotlib.pyplot as P
    try:
        import seaborn
    except ImportError:
        pass

    from iminuit.frontends import ConsoleFrontend

    # Initial grid
    grid = StructuredGrid.create(15, 15, step=1., rotation=N.deg2rad(5.))

    # Add distortions
    gdist = GeometricDistortion(3 + 4j,
                                Kcoeffs=[1e-3], Pcoeffs=[1e-3, 0, 1e-3])
    grid = StructuredGrid(gdist.forward(grid.xy))
    grid.xy += complex(-1, 2)   # Offset

    length, angle, offset, center = grid.estimate_parameters(fig=False)

    refgrid = StructuredGrid.create(grid.nx, grid.ny,
                                    step=length, rotation=angle, offset=offset)
    refgrid.reorder(grid.signature)

    refrms = refgrid.rms(grid.xy)
    print("RMS wrt. reference grid: {}".format(refrms))
    print("Estimated center of distortion: {}".format(center))

    if True:
        ax = grid.plot_offsets(refgrid)
        ax.set_title("Reference RMS = {}".format(refrms))

    dist = GeometricDistortion(center, Kcoeffs=[0.])
    minuit = refgrid.adjust_distortion(grid, dist,
                                       #scale=True, rotation=True, offset=True,
                                       print_level=1,
                                       frontend=ConsoleFrontend())

    if minuit.migrad_ok():
        adjrms = minuit.fval ** 0.5  # objfun is RMS**2
        print("RMS wrt. adjusted grid: {}".format(adjrms))

        dist = GeometricDistortion.from_kwargs(**minuit.values)
        adjgrid = StructuredGrid(dist.forward(refgrid.xy))

        ax = grid.plot(label="Input", color='k')
        refgrid.plot(ax=ax, color='b',
                     label="Reference (RMS={:.3g})".format(refrms))
        adjgrid.plot(ax=ax, color='r',
                     label="Adjusted (RMS={:.3g})".format(adjrms))
        ax.legend(loc='lower right', frameon=True, framealpha=0.5)

    if minuit.migrad_ok() and True:
        ax = grid.plot_offsets(adjgrid)
        ax.set_title("Adjusted RMS = {}".format(adjrms))

    P.show()
