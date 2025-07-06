# Copyright 2016, Sofia Makgiriadou, Vinothan N. Manoharan, Victoria Hwang,
# Annie Stephenson
#
# This file is part of the structural-color python package.
#
# This package is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This package is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this package. If not, see <http://www.gnu.org/licenses/>.
"""
Routines for simulating and calculating structures, structural parameters, and
structure factors

.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>
.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor :: Annie Stephenson <stephenson@g.harvard.edu>
"""

import numpy as np
# Quantity constructor from pint
from . import Quantity
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.special import jv
import xarray as xr
import os
import structcol as sc

class StructureFactor:
    """Base class for different types of structure factors.

    In general, a structure factor is a function S(q), where q is the
    wavevector. The value of S(q) depends also on the details of the structure
    of the material, but not on its scattering properties. This class
    encapsulates the structural details and uses them in the calculation of
    S(q). We use a class instead of a series of functions because some
    parameters may need to be precomputed and stored, so that they aren't
    computed each time a structure factor is calculated.

    """

    def __init__(self):
        pass

    def __call__(self, ql):
        # ensure ql is one-dimensional DataArray so that we can
        # generate results for all combinations of ql and structural
        # parameters.
        if not isinstance(ql, xr.DataArray):
            ql = xr.DataArray(ql, coords={"ql": ql})

        return self.calculate(ql)

    def calculate(self, ql):
        """
        Virtual method to calculate the structure factor at nondimensional
        wavevector ql

        Parameters
        ----------
        ql : xr.DataArray
            dimensionless product of wavevector (in vacuum) and some length
            scale (usually diameter of particles. Must be specified as a
            labeled DataArray if you call the `structure_factor.calculate(ql)`
            method directly, where `structure_factor` is an instance of the
            StructureFactor class. If you use the convenience method
            `structure_factor(ql)`, you can specify `ql` as a numpy array or
            xarray DataArray.

        Returns:
        -------
        xr.DataArray
            The structure factor as a function of ql and the other structural
            parameters specified in the derived class (for example, volume
            fraction).

        """
        raise NotImplementedError("The StructureFactor class is a base "
                                  "class. Use a derived class such as "
                                  "PercusYevick.")

class Constant(StructureFactor):
    """Structure factor that is always equal to a constant.

    Useful for simulating a very dilute system or testing the effect of the
    form factor alone in calculations.

    Attributes
    ----------
    constant : float
        value of the structure factor

    """

    def __init__(self, constant):
        self.constant = constant

    def calculate(self, ql):
        """Yields constant value irrespective of ql.

        """
        # quick way to ensure we have the right shape
        s = self.constant + 0*ql

        return s.squeeze()

class PercusYevick(StructureFactor):
    """Analytical structure factor for monodisperse hard-sphere liquids.

    The calculation is based on the Ornstein-Zernike equation and Percus-Yevick
    approximation. The analytical formula is derived in [1]_, though this is
    far from the first reference on the subject.

    Attributes
    ----------
    volume_fraction : array-like
        volume fraction of particles or voids in matrix
    ql_cutoff : float (optional)
        ql below which an approximate solution is used

    Notes
    -----
    Might not be accurate for volume fractions above 0.5 (see discussion in
    [2]_). Also, for small q, the analytical solution is numerically unstable,
    so we use a Taylor expansion of the direct correlation function to
    calculate the structure factor at ql values below `ql_cutoff`. The
    derivation of the approximate solution is in the docstring for the
    `approximate_dcf()` method.

    This code is fully vectorized, so you can feed it arrays for
    both ql and phi and it will produce a 2D output (see Examples).

    Examples
    --------
    >>>  ql = np.arange(0.1, 20, 0.01)
    >>>  volume_fraction = np.array([0.15, 0.3, 0.45])
    >>>  structure_factor = PercusYevick(volume_fraction)
    >>>  s_of_q = structure_factor(ql)

    References
    ----------
    [1] Xiao, Ming, Anna B. Stephenson, Andreas Neophytou, Victoria Hwang,
    Dwaipayan Chakrabarti, and Vinothan N. Manoharan. “Investigating the
    Trade-off between Color Saturation and Angle-Independence in Photonic
    Glasses.” Optics Express 29, no. 14 (July 5, 2021): 21212–24.
    https://doi.org/10.1364/OE.425399.

    [2] Scheffold, F., and T. G. Mason. “Scattering from Highly Packed
    Disordered Colloids.” Journal of Physics: Condensed Matter 21, no. 33 (July
    2009): 332102. https://doi.org/10.1088/0953-8984/21/33/332102.

    """

    def __init__(self, volume_fraction, ql_cutoff=0.01):
        # convert volume fraction to DataArray so that calculations will
        # vectorize easily
        phi = np.atleast_1d(volume_fraction)
        self.volume_fraction = xr.DataArray(phi, coords={"volfrac": phi})
        self.ql_cutoff = ql_cutoff

    def calculate(self, ql):
        """Calculates structure factor using the Percus-Yevick analytical
        approximation for hard-sphere liquids.

        """
        phi = self.volume_fraction

        # constants in the direct correlation function (eqs. 3-5 of [1]_)
        alpha = (1 + 2*phi)**2 / (1 - phi)**4
        beta = -6*phi*(1 + phi/2)**2 / (1 - phi)**4
        gamma = 0.5*phi*(1 + 2*phi)**2 / (1 - phi)**4

        # Fourier transform of the direct correlation function multiplied by
        # the number density (eq. 6 of [1]_)
        rho_c = ((-24*phi/ql**6)
                 * (ql*np.sin(ql) * (ql**2 * (alpha + 2*beta + 4*gamma)
                                     - 24*gamma)
                    - 2*(ql**2)*beta
                    - np.cos(ql) * (ql**4 * (alpha + beta + gamma)
                                    - 2*ql**2 * (beta + 6*gamma) + 24*gamma)
                    + 24*gamma))

        # the above expression is not numerically stable near low ql.  We
        # replace the values with a small-ql approximation
        rho_c_low = self.approximate_dcf(ql, phi, alpha, beta, gamma)
        rho_c = xr.where(rho_c.ql < self.ql_cutoff, rho_c_low, rho_c)

        # Structure factor at ql (eq. 1 of [1]_)
        s = 1.0/(1-rho_c)

        # squeeze but keeps coordinates as scalar values so that the returned
        # DataArray still has the value of the volume fraction recorded
        return s.squeeze()

    def approximate_dcf(self, ql, phi, alpha, beta, gamma):
        r"""Calculates an approximation to the direct correlation function
        valid for small ql (ql << 1), where the analytical solution may be
        numerically unstable.

        Notes
        -----

        Derivation is below.  See also SasView's hardsphere.c, which makes a
        similar approximation [1]_.

        Let :math:`x = ql`.

        Then the dimensionless direct correlation function is

        .. math::

            \rho c(x) = -\frac{24\phi}{x^6} \Big\{&x\sin(x)\left[x^2(\alpha +
            2\beta + 4\gamma) - 24\gamma\right] - 2x^2\beta \\
            &- \cos(x)\left[x^4(\alpha + \beta + \gamma)
             - 2x^2(\beta + 6\gamma) + 24\gamma\right] + 24\gamma\Big\}

        We expand the sine and cosine to a sufficient number of terms such that
        we capture terms of up to :math:`x^2`:

        .. math::

            \rho c(x) = -\frac{24\phi}{x^6} \Bigg\{&x\left(x-\frac{x^3}{6}
             +\frac{x^5}{120}-\frac{x^7}{5040}\right)\left[x^2(\alpha + 2\beta
             + 4\gamma) - 24\gamma\right] - 2x^2\beta \\
            &- \left(1-\frac{x^2}{2}+\frac{x^4}{24}-\frac{x^6}{720}+
             \frac{x^8}{40320}\right)\left[x^4(\alpha + \beta + \gamma)
             - 2x^2(\beta + 6\gamma) + 24\gamma\right] + 24\gamma\Bigg\}

        Distributing:

        .. math::

            \rho c(x) = \phi \Bigg\{&\left(-24x^{-4}+4x^{-2}-\frac{1}{5}
             +\frac{x^2}{210}\right)\left[x^2(\alpha + 2\beta + 4\gamma)
             - 24\gamma\right] - 2x^{-4}\beta \\
            &+ \left(24x^{-6}-12x^{-4}+x^{-2}-\frac{1}{30}
             +\frac{1}{1680}x^2\right)\left[x^4(\alpha + \beta + \gamma)
             - 2x^2(\beta + 6\gamma) + 24\gamma\right] + 24x^{-6}\gamma\Bigg\}

        We need only keep the terms of order :math:`x^0` and higher. Terms with
        negative powers of :math:`x` should all cancel one another (if they
        didn't, the direct correlation function would go to :math:`-\infty` and
        :math:`S(q)` would go to 0 at small :math:`q` instead of a finite
        value, which is required by thermodynamics). After dropping terms with
        negative powers of $x$, we have

        .. math::

            \rho c(x) = \phi \Bigg\{&\left[(4\alpha + 8\beta + 16\gamma)
            + \frac{24\gamma}{5}\right] \\
            &+ x^2\left[\left(-\frac{\alpha}{5} - \frac{2\beta}{5}
            - \frac{4\gamma}{5}\right) - \frac{4\gamma}{35}\right] \\
            &+ \left[-\left(12\alpha + 12\beta + 12\gamma)
             - (2\beta + 12\gamma) - \frac{4\gamma}{5}\right)\right]  \\
            &+ x^2\left[(\alpha + \beta + \gamma) + \left(\frac{\beta}{15}
             + \frac{2\gamma}{5}\right) + \frac{\gamma}{70}\right]\Bigg\}

        Simplifying, we arrive at

        .. math::

            \rho c(x) = \phi \left[\left(-8\alpha - 6\beta - 4\gamma\right)
            + x^2\left(\frac{4\alpha}{5} + \frac{2\beta}{3}
                       + \frac{\gamma}{2}\right)\right]

        References
        ----------

        [1] https://www.sasview.org/docs/user/models/hardsphere.html

        """
        return phi * ((-8*alpha -6*beta -4*gamma)
                      + ql**2 *(4*alpha/5 + 2*beta/3 + gamma/2))

class Paracrystal(StructureFactor):
    """Calculate structure factor of a structure characterized by disorder of
    the second kind.

    Disorder of the second time is defined in Guinier [1]. This type of
    structure is referred to as paracrystalline by Hoseman [2]. See also [3]
    for concise description.

    Attributes
    ----------
    volume_fraction : array-like
        volume fraction of particles or voids in matrix
    sigma: float
        The standard deviation of a Gaussian representing the distribution of
        particle/void spacings in the structure. Sigma has implied units of
        particle diamter squared. A larger sigma will give more broad peaks,
        and a smaller sigma more sharp peaks.

    Returns:
    -------
    1D numpy array:
        The structure factor as a function of ql.

    Notes
    -----
    Not fully tested.

    References
    ----------
    [1] Guinier, A (1963). X-Ray Diffraction. San Francisco and London: WH
    Freeman.

    [2] Lindenmeyer, PH; Hosemann, R (1963). "Application of the Theory of
    Paracrystals to the Crystal Structure Analysis of Polyacrylonitrile".
    J. Applied Physics. 34: 42

    [3] https://en.wikipedia.org/wiki/Structure_factor#Disorder_of_the_second_kind

    """

    def __init__(self, volume_fraction, sigma=0.15):
        # convert arguments to DataArray so that calculations will vectorize
        # easily
        phi = np.atleast_1d(volume_fraction)
        self.volume_fraction = xr.DataArray(phi, coords={"volfrac": phi})
        sigma = np.atleast_1d(sigma)
        self.sigma = xr.DataArray(sigma, coords={"sigma": sigma})

    def calculate(self, ql):
        """Calculates paracrystalline structure factor.

        """
        phi = self.volume_fraction

        r = np.exp(-(ql*phi**(-1/3) * self.sigma)**2/2)
        s = (1 - r**2) / (1 + r**2 - 2*r*np.cos(ql*phi**(-1/3)))

        return s.squeeze()

class Polydisperse(StructureFactor):
    """Object to calculate polydisperse structure factor for a monospecies (one
    mean particle size) or a bispecies (two different mean particle sizes)
    system, each with its own polydispersity. Uses formulae from [1]_. The size
    distribution is assumed to be the Schulz distribution, which tends to
    Gaussian when the polydispersity goes to zero, and skews to larger sizes
    when the polydispersity becomes large.

    Attributes
    ----------
    volume_fraction : array-like
        volume fraction of all the particles or voids in matrix
    diameters: array of structcol.Quantity [length]
        mean diameters of each species of particles (can be one for a
        monospecies or two for bispecies).
    concentrations:  array-like
        number fraction of each species. For example, a system composed
        of 90 A particles and 10 B particles would have c = [0.9, 0.1].
    pdi: array of float
        polydispersity index of each species

    References
    ----------
    [1] M. Ginoza and M. Yasutomi, "Measurable Structure Factor of a
    Multi-Species Polydisperse Percus-Yevick Fluid with Schulz Distributed
    Diameters", J. Phys. Soc. Japan, 68, 7, 2292-2297 (1999).

    """

    def __init__(self, volume_fraction, diameters, concentrations, pdi):
        # this structure factor doesn't broadcast
        phi = np.atleast_1d(volume_fraction)
        self.volume_fraction = phi

        c = np.atleast_1d(concentrations)
        self.concentrations = c

        d = np.atleast_1d(diameters)
        self.diameters = d

        self.pdi = np.atleast_1d(pdi).astype(float)
        if isinstance(self.pdi, Quantity):
            self.pdi = self.pdi.magnitude
        # if the pdi is zero, assume it's very small (we get the same results)
        # because otherwise we get a divide by zero error
        self.pdi[self.pdi < 1e-5] = 1e-5

    def __call__(self, q):
        return self.calculate(q)

    def fm(self, x, t, tm, m):
        """Evaluates the function in eq. 25 of [1]_, which is used to integrate
        the Schulz distribution. Here x is "a" in the reference, which is the
        width parameter of the distribution. t is t(tau), where tau is an index
        over species, and t is a nonnegative integer.  tm is the normalized
        moment of the distribution, defined by eq. 24.

        """

        if isinstance(x, Quantity):
            x = x.to('').magnitude
        if isinstance(t, Quantity):
            t = t.to('').magnitude
        if isinstance(tm, Quantity):
            tm = tm.to('').magnitude
        t = np.reshape(t, (len(np.atleast_1d(t)),1))
        tm = np.reshape(tm, (len(tm),1))
        return (tm * (1 + x/(t+1))**(-(t+1+m)))

    def tm(self, m, t):
        """Evaluates the moments in eq. 24 of [1]_.  m is an integer.
        """
        t = np.reshape(t, (len(np.atleast_1d(t)),1))
        num_array = np.arange(m, 0, -1) + t
        prod = np.prod(num_array, axis=1).reshape((len(t), 1))
        return (prod / (t + 1)**m)

    def calculate(self, q):
        """Calculate the measurable polydisperse structure using the approach
        in [1]_
        """

        c = self.concentrations
        diameters = self.diameters
        phi = self.volume_fraction

        Dsigma = self.pdi**2
        Delta = 1 - phi
        t = 1/Dsigma - 1

        t0 = self.tm(0, t)
        t1 = self.tm(1, t)
        # from eq. 24 of reference and simplifying
        t2 = Dsigma + 1
        # from eq. 24 and also on page 2295
        t3 = (Dsigma + 1) * (2*Dsigma + 1)

        # If monospecies, no need to calculate individual species parameters.
        # concentration c should always be a 2-element array because
        # polydisperse calculations assume the format of a bispecies particle
        # mixture, so if either element in c is 0, we assume the form factor is
        # monospecies We include the second monospecies test in case the user
        # enters a 1d concentration, even though the docstring advises that
        # concentration should have two elements.
        if np.any(c == 0) or (len(np.atleast_1d(c)) == 1):
            if len(np.atleast_1d(c)) == 1:
                t3_1d = t3
                diam_1d = diameters
            else:
                ind0 = np.where(c != 0)[0]
                t3_1d = t3[ind0]
                diam_1d = diameters[ind0]
            rho = 6*phi/(t3_1d*np.pi*diam_1d**3)
        else:
            phi_ratio = 1 / (c[0]/c[1] * (diameters[0] / diameters[1]) ** 3 *
                             t3[0] / t3[1] + 1)
            phi_tau1 = phi_ratio * phi
            phi_tau0 = phi - phi_tau1

            rho_tau0 = 6*phi_tau0/(t3[0]*np.pi*diameters[0]**3)
            rho_tau1 = 6*phi_tau1/(t3[1]*np.pi*diameters[1]**3)
            rho = rho_tau0 + rho_tau1

        # this is the "effective" mean interparticle spacing
        sigma0 = (6*phi / (np.pi*rho))**(1/3)

        #q = qd / sigma0

        t2 = np.reshape(t2, (len(np.atleast_1d(t2)), 1))
        c = np.reshape(c, (len(np.atleast_1d(c)), 1))
        diameters = np.reshape(diameters, (len(np.atleast_1d(diameters)), 1))

        if hasattr(q, 'shape'):
            q_shape = q.shape
        else:
            q_shape = np.array([])
        if len(q_shape) == 2:
            q = Quantity(np.ndarray.flatten(q.magnitude), q.units)  # added
        s = 1j*q
        x = s*diameters
        F0 = rho
        zeta2 = rho * sigma0**2

        f0 = self.fm(x,t,t0,0)
        f1 = self.fm(x,t,t1,1)
        f2 = self.fm(x,t,t2,2)
        f0_inv = self.fm(-x,t,t0,0)
        f1_inv = self.fm(-x,t,t1,1)
        f2_inv = self.fm(-x,t,t2,2)

        # from eqs 29a-29d
        fa = 1/x**3 * (1 - x/2 - f0 - x/2 * f1)
        fb = 1/x**3 * (1 - x/2 * t2 - f1 - x/2 * f2)
        fc = 1/x**2 * (1 - x - f0)
        fd = 1/x**2 * (1 - x*t2 - f1)

        # eqs 26a, 26b
        Ialpha1 = 24/s**3 * np.sum(c * F0 * (-1/2*(1-f0) + x/4 * (1 + f1)),
                                   axis=0)
        Ialpha2 = 24/s**3 * np.sum(c * F0 * (-diameters/2 * (1-f1) +
                                   s*diameters**2/4 * (t2 + f2)), axis=0)

        Iw1 = 2*np.pi*rho/(Delta*s**3) * (Ialpha1 + s/2*Ialpha2)
        Iw2 = (np.pi*rho/(Delta*s**2) * (1 + np.pi*zeta2/(Delta*s))*Ialpha1 +
               np.pi**2*zeta2*rho/(2*Delta**2*s**2) * Ialpha2)

        F11 = np.sum(c*2*np.pi*rho*diameters**3/Delta * fa, axis=0)
        F12 = np.sum(c/diameters * ((np.pi/Delta)**2 * rho * zeta2
                                    * diameters**4*fa
                                    + np.pi*rho*diameters**3/Delta * fc),
                     axis=0)
        F21 = np.sum(c * diameters * 2*np.pi*rho*diameters**3/Delta * fb,
                     axis=0)
        F22 = np.sum(c * ((np.pi/Delta)**2 *rho*zeta2*diameters**4*fb +
                     np.pi*rho*diameters**3/Delta * fd), axis=0)

        FF11 = 1 - F11
        FF12 = -F12
        FF21 = -F21
        FF22 = 1 - F22

        G11 = FF22 / (FF11 * FF22 - FF12 * FF21)
        G12 = -FF12 / (FF11 * FF22 - FF12 * FF21)
        G21 = -FF21 / (FF11 * FF22 - FF12 * FF21)
        G22 = FF11 / (FF11 * FF22 - FF12 * FF21)

        I0 = -9/2*(2/s)**6 * np.sum(c * F0**2 * (1-1/2*(f0_inv + f0) +
                                    x/2 *(f1_inv - f1) -
                                    (s**2*diameters**2)/8
                                                 * (f2_inv + f2 + 2*t2)),
                                    axis=0)

        term1 = Iw1 * G11 * Ialpha1 / I0
        term2 = Iw1 * G12 * Ialpha2 / I0
        term3 = Iw2 * G21 * Ialpha1 / I0
        term4 = Iw2 * G22 * Ialpha2 / I0

        h2 = (term1 + term2 + term3 + term4).real

        SM = 1 - 2*h2
        SM[SM<0] = 0
        if len(q_shape)==2:
            SM = np.reshape(SM,q_shape)
        return(SM)

class Interpolated(StructureFactor):
    """Object to calculate an interpolated structure factor using data

    Attributes
    ----------
    interpolation_func : Function
        interpolation function generated by `scipy.interp1d`
    data : xr.DataArray
        data used to generate interpolation function
    """

    def __init__(self, s_data, ql_data):
        """Construct interpolation for a structure factor from data

        Parameters
        ----------
        s_data: 1D numpy array
            structure factor values from data
        ql_data: 1D numpy array
            ql values from data
        """
        self.data = xr.DataArray(s_data, coords={"ql": ql_data})
        func = interp1d(ql_data, s_data, kind = 'linear', bounds_error=False,
                        fill_value=s_data[0])
        self.interpolation_func = func

    def calculate(self, ql):
        """Calculates paracrystalline structure factor.

        """
        return self.interpolation_func(ql).squeeze()

def field_phase_data(qd, filename='spf.dat'):
    s_file = os.path.join(os.getcwd(),filename)
    s_phase_data=np.loadtxt(s_file)
    qd_data = s_phase_data[:,0]
    s_phase = s_phase_data[:,1]
    s_phase_func = interp1d(qd_data, s_phase, kind = 'linear',
                            bounds_error=False,
                            fill_value=s_phase_data[0,1])
    return s_phase_func(qd)

def phase_factor(qd, phi, n=1000):
    # define r/d
    r_d = np.linspace(0,10, n)

    # calculate g
    g = radial_dist_py(phi, x = r_d)
    integral = np.zeros(qd.shape)
    rho = 3.0 * phi / (4.0 * np.pi) # dimensionless rho*sigma**3

    # calculate the integral for each qd
    for i in range(qd.shape[0]):
        for j in range(qd.shape[1]):
            bessel = rho*4*np.pi*r_d**2*np.pi*jv(0, qd[i,j]*r_d)
            integral[i,j] = np.trapz(bessel*g, x=r_d)

    return integral


def field_phase_py(qd, phi, n=10000, r_d=np.arange(1,5,0.005), rng=None):
    '''
    Calculate the phase shift contribution based on the radial distribution
    function calculated using the Percus-Yevick approximation

    Parameters:
    ----------
    qd: 1D numpy array
        dimensionless quantity q times diameter
    phi: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix
    n: float
        number of samples of g(r)
    r_d: 1D numpy array
        range of radial positions normalized by particle diameter.
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package

    Returns:
    --------
    field_s: 1D numpy array
        phase shift contributions based on the structure
    '''
    if rng is None:
        rng = sc.rng

    # calculate radial distribution function up to r/R= 5
    #g_file = os.path.join(os.getcwd(),'g_4.csv')
    #df=pd.read_csv(g_file, sep=',',header=None)
    #r_d = np.array(df[0])
    #g = np.array(df[1])
    g = radial_dist_py(phi, x = r_d)

    # sample the g of r probability distribution
    r_samp = rng.choice(r_d, n, p = g/np.sum(g))

    # calculate the field term
    field_s = np.zeros(qd.shape, dtype='complex')
    for i in range(qd.shape[0]):
        for j in range(qd.shape[1]):
            field_s[i,j] = 1/n*np.sum(np.exp(1j*qd[i,j]*r_samp))

    return field_s

def radial_dist_py(phi, x=np.arange(1,5,0.005)):
    '''
    Calculate the radial distribution function for hard spheres using the
    Percus-Yevick approximation.

    This function and its helper functions is based on the code found here:
    https://github.com/FTurci/hard-spheres-utilities/blob/master/Percus-Yevick.py
    This method for calculating g(r) is described in the SI of:
    J. W. E. Drewitt, F. Turci, B. J. Heinen, S. G. Macleod, F. Qin, A. K.
    Kleppe, and O. T. Lord. Phys. Rev. Lett. 124

    Parameters:
    -----------
    phi: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix
    x: 1D numpy array
       dimensionless value defined as position over particle diameter (r/d)

    Returns:
    --------
    g_fcn(x): 1D numpy array
       The radial distribution function calculated at the specified x
       values.
    '''
    # number density
    if isinstance(phi,Quantity):
        phi = phi.magnitude
    rho=6./np.pi*phi

    # get the direct correlation function c(r) from the analytic Percus-Yevick
    # solution, vectorizing the function
    c=np.vectorize(cc)

    # space discretization
    dr=0.005
    r=np.arange(1,1024*2+1,1 )*dr

    # reciprocal space discretization (highest available frequency)
    dk=1/r[-1]
    k=np.arange(1,1024*2+1,1 )*dk

    # direct correlation function c(r)
    c_direct=c(r,phi)

    # calculate the Fourier transform
    ft_c_direct=spherical_FT(c_direct, k,r,dr)

    # using the Ornstein-Zernike equation, calculate the structure factor
    ft_h=ft_c_direct/(1.-rho*ft_c_direct)

    # inverse Fourier transform
    h=inverse_spherical_FT(ft_h, k,r,dk)

    # radial distribution function
    gg=h+1

    # clean the r<1 region
    g=np.zeros(len(gg))
    g[r>=1]=gg[r>=1]

    # make g function from interpolation
    g_fcn=InterpolatedUnivariateSpline(r, g)

    return g_fcn(x)

def spherical_FT(f,k,r,dr):
    '''
    Spherical Fourier Transform (using the liquid isotropicity)
    '''
    ft=np.zeros(len(k))
    for i in range(len(k)):
        ft[i]=4.*np.pi*np.sum(r*np.sin(k[i]*r)*f*dr)/k[i]
    return ft

def inverse_spherical_FT(ff,k,r,dk):
    '''
    Inverse spherical Fourier Transform (using the liquid isotropicity)
    '''
    ift=np.zeros(len(r))
    for i in range(len(r)):
        ift[i]=np.sum(k*np.sin(k*r[i])*ff*dk)/r[i]/(2*np.pi**2)
    return ift

# functions to calcualte direct correlation function
# from Percus-Yevick. See D. Henderson "Condensed Matter Physics" 2009, Vol.
# 12, No. 2, pp. 127-135
# or M. S Wertheim "Exact Solutions of the Percus-Yevick Integral for Hard
# Spheres" PRL. Vol. 10, No. 8, 1963
def c0(eta):
    return -(1.+2.*eta)**2/(1.-eta)**4
def c1(eta):
    return 6.*eta*(1.+eta*0.5)**2/(1.-eta)**4
def c3(eta):
    return eta*0.5*c0(eta)
def cc(r,eta):
    if r>1:
        return 0
    else:
        return c0(eta)+c1(eta)*r +c3(eta)*r**3
