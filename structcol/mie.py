# Copyright 2011-2013, 2016 Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, and Sofia Magkiriadou
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
Functions for Mie scattering calculations.

Notes
-----
Based on miescatlib.py in holopy. Also includes some functions from Jerome's
old miescat_1d.py library.

Numerical stability not guaranteed for large nstop, so be careful when
calculating very large size parameters. A better-tested (and faster) version of
this code is in the HoloPy package (http://manoharan.seas.harvard.edu/holopy).

References
----------
[1] Bohren, C. F. and Huffman, D. R. ""Absorption and Scattering of Light by
Small Particles" (1983)
[2] Wiscombe, W. J. "Improved Mie Scattering Algorithms" Applied Optics 19, no.
9 (1980): 1505. doi:10.1364/AO.19.001505

.. moduleauthor :: Jerome Fung <jerome.fung@gmail.com>
.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>
"""
import numpy as np
from scipy.special import lpn, riccati_jn, riccati_yn, sph_jn, sph_yn
from . import mie_specfuncs, ureg, Quantity
from .mie_specfuncs import DEFAULT_EPS1, DEFAULT_EPS2   # default tolerances

# User-facing functions for the most often calculated quantities (form factor,
# efficiencies, asymmetry parameter)

@ureg.check('[]', '[]', '[]') # all arguments should be dimensionless
def calc_ang_dist(m, x, angles, mie = True, check = False):
    """
    Calculates the angular distribution of light intensity for parallel and
    perpendicular polarization for a sphere.

    Parameters
    ----------
    m: complex particle relative refractive index, n_part/n_med
    x: size parameter, x = ka = 2*\pi*n_med/\lambda * a (sphere radius a)
    angles: ndarray(structcol.Quantity [dimensionless])
        array of angles. Must be entered as a Quantity to allow specifying
        units (degrees or radians) explicitly
    mie: Boolean (optional)
        if true (default) does full Mie calculation; if false, )uses RG
        approximation
    check: Boolean (optional)
        if true, outputs scattering efficiencies

    Returns
    -------
    ipar: |S_2|^2
    iperp: |S_1|^2
    (These are the differential scattering X-section*k^2 for polarization
    parallel and perpendicular to scattering plane, respectively.  See
    Bohren & Huffman ch. 3 for details.)
    """

    # convert to radians from whatever units the user specifies
    angles = angles.to('rad').magnitude
    #initialize arrays for holding ipar and iperp
    ipar = np.array([])
    iperp = np.array([])

    # TODO vectorize and remove loop
    # loop over input angles
    if mie:
        # Mie scattering preliminaries
        nstop = _nstop(x)
        coeffs = _scatcoeffs(m, x, nstop)
        n = np.arange(nstop)+1.
        prefactor = (2*n+1.)/(n*(n+1.))

        for theta in angles:
            asmat = _amplitude_scattering_matrix(nstop, prefactor, coeffs, theta)
            par = np.absolute(asmat[0])**2
            ipar = np.append(ipar, par)
            perp = np.absolute(asmat[1])**2
            iperp = np.append(iperp, perp)

        if check:
            opt = _amplitude_scattering_matrix(nstop, prefactor, coeffs, 0).real
            qscat, qext, qback = calc_efficiencies(m, x)
            print('Number of terms:')
            print(nstop)
            print('Scattering, extinction, and backscattering efficiencies:')
            print(qscat, qext, qback)
            print('Extinction efficiency from optical theorem:')
            print((4./x**2)*opt)
            print('Asymmetry parameter')
            print(calc_g(m, x))

    else:
        prefactor = -1j * (2./3.) * x**3 * np.absolute(m - 1)
        for theta in angles:
            asmat = _amplitude_scattering_matrix_RG(prefactor, x, theta)
            ipar = np.append(ipar, np.absolute(asmat[0])**2)
            iperp = np.append(iperp, np.absolute(asmat[1])**2)

    return ipar, iperp

@ureg.check(None, None, '[length]', None, None)
def calc_cross_sections(m, x, wavelen_media, eps1 = DEFAULT_EPS1,
                        eps2 = DEFAULT_EPS2):
    """
    Calculate (dimensional) scattering, absorption, and extinction cross
    sections, and asymmetry parameter for spherically symmetric scatterers.

    Parameters
    ----------
    m: complex relative refractive index
    x: size parameter
    wavelen_media: structcol.Quantity [length]
        wavelength of incident light *in media* (usually this would be the
        wavelength in the effective index of the particle-matrix composite)

    Returns
    -------
    cross_sections : tuple (5)
        Dimensional scattering, absorption, extinction, and backscattering cross
        sections, and <cos \theta> (asymmetry parameter g)

    Notes
    -----
    The backscattering cross-section is 1/(4*pi) times the radar backscattering
    cross-section; that is, it corresponds to the differential scattering
    cross-section in the backscattering direction.  See B&H 4.6.

    The radiation pressure cross section C_pr is given by
    C_pr = C_ext - <cos \theta> C_sca.

    The radiation pressure force on a sphere is

    F = (n_med I_0 C_pr) / c

    where I_0 is the incident intensity.  See van de Hulst, p. 14.
    """
    # This is adapted from mie.py in holopy
    # TODO take arrays for m and x to describe a multilayer sphere and return
    # multilayer scattering coefficients

    lmax = _nstop(x)
    albl = _scatcoeffs(m, x, lmax, eps1=eps1, eps2=eps2)

    cscat, cext, cback =  tuple(wavelen_media**2 * c/2/np.pi for c in
                                _cross_sections(albl[0], albl[1]))

    cabs = cext - cscat # conservation of energy

    asym = wavelen_media**2 / np.pi / cscat * \
           _asymmetry_parameter(albl[0], albl[1])

    return cscat, cext, cabs, cback, asym

def calc_efficiencies(m, x):
    """
    Scattering, extinction, backscattering efficiencies

    Note that the backscattering efficiency is 1/(4*pi) times the radar
    backscattering efficiency; that is, it corresponds to the differential
    scattering cross-section in the backscattering direction, divided by the
    geometrical cross-section
    """
    nstop = _nstop(x)
    cscat, cext, cback = _cross_sections(_scatcoeffs(m, x, nstop)[0],
                                         _scatcoeffs(m, x, nstop)[1])
    qscat = cscat * 2./x**2
    qext = cext * 2./x**2
    qback = cback * 1./x**2
    # in order: scattering, extinction and backscattering efficiency
    return qscat, qext, qback

def calc_g(m, x):
    """
    Asymmetry parameter
    """
    nstop = _nstop(x)
    coeffs = _scatcoeffs(m, x, nstop)
    cscat = _cross_sections(coeffs[0], coeffs[1])[0] * 2./x**2
    g = (4/(x**2 * cscat)) * _asymmetry_parameter(coeffs[0], coeffs[1])
    return g

@ureg.check(None, None, '[length]', ('[]','[]', '[]'))
def calc_integrated_cross_section(m, x, wavelen_media, theta_range):
    """
    Calculate (dimensional) integrated cross section using quadrature

    Parameters
    ----------
    m: complex relative refractive index
    x: size parameter
    wavelen_media: structcol.Quantity [length]
        wavelength of incident light *in media*
    theta_range: tuple of structcol.Quantity [dimensionless]
        first two elements specify the range of polar angles over which to
        integrate the scattering. Last element specifies the number of angles.

    Returns
    -------
    cross_section : float
        Dimensional integrated cross-section
    """
    theta_min = theta_range[0].to('rad').magnitude
    theta_max = theta_range[1].to('rad').magnitude
    angles = Quantity(np.linspace(theta_min, theta_max, theta_range[2]), 'rad')
    form_factor = calc_ang_dist(m, x, angles)

    integrand_par = form_factor[0]*np.sin(angles)
    integrand_perp = form_factor[1]*np.sin(angles)

    # np.trapz does not preserve units, so need to state explicitly that we are
    # in the same units as the integrand
    integral_par = 2 * np.pi * np.trapz(integrand_par, x=angles)*integrand_par.units
    integral_perp = 2 * np.pi * np.trapz(integrand_perp, x=angles)*integrand_perp.units

    # multiply by 1/k**2 to get the dimensional value
    return wavelen_media**2/4/np.pi/np.pi * (integral_par + integral_perp)/2.0

# Mie functions used internally

def _lpn_vectorized(n, z):
    # scipy.special.lpn (Legendre polynomials and derivatives) is not a ufunc,
    # so cannot handle array arguments for n and z. It does, however, calculate
    # all orders of the polynomial and derivatives up to order n. So we pick
    # the max value of the array n that is passed to the function.
    nmax = np.max(n)
    z = np.atleast_1d(z)
    # now vectorize over z; this is in general slow because it runs a python loop.
    # Need to figure out a better way to do this; one possibility is to use
    # scipy.special.sph_harm, which is a ufunc, and use special cases to recover
    # the legendre polynomials and derivatives
    return np.array([lpn(nmax, z) for z in z])

def _pis_and_taus(nstop, theta):
    """
    Calculate pi_n and tau angular functions at theta out to order n by up
    recursion.

    Parameters
    ----------
    nstop: maximum order
    theta: angle

    Returns
    -------
    pis, taus (order 1 to n)

    Notes
    -----
    Pure python version of mieangfuncs.pisandtaus in holopy.  See B/H eqn 4.46,
    Wiscombe eqns 3-4.
    """
    mu = np.cos(theta)
    # returns P_n and derivative, as a list of 2 arrays, the second
    # being the derivative
    legendre0 = lpn(nstop, mu)
    # use definition in terms of d/dmu (P_n(mu)), where mu = cos theta
    pis = (legendre0[1])[0:nstop+1]
    pishift = np.concatenate((np.zeros(1), pis))[0:nstop+1]
    n = np.arange(nstop+1)
    mus = mu*np.ones(nstop+1)
    taus = n*pis*mus - (n+1)*pishift
    return np.array([pis[1:nstop+1], taus[1:nstop+1]])

def _scatcoeffs(m, x, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    # see B/H eqn 4.88
    # implement criterion used by BHMIE plus a couple more orders to be safe
    # nmx = np.array([nstop, np.round_(np.absolute(m*x))]).max() + 20
    # Dnmx = mie_specfuncs.log_der_1(m*x, nmx, nstop)
    # above replaced with Lentz algorithm
    Dnmx = mie_specfuncs.dn_1_down(m * x, nstop + 1, nstop,
                                   mie_specfuncs.lentz_dn1(m * x, nstop + 1,
                                                           eps1, eps2))
    n = np.arange(nstop+1)
    psi, xi = mie_specfuncs.riccati_psi_xi(x, nstop)
    psishift = np.concatenate((np.zeros(1), psi))[0:nstop+1]
    xishift = np.concatenate((np.zeros(1), xi))[0:nstop+1]
    an = ( (Dnmx/m + n/x)*psi - psishift ) / ( (Dnmx/m + n/x)*xi - xishift )
    bn = ( (Dnmx*m + n/x)*psi - psishift ) / ( (Dnmx*m + n/x)*xi - xishift )
    return np.array([an[1:nstop+1], bn[1:nstop+1]]) # output begins at n=1

def _internal_coeffs(m, x, n_max, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate internal Mie coefficients c_n and d_n given
    relative index, size parameter, and maximum order of expansion.

    Follow Bohren & Huffman's convention. Note that van de Hulst and Kerker
    have different conventions (labeling of c_n and d_n and factors of m)
    for their internal coefficients.
    '''
    ratio = mie_specfuncs.R_psi(x, m * x, n_max, eps1, eps2)
    D1x, D3x = mie_specfuncs.log_der_13(x, n_max, eps1, eps2)
    D1mx = mie_specfuncs.dn_1_down(m * x, n_max + 1, n_max,
                                   mie_specfuncs.lentz_dn1(m * x, n_max + 1,
                                                           eps1, eps2))
    cl = m * ratio * (D3x - D1x) / (D3x - m * D1mx)
    dl = m * ratio * (D3x - D1x) / (m * D3x - D1mx)
    return array([cl[1:], dl[1:]]) # start from l = 1

def _nstop(x):
    # takes size parameter, outputs order to compute to according to
    # Wiscombe, Applied Optics 19, 1505 (1980).
    # 7/7/08: generalize to apply same criterion when x is complex
    return (np.round_(np.absolute(x+4.05*x**(1./3.)+2))).astype('int')

def _asymmetry_parameter(al, bl):
    '''
    Inputs: an, bn coefficient arrays from Mie solution

    See discussion in Bohren & Huffman p. 120.
    The output of this function omits the prefactor of 4/(x^2 Q_sca).
    '''
    lmax = al.shape[0]
    l = np.arange(lmax) + 1
    selfterm = (l[:-1] * (l[:-1] + 2.) / (l[:-1] + 1.) *
                np.real(al[:-1] * np.conj(al[1:]) +
                        bl[:-1] * np.conj(bl[1:]))).sum()
    crossterm = ((2. * l + 1.)/(l * (l + 1)) *
                 np.real(al * np.conj(bl))).sum()
    return selfterm + crossterm

def _cross_sections(al, bl):
    '''
    Calculates scattering and extinction cross sections
    given arrays of Mie scattering coefficients al and bl.

    See Bohren & Huffman eqns. 4.61 and 4.62.

    The output omits a scaling prefactor of 2 * pi / k^2 = lambda_media^2/2/pi.
    '''
    lmax = al.shape[0]

    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)
    cscat = (prefactor * (np.abs(al)**2 + np.abs(bl)**2)).sum()
    cext = (prefactor * np.real(al + bl)).sum()

    # see p. 122 and discussion in that section. The formula on p. 122
    # calculates the backscattering cross-section according to the traditional
    # definition, which includes a factor of 4*pi for historical reasons. We
    # jettison the factor of 4*pi to get values that correspond to the
    # differential scattering cross-section in the backscattering direction.
    alts = 2. * (np.arange(lmax) % 2) - 1
    cback = (np.abs((prefactor * alts * (al - bl)).sum())**2)/4.0/np.pi

    return cscat, cext, cback

def _amplitude_scattering_matrix(n_stop, prefactor, coeffs, theta):
    # amplitude scattering matrix from Mie coefficients
    angfuncs = _pis_and_taus(n_stop, theta)
    pis = angfuncs[0]
    taus = angfuncs[1]
    S1 = (prefactor*(coeffs[0]*pis + coeffs[1]*taus)).sum()
    S2 = (prefactor*(coeffs[0]*taus + coeffs[1]*pis)).sum()
    return np.array([S2,S1])

def _amplitude_scattering_matrix_RG(prefactor, x, theta):
    # amplitude scattering matrix from Rayleigh-Gans approximation
    u = 2 * x * np.sin(thet/2.)
    S1 = prefactor * (3./u**3) * (np.sin(u) - u*np.cos(u))
    S2 = prefactor * (3./u**3) * (np.sin(u) - u*np.cos(u)) * np.cos(theta)
    return np.array([S2, S1])


# TODO: copy multilayer code from multilayer_sphere_lib.py in holopy and
# integrate with functions for calculating scattering cross sections and form
# factor.
