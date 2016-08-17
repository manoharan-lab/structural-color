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

def pis_and_taus(nstop, theta):
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

def scatcoeffs(m, x, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
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

def internal_coeffs(m, x, n_max, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
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

def nstop(x):
    # takes size parameter, outputs order to compute to according to
    # Wiscombe, Applied Optics 19, 1505 (1980).
    # 7/7/08: generalize to apply same criterion when x is complex
    return int(np.round_(np.absolute(x+4.05*x**(1./3.)+2)))

def asymmetry_parameter(al, bl):
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

def cross_sections(al, bl):
    '''
    Calculates scattering and extinction cross sections
    given arrays of Mie scattering coefficients an and bn.

    See Bohren & Huffman eqns. 4.61 and 4.62.

    The output omits a scaling prefactor of 2 * pi / k^2 = lambda_medium^2/2/pi.
    '''
    lmax = al.shape[0]

    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)
    cscat = (prefactor * (np.abs(al)**2 + np.abs(bl)**2)).sum()
    cext = (prefactor * np.real(al + bl)).sum()

    # see p. 122
    alts = 2. * (np.arange(lmax) % 2) - 1
    cback = np.abs((prefactor * alts * (al - bl)).sum())**2

    return cscat, cext, cback

# Convenience functions for the most often calculated quantities (form factor,
# efficiencies, asymmetry parameter)

def calc_ang_dist(m, x, angles = None, degrees = True,
                  mie = True, check = True):
    """
    Calculates the angular distribution of light intensity for parallel and
    perpendicular polarization for a sphere.

    Parameters
    ----------
    m: complex particle relative refractive index, n_part/n_med
    x: size parameter, x = ka = 2*\pi*n_med/\lambda * a (sphere radius a)
    angles: ndarray for range of angles. Default is 0-180 degrees.
    degrees: Boolean, set false for angles in radians.
    mie: Boolean, default true, uses RG approximation if false
    check: Boolean, if using Mie solution display scat. efficiencies

    Returns
    -------
    ipar: |S_2|^2
    iperp: |S_1|^2
    (These are the differential scattering X-section*k^2 for polarization
    parallel and perpendicular to scattering plane, respectively.  See
    Bohren & Huffman ch. 3 for details.)
    """

    if angles == None:
        angles = np.linspace(0, 180., 1801)

    if degrees: # convert to radians
        angles = angles * (np.pi / 180.)

    #initialize arrays for holding ipar and iperp
    ipar = np.array([])
    iperp = np.array([])

    def AScatMatrixMie(thet): # amplitude scat matrix from Mie scattering
        angfuncs = pis_and_taus(n_stop, thet)
        pis = angfuncs[0]
        taus = angfuncs[1]
        S1 = (prefactor*(coeffs[0]*pis + coeffs[1]*taus)).sum()
        S2 = (prefactor*(coeffs[0]*taus + coeffs[1]*pis)).sum()
        return np.array([S2,S1])

    def AScatMatrixRG(thet): # amplitude scat matrix from Rayleigh-Gans
        u = 2 * x * np.sin(thet/2.)
        S1 = prefactor * (3./u**3) * (np.sin(u) - u*np.cos(u))
        S2 = prefactor * (3./u**3) * (np.sin(u) - u*np.cos(u)) * np.cos(theta)
        return np.array([S2, S1])

    def MieChecks(): # display and print cross sections
        xsects = cross_sections(coeffs[0], coeffs[1])
        opt = AScatMatrixMie(0).real
        print('Number of terms:')
        print(n_stop)
        print('Scattering, extinction, and backscattering efficiencies:')
        efficiencies = xsects * (2./x**2) * np.array([1,1,0.5]) # save the result
        eff = efficiencies
        print(efficiencies)
        print('Extinction efficiency from optical theorem:')
        print((4./x**2)*opt)
        return efficiencies

# loop over input angles
    if mie:
        # Mie scattering preliminaries
        n_stop = nstop(x)
        coeffs = scatcoeffs(m, x, n_stop)
        n = np.arange(n_stop)+1.
        prefactor = (2*n+1.)/(n*(n+1.))

        for theta in angles:
            asmat = AScatMatrixMie(theta)
            par = np.absolute(asmat[0])**2
            ipar = np.append(ipar, par)
            perp = np.absolute(asmat[1])**2
            iperp = np.append(iperp, perp)

        if check:
            efficiencies = MieChecks()
            g = (4/(x**2 * efficiencies[0])) * asymmetry_parameter(coeffs[0], coeffs[1])

    else:
        prefactor = -1j * (2./3.) * x**3 * np.absolute(m - 1)
        for theta in angles:
            asmat = AScatMatrixRG(theta)
            ipar = np.append(ipar, np.absolute(asmat[0])**2)
            iperp = np.append(iperp, np.absolute(asmat[1])**2)

    return ipar, iperp

@ureg.check(None, None, '[length]', None, None)
def calc_cross_sections(m, x, wavelen_medium, eps1 = DEFAULT_EPS1,
                        eps2 = DEFAULT_EPS2):
    """
    Calculate (dimensional) scattering, absorption, and extinction cross
    sections, and asymmetry parameter for spherically symmetric scatterers.

    Parameters
    ----------
    m: complex relative refractive index
    x: size parameter
    wavelen_medium: structcol.Quantity [length]
        wavelength of incident light *in medium*

    Returns
    -------
    cross_sections : tuple (5)
        Dimensional scattering, absorption, extinction, and backscattering cross
        sections, and <cos \theta> (asymmetry parameter g)

    Notes
    -----
    The radiation pressure cross section C_pr is given by
    C_pr = C_ext - <cos \theta> C_sca.

    The radiation pressure force on a sphere is

    F = (n_med I_0 C_pr) / c

    where I_0 is the incident intensity.  See van de Hulst, p. 14.
    """
    # This is adapted from mie.py in holopy
    # TODO take arrays for m and x to describe a multilayer sphere and return
    # multilayer scattering coefficients

    lmax = nstop(x)
    albl = scatcoeffs(m, x, lmax, eps1=eps1, eps2=eps2)

    cscat, cext, cback =  tuple(wavelen_medium**2 * c/2/np.pi for c in
                                cross_sections(albl[0], albl[1]))

    cabs = cext - cscat # conservation of energy

    asym = wavelen_medium**2 / np.pi / cscat * \
           asymmetry_parameter(albl[0], albl[1])

    return cscat, cext, cabs, cback, asym

def calc_efficiencies(m, x):
    """
    Scattering, extinction, backscattering efficiencies
    """
    n_stop = nstop(x)
    cscat, cext, cback = cross_sections(scatcoeffs(m, x, n_stop)[0],
                                        scatcoeffs(m, x, n_stop)[1])
    qscat = cscat * 2./x**2
    qext = cext * 2./x**2
    qback = cback * 1./x**2
    # in order: scattering, extinction and backscattering efficiency
    return qscat, qext, qback

def calc_g(m, x):
    """
    Asymmetry parameter
    """
    n_stop = nstop(x)
    coeffs = scatcoeffs(m, x, n_stop)
    cscat = cross_sections(coeffs[0], coeffs[1])[0] * 2./x**2
    g = (4/(x**2 * cscat)) * asymmetry_parameter(coeffs[0], coeffs[1])
    return g

# TODO: copy multilayer code from multilayer_sphere_lib.py in holopy and
# integrate with functions for calculating scattering cross sections and form
# factor.
