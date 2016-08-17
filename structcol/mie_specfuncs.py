# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, and Anna Wang
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
Compute special functions needed for the computation of scattering coefficients
in the Lorenz-Mie scattering solution and related problems such as layered
spheres.

Notes
-----
Forked from holopy on 16 Aug 2016; added python version of lentz_dn1 and
changed log_der_1 to dn_1_down, following the code in mieangfuncs.f90. These
changes implement the Lentz continued fraction algorithm.

These functions are not to be used for calculations at each field point.
Rather, they should be used once for the calculation of scattering
coefficients, which can then be used for field calculations.

References
----------
[1] D. W. Mackowski, R. A. Altenkirch, and M. P. Menguc, "Internal absorption
cross sections in a stratified sphere," Applied Optics 29, 1551-1559, (1990).

[2] Yang, "Improved recursive algorithm for light scattering by a multilayered
sphere," Applied Optics 42, 1710-1720, (1993).

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
import numpy as np
from numpy import array, sin, cos, zeros, arange, real, imag, exp

import scipy
from scipy.special import riccati_jn, riccati_yn

# default tolerances
DEFAULT_EPS1 = 1e-3
DEFAULT_EPS2 = 1e-16

def riccati_psi_xi(x, nstop):
    """
    Construct riccati hankel function of 1st kind by linear combination of
    RB's based on j_n and y_n
    """
    if np.imag(x) != 0.:
        raise TypeError('Cannot handle complex arguments.')
    psin = riccati_jn(nstop, x)
    # scipy sign on y_n consistent with B/H
    xin = psin[0] + 1j*riccati_yn(nstop, x)[0]
    rbh = array([psin[0], xin])
    return rbh

def lentz_dn1(z, n, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    """
    Calculate logarithmic derivative D_n(z) of the Riccati-Bessel
    function for a single value of n using the Lentz (1976)
    continued fraction method.

    Notes
    -----
    Implements check/workaround for ill-conditioning described under "Algorithm
    Improvement" in Lentz (1976); see also Wiscombe/NCAR Mie report.

    Parameters
    ----------
    z: complex argument
    n: order of the logarithmic derivative
    eps1: value of continued fraction numerator or denominator
          triggering ill-conditioning workaround. Recommend
          1e-3.
    eps2: converge when additional products in continued fraction
          differ by less than eps2 from 1. Recommend 1e-16.

    Returns
    -------
    value of D_n(z)
    """
    def a_i(i):
        return (-1.)**(i + 1) * 2. * (n + i - 0.5) / z

    numerator = a_i(2) + 1. / a_i(1)
    denominator = a_i(2)

    product = a_i(1) * numerator / denominator
    ratio = product

    ctr = 3

    while abs(product.real - 1) > eps2 or abs(product.imag) > eps2:
        ai = a_i(ctr)
        numerator = ai + 1. / numerator
        denominator = ai + 1. / denominator

        if abs(numerator / ai) < eps1 or abs(denominator / ai) < eps1:
            # ill conditioning
            xi1 = 1. + a_i(ctr + 1) * numerator
            xi2 = 1. + a_i(ctr + 1) * denominator
            ratio = ratio * xi1 / xi2
            #print('ill conditioned', ratio)
            numerator = a_i(ctr + 2) + numerator / xi1
            denominator = a_i(ctr + 2) + denominator / xi2
            ctr = ctr + 2
        product = numerator / denominator
        ratio = ratio * product
        ctr = ctr + 1
    return ratio - n / z

def dn_1_down(z, nmx, nstop, start_val):
    '''
    Computes logarithmic derivative of Riccati-Bessel function \psi_n(z)
    by downward recursion as in BHMIE.

    Parameters
    ----------
    z: complex argument
    nmx: order from which downward recursion begins.
    nstop: integer, maximum order
    start_val: value from which recursion begins

    Notes
    -----
    \psi_n(z) is related to the spherical Bessel function j_n(z).
    '''
    dn = zeros(nmx+1, dtype = 'complex128')
    dn[nmx] = start_val

    for i in np.arange(nmx-1, -1, -1):
        dn[i] = (i+1.)/z - 1.0/(dn[i+1] + (i+1.)/z)
    return dn[0:nstop+1]


def log_der_13(z, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate logarithmic derivatives of Riccati-Bessel functions psi
    and xi for complex arguments.  Riccati-Bessel conventions follow
    Bohren & Huffman.

    See Mackowski et al., Applied Optics 29, 1555 (1990).

    Parameters
    ----------
    z: complex number
    nstop: maximum order of computation
    eps1: underflow criterion for Lentz continued fraction for Dn1
    eps2: convergence criterion for Lentz continued fraction for Dn1
    '''
    z = np.complex128(z) # convert to double precision

    # Calculate Dn_1 (based on \psi(z)) using downward recursion.
    # See Mackowski eqn. 62
    #nmx = np.maximum(nstop, int(np.round_(np.absolute(z)))) + 25
    #dn1 = log_der_1(z, nmx, nstop)
    dn1 = dn_1_down(z, nstop + 1, nstop, lentz_dn1(z, nstop + 1, eps1, eps2))

    # Calculate Dn_3 (based on \xi) by up recurrence
    # initialize
    dn3 = zeros(nstop+1, dtype = 'complex128')
    psixi = zeros(nstop+1, dtype = 'complex128')
    dn3[0] = 1.j
    psixi[0] = -1j*exp(1.j*z)*sin(z)
    for dindex in arange(1, nstop+1):
        # Mackowski eqn 63
        psixi[dindex] = psixi[dindex-1] * ( (dindex/z) - dn1[dindex-1]) * (
            (dindex/z) - dn3[dindex-1])
        # Mackowski eqn 64
        dn3[dindex] = dn1[dindex] + 1j/psixi[dindex]

    return dn1, dn3

# calculate ratio of RB's defined in Yang eqn. 23 by up recursion relation
def Qratio(z1, z2, nstop, dns1 = None, dns2 = None, eps1 = DEFAULT_EPS1,
           eps2 = DEFAULT_EPS2):
    '''
    Calculate ratio of Riccati-Bessel functions defined in Yang eq. 23
    by up recursion.

    Logarithmic derivatives calculated automatically if not specified.
    '''
    # convert z1 and z2 to 128 bit complex to prevent division problems
    z1 = np.complex128(z1)
    z2 = np.complex128(z2)

    if dns1 == None:
        logdersz1 = LogDer13(z1, nstop, eps1, eps2)
        logdersz2 = LogDer13(z2, nstop, eps1, eps2)
        d1z1 = logdersz1[0]
        d3z1 = logdersz1[1]
        d1z2 = logdersz2[0]
        d3z2 = logdersz2[1]
    else:
        d1z1 = dns1[0]
        d3z1 = dns1[1]
        d1z2 = dns2[0]
        d3z2 = dns2[1]

    qns = zeros(nstop+1, dtype = 'complex128')

    # initialize according to Yang eqn. 34
    a1 = real(z1)
    a2 = real(z2)
    b1 = imag(z1)
    b2 = imag(z2)
    qns[0] = exp(-2.*(b2-b1)) * (exp(-1j*2.*a1)-exp(-2.*b1)) / (exp(-1j*2.*a2)
                                                                - exp(-2.*b2))
    # Loop to do upwards recursion in eqn. 33
    for i in arange(1, nstop+1):
        qns[i] = qns[i-1]* ( (d3z1[i] + i/z1) * (d1z2[i] + i/z2)
	       		     )  / ((d3z2[i] + i/z2) * (d1z1[i] + i/z1) )
    return qns

def R_psi(z1, z2, nmax, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate ratio of Riccati-Bessel function \psi: \psi(z1)/\psi(z2).

    See Mackowski eqns. 65-66.
    '''
    output = zeros(nmax + 1, dtype = 'complex128')
    output[0] = sin(z1) / sin(z2)
    dnz1 = dn_1_down(z1, nmax + 1, nmax, lentz_dn1(z1, nmax + 1, eps1, eps2))
    dnz2 = dn_1_down(z2, nmax + 1, nmax, lentz_dn1(z2, nmax + 1, eps1, eps2))

    # use up recursion
    for i in arange(1, nmax + 1):
        output[i] = output[i - 1] * (dnz2[i] + i / z2) / (dnz1[i] + i / z1)
    return output
