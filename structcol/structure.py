# Copyright 2016, Sofia Makgiriadou, Vinothan N. Manoharan
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

.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>.
.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import numpy as np
from . import ureg, Quantity  # unit registry and Quantity constructor from pint

@ureg.check('[]','[]')    # inputs should be dimensionless
def factor_py(qd, phi):
    """
    Calculate structure factor of hard spheres using the Ornstein-Zernike equation
    and Percus-Yevick approximation [1]_ [2]_.

    Notes
    -----
    Might not be accurate for volume fractions above 0.5 (need to check against
    some simulated structure factors).

    This code is fully vectorized, so you can feed it orthogonal vectors for
    both qd and phi and it will produce a 2D output:
        qd = np.arange(0.1, 20, 0.01)
        phi = np.array([0.15, 0.3, 0.45])
        s = structure.factor_py(qd.reshape(-1,1), phi.reshape(1,-1))

    References
    ----------
    [1] Boualem Hammouda, "Probing Nanoscale Structures -- The SANS Toolbox"
    http://www.ncnr.nist.gov/staff/hammouda/the_SANS_toolbox.pdf Chapter 32
    (http://www.ncnr.nist.gov/staff/hammouda/distance_learning/chapter_32.pdf)

    [2] Boualem Hammouda, "A Tutorial on Small-Angle Neutron Scattering from
    Polymers", http://www.ncnr.nist.gov/programs/sans/pdf/polymer_tut.pdf, pp.
    51--53.
    """

    # constants in the direct correlation function
    lambda1 = (1 + 2*phi)**2 / (1 - phi)**4
    lambda2 = -(1 + phi/2.)**2 /  (1 - phi)**4
    # Fourier transform of the direct correlation function (eq X.33 of [2]_)
    c = -24*phi*(lambda1 * (np.sin(qd) - qd*np.cos(qd)) / qd**3  -
                 6*phi*lambda2 * (qd**2 * np.cos(qd) - 2*qd*np.sin(qd) -
                                  2*np.cos(qd)+2.0) / qd**4 -
                 (phi*lambda1/2.) * (qd**4 * np.cos(qd) - 4*qd**3 * np.sin(qd) -
                                     12 * qd**2 * np.cos(qd) + 24*qd * np.sin(qd) +
                                     24 * np.cos(qd) - 24.0) / qd**6)
    # Structure factor at qd (eq X.34 of [2]_)
    return 1.0/(1-c)
