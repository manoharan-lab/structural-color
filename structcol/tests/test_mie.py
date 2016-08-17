# Copyright 2016, Vinothan N. Manoharan
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
Tests for the mie module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, np, mie
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError

def test_cross_sections():
    # Test cross sections against values calculated from BHMIE code (originally
    # calculated for testing fortran-based Mie code in holopy)

    # test case is PS sphere in water
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_medium = 1.33
    n_particle = 1.59 + 1e-4 * 1.0j
    m = n_particle/n_medium
    # scipy special functions cannot handle pint Quantities, so must first put
    # in non-dimensional form and then pass the magnitude
    x = (2*np.pi*n_medium/wavelen * radius).to('dimensionless').magnitude
    qscat, qext, _ = mie.calc_efficiencies(m, x)
    g = mie.calc_g(m,x)   # asymmetry parameter

    qscat_std, qext_std, g_std = 3.6647, 3.6677, 0.92701
    assert_almost_equal(qscat, qscat_std, decimal=4)
    assert_almost_equal(qext, qext_std, decimal=4)
    assert_almost_equal(g, g_std, decimal=4)

    wavelen_scalar = wavelen.to('m').magnitude
    radius_scalar = radius.to('m').magnitude
    # test to make sure calc_cross_sections returns the same values as
    # calc_efficiencies and calc_g
    cross_sections_1 = mie.calc_cross_sections(m, x, wavelen_scalar)[0:2]
    cross_sections_2 = mie.calc_efficiencies(m, x)[0:2] * np.pi * radius_scalar**2
    assert_almost_equal(cross_sections_1, cross_sections_2)

    g_1 = mie.calc_cross_sections(m, x, wavelen_scalar)[4]
    g_2 = mie.calc_g(m, x)
    assert_almost_equal(g_1, g_2)

# TODO: need to add more tests here to test for correctness of Mie results
# (form factors and cross sections at a variety of different size parameters)
