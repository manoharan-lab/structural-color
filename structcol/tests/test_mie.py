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

from .. import Quantity, ureg, q, index_ratio, size_parameter, np, mie
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError

def test_cross_sections():
    # Test cross sections against values calculated from BHMIE code (originally
    # calculated for testing fortran-based Mie code in holopy)

    # test case is PS sphere in water
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_medium = Quantity(1.33, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_medium)
    x = size_parameter(wavelen, n_medium, radius)
    qscat, qext, qback = mie.calc_efficiencies(m, x)
    g = mie.calc_g(m,x)   # asymmetry parameter

    qscat_std, qext_std, g_std = 3.6647, 3.6677, 0.92701
    assert_almost_equal(qscat, qscat_std, decimal=4)
    assert_almost_equal(qext, qext_std, decimal=4)
    assert_almost_equal(g, g_std, decimal=4)

    # test to make sure calc_cross_sections returns the same values as
    # calc_efficiencies and calc_g
    cscat = qscat * np.pi * radius**2
    cext = qext * np.pi * radius**2
    cback  = qback * np.pi * radius**2
    cscat2, cext2, _, cback2, g2 = mie.calc_cross_sections(m, x, wavelen/n_medium)
    assert_almost_equal(cscat.to('m^2').magnitude, cscat2.to('m^2').magnitude)
    assert_almost_equal(cext.to('m^2').magnitude, cext2.to('m^2').magnitude)
    assert_almost_equal(cback.to('m^2').magnitude, cback2.to('m^2').magnitude)
    assert_almost_equal(g, g2)

    # test that calc_cross_sections throws an exception when given an argument
    # with the wrong dimensions
    assert_raises(DimensionalityError, mie.calc_cross_sections,
                  m, x, Quantity('0.25 J'))
    assert_raises(DimensionalityError, mie.calc_cross_sections,
                  m, x, Quantity('0.25'))

def test_form_factor():
    pass
# TODO: need to add more tests here to test for correctness of Mie results
# (form factors and cross sections at a variety of different size parameters)
