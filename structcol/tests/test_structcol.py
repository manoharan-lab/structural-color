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
Tests various features of the structcol package not found in submodules

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
from pytest import raises
from pint.errors import DimensionalityError

def test_q():
    # make sure that the q function works correctly on arrays and quantities
    # with dimensions

    # test angle conversion
    assert_equal(q(Quantity('450 nm'), Quantity('pi/2 rad')).magnitude,
                 q(Quantity('450 nm'), Quantity('90 degrees')).magnitude)

    # test to make sure function returns an array if given an array argument
    wavelen = Quantity(np.arange(500.0, 800.0, 10.0), 'nm')
    assert_equal(wavelen.shape, (30,))
    q_values = q(wavelen, Quantity('90 degrees'))
    assert_equal(q_values.shape, wavelen.shape)
    angle = np.transpose(Quantity(np.arange(0, 180., 1.0), 'degrees'))
    assert_equal(angle.shape, (180,))
    q_values = q(Quantity('0.5 um'), angle)
    assert_equal(q_values.shape, angle.shape)

    # test to make sure function returns a 2D array if given arrays for both
    # theta and wavelen
    q_values = q(wavelen.reshape(-1,1), angle.reshape(1,-1))
    assert_equal(q_values.shape, (wavelen.shape[0], angle.shape[0]))

    # test dimension checking
    raises(DimensionalityError, q, Quantity('0.5 J'), Quantity('0.5 rad'))
    raises(DimensionalityError, q, Quantity('450 nm'), Quantity('0.5 m'))
