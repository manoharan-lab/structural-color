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
Tests for the refractive_index module of structcol

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import refractive_index as ri
from .. import Quantity
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError


def test_n():
    # make sure that a material not in the dictionary raises a KeyError
    assert_raises(KeyError, ri.n, 'badkey', Quantity('0.5 um'))

    # make sure that specifying no units throws an exception
    assert_raises(AttributeError, ri.n, 'polystyrene', 0.5)
    # and specifying the wrong units, too
    assert_raises(DimensionalityError, ri.n, 'polystyrene', Quantity('0.5 J'))

# the next few tests make sure that the various dispersion formulas give values
# of n close to those listed by refractiveindex.info (or other source) at the
# boundaries of the visible spectrum.  This is mostly to make sure that the
# coefficients of the dispersion formulas are entered properly

def test_npmma():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('pmma', Quantity('0.42 um')),
                        Quantity('1.5049521933717'))
    assert_almost_equal(ri.n('pmma', Quantity('0.804 um')),
                        Quantity('1.4866523830528'))

def test_nps():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('polystyrene', Quantity('0.4491 um')),
                        Quantity('1.6137854760669'))
    assert_almost_equal(ri.n('polystyrene', Quantity('0.7998 um')),
                        Quantity('1.5781660671827'))

def test_rutile():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('rutile', Quantity('0.4300 um')),
                        Quantity('2.8716984534676'))
    assert_almost_equal(ri.n('rutile', Quantity('0.8040 um')),
                        Quantity('2.5187663081355'))

def test_fused_silica():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('fused silica', Quantity('0.3850 um')),
                        Quantity('1.4718556531995'))
    assert_almost_equal(ri.n('fused silica', Quantity('0.8050 um')),
                        Quantity('1.4532313266004'))

def test_vacuum():
    assert_almost_equal(ri.n('vacuum', Quantity('0.400 um')), Quantity('1.0'))
    assert_almost_equal(ri.n('vacuum', Quantity('0.800 um')), Quantity('1.0'))
