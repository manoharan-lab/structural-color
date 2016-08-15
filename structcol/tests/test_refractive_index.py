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

from .. import refractive_index as ri
from .. import Quantity
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal


def test_n():
    # make sure that a material not in the dictionary raises a KeyError
    assert_raises(KeyError, ri.n, 'badkey', Quantity('0.5 um'))

# the next few tests make sure that the various dispersion formulas give values
# of n close to those listed by refractiveindex.info (or other source) at the
# boundaries of the visible spectrum.  This is mostly to make sure that the
# coefficients of the dispersion formulas are entered properly

def test_npmma():
    # values from refractiveindex.info
    n450 = ri.n('pmma', Quantity('0.42 um'))
    n800 = ri.n('pmma', Quantity('0.804 um'))
    assert_almost_equal(n450, Quantity('1.5049521933717'))
    assert_almost_equal(n800, Quantity('1.4866523830528'))

def test_nps():
    # values from refractiveindex.info
    n450 = ri.n('polystyrene', Quantity('0.4491 um'))
    n800 = ri.n('polystyrene', Quantity('0.7998 um'))
    assert_almost_equal(n450, Quantity('1.6137854760669'))
    assert_almost_equal(n800, Quantity('1.5781660671827'))

def test_rutile():
    # values from refractiveindex.info
    n450 = ri.n('rutile', Quantity('0.4300 um'))
    n800 = ri.n('rutile', Quantity('0.8040 um'))
    assert_almost_equal(n450, Quantity('2.8716984534676'))
    assert_almost_equal(n800, Quantity('2.5187663081355'))

def test_fused_silica():
    # values from refractiveindex.info
    n450 = ri.n('fused silica', Quantity('0.3850 um'))
    n800 = ri.n('fused silica', Quantity('0.8050 um'))
    assert_almost_equal(n450, Quantity('1.4718556531995'))
    assert_almost_equal(n800, Quantity('1.4532313266004'))
