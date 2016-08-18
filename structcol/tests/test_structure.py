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
Tests for the structure module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, np, structure
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError

def test_structure_factor_percus_yevick():
    # Test structure factor as calculated by solution of Ornstein-Zernike
    # integral equation and Percus-Yevick closure approximation

    # test that function handles dimensionless arguments, and only
    # dimensionless arguments
    structure.factor_py(Quantity('0.1'), Quantity('0.4'))
    structure.factor_py(0.1, 0.4)
    assert_raises(DimensionalityError, structure.factor_py,
                  Quantity('0.1'), Quantity('0.1 m'))
    assert_raises(DimensionalityError, structure.factor_py,
                  Quantity('0.1 m'), Quantity('0.1'))

    # test vectorization by doing calculation over range of qd and phi
    qd = np.arange(0.1, 20, 0.01)
    phi = np.array([0.15, 0.3, 0.45])
    # this little trick allows us to calculate the structure factor on a 2d
    # grid of points (turns qd into a column vector and phi into a row vector).
    # Could also use np.ogrid
    s = structure.factor_py(qd.reshape(-1,1), phi.reshape(1,-1))

    # compare to values from Cipelletti, Trappe, and Pine, "Scattering
    # Techniques", in "Fluids, Colloids and Soft Materials: An Introduction to
    # Soft Matter Physics", 2016 (plot on page 137)
    # (I extracted values from the plot using a digitizer
    # (http://arohatgi.info/WebPlotDigitizer/app/). They are probably good to
    # only one decimal place, so this is a fairly crude test.)
    max_vals = np.max(s, axis=0)    # max values of S(qd) at different phi
    max_qds = qd[np.argmax(s, axis=0)]  # values of qd at which S(qd) has max
    assert_almost_equal(max_vals[0], 1.17, decimal=1)
    assert_almost_equal(max_vals[1], 1.52, decimal=1)
    assert_almost_equal(max_vals[2], 2.52, decimal=1)
    assert_almost_equal(max_qds[0], 6.00, decimal=1)
    assert_almost_equal(max_qds[1], 6.37, decimal=1)
    assert_almost_equal(max_qds[2], 6.84, decimal=1)
