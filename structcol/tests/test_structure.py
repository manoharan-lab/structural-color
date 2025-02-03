# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang, Annie Stephenson
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
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
"""

from .. import Quantity, np, structure
from .. import size_parameter
from .. import refractive_index as ri
from numpy.testing import assert_equal, assert_almost_equal
from pytest import raises
from pint.errors import DimensionalityError


def test_structure_factor_percus_yevick():
    # Test structure factor as calculated by solution of Ornstein-Zernike
    # integral equation and Percus-Yevick closure approximation

    # test that function handles dimensionless arguments, and only
    # dimensionless arguments
    structure.factor_py(Quantity('0.1'), Quantity('0.4'))
    structure.factor_py(0.1, 0.4)
    raises(DimensionalityError, structure.factor_py,
                  Quantity('0.1'), Quantity('0.1 m'))
    raises(DimensionalityError, structure.factor_py,
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

def test_structure_factor_percus_yevick_core_shell():
    # Test that the structure factor is the same for core-shell particles and
    # non-core-shell particles at low volume fraction (assuming the core diameter
    # is the same as the particle diameter for the non-core-shell case)

    wavelen = Quantity('400.0 nm')
    angles = Quantity(np.pi, 'rad')
    n_matrix = Quantity(1.0, '')

    # Structure factor for non-core-shell particles
    radius = Quantity('100.0 nm')
    n_particle = Quantity(1.5, '')
    volume_fraction = Quantity(0.0001, '')         # IS VF TOO LOW?
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
    x = size_parameter(wavelen, n_sample, radius)
    qd = 4*x*np.sin(angles/2)
    s = structure.factor_py(qd, volume_fraction)

    # Structure factor for core-shell particles with core size equal to radius
    # of non-core-shell particle
    radius_cs = Quantity(np.array([100.0, 105.0]), 'nm')
    n_particle_cs = Quantity(np.array([1.5, 1.0]), '')
    volume_fraction_shell = volume_fraction * (radius_cs[1]**3 / radius_cs[0]**3 -1)
    volume_fraction_cs = Quantity(np.array([volume_fraction.magnitude, volume_fraction_shell.magnitude]), '')

    n_sample_cs = ri.n_eff(n_particle_cs, n_matrix, volume_fraction_cs)
    x_cs = size_parameter(wavelen, n_sample_cs, radius_cs[1]).flatten()
    qd_cs = 4*x_cs*np.sin(angles/2)
    s_cs = structure.factor_py(qd_cs, np.sum(volume_fraction_cs))

    assert_almost_equal(s.magnitude, s_cs.magnitude, decimal=5)


def test_structure_factor_polydisperse():
    # test that the analytical structure factor for polydisperse systems matches
    # Percus-Yevick in the monodisperse limit

    # Percus-Yevick
    qd = Quantity(5.0, '')
    phi = Quantity(0.5, '')
    S_py = structure.factor_py(qd, phi)

    # Polydisperse S
    d = Quantity('100.0 nm')
    c = Quantity(1.0, '')
    pdi = Quantity(1e-5, '')
    q2 = qd / d

    S_poly = structure.factor_poly(q2, phi, d, c, pdi)

    assert_almost_equal(S_py.magnitude, S_poly.magnitude)


def test_structure_factor_data():
    qd = np.array([1, 2])
    qd_data = np.array([0.5, 2.5])
    s_data = np.array([1, 1])
    s = structure.factor_data(qd, s_data, qd_data)
    assert_equal(s[0], 1)
