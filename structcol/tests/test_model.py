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
Tests for the single-scattering model (in structcol/model.py)

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, index_ratio, size_parameter, np, mie, model
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError

def test_fresnel():
    # test the fresnel reflection and transmission coefficients
    n1 = Quantity(1.00, '')
    n2 = Quantity(1.5, '')

    # quantities calculated from
    # http://www.calctool.org/CALC/phys/optics/reflec_refrac
    rpar, rperp = model.fresnel_reflection(n1, n2, Quantity('0 deg'))
    assert_almost_equal(rpar, 0.04)
    assert_almost_equal(rperp, 0.04)
    rpar, rperp = model.fresnel_reflection(n1, n2, Quantity('45 deg'))
    assert_almost_equal(rpar, 0.00846646)
    assert_almost_equal(rperp, 0.0920134)

    # test total internal reflection
    rpar, rperp = model.fresnel_reflection(n2, n1, Quantity('45 deg'))
    assert_equal(rpar, 1.0)
    assert_equal(rperp, 1.0)

    # test no total internal reflection (just below critical angle)
    rpar, rperp = model.fresnel_reflection(n2, n1, Quantity('41.810 deg'))
    assert_almost_equal(rpar, 0.972175, decimal=6)
    assert_almost_equal(rperp, 0.987536, decimal=6)

    # test vectorized computation
    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # check for value error
    assert_raises(ValueError, model.fresnel_reflection, n2, n1, angles)
    angles = Quantity(np.linspace(0, 90., 10), 'deg')
    rpar, rperp = model.fresnel_reflection(n2, n1, angles)
    rpar_std = np.array([0.04, 0.0362780, 0.0243938, 0.00460754, 0.100064, 1.0,
                         1.0, 1.0, 1.0, 1])
    rperp_std = np.array([0.04, 0.0438879, 0.0590632, 0.105773, 0.390518, 1.0,
                         1.0, 1.0, 1.0, 1.0])
    assert_array_almost_equal(rpar, rpar_std)
    assert_array_almost_equal(rperp, rperp_std)

    # test transmission
    tpar, tperp = model.fresnel_transmission(n2, n1, angles)
    tpar_std = 1.0-rpar_std
    tperp_std = 1.0-rperp_std
    assert_array_almost_equal(tpar, tpar_std)
    assert_array_almost_equal(tperp, tperp_std)

def test_reflection():
    # test reflection, anisotropy factor, and transport length calculations 
    # make sure the values for refl, g, and lstar remain the same after adding
    # core-shell capability into the model
    
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120 nm')
    wavelength = Quantity(500, 'nm')
    n_particle = Quantity(1.5, '')
    n_matrix = Quantity(1.0, '')
    n_medium = n_matrix

    refl1, _, _, g1, lstar1 = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction, 
                                            thickness = Quantity('15000.0 nm'), 
                                            theta_min = Quantity('90 deg'))
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction, 
                                            thickness = Quantity('15000.0 nm'), 
                                            theta_min = Quantity('90 deg'), 
                                            shell_radius = None)
    refl3, _, _, g3, lstar3 = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction, 
                                            thickness = Quantity('15000.0 nm'), 
                                            theta_min = Quantity('90 deg'),
                                            shell_radius = radius)
    
    # outputs for refl, g, and lstar before adding core-shell capability
    refl = Quantity(0.20772170840902376, '')
    g = Quantity(-0.18931942267032678, '')
    lstar = Quantity(10810.088573316663, 'nm')
    
    assert_array_almost_equal(refl, refl1)
    assert_array_almost_equal(refl1, refl2)
    assert_array_almost_equal(refl2, refl3)

    assert_array_almost_equal(g, g1)    
    assert_array_almost_equal(g1, g2)
    assert_array_almost_equal(g2, g3)
    
    assert_array_almost_equal(lstar, lstar1)
    assert_array_almost_equal(lstar1, lstar2)
    assert_array_almost_equal(lstar2, lstar3)
    