# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang
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
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
"""

import structcol as sc
from .. import refractive_index as ri
from .. import Quantity
from numpy.testing import assert_equal, assert_almost_equal, assert_warns
from pytest import raises
from pint.errors import DimensionalityError
import numpy as np
import pytest

def test_index_from_function():
    # test that making an index object from a function works
    def fake_index_relation(wavelen, fake_index=None):
        if fake_index is None:
            return np.ones_like(wavelen) * 1.0
        else:
            return np.ones_like(wavelen) * fake_index
    wavelen = sc.Quantity(np.linspace(400, 800, 100), 'nm')

    my_index = sc.Index(fake_index_relation)
    assert_equal(my_index(wavelen), np.ones_like(wavelen) * 1.0)

    # check that scalar wavelength works
    assert_equal(my_index(sc.Quantity('400.0 nm')), 1.0)

    # check that keyword is set when creating Index object
    my_index = sc.Index(fake_index_relation, fake_index=3.33)
    assert_equal(my_index(wavelen), np.ones_like(wavelen) * 3.33)

    # check that wavelengths with no units give error
    with pytest.raises(DimensionalityError):
        my_index(np.linspace(400, 800, 100))

def test_index_from_constant():
    # test that making an index object from a constant works
    wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')

    my_index = sc.Index.constant(1.888)
    assert_equal(my_index(wavelen), np.ones_like(wavelen) * 1.888)

    # check that wavelengths with wrong units gives error
    with pytest.raises(DimensionalityError):
        my_index(sc.Quantity('400 kg'))

def test_n():
    # make sure that a material not in the dictionary raises a KeyError
    raises(KeyError, ri.n, 'badkey', Quantity('0.5 um'))

    # make sure that specifying no units throws an exception
    raises(DimensionalityError, ri.n, 'polystyrene', 0.5)

    # and specifying the wrong units, too
    raises(DimensionalityError, ri.n, 'polystyrene', Quantity('0.5 J'))

# the next few tests make sure that the various dispersion formulas give values
# of n close to those listed by refractiveindex.info (or other source) at the
# boundaries of the visible spectrum.  This is mostly to make sure that the
# coefficients of the dispersion formulas are entered properly

def test_water():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('water', Quantity('0.40930 um')).magnitude,
                        Quantity('1.3427061376724').magnitude)
    assert_almost_equal(ri.n('water', Quantity('0.80700 um')).magnitude,
                        Quantity('1.3284883366632').magnitude)

def test_npmma():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('pmma', Quantity('0.42 um')).magnitude,
                        Quantity('1.5049521933717').magnitude)
    assert_almost_equal(ri.n('pmma', Quantity('0.804 um')).magnitude,
                        Quantity('1.4866523830528').magnitude)

def test_nps():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('polystyrene', Quantity('0.4491 um')).magnitude,
                        Quantity('1.6137854760669').magnitude)
    assert_almost_equal(ri.n('polystyrene', Quantity('0.7998 um')).magnitude,
                        Quantity('1.5781660671827').magnitude)

def test_rutile():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('rutile', Quantity('0.4300 um')).magnitude,
                        Quantity('2.8716984534676').magnitude)
    assert_almost_equal(ri.n('rutile', Quantity('0.8040 um')).magnitude,
                        Quantity('2.5187663081355').magnitude)

def test_fused_silica():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('fused silica', Quantity('0.3850 um')).magnitude,
                        Quantity('1.4718556531995').magnitude)
    assert_almost_equal(ri.n('fused silica', Quantity('0.8050 um')).magnitude,
                        Quantity('1.4532313266004').magnitude)
def test_zirconia():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('zirconia', Quantity('.405 um')).magnitude,
                       Quantity('2.3135169070958').magnitude)
    assert_almost_equal(ri.n('zirconia', Quantity('.6350 um')).magnitude,
                        Quantity('2.1593242574339').magnitude)


def test_vacuum():
    assert_almost_equal(ri.n('vacuum', Quantity('0.400 um')).magnitude, Quantity('1.0').magnitude)
    assert_almost_equal(ri.n('vacuum', Quantity('0.800 um')).magnitude, Quantity('1.0').magnitude)

def test_cargille():
    assert_almost_equal(ri.n_cargille(1,'AAA',Quantity('0.400 um')).magnitude,
                        Quantity('1.3101597437500001').magnitude)
    assert_almost_equal(ri.n_cargille(1,'AAA',Quantity('0.700 um')).magnitude,
                        Quantity('1.303526242857143').magnitude)
    assert_almost_equal(ri.n_cargille(1,'AA',Quantity('0.400 um')).magnitude,
                        Quantity('1.4169400062500002').magnitude)
    assert_almost_equal(ri.n_cargille(1,'AA',Quantity('0.700 um')).magnitude,
                        Quantity('1.3987172673469388').magnitude)
    assert_almost_equal(ri.n_cargille(1,'A',Quantity('0.400 um')).magnitude,
                        Quantity('1.4755715625000001').magnitude)
    assert_almost_equal(ri.n_cargille(1,'A',Quantity('0.700 um')).magnitude,
                        Quantity('1.458145836734694').magnitude)
    assert_almost_equal(ri.n_cargille(1,'B',Quantity('0.400 um')).magnitude,
                        Quantity('1.6720350625').magnitude)
    assert_almost_equal(ri.n_cargille(1,'B',Quantity('0.700 um')).magnitude,
                        Quantity('1.6283854489795917').magnitude)
    assert_almost_equal(ri.n_cargille(1,'E',Quantity('0.400 um')).magnitude,
                        Quantity('1.5190772875').magnitude)
    assert_almost_equal(ri.n_cargille(1,'E',Quantity('0.700 um')).magnitude,
                        Quantity('1.4945156653061225').magnitude)
    assert_almost_equal(ri.n_cargille(0,'acrylic',Quantity('0.400 um')).magnitude,
                        Quantity('1.50736788125').magnitude)
    assert_almost_equal(ri.n_cargille(0,'acrylic',Quantity('0.700 um')).magnitude,
                        Quantity('1.4878716959183673').magnitude)

def test_neff():
    # test that at low volume fractions, Maxwell-Garnett and Bruggeman roughly
    # match for a non-core-shell particle
    n_particle = Quantity(2.7, '')
    n_matrix = Quantity(2.2, '')
    vf = Quantity(0.001, '')

    neff_mg = ri.n_eff(n_particle, n_matrix, vf, maxwell_garnett=True)
    neff_bg = ri.n_eff(n_particle, n_matrix, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg.magnitude, neff_bg.magnitude)

    # test that the non-core-shell particle with Maxwell-Garnett matches with
    # the core-shell of shell index of air with Bruggeman at low volume fractions
    n_particle2 = Quantity(np.array([2.7, 2.2]), '')
    vf2 = Quantity(np.array([0.001, 0.1]), '')
    neff_bg2 = ri.n_eff(n_particle2, n_matrix, vf2, maxwell_garnett=False)

    assert_almost_equal(neff_mg.magnitude, neff_bg2.magnitude)
    assert_almost_equal(neff_bg.magnitude, neff_bg2.magnitude)

    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    vf3 = Quantity(0.5, '')
    neff_bg3 = ri.n_eff(n_particle, n_matrix, vf3, maxwell_garnett=False)

    vf3_cs = Quantity(np.array([0.5, 0.1]), '')
    neff_bg3_cs = ri.n_eff(n_particle2, n_matrix, vf3_cs, maxwell_garnett=False)

    assert_almost_equal(neff_bg3.magnitude, neff_bg3_cs.magnitude)

    # repeat the tests using complex indices
    n_particle_complex = Quantity(2.7+0.001j, '')
    n_matrix_complex = Quantity(2.2+0.001j, '')

    neff_mg_complex = ri.n_eff(n_particle_complex, n_matrix_complex, vf, maxwell_garnett=True)
    neff_bg_complex = ri.n_eff(n_particle_complex, n_matrix_complex, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex.magnitude, neff_bg_complex.magnitude)

    # test that the non-core-shell particle with Maxwell-Garnett matches with
    # the core-shell of shell index of air with Bruggeman at low volume fractions
    n_particle2_complex = Quantity(np.array([2.7+0.001j, 2.2+0.001j]), '')
    neff_bg2_complex = ri.n_eff(n_particle2_complex, n_matrix_complex, vf2, maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex.magnitude, neff_bg2_complex.magnitude)
    assert_almost_equal(neff_bg_complex.magnitude, neff_bg2_complex.magnitude)

    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    neff_bg3_complex = ri.n_eff(n_particle_complex, n_matrix_complex, vf3, maxwell_garnett=False)

    neff_bg3_cs_complex = ri.n_eff(n_particle2_complex, n_matrix_complex, vf3_cs, maxwell_garnett=False)

    assert_almost_equal(neff_bg3_complex.magnitude, neff_bg3_cs_complex.magnitude)

def test_data():
    # Test that we can input data for refractive index
    wavelength = Quantity(np.array([400.0, 500.0, 600.0]), 'nm')
    data = Quantity(np.array([1.5,1.55,1.6]), '')
    assert_equal(ri.n('data', wavelength, index_data=data, wavelength_data=wavelength).magnitude.all(), data.magnitude.all())

    # Test that it also works for complex values
    data_complex = np.array([1.5+0.01j,1.55+0.02j,1.6+0.03j])
    assert_equal(ri.n('data', wavelength, index_data=data, wavelength_data=wavelength).all(), data_complex.all())

    # Test that keyerror is raised when no index is specified for 'data'
    raises(KeyError, ri.n, 'data', Quantity('0.5 um'), index_data=None)

    # Test warning message when user specifies index for a material other than 'data'
    assert_warns(Warning, ri.n, 'water', Quantity('0.5 um'), index_data=data)
