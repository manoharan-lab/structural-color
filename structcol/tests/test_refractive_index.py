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
from .. import Quantity
from numpy.testing import assert_equal, assert_almost_equal, assert_warns
from pytest import raises
from pint.errors import DimensionalityError
import numpy as np
import pytest

class TestIndex:
    """Tests for the Index class"""
    wavelen = sc.Quantity(np.linspace(400, 800, 100), 'nm')

    def test_index_from_function(self):
        # test that making an Index object from a function works
        def fake_index_relation(wavelen, fake_index=None):
            if fake_index is None:
                return np.ones_like(wavelen) * 1.0
            else:
                return np.ones_like(wavelen) * fake_index

        my_index = sc.Index(fake_index_relation)
        assert_equal(my_index(self.wavelen), np.ones_like(self.wavelen) * 1.0)

        # check that scalar wavelength works
        assert_equal(my_index(sc.Quantity('400.0 nm')), 1.0)

        # check that keyword is set when creating Index object
        my_index = sc.Index(fake_index_relation, fake_index=3.33)
        assert_equal(my_index(self.wavelen), np.ones_like(self.wavelen) * 3.33)

        # check that wavelengths with no units give error
        with pytest.raises(DimensionalityError):
            my_index(np.linspace(400, 800, 100))

    def test_index_from_constant(self):
        # test that making an index object from a constant works
        my_index = sc.Index.constant(1.88)
        assert_equal(my_index(self.wavelen), np.ones_like(self.wavelen) * 1.88)

        # check that giving dimensional constant gives error
        with pytest.raises(ValueError):
            my_index = sc.Index.constant(sc.Quantity(1.45, 'um'))
            my_index(self.wavelen)

        # check that wavelengths with wrong units gives error
        with pytest.raises(DimensionalityError):
            my_index(sc.Quantity('400 kg'))

        # test that dimensions of index are stripped
        my_index = sc.Index.constant(sc.Quantity(1.33, ''))
        assert not isinstance(my_index(sc.Quantity('400 nm')), Quantity)

    def test_index_from_data(self):
        # test that making an index object from a data works as expected

        wavelength_data = sc.Quantity(np.linspace(400, 800, 10), 'nm')
        index_data = sc.index.water(wavelength_data)

        # next construct interpolating function
        interpolated_index_linear = sc.Index.from_data(wavelength_data,
                                                       index_data,
                                                       kind="linear")
        interpolated_index_cubic = sc.Index.from_data(wavelength_data,
                                                      index_data, kind='cubic')

        # linear should roughly agree; cubic should have better agreement
        assert_almost_equal(interpolated_index_linear(self.wavelen),
                            sc.index.water(self.wavelen), decimal=3)
        assert_almost_equal(interpolated_index_cubic(self.wavelen),
                            sc.index.water(self.wavelen), decimal=5)

        # test that specifying wavelength in the wrong units gives error
        with pytest.raises(ValueError):
            index_data = sc.index.water(wavelength_data) * sc.Quantity('kg')
            index = sc.Index.from_data(wavelength_data, index_data)

        # test that dimensions of index are stripped
        index_data = sc.Quantity(sc.index.water(wavelength_data), '')
        interpolated_index = sc.Index.from_data(wavelength_data, index_data)
        assert not isinstance(interpolated_index(sc.Quantity('400 nm')),
                              Quantity)

        # test that exception is thrown if lengths of arrays are not identical
        index_data = index_data[1:]
        with pytest.raises(ValueError):
            interpolated_index = sc.Index.from_data(wavelength_data,
                                                    index_data)

    def test_dimensions(self):
        # try inputs with various dimensions to make sure the appropriate
        # exceptions are thrown and the index is always dimensionless
        with pytest.raises(ValueError):
            def fake_index_func(wavelen):
                return 1.4 * sc.Quantity(500, 'nm')/(wavelen*wavelen)
            my_index = sc.Index(fake_index_func)
            my_index(sc.Quantity('400 nm'))

        # test with weird but OK units
        def fake_index_func(wavelen):
            return 1.4 + sc.Quantity(0.5, 'nm*um')/(wavelen*wavelen)
        my_index = sc.Index(fake_index_func)
        assert_equal(my_index(sc.Quantity('400 nm')), 1.403125)

        def fake_index_func(wavelen):
            return 1.4 + sc.Quantity(0.000005, 'nm*m')/(wavelen*wavelen)
        assert_equal(my_index(sc.Quantity('400 nm')), 1.403125)

        # ensure that we get the same result with different units for the
        # wavelength
        assert_equal(my_index(sc.Quantity('500 nm')),
                     my_index(sc.Quantity('0.5 um')))

        assert_equal(my_index(sc.Quantity('5e-5 cm')),
                     my_index(sc.Quantity('5e-7 m')))

def test_n():
    # make sure that a material not in the dictionary raises a KeyError
    raises(KeyError, sc.index.n, 'badkey', Quantity('0.5 um'))

    # make sure that specifying no units throws an exception
    raises(DimensionalityError, sc.index.n, 'polystyrene', 0.5)

    # and specifying the wrong units, too
    raises(DimensionalityError, sc.index.n, 'polystyrene', Quantity('0.5 J'))

# the next few tests make sure that the various dispersion formulas give values
# of n close to those listed by refractiveindex.info (or other source) at the
# boundaries of the visible spectrum.  This is mostly to make sure that the
# coefficients of the dispersion formulas are entered properly

def test_water():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.water(Quantity('0.40930 um')),
                        1.3427061376724)
    assert_almost_equal(sc.index.water(Quantity('0.80700 um')),
                        1.3284883366632)

def test_npmma():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.pmma(Quantity('0.42 um')),
                        1.5049521933717)
    assert_almost_equal(sc.index.pmma(Quantity('0.804 um')),
                        1.4866523830528)

def test_nps():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.n('polystyrene', Quantity('0.4491 um')),
                        1.6137854760669)
    assert_almost_equal(sc.index.n('polystyrene', Quantity('0.7998 um')),
                        1.5781660671827)

def test_rutile():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.n('rutile', Quantity('0.4300 um')),
                        2.8716984534676)
    assert_almost_equal(sc.index.n('rutile', Quantity('0.8040 um')),
                        2.5187663081355)

def test_fused_silica():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.n('fused silica', Quantity('0.3850 um')),
                        1.4718556531995)
    assert_almost_equal(sc.index.n('fused silica', Quantity('0.8050 um')),
                        1.4532313266004)

def test_zirconia():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.n('zirconia', Quantity('.405 um')),
                       2.3135169070958)
    assert_almost_equal(sc.index.n('zirconia', Quantity('.6350 um')),
                        2.1593242574339)

def test_vacuum():
    assert_equal(sc.index.n('vacuum', Quantity('0.400 um')), 1.0)
    assert_equal(sc.index.n('vacuum', Quantity('0.800 um')), 1.0)

def test_cargille():
    assert_almost_equal(sc.index.n_cargille(1,'AAA',Quantity('0.400 um')),
                        1.3101597437500001)
    assert_almost_equal(sc.index.n_cargille(1,'AAA',Quantity('0.700 um')),
                        1.303526242857143)
    assert_almost_equal(sc.index.n_cargille(1,'AA',Quantity('0.400 um')),
                        1.4169400062500002)
    assert_almost_equal(sc.index.n_cargille(1,'AA',Quantity('0.700 um')),
                        1.3987172673469388)
    assert_almost_equal(sc.index.n_cargille(1,'A',Quantity('0.400 um')),
                        1.4755715625000001)
    assert_almost_equal(sc.index.n_cargille(1,'A',Quantity('0.700 um')),
                        1.458145836734694)
    assert_almost_equal(sc.index.n_cargille(1,'B',Quantity('0.400 um')),
                        1.6720350625)
    assert_almost_equal(sc.index.n_cargille(1,'B',Quantity('0.700 um')),
                        1.6283854489795917)
    assert_almost_equal(sc.index.n_cargille(1,'E',Quantity('0.400 um')),
                        1.5190772875)
    assert_almost_equal(sc.index.n_cargille(1,'E',Quantity('0.700 um')),
                        1.4945156653061225)
    assert_almost_equal(sc.index.n_cargille(0,'acrylic',Quantity('0.400 um')),
                        1.50736788125)
    assert_almost_equal(sc.index.n_cargille(0,'acrylic',Quantity('0.700 um')),
                        1.4878716959183673)

def test_neff():
    # test that at low volume fractions, Maxwell-Garnett and Bruggeman roughly
    # match for a non-core-shell particle
    n_particle = 2.7
    n_matrix = 2.2
    vf = Quantity(0.001, '')

    neff_mg = sc.index.n_eff(n_particle, n_matrix, vf, maxwell_garnett=True)
    neff_bg = sc.index.n_eff(n_particle, n_matrix, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg, neff_bg)

    # test that the non-core-shell particle with Maxwell-Garnett matches with
    # the core-shell of shell index of air with Bruggeman at low volume fractions
    n_particle2 = np.array([2.7, 2.2])
    vf2 = Quantity(np.array([0.001, 0.1]), '')
    neff_bg2 = sc.index.n_eff(n_particle2, n_matrix, vf2, maxwell_garnett=False)

    assert_almost_equal(neff_mg, neff_bg2)
    assert_almost_equal(neff_bg, neff_bg2)

    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    vf3 = Quantity(0.5, '')
    neff_bg3 = sc.index.n_eff(n_particle, n_matrix, vf3, maxwell_garnett=False)

    vf3_cs = Quantity(np.array([0.5, 0.1]), '')
    neff_bg3_cs = sc.index.n_eff(n_particle2, n_matrix, vf3_cs, maxwell_garnett=False)

    assert_almost_equal(neff_bg3, neff_bg3_cs)

    # repeat the tests using complex indices
    n_particle_complex = 2.7+0.001j
    n_matrix_complex = 2.2+0.001j

    neff_mg_complex = sc.index.n_eff(n_particle_complex, n_matrix_complex, vf, maxwell_garnett=True)
    neff_bg_complex = sc.index.n_eff(n_particle_complex, n_matrix_complex, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex, neff_bg_complex)

    # test that the non-core-shell particle with Maxwell-Garnett matches with
    # the core-shell of shell index of air with Bruggeman at low volume fractions
    n_particle2_complex = np.array([2.7+0.001j, 2.2+0.001j])
    neff_bg2_complex = sc.index.n_eff(n_particle2_complex, n_matrix_complex, vf2, maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex, neff_bg2_complex)
    assert_almost_equal(neff_bg_complex, neff_bg2_complex)

    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    neff_bg3_complex = sc.index.n_eff(n_particle_complex, n_matrix_complex, vf3, maxwell_garnett=False)

    neff_bg3_cs_complex = sc.index.n_eff(n_particle2_complex, n_matrix_complex, vf3_cs, maxwell_garnett=False)

    assert_almost_equal(neff_bg3_complex, neff_bg3_cs_complex)

def test_data():
    # Test that we can input data for refractive index
    wavelength = Quantity(np.array([400.0, 500.0, 600.0]), 'nm')
    data = Quantity(np.array([1.5,1.55,1.6]), '')
    assert_equal(sc.index.n('data', wavelength, index_data=data, wavelength_data=wavelength).all(), data.magnitude.all())

    # Test that it also works for complex values
    data_complex = np.array([1.5+0.01j,1.55+0.02j,1.6+0.03j])
    assert_equal(sc.index.n('data', wavelength, index_data=data, wavelength_data=wavelength).all(), data_complex.all())

    # Test that keyerror is raised when no index is specified for 'data'
    raises(KeyError, sc.index.n, 'data', Quantity('0.5 um'), index_data=None)

    # Test warning message when user specifies index for a material other than 'data'
    assert_warns(Warning, sc.index.n, 'water', Quantity('0.5 um'), index_data=data)
