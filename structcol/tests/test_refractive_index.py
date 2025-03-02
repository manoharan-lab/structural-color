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
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from pint.errors import DimensionalityError
import numpy as np
import xarray as xr
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
        single_wavelength_n = my_index(sc.Quantity('400.0 nm'))
        assert_equal(single_wavelength_n.to_numpy(), 1.0)
        # and that we return an xarray DataArray object even for single
        # wavelength
        assert isinstance(single_wavelength_n, xr.DataArray)
        assert (single_wavelength_n.attrs[sc.Attr.LENGTH_UNIT] ==
                Quantity(1, 'nm').to_preferred().units)
        assert (sc.Coord.WAVELEN in single_wavelength_n.coords.keys())

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

        # Test that output calculations match input data
        wavelength = Quantity(np.array([400.0, 500.0, 600.0]), 'nm')
        data = np.array([1.5,1.55,1.6])
        assert_equal(sc.Index.from_data(wavelength, data)(wavelength), data)

        # Test that it also works for complex values
        data_complex = np.array([1.5+0.01j,1.55+0.02j,1.6+0.03j])
        assert_equal(sc.Index.from_data(wavelength, data_complex)(wavelength),
                     data_complex)

        # generate "data" over visible range from existing dispersion formula
        wavelength_data = sc.Quantity(np.linspace(400, 800, 10), 'nm')
        index_data = sc.index.water(wavelength_data)

        # next construct interpolating function
        interpolated_index_linear = sc.Index.from_data(wavelength_data,
                                                       index_data,
                                                       kind="linear")
        interpolated_index_cubic = sc.Index.from_data(wavelength_data,
                                                      index_data, kind='cubic')

        # linear should roughly agree; cubic should have better agreement
        assert_almost_equal(interpolated_index_linear(self.wavelen).to_numpy(),
                            sc.index.water(self.wavelen).to_numpy(), decimal=3)
        assert_almost_equal(interpolated_index_cubic(self.wavelen).to_numpy(),
                            sc.index.water(self.wavelen).to_numpy(), decimal=5)

        # test that specifying wavelength in the wrong units gives error
        with pytest.raises(DimensionalityError):
            index_data = sc.index.water(wavelength_data)
            wavelength_wrong = wavelength_data.magnitude *  sc.Quantity('kg')
            index = sc.Index.from_data(wavelength_wrong, index_data)

        # test that dimensions of index are stripped
        index_data = sc.Quantity(sc.index.water(wavelength_data).to_numpy(),
                                 '')
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
        assert_equal(my_index(sc.Quantity('400 nm')).to_numpy(), 1.403125)

        def fake_index_func(wavelen):
            return 1.4 + sc.Quantity(0.000005, 'nm*m')/(wavelen*wavelen)
        assert_equal(my_index(sc.Quantity('400 nm')).to_numpy(), 1.403125)

        # ensure that we get the same result with different units for the
        # wavelength
        assert_equal(my_index(sc.Quantity('500 nm')),
                     my_index(sc.Quantity('0.5 um')))

        assert_equal(my_index(sc.Quantity('5e-5 cm')),
                     my_index(sc.Quantity('5e-7 m')))

def test_ratio():
    """Tests calculation of index ratios

    """
    wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
    n_particle = sc.index.polystyrene(wavelen)
    n_matrix = sc.index.water(wavelen)
    ratio = sc.index.ratio(n_particle, n_matrix)
    # make sure we get a plain numpy array
    assert isinstance(ratio, np.ndarray)
    assert not isinstance(ratio, xr.DataArray)
    assert_equal(ratio, n_particle/n_matrix)

    # make sure we get exceptions if we don't give the right inputs
    with pytest.raises(ValueError):
        ratio = sc.index.ratio(n_particle.to_numpy(), n_matrix)
    # exception if indexes evaluated at different sets of wavelengths
    with pytest.raises(ValueError):
        ratio = sc.index.ratio(sc.index.polystyrene(wavelen[:-1]), n_matrix)
    with pytest.raises(ValueError):
        wavelen[0] = sc.Quantity(401, 'nm')
        ratio = sc.index.ratio(sc.index.polystyrene(wavelen), n_matrix)

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
    assert_almost_equal(sc.index.polystyrene(Quantity('0.4491 um')),
                        1.6137854760669)
    assert_almost_equal(sc.index.polystyrene(Quantity('0.7998 um')),
                        1.5781660671827)

def test_rutile():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.rutile(Quantity('0.4300 um')),
                        2.8716984534676)
    assert_almost_equal(sc.index.rutile(Quantity('0.8040 um')),
                        2.5187663081355)

def test_fused_silica():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.fused_silica(Quantity('0.3850 um')),
                        1.4718556531995)
    assert_almost_equal(sc.index.fused_silica(Quantity('0.8050 um')),
                        1.4532313266004)

def test_zirconia():
    # values from refractiveindex.info
    assert_almost_equal(sc.index.zirconia(Quantity('.405 um')),
                       2.3135169070958)
    assert_almost_equal(sc.index.zirconia(Quantity('.6350 um')),
                        2.1593242574339)

def test_vacuum():
    assert_equal(sc.index.vacuum(Quantity('0.400 um')).to_numpy(), 1.0)
    assert_equal(sc.index.vacuum(Quantity('0.800 um')).to_numpy(), 1.0)

def test_cargille():
    cargille = sc.Index(sc.index.n_cargille, i=1, series="AAA")
    assert_almost_equal(cargille(Quantity('0.400 um')), 1.31088240)
    assert_almost_equal(cargille(Quantity('0.700 um')), 1.30360329)

    cargille = sc.Index(sc.index.n_cargille, i=1, series="AA")
    assert_almost_equal(cargille(Quantity('0.400 um')), 1.415307193)
    assert_almost_equal(cargille(Quantity('0.700 um')), 1.398543173)

    cargille = sc.Index(sc.index.n_cargille, i=1, series="A")
    assert_almost_equal(cargille(Quantity('0.400 um')), 1.47737625)
    assert_almost_equal(cargille(Quantity('0.700 um')), 1.45833825)

    cargille = sc.Index(sc.index.n_cargille, i=1, series="B")
    assert_almost_equal(cargille(Quantity('0.400 um')), 1.70461318)
    assert_almost_equal(cargille(Quantity('0.700 um')), 1.63185900)

    cargille = sc.Index(sc.index.n_cargille, i=1, series="E")
    assert_almost_equal(cargille(Quantity('0.400 um')), 1.54540541)
    assert_almost_equal(cargille(Quantity('0.700 um')), 1.4973228289)

    cargille = sc.Index(sc.index.n_cargille, i=0, series="acrylic")
    assert_almost_equal(cargille(Quantity('0.400 um')), 1.50703048)
    assert_almost_equal(cargille(Quantity('0.700 um')), 1.48783572)

def test_neff():
    # test that at low volume fractions, Maxwell-Garnett and Bruggeman roughly
    # match for a non-core-shell particle
    wavelen = sc.Quantity(500, 'nm')
    n_particle = sc.Index.constant(2.7)(wavelen)
    n_matrix = sc.Index.constant(2.2)(wavelen)
    vf = 0.001

    neff_mg = sc.index.n_eff(n_particle, n_matrix, vf, maxwell_garnett=True)
    neff_bg = sc.index.n_eff(n_particle, n_matrix, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg, neff_bg)

    # test that the non-core-shell particle with Maxwell-Garnett matches with
    # the core-shell of shell index of air with Bruggeman at low volume
    # fractions
    indices = np.array([sc.Index.constant(2.7), sc.Index.constant(2.2)])
    n_particle2 = np.array([index(wavelen) for index in indices])
    vf2 = np.array([0.001, 0.1])
    neff_bg2 = sc.index.n_eff(n_particle2, n_matrix, vf2,
                              maxwell_garnett=False)

    assert_almost_equal(neff_mg, neff_bg2)
    assert_almost_equal(neff_bg, neff_bg2)

    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    vf3 = 0.5
    neff_bg3 = sc.index.n_eff(n_particle, n_matrix, vf3, maxwell_garnett=False)

    vf3_cs = np.array([0.5, 0.1])
    neff_bg3_cs = sc.index.n_eff(n_particle2, n_matrix, vf3_cs,
                                 maxwell_garnett=False)

    assert_almost_equal(neff_bg3, neff_bg3_cs)

    # repeat the tests using complex indices
    n_particle_complex = sc.Index.constant(2.7+0.001j)(wavelen)
    n_matrix_complex = sc.Index.constant(2.2+0.001j)(wavelen)

    neff_mg_complex = sc.index.n_eff(n_particle_complex, n_matrix_complex, vf,
                                     maxwell_garnett=True)
    neff_bg_complex = sc.index.n_eff(n_particle_complex, n_matrix_complex, vf,
                                     maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex, neff_bg_complex)

    # test that the non-core-shell particle with Maxwell-Garnett matches with
    # the core-shell of shell index of air with Bruggeman at low volume
    # fractions
    indices = np.array([sc.Index.constant(2.7+0.001j),
                        sc.Index.constant(2.2+0.001j)])
    n_particle2_complex = np.array([index(wavelen) for index in indices])
    neff_bg2_complex = sc.index.n_eff(n_particle2_complex, n_matrix_complex,
                                      vf2, maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex, neff_bg2_complex)
    assert_almost_equal(neff_bg_complex, neff_bg2_complex)

    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    neff_bg3_complex = sc.index.n_eff(n_particle_complex, n_matrix_complex,
                                      vf3, maxwell_garnett=False)

    neff_bg3_cs_complex = sc.index.n_eff(n_particle2_complex, n_matrix_complex,
                                         vf3_cs, maxwell_garnett=False)

    assert_almost_equal(neff_bg3_complex, neff_bg3_cs_complex)

def test_multimaterial_bruggeman():
    """Tests the Bruggeman approximation for three or more materials
    """
    # five layers, all same index.  Total volume fraction is 1, so result
    # should not depend on volume fraction of matrix
    wavelen = sc.Quantity(500, 'nm')
    index = sc.Index.constant(1.33)
    layers = 5
    n_particle = index(wavelen).expand_dims(dim={sc.Coord.LAYER: layers})
    vf = np.ones(layers) * 1/layers
    n_matrix = sc.Index.constant(1.0)(wavelen)
    assert_equal(sc.index.n_eff(n_particle, n_matrix, vf), index(wavelen))

    # three layers, outer layer same as matrix.  Should return same as two
    # layers
    n_matrix = sc.Index.constant(1.33)(wavelen)
    indices = [sc.Index.constant(1.0), sc.Index.constant(1.59),
               sc.Index.constant(1.33)]
    n_threelayer = np.array([index(wavelen) for index in indices])
    vf = np.array([0.2, 0.2, 0.2])
    n_threelayer_eff = sc.index.n_eff(n_threelayer, n_matrix, vf)
    n_twolayer_eff = sc.index.n_eff(n_threelayer[:-1], n_matrix, vf[:-1])
    assert_almost_equal(n_threelayer_eff, n_twolayer_eff)

def test_vectorized_maxwell_garnett():
    """Tests whether Maxwell-Garnett works on multiple wavelengths at once

    """
    wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
    index_particle = sc.Index.constant(1.33)
    index_matrix = sc.Index.constant(1.00)
    vf = 0.5
    n_mg_vector = sc.index.n_eff(index_particle(wavelen),
                                 index_matrix(wavelen), vf,
                                 maxwell_garnett=True)
    single_wavelen = sc.Quantity(400, 'nm')
    n_mg_single = sc.index.n_eff(index_particle(single_wavelen),
                                 index_matrix(single_wavelen), vf,
                                 maxwell_garnett=True)
    assert_equal(n_mg_vector.to_numpy(),
                 n_mg_single.to_numpy()*np.ones(len(wavelen)))

def test_vectorized_bruggeman():
    """Tests that Bruggeman effective index works on multiple wavelengths at
    once.

    """
    num_wavelengths = 10
    wavelen = sc.Quantity(np.linspace(400, 800, num_wavelengths), 'nm')
    index_particle = sc.Index.constant(1.33)
    num_layers = 5
    n_particle = index_particle(wavelen).expand_dims(dim={sc.Coord.LAYER:
                                                          num_layers})
    n_particle = n_particle.transpose()
    n_matrix = sc.Index.constant(1.00)(wavelen)
    vf = np.ones(num_layers)* 1/num_layers
    n_eff = sc.index.n_eff(n_particle, n_matrix, vf, maxwell_garnett=False)
    # ensure that result is purely real
    assert np.issubdtype(n_eff.dtype, np.floating)
    # test that we get the right values; since the matrix volume fraciton is
    # zero, the effective index should be 1.33
    assert_equal(n_eff, np.ones(num_wavelengths)*1.33)

    # now test that this works with an actual dispersion relation
    wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
    # we add a small imaginary part to the particle index
    index = sc.index.polystyrene
    n_particle = (index(wavelen).expand_dims(dim={sc.Coord.LAYER: 1})
                   + 0.0001j).transpose()
    n_matrix = sc.index.water(wavelen)
    vf = 0.5
    n_eff_vectorized = sc.index.n_eff(n_particle, n_matrix, vf,
                                      maxwell_garnett=False)
    # ensure that result is complex
    assert np.issubdtype(n_eff_vectorized.dtype, np.complexfloating)

    # do same calculation using a for loop
    n_particle = n_particle.squeeze()
    n_matrix = n_matrix.squeeze()
    n_eff = [sc.index.n_eff(n_particle[i], n_matrix[i], vf,
                            maxwell_garnett=False)
             for i in range(len(n_particle))]

    # agreement shouldn't necessarily be exact because of tolerance of fsolve
    assert_allclose(n_eff_vectorized, np.array(n_eff))
