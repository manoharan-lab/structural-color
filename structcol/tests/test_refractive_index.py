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
from numpy.testing import assert_equal, assert_almost_equal
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

    def test_add(self):
        """Test that addition of Index objects and addition of scalars to Index
        objects works correctly.

        """
        self.wavelen = sc.Quantity(np.linspace(400, 800, 5), 'nm')
        def check_add_index(initial, constant):
            index = initial + constant
            if isinstance(constant, str):
                expected = initial(self.wavelen) + float(constant)
            else:
                expected = initial(self.wavelen) + constant
            xr.testing.assert_equal(index(self.wavelen), expected)
            # make sure that computed indexes are not Quantity objects
            assert not isinstance(index(self.wavelen).data, sc.Quantity)


        # adding complex, float, string, int, Quantity should all work
        check_add_index(sc.index.polystyrene, constant = 1.33 + 0.1j)
        check_add_index(sc.index.water, constant = 0.1555)
        check_add_index(sc.index.pmma, constant = "0.55")
        check_add_index(sc.index.vacuum, constant = int(1))
        check_add_index(sc.index.fused_silica, constant = sc.Quantity('0.1'))
        check_add_index(sc.index.fused_silica,
                        constant = sc.Quantity(0.1j, ''))

        # test with Index object from dispersion relation
        index = sc.index.polystyrene + sc.index.vacuum
        expected = (sc.index.polystyrene(self.wavelen) +
                    sc.index.vacuum(self.wavelen))
        xr.testing.assert_equal(index(self.wavelen), expected)

        # test with constant Index object
        constant = sc.Index.constant(0.01j)
        index = sc.index.fused_silica + constant
        expected = (sc.index.fused_silica(self.wavelen) +
                    constant(self.wavelen))
        xr.testing.assert_equal(index(self.wavelen), expected)

        # adding another type of object should not work
        with pytest.raises(ValueError):
            index = sc.index.polystyrene + np.ones(3)
        with pytest.raises(ValueError):
            index = sc.index.polystyrene + {}
        with pytest.raises(ValueError):
            index = sc.index.polystyrene + [1.3]

class TestEffectiveIndex():
    """Tests for the EffectiveIndex class and related effective_index function
    """
    def test_effective_index_objects(self):
        # test that we can build an effective index object and that it returns
        # expected results
        num_wavelengths = 10
        wavelen = sc.Quantity(np.linspace(400, 800, num_wavelengths), 'nm')
        index_particle = [sc.index.polystyrene, sc.index.fused_silica,
                          sc.index.ethanol, sc.index.pmma, sc.index.rutile]
        index_matrix = sc.index.water + 0.001j

        radius = sc.Quantity(np.array([100, 120, 140, 150, 160]), 'nm')
        sphere = sc.Sphere(index_particle, radius)
        vf_array = sphere.volume_fraction(total_volume_fraction=0.5)
        index_list = index_particle + [index_matrix]

        index_eff = sc.EffectiveIndex(index_list, vf_array,
                                      maxwell_garnett=False)
        n_eff_from_object = index_eff(wavelen)

        n_eff_from_func = sc.index.effective_index(index_list, vf_array,
                                                   wavelen,
                                                   maxwell_garnett=False)

        xr.testing.assert_equal(n_eff_from_object, n_eff_from_func)


    def test_effective_index(self):
        # test that at low volume fractions, Maxwell-Garnett and Bruggeman
        # roughly match for a non-core-shell particle
        wavelen = sc.Quantity(500.0, 'nm')
        index_particle = sc.Index.constant(2.7)
        index_matrix = sc.Index.constant(2.2)
        vf = xr.DataArray(np.array([0.001, 1-0.001]),
                          coords={sc.Coord.MAT: range(2)})

        neff_mg = sc.index.effective_index([index_particle, index_matrix], vf,
                                           wavelen, maxwell_garnett=True)
        neff_bg = sc.index.effective_index([index_particle, index_matrix], vf,
                                           wavelen, maxwell_garnett=False)

        xr.testing.assert_allclose(neff_mg, neff_bg)

        # test that the non-core-shell particle with Maxwell-Garnett matches
        # with core-shell (with shell index matched to matrix) with Bruggeman
        # at low volume fractions
        indices = np.array([index_particle, index_matrix, index_matrix])
        vf2 = xr.DataArray(np.array([0.001, 0.1, (1-0.001-0.1)]),
                           coords={sc.Coord.MAT: range(3)})
        neff_bg2 = sc.index.effective_index(indices, vf2, wavelen,
                                            maxwell_garnett=False)

        xr.testing.assert_allclose(neff_mg, neff_bg2)
        xr.testing.assert_allclose(neff_bg, neff_bg2)

        # test that the effective indices for a non-core-shell and a core-shell
        # of shell index matched to matrix using Bruggeman at intermediate
        # volume fractions
        vf3 = xr.DataArray([0.5, 1-0.5],
                           coords = {sc.Coord.MAT: range(2)})
        neff_bg3 = sc.index.effective_index([index_particle, index_matrix],
                                            vf3, wavelen,
                                            maxwell_garnett=False)

        vf3_cs = xr.DataArray(np.array([0.5, 0.1, (1-0.5-0.1)]),
                              coords = {sc.Coord.MAT: range(3)})
        neff_bg3_cs = sc.index.effective_index([index_particle, index_matrix,
                                                index_matrix], vf3_cs, wavelen,
                                               maxwell_garnett=False)

        xr.testing.assert_allclose(neff_bg3, neff_bg3_cs)

        # repeat the tests using complex indices
        index_particle_complex = sc.Index.constant(2.7+0.001j)
        index_matrix_complex = sc.Index.constant(2.2+0.001j)

        neff_mg_complex = sc.index.effective_index([index_particle_complex,
                                                    index_matrix_complex], vf,
                                                   wavelen,
                                                   maxwell_garnett=True)
        neff_bg_complex = sc.index.effective_index([index_particle_complex,
                                                    index_matrix_complex], vf,
                                                   wavelen,
                                                   maxwell_garnett=False)

        xr.testing.assert_allclose(neff_mg_complex, neff_bg_complex)

        # test that the non-core-shell particle with Maxwell-Garnett matches
        # with the core-shell of shell index matched to matrix with Bruggeman
        # at low volume fractions
        indices = [sc.Index.constant(2.7+0.001j),
                   sc.Index.constant(2.2+0.001j)]
        neff_bg2_complex = sc.index.effective_index(indices +
                                                    [index_matrix_complex],
                                                    vf2, wavelen,
                                                    maxwell_garnett=False)

        xr.testing.assert_allclose(neff_mg_complex, neff_bg2_complex)
        xr.testing.assert_allclose(neff_bg_complex, neff_bg2_complex)

        # test that the effective indices for a non-core-shell and a core-shell
        # of shell index matched to matrix match using Bruggeman at
        # intermediate volume fractions
        neff_bg3_complex = sc.index.effective_index([index_particle_complex,
                                                     index_matrix_complex],
                                                    vf3, wavelen,
                                                    maxwell_garnett=False)

        neff_bg3_cs_complex = sc.index.effective_index(indices +
                                                       [index_matrix_complex],
                                                       vf3_cs, wavelen,
                                                       maxwell_garnett=False)

        xr.testing.assert_allclose(neff_bg3_complex, neff_bg3_cs_complex)

    def test_multimaterial_bruggeman(self):
        """Tests the Bruggeman approximation for three or more materials
        """
        # five layers, all same index.  Total volume fraction is 1, so result
        # should not depend on index of matrix
        wavelen = sc.Quantity(500.0, 'nm')
        index = sc.Index.constant(1.33)
        layers = 5
        index_particle = [index]*layers
        index_matrix = sc.Index.constant(1.0)
        vf = xr.DataArray([1/layers]*layers + [0],
                          coords={sc.Coord.MAT: range(layers+1)})
        n_eff = sc.index.effective_index(index_particle + [index_matrix], vf,
                                         wavelen)
        xr.testing.assert_allclose(n_eff, index(wavelen))

        wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
        # three layers, outer layer same as matrix.  Should return same as two
        # layers
        index_matrix = sc.Index.constant(1.33)
        index_particle = [sc.Index.constant(1.0), sc.Index.constant(1.59),
                          sc.Index.constant(1.33)]
        vf = xr.DataArray(np.array([0.2, 0.2, 0.2, 1-0.6]),
                          coords={sc.Coord.MAT: range(4)})
        n_threelayer_eff = sc.index.effective_index(index_particle +
                                                    [index_matrix], vf,
                                                    wavelen)

        # two layers
        index_particle = [sc.Index.constant(1.0), sc.Index.constant(1.59)]
        vf = xr.DataArray(np.array([0.2, 0.2, 1-0.4]),
                          coords={sc.Coord.MAT: range(3)})
        n_twolayer_eff = sc.index.effective_index(index_particle +
                                                  [index_matrix], vf,
                                                  wavelen)
        xr.testing.assert_allclose(n_threelayer_eff, n_twolayer_eff)

    def test_vectorized_maxwell_garnett(self):
        """Tests whether Maxwell-Garnett works on multiple wavelengths at once

        """
        # since indices are constant, should get same result at all wavelengths
        # as at a single wavelength
        wavelen = sc.Quantity(np.linspace(400.0, 800.0, 10), 'nm')
        index_particle = sc.Index.constant(1.33)
        index_matrix = sc.Index.constant(1.00)
        vf = xr.DataArray(np.array([0.5, 0.5]),
                          coords={sc.Coord.MAT: range(2)})
        n_mg_vector = sc.index.effective_index([index_particle, index_matrix],
                                               vf, wavelen,
                                               maxwell_garnett=True)
        single_wavelen = sc.Quantity(400.0, 'nm')
        n_mg_single = sc.index.effective_index([index_particle, index_matrix],
                                               vf, single_wavelen,
                                               maxwell_garnett=True)
        xr.testing.assert_equal(n_mg_vector,
                                (n_mg_single.to_numpy()
                                 * xr.ones_like(n_mg_vector)))

    def test_vectorized_bruggeman(self):
        """Tests that Bruggeman effective index works on multiple wavelengths
        at once.

        """
        num_wavelengths = 10
        wavelen = sc.Quantity(np.linspace(400, 800, num_wavelengths), 'nm')
        index_particle = sc.Index.constant(1.33)
        num_layers = 5
        index_list = [index_particle] * num_layers
        radius = sc.Quantity(np.array([100, 120, 140, 150, 160]), 'nm')
        # radius shouldn't matter for this calculation
        sphere = sc.Sphere(index_list, radius)
        vf_array = sphere.volume_fraction(total_volume_fraction=1.0)
        index_matrix = sc.Index.constant(1.00)

        n_effective = sc.index.effective_index(index_list + [index_matrix],
                                               vf_array, wavelen,
                                               maxwell_garnett=False)
        # ensure that result is purely real
        assert np.issubdtype(n_effective.dtype, np.floating)
        # test that we get the right values; since the matrix volume fraction
        # is zero, the effective index should be 1.33. Test also that coords
        # are correct.
        coords = {sc.Coord.WAVELEN: wavelen.to_preferred().magnitude}
        xr.testing.assert_equal(n_effective,
                                xr.DataArray(np.ones(num_wavelengths)*1.33,
                                             coords = coords))

        # now test that this works with an actual dispersion relation
        wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
        # we add a small imaginary part to the particle index
        index = sc.index.polystyrene + 0.0001j
        index_matrix = sc.index.water
        vf = 0.5
        sphere = sc.Sphere(index, sc.Quantity(100, 'nm'))
        vf_array = sphere.volume_fraction(total_volume_fraction=vf)
        n_effective_vectorized = sc.index.effective_index([index,
                                                           index_matrix],
                                                          vf_array, wavelen,
                                                          maxwell_garnett =
                                                          False)
        # ensure that result is complex
        assert np.issubdtype(n_effective_vectorized.dtype, np.complexfloating)

        # do same calculation using a for loop
        n_effective = [sc.index.effective_index([index, index_matrix],
                                                vf_array, wl,
                                                maxwell_garnett=False)
                       for wl in wavelen]

        # agreement shouldn't necessarily be exact because of tolerance of
        # fsolve
        xr.testing.assert_allclose(n_effective_vectorized,
                                   xr.concat(n_effective,
                                             dim=sc.Coord.WAVELEN))


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

    # multiple wavelengths, single layer should give shape
    # [num_wavelengths, 1]
    assert ratio.shape == (len(wavelen), 1)
    assert_equal(ratio.squeeze(), n_particle/n_matrix)

    # single wavelength, single layer should give scalar
    ratio = sc.index.ratio(n_particle[0], n_matrix[0])
    assert np.isscalar(ratio)
    assert_equal(ratio, (n_particle[0]/n_matrix[0]).to_numpy().item())

    # single wavelength, multiple layers should give shape
    # [1, num_layers]
    num_layers = 35
    index_particle = num_layers*[sc.index.polystyrene]
    n_particle = sc.index._indexes_from_list(index_particle, wavelen[0])
    ratio = sc.index.ratio(n_particle.isel(wavelength=0), n_matrix[0])
    assert ratio.shape == (1, num_layers)

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

