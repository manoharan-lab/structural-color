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

from .. import Quantity, q, np
from numpy.testing import assert_equal
import xarray as xr
import pytest
import structcol as sc
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
    with pytest.raises(DimensionalityError):
        q(Quantity('0.5 J'), Quantity('0.5 rad'))
    with pytest.raises(DimensionalityError):
        q(Quantity('450 nm'), Quantity('0.5 m'))


# test both scalars and vectors
@pytest.mark.parametrize("wavelen", [400, np.linspace(400, 800, 20)])
@pytest.mark.parametrize("volume_fraction", [0.5, np.linspace(0, 1.0, 20)])
def test_size_parameter(wavelen, volume_fraction):
    wavelen = sc.Quantity(wavelen, "nm")

    # first look at non-effective index, single-layer particle
    index_matrix = sc.index.water
    index_particle = sc.Index.constant(1.5 + 0.1j)
    radius = sc.Quantity(0.2, "um")
    sphere = sc.Sphere(index_particle, radius)
    n_particle = sphere.n(wavelen)

    x = sc.size_parameter(n_particle, radius)
    # should be 1 material, no volume fraction dimension
    assert x.sizes == {sc.Coord.WAVELEN: len(np.atleast_1d(wavelen)),
                       sc.Coord.MAT: 1}

    # now try effective index, single-layer particle
    index_effective = sc.EffectiveIndex.from_particle(sphere, volume_fraction,
                                                      index_matrix)
    n_eff = index_effective(wavelen)
    x = sc.size_parameter(n_eff, radius)

    if not np.isscalar(volume_fraction):
        assert x.sizes[sc.Coord.VOLFRAC] == len(volume_fraction)
    assert x.sizes[sc.Coord.WAVELEN] == len(np.atleast_1d(wavelen))
    assert x.sizes[sc.Coord.MAT] == 1

    # now try core-shell, non-effective index
    index_core = sc.index.polystyrene
    index_shell = sc.index.water
    index_matrix = sc.index.vacuum
    radii = sc.Quantity(np.array([50, 100]), "nm")

    cs_sphere = sc.Sphere([index_core, index_shell], radii)
    n_particle = cs_sphere.n(wavelen)
    x = sc.size_parameter(n_particle, radii)
    # should be 2 materials, no volume fraction dimension
    assert x.sizes == {sc.Coord.WAVELEN: len(np.atleast_1d(wavelen)),
                       sc.Coord.MAT: 2}

    # core-shell, effective index
    index_effective = sc.EffectiveIndex.from_particle(cs_sphere,
                                                      volume_fraction,
                                                      index_matrix)
    n_eff = index_effective(wavelen)
    x = sc.size_parameter(n_eff, radii)
    if not np.isscalar(volume_fraction):
        assert x.sizes[sc.Coord.VOLFRAC] == len(volume_fraction)
    assert x.sizes[sc.Coord.WAVELEN] == len(np.atleast_1d(wavelen))
    assert x.sizes[sc.Coord.MAT] == 2


@pytest.mark.parametrize("angles", [10, np.linspace(0, 180, 90)])
@pytest.mark.parametrize("wavelen", [400, np.linspace(400, 800, 100)])
def test_ql(wavelen, angles):
    """Test that the function for computing the nondimensional
    momentum-transfer vector returns expected values and is properly
    vectorized.

    """
    angles = sc.Quantity(angles, 'deg')
    index = sc.index.polystyrene
    wavelen = sc.Quantity(wavelen, 'nm')
    lengthscale = sc.Quantity(0.2, 'um')

    n_medium = index(wavelen)

    ql = sc.ql(n_medium, lengthscale, angles)

    # expected ql as calculated using numpy
    x = sc.size_parameter(n_medium, lengthscale).to_numpy()
    ql_expected = (4*np.abs(x).max(axis=1)[..., np.newaxis]
                   * np.sin(angles.to('rad').magnitude/2))

    assert_equal(ql.to_numpy(), ql_expected)
    assert ql.dims == (sc.Coord.WAVELEN, sc.Coord.THETA)

    # make sure ql represents the outer radius for layered particles
    radii = sc.Quantity(np.array([0.1, 0.2, 0.3]), 'um')
    ql = sc.ql(n_medium, radii, angles)
    x = sc.size_parameter(n_medium, radii).to_numpy()
    ql_expected = (4*np.abs(x).max(axis=1)[..., np.newaxis]
                   * np.sin(angles.to('rad').magnitude/2))

    assert_equal(ql.to_numpy(), ql_expected)
    assert ql.dims == (sc.Coord.WAVELEN, sc.Coord.THETA)

@pytest.mark.parametrize("wavelen", [sc.Quantity(400, "nm"),
                                     0.4,
                                     sc.Quantity(np.linspace(400, 800, 10),
                                                 "nm"),
                                     np.linspace(0.4, 0.8, 20)])
def test_make_input_coords(wavelen):
    """Tests the convenience function to make DataArray coords from numpy
    arrays or scalars"""
    thetas = sc.Quantity(np.linspace(0, np.pi, 10), "rad")
    phis = sc.Quantity(np.linspace(0, 360, 12), "deg")

    coords = sc._make_input_coords(wavelen, thetas)
    assert np.ndim(coords[sc.Coord.WAVELEN]) == 1
    assert sc.Coord.THETA in coords
    assert sc.Coord.PHI not in coords

    # ensure that coords work with thetas, phis specified from either meshgrid
    # or as separate 1D arrays.  First 1D:
    coords_1d = sc._make_input_coords(wavelen, thetas, phis=phis)
    assert sc.Coord.THETA in coords_1d
    assert sc.Coord.PHI in coords_1d

    # we need to specify phis if thetas comes from meshgrid
    thetas, phis = np.meshgrid(thetas, phis, indexing="ij")
    with pytest.raises(ValueError, match="thetas specified as"):
        coords = sc._make_input_coords(wavelen, thetas)
    coords_2d = sc._make_input_coords(wavelen, thetas, phis=phis)
    xr.testing.assert_equal(coords_2d, coords_1d)
