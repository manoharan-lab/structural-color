# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang, Sofia Magkiriadou,
# Anna B. Stephenson
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
Tests for the Particle class and subclasses (in structcol/particle.py)

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import np, mie
from numpy.testing import assert_allclose, assert_equal
import pytest
import structcol as sc
import xarray as xr
from pint.errors import DimensionalityError

class TestParticle():
    """Tests for the Particle class and derived classes
    """
    wavelen = sc.Quantity(np.linspace(400, 800, 100), 'nm')
    def test_particle_construction(self):
        index = 1.445
        size = sc.Quantity(150, 'nm')

        # particle construction without units or with wrong units should fail
        with pytest.raises(DimensionalityError):
            my_particle = sc.Particle(sc.Index.constant(index),
                                                0.15)
        with pytest.raises(DimensionalityError):
            my_particle = sc.Particle(sc.Index.constant(index),
                                                sc.Quantity(0.15, 'kg'))

        my_particle = sc.Particle(sc.Index.constant(index), size)
        # make sure index is stored and calculated correctly
        n = my_particle.n(self.wavelen)
        assert_equal(n, np.ones_like(self.wavelen)*index)
        # check stored units
        assert n.attrs[sc.Attr.LENGTH_UNIT] == size.to_preferred().units

        # make sure size is stored as DataArray with the correct dimensions and
        # units
        assert isinstance(my_particle.size, xr.DataArray)
        assert my_particle.size.shape == ()
        assert (my_particle.size.attrs[sc.Attr.LENGTH_UNIT]
                == size.to_preferred().units)

        # make sure reported units of size are correct
        assert_equal(size.to_preferred().magnitude, my_particle.size_q.magnitude)

    def test_sphere_construction(self):
        radius = sc.Quantity(150, 'nm')

        # test that both index and radius must be specified:
        with pytest.raises(KeyError):
            my_sphere = sc.Sphere(sc.index.polystyrene)
        with pytest.raises(DimensionalityError):
            my_sphere = sc.Sphere(sc.index.polystyrene, 0.15)

        my_sphere = sc.Sphere(sc.index.polystyrene, radius)

        # test that index works as expected
        n = my_sphere.n(self.wavelen)
        xr.testing.assert_equal(n, sc.index.polystyrene(self.wavelen))
        assert n.attrs[sc.Attr.LENGTH_UNIT] == radius.to_preferred().units

        # make sure diameter is correct
        assert_equal(radius.to_preferred().magnitude * 2,
                     my_sphere.diameter_q.magnitude)
        assert_equal(radius.to_preferred().magnitude,
                     my_sphere.radius_q.magnitude)
        assert_equal(radius.to_preferred().magnitude * 2,
                     my_sphere.diameter.to_numpy())
        assert_equal(radius.to_preferred().magnitude,
                     my_sphere.radius.to_numpy())

        # check that the particle is not marked as layered either in object or
        # in size array
        assert not my_sphere.layered
        assert sc.Coord.LAYER not in my_sphere.radius

        # test outer diameter
        assert (my_sphere.outer_diameter == 2*radius.to_preferred().magnitude)
        assert my_sphere.current_units == radius.to_preferred().units

    def test_core_shell_single_wavelength(self):
        index = [sc.index.vacuum, sc.index.polystyrene]
        radii = sc.Quantity([0.15, 0.16], 'um')

        my_core_shell = sc.Sphere(index, radii)
        assert my_core_shell.layered
        assert len(my_core_shell.radius.coords[sc.Coord.LAYER]) == 2

        # test that index works as expected
        wavelen = sc.Quantity(400, 'nm')
        n = my_core_shell.n(wavelen)
        xr.testing.assert_equal(n.sel({sc.Coord.LAYER: 0}, drop=True),
                                sc.index.vacuum(wavelen))

    def test_layered_sphere(self):
        index = [sc.index.vacuum, sc.index.polystyrene, sc.index.water]
        radii = sc.Quantity([0.15, 0.16, 0.18], 'um')
        radii_wrong_order = sc.Quantity([0.15, 0.17, 0.14], 'um')
        radii_too_many = sc.Quantity([0.15, 0.16, 0.17, 0.18], 'um')
        radii_too_few = sc.Quantity(0.15, 'um')

        with pytest.raises(ValueError):
            my_layered_sphere = sc.Sphere(index, radii_wrong_order)
        with pytest.raises(ValueError):
            my_layered_sphere = sc.Sphere(index, radii_too_many)
        with pytest.raises(ValueError):
            my_layered_sphere = sc.Sphere(index, radii_too_few)

        my_layered_sphere = sc.Sphere(index, radii)
        assert_equal(radii.to_preferred().magnitude, my_layered_sphere.size)
        assert_equal((radii.to_preferred() * 2).magnitude,
                     my_layered_sphere.diameter_q.magnitude)
        assert_equal((radii.to_preferred() * 2).units,
                     my_layered_sphere.diameter_q.units)
        assert my_layered_sphere.layered

        # test that index works as expected
        n = my_layered_sphere.n(self.wavelen)
        xr.testing.assert_equal(n.sel({sc.Coord.LAYER: 0}, drop=True),
                                sc.index.vacuum(self.wavelen))
        xr.testing.assert_equal(n.sel({sc.Coord.LAYER: 1}, drop=True),
                                sc.index.polystyrene(self.wavelen))
        xr.testing.assert_equal(n.sel({sc.Coord.LAYER: 2}, drop=True),
                                sc.index.water(self.wavelen))
        assert n.attrs[sc.Attr.LENGTH_UNIT] == radii.to_preferred().units

        # test number of layers
        assert my_layered_sphere.layers == len(radii)
        # test that size array has the proper number of layers too
        num_layers = len(my_layered_sphere.radius.coords[sc.Coord.LAYER])
        assert num_layers == len(radii)

        # test outer diameter
        assert (my_layered_sphere.outer_diameter ==
                2*radii[-1].to_preferred().magnitude)
        assert my_layered_sphere.current_units == radii.to_preferred().units

    def test_volume_fraction(self):
        """test that calculations of volume fraction for each layer work

        """
        index = [sc.index.vacuum, sc.index.polystyrene, sc.index.fused_silica,
                 sc.index.water]
        # can do the calculation by hand for the following radii
        radii = sc.Quantity([0.1, 0.2, 0.3, 1.0], 'um')
        my_layered_sphere = sc.Sphere(index, radii)
        vf = my_layered_sphere.volume_fraction()
        vf_expected = xr.DataArray([0.1**3, 0.2**3 - 0.1**3, 0.3**3 - 0.2**3,
                                    1 - 0.3**3],
                                   coords = {sc.Coord.MAT : range(4)})
        xr.testing.assert_equal(vf, vf_expected)

        # try with total volume fraction specified
        vf = my_layered_sphere.volume_fraction(total_volume_fraction=1)
        vf_expected = xr.DataArray([0.1**3, 0.2**3 - 0.1**3, 0.3**3 - 0.2**3,
                                    1 - 0.3**3, 0],
                                   coords = {sc.Coord.MAT : range(5)})
        xr.testing.assert_equal(vf, vf_expected)

        # try with a different value of total volume fraction
        vf = my_layered_sphere.volume_fraction(total_volume_fraction=0.5)
        vf_expected = vf_expected * 0.5
        vf_expected[-1] = 1-0.5
        xr.testing.assert_equal(vf, vf_expected)

        # test with a nonlayered sphere
        radius = sc.Quantity(150, 'nm')
        sphere = sc.Sphere(sc.index.polystyrene, radius)

        vf = sphere.volume_fraction()
        vf_expected = xr.DataArray([1.0], coords={sc.Coord.MAT: range(1)})
        xr.testing.assert_equal(vf, vf_expected)

        phi = 0.3256687
        vf = sphere.volume_fraction(total_volume_fraction=phi)
        vf_expected = xr.DataArray([phi, 1-phi],
                                   coords={sc.Coord.MAT: range(2)})
        xr.testing.assert_equal(vf, vf_expected)

        # should not work with a generic Particle
        particle = sc.Particle(sc.index.polystyrene, radius)
        with pytest.raises(NotImplementedError):
            particle.volume_fraction()

    def test_index_list(self):
        """test that index_list method reports correct results

        """
        # test for multilayer sphere
        indexes = [sc.index.vacuum, sc.index.polystyrene,
                   sc.index.fused_silica, sc.index.water]
        radii = sc.Quantity([0.1, 0.2, 0.3, 1.0], 'um')
        my_layered_sphere = sc.Sphere(indexes, radii)
        index_list = my_layered_sphere.index_list()
        assert index_list == indexes
        assert isinstance(index_list, list)

        # multilayer sphere, method used with matrix index specified
        index_matrix = sc.index.vacuum
        index_list = my_layered_sphere.index_list(index_matrix)
        assert index_list == list(indexes) + [index_matrix]
        # make sure that lists/arrays are not nested
        for index in index_list:
            assert isinstance(index, sc.Index)

        # should work also for a generic particle
        my_particle = sc.Particle(indexes, radii)
        index_list = my_particle.index_list(index_matrix)
        assert index_list == list(indexes) + [index_matrix]

        # test with a nonlayered sphere
        radius = sc.Quantity(150, 'nm')
        sphere = sc.Sphere(sc.index.polystyrene, radius)
        index_list = sphere.index_list(index_matrix)
        assert index_list == [sc.index.polystyrene, index_matrix]

    def test_form_factor(self):
        """Test that we get the same results from calling the
        Sphere.form_factor() method as we do from calling pymie directly.

        """
        # The pymie/tests/test_mie.py::test_form_factor test checks that the
        # Mie calculation gives the correct results for these parameters. Here
        # we just check to see if we get the same results as pymie
        wavelen = sc.Quantity('658.0 nm')
        radius = sc.Quantity('0.85 um')
        index_matrix = sc.Index.constant(1.00)
        n_matrix = index_matrix(wavelen)
        index_particle = sc.Index.constant(1.59 + 1e-4 * 1.0j)
        sphere = sc.Sphere(index_particle, radius)
        angles = sc.Quantity(np.linspace(0, 180., 19), 'deg')
        ipar_sphere, iperp_sphere = sphere.form_factor(wavelen, angles,
                                                       index_matrix)

        m = sc.index.ratio(sphere.n(wavelen), index_matrix(wavelen))
        x = sc.size_parameter(index_matrix(wavelen), radius).to_numpy()
        ipar_mie, iperp_mie = mie.calc_ang_dist(m, x, angles)

        assert_equal(ipar_sphere, ipar_mie)
        assert_equal(iperp_sphere, iperp_mie)

        # test calculations for gold, which has a high imaginary refractive
        # index.  Again, pymie/tests/test_mie.py::test_absorbing_materials()
        # checks that the Mie calculation gives the correct results for these
        # parameters. Here we just check to see if we get the same results as
        # pymie
        wavelen = sc.Quantity('658.0 nm')
        x = 10.0
        radius = x/(2*np.pi/wavelen)
        index_matrix = sc.Index.constant(1.00)
        gold_index = sc.Index.constant(0.1425812 + 3.6813284 * 1.0j)
        sphere = sc.Sphere(gold_index, radius)
        angles = sc.Quantity(np.linspace(0, 90., 10), 'deg')
        ipar_sphere, iperp_sphere = sphere.form_factor(wavelen, angles,
                                                       index_matrix)

        m = sc.index.ratio(sphere.n(wavelen), index_matrix(wavelen))
        ipar_mie, iperp_mie = mie.calc_ang_dist(m, x, angles)

        assert_equal(ipar_sphere, ipar_mie)
        assert_equal(iperp_sphere, iperp_mie)

        # Test absorbing matrix.
        # Although Sphere.form_factor() calls the same function
        # (diff_scat_intensity_complex_medium) used here, the results may not
        # be equal if units are converted in different ways.  So to test for
        # equality, we first convert radius and distance to preferred units.
        radius = sc.Quantity('120.0 nm').to_preferred()
        sphere = sc.Sphere(sc.Index.constant(1.5+0.001j), radius)
        distance = sc.Quantity(10000.0,'nm').to_preferred()
        index_matrix = sc.Index.constant(1.0+0.001j)
        angles = sc.Quantity(np.linspace(0, 90., 10), 'deg')

        # not specifying distance should throw exception
        with pytest.raises(ValueError):
            _ = sphere.form_factor(wavelen, angles, index_matrix)

        m = sc.index.ratio(sphere.n(wavelen), index_matrix(wavelen))
        x = sc.size_parameter(index_matrix(wavelen), radius)
        k = 2 * np.pi * index_matrix(wavelen).to_numpy() / wavelen

        ipar_sphere, iperp_sphere = sphere.form_factor(wavelen, angles,
                                                       index_matrix,
                                                       kd=k*distance)


        ipar_mie, iperp_mie = mie.diff_scat_intensity_complex_medium(
            m, x, angles, k*distance)

        assert_equal(ipar_sphere, ipar_mie)
        assert_equal(iperp_sphere, iperp_mie)

        # test layered particle
        index = [sc.index.vacuum, sc.index.polystyrene, sc.index.pmma]
        wavelen = sc.Quantity('658.0 nm').to_preferred()
        radii = sc.Quantity([0.10, 0.16, 0.25], 'um').to_preferred()
        sphere = sc.Sphere(index, radii)
        angles = sc.Quantity(np.linspace(0, 180., 19), 'deg')
        index_matrix = sc.index.water

        ipar_sphere, iperp_sphere = sphere.form_factor(wavelen, angles,
                                                       index_matrix)

        m = sc.index.ratio(sphere.n(wavelen), index_matrix(wavelen))
        x = sc.size_parameter(index_matrix(wavelen), radii).to_numpy()
        ipar_mie, iperp_mie = mie.calc_ang_dist(m, x, angles)

        assert_equal(ipar_sphere, ipar_mie)
        assert_equal(iperp_sphere, iperp_mie)

    def test_vectorized_form_factor(self):
        # test that we can calculate the form factor for several wavelengths
        # simultaneously.
        num_wavelengths = 10
        num_angles = 19
        wavelen = sc.Quantity(np.linspace(400, 800, num_wavelengths), 'nm')
        sphere = sc.Sphere(sc.index.polystyrene,
                                 sc.Quantity('0.125 um'))
        index_matrix = sc.index.water
        angles = sc.Quantity(np.linspace(0, 180., num_angles), 'deg')
        form_sphere = sphere.form_factor(wavelen, angles, index_matrix)

        # make sure shape is correct
        for i in range(2):
            assert form_sphere[i].shape == (num_wavelengths, num_angles)

        # test that we get same values from a loop
        ipar = np.zeros((num_wavelengths, num_angles))
        iperp = np.zeros((num_wavelengths, num_angles))
        for i in range(num_wavelengths):
            ipar[i], iperp[i] = sphere.form_factor(wavelen[i], angles,
                                                   index_matrix)

        assert_equal(form_sphere[0], ipar)
        assert_equal(form_sphere[1], iperp)

class TestSphereDistribution():
    """Tests for the SphereDistribution class.
    """
    wavelen = sc.Quantity(400, 'nm')
    angles = sc.Quantity(np.linspace(0, 180., 19), 'deg')

    def test_spheredistribution_construction(self):
        radius = sc.Quantity(150, 'nm')
        concentrations = [1.0, 0.0]
        pdi = 0.15

        # test construction with monospecies
        sphere1 = sc.Sphere(sc.index.polystyrene, radius)
        dist = sc.SphereDistribution(sphere1, concentrations, pdi)

        with pytest.raises(ValueError, match=r"Concentrations must"):
            dist = sc.SphereDistribution(sphere1, 0.5, pdi)
        dist = sc.SphereDistribution(sphere1, 1.0, pdi)
        assert not dist.has_layered

        # test construction with layered sphere
        index = [sc.index.vacuum, sc.index.polystyrene]
        radii = sc.Quantity([0.15, 0.16], 'um')

        sphere2 = sc.Sphere(index, radii)
        dist = sc.SphereDistribution(sphere2, concentrations, pdi)
        assert dist.has_layered

        # test construction with two spheres, one layered
        concentrations = [0.1, 0.5]
        pdi = [0.15, 0]
        with pytest.raises(ValueError, match=r"Concentrations must"):
            dist = sc.SphereDistribution([sphere1, sphere2],
                                                  concentrations, pdi)
        concentrations = [0.5, 0.5]
        dist = sc.SphereDistribution([sphere1, sphere2],
                                              concentrations, pdi)
        assert dist.has_layered

        assert_equal(dist.diameters, [2*radius.to_preferred().magnitude,
                                      2*radii[-1].to_preferred().magnitude])
        assert_equal(dist.concentrations, concentrations)

        # zero polydispersity for species 2 should have been replaced with
        # finite polydispersity
        assert_equal(dist.pdi, [pdi[0], dist.polydispersity_bound])

        # at the moment, works only with one or two species, not more
        with pytest.raises(ValueError, match=r"Can only handle one or two"):
            dist = sc.SphereDistribution([sphere1, sphere2, sphere1],
                                                  concentrations, pdi)

    def test_spheredistribution_formfactor(self):
        concentrations = [0.5, 0.5]
        pdi = [0.1, 0.2]
        index_external = sc.index.water

        radius1 = sc.Quantity(150, 'nm')
        index1 = sc.index.polystyrene
        sphere1 = sc.Sphere(index1, radius1)

        # shouldn't work with two spheres with different refractive indices
        radius2 = sc.Quantity(180, 'nm')
        index2 = sc.index.fused_silica
        sphere2 = sc.Sphere(index2, radius2)

        dist = sc.SphereDistribution([sphere1, sphere2],
                                              concentrations, pdi)
        with pytest.raises(ValueError, match=r"Currently can handle"):
            dist.form_factor(self.wavelen, self.angles, index_external)

        # shouldn't work when one or both spheres are layered
        radius2 = sc.Quantity([0.15, 0.16], 'um')
        index2 = [sc.index.vacuum, sc.index.polystyrene]

        sphere2 = sc.Sphere(index2, radius2)
        dist = sc.SphereDistribution(sphere2, [1.0, 0], pdi[0])
        with pytest.raises(ValueError, match=r"Cannot handle polydispersity"):
            dist.form_factor(self.wavelen, self.angles, index_external)

        dist = sc.SphereDistribution([sphere1, sphere2],
                                              concentrations, pdi)
        with pytest.raises(ValueError, match=r"Cannot handle polydispersity"):
            dist.form_factor(self.wavelen, self.angles, index_external)

        # should work with one species
        dist = sc.SphereDistribution(sphere1, 1.0, 0.0)
        # when polydispersity is small, should be close to monodisperse form
        # factor
        polyff = dist.form_factor(self.wavelen, self.angles, index_external)
        monoff = sphere1.form_factor(self.wavelen, self.angles, index_external)
        assert_allclose(polyff, monoff)
