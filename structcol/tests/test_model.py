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
Tests for the single-scattering model (in structcol/model.py)

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
"""

from .. import Quantity, np, mie
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_allclose)
import pytest
import structcol as sc
import xarray as xr

class TestModel():
    """Tests for the Model class and derived classes.
    """
    wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
    ps_radius = sc.Quantity('0.125 um')
    index_particle = sc.index.polystyrene
    ps_sphere = sc.Sphere(index_particle, ps_radius)
    hollow_sphere = sc.Sphere([sc.index.vacuum, sc.index.polystyrene],
                                         sc.Quantity([125, 135], 'nm'))
    qd = np.arange(0.1, 20, 0.01)
    # test vectorization across volume fractions
    phi = np.array([0.15, 0.3, 0.45])
    my_units = sc.ureg.millimeter
    thickness = 0.050 * sc.ureg.millimeter

    angles = sc.Quantity(np.linspace(0, np.pi, 100), 'rad')
    coords = sc._make_input_coords(wavelen, angles)
    angles_da = xr.DataArray(angles.magnitude, {sc.Coord.THETAIDX:
                                                range(len(angles))})

    def test_base_model(self):
        """tests for Model base class"""
        model = sc.model.Model(sc.index.vacuum)
        with pytest.raises(NotImplementedError):
            model.differential_cross_section(self.coords)

    def test_formstructure_model(self):
        """tests for the FormStructureModel"""
        wavelen = self.wavelen

        # construction with particle and without volume_fraction should fail
        with pytest.raises(ValueError, match="volume_fraction must be"):
            model = sc.model.FormStructureModel(None, None, self.ps_radius,
                                                sc.index.vacuum,
                                                sc.index.vacuum,
                                                particle = self.ps_sphere)

        # not specifying particle/volume_fraction should lead
        # to an error when calculating number density
        model = sc.model.FormStructureModel(None, None, self.ps_radius,
                                            sc.index.vacuum, sc.index.vacuum)
        with pytest.raises(ValueError, match="Number density cannot"):
            _ = model.number_density

        # only specifying volume_fraction should also lead to an error
        model = sc.model.FormStructureModel(None, None, self.ps_radius,
                                            sc.index.vacuum, sc.index.vacuum,
                                            volume_fraction = 0.5)
        with pytest.raises(ValueError, match="Number density cannot"):
            _ = model.number_density

        # if form factor is None and structure factor is a constant, should
        # have a constant differential scattering cross section corresponding
        # to 1/|k|^2
        k = sc.wavevector(sc.index.vacuum(wavelen))
        const = 1.0
        model = sc.model.FormStructureModel(None, sc.structure.Constant(const),
                                            self.ps_radius,
                                            sc.index.vacuum,
                                            sc.index.vacuum)
        coords = model.make_input_coords(wavelen, self.angles)
        dscat = model.differential_cross_section(coords)

        xr.testing.assert_equal(dscat, xr.ones_like(dscat)*const/np.abs(k)**2)

        # Test that constant structure factor yields the same results as form
        # factor.  There is no effective index for a FormStructureModel, unless
        # one is explicitly given
        model = sc.model.FormStructureModel(self.ps_sphere.form_factor,
                                            sc.structure.Constant(const),
                                            self.ps_radius,
                                            sc.index.vacuum,
                                            sc.index.vacuum)
        dscat = model.differential_cross_section(coords)
        ff = self.ps_sphere.form_factor(coords, sc.index.vacuum)

        # dscat contains avg polarization; ff does not.  Also ff is
        # nondimensional, so have to divide by k^2 to compare
        xr.testing.assert_equal(dscat.loc["par":"perp"], ff/np.abs(k**2))

        # Test that constant form factor yields the same results as structure
        # factor.
        structure_factor = sc.structure.PercusYevick(0.5)
        index_matrix = sc.index.water
        model = sc.model.FormStructureModel(None,
                                            structure_factor,
                                            self.ps_radius,
                                            index_matrix,
                                            sc.index.vacuum)
        dscat = model.differential_cross_section(coords)

        ql = sc.ql(index_matrix(wavelen), self.ps_radius, self.angles_da)
        s = structure_factor(ql).to_numpy().squeeze()

        # test numpy versions because DataArrays will have different coords
        #
        # dscat for both polarizations should be equal to s, after
        # nondimensionalizing dscat by k^2
        k = sc.wavevector(index_matrix(wavelen))
        assert_allclose((dscat[0]*np.abs(k)**2).to_numpy().squeeze(), s,
                        rtol=1e-15)
        assert_allclose((dscat[1]*np.abs(k)**2).to_numpy().squeeze(), s,
                        rtol=1e-15)

    def test_hardsphere_model(self):
        """tests that HardSphere model construction and differential cross
        section method work.  The functions are vectorized over both wavelength
        and volume fraction

        """
        index_matrix = sc.index.water
        glass = sc.model.HardSpheres(self.ps_sphere, self.phi, sc.index.water,
                                     sc.index.vacuum)

        # make sure form factor is calculated correctly
        angles = sc.Quantity(np.linspace(0, 180., 19), 'deg')
        coords = glass.make_input_coords(self.wavelen, angles)
        # use the effective index, which should depend on volume fraction
        form_model = glass.form_factor(coords, glass.index_external)
        form_sphere = glass.sphere.form_factor(coords, glass.index_external)
        xr.testing.assert_equal(form_model, form_sphere)
        # make sure function vectorized properly in both dimensions
        assert form_model.sizes[sc.Coord.VOLFRAC] == len(self.phi)
        assert form_model.sizes[sc.Coord.WAVELEN] == len(self.wavelen)

        # make sure structure factor is calculated correctly
        s_ps = glass.structure_factor(self.qd)
        structure_factor = sc.structure.PercusYevick(self.phi)
        assert_equal(s_ps.to_numpy(), structure_factor(self.qd).to_numpy())

        # make sure structure factor is the same for layered spheres as for
        # solid spheres
        glass = sc.model.HardSpheres(self.hollow_sphere, self.phi,
                                     sc.index.water, sc.index.vacuum)
        s_hollow = glass.structure_factor(self.qd)
        xr.testing.assert_equal(s_hollow, structure_factor(self.qd))

    def test_polydispersehardsphere_model(self):
        """tests that PolydisperseHardSphere model construction and
        differential cross section method work

        """
        index_matrix = sc.index.water
        index_medium = sc.index.vacuum

        # single particle species, low volume fraction
        volume_fraction = 1e-8
        pdi = 1e-5
        concentration = 1.0
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)

        # for this low volume fraction, form factor should dominate.  Note that
        # we vectorize over wavelength
        wavelen = self.wavelen
        # start at a few degrees to avoid division by zero error
        angles = sc.Quantity(np.linspace(2, 180., 19),
                             'deg').to("rad").magnitude
        angles = xr.DataArray(angles,
                              coords={sc.Coord.THETAIDX: range(len(angles))})
        coords = model.make_input_coords(wavelen, angles)
        form_model = model.form_factor(coords, index_matrix)
        form_sphere = dist.spheres[0].form_factor(coords, index_matrix)
        # monodisperse and polydisperse form factors should be equal at low
        # polydispersity
        xr.testing.assert_allclose(form_model, form_sphere)

        # differential scattering cross sections should be very close for
        # monodisperse and polydisperse models in the limit of low
        # polydispersity
        mono_model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                          index_matrix, index_medium)
        dscat = model.differential_cross_section(coords)
        dscat_mono = mono_model.differential_cross_section(coords)
        xr.testing.assert_allclose(dscat, dscat_mono)

        # and structure factor should be close to 1
        n_ext = index_matrix(wavelen)
        lengthscale = dist.spheres[0].radius_q
        ql = sc.ql(n_ext, lengthscale, angles)
        s = model.structure_factor(ql)
        xr.testing.assert_allclose(s, xr.ones_like(s))

        # we check also that the scattering cross sections are the same for the
        # monodisperse and polydisperse models
        cscat = model.scattering_cross_section(dscat)
        cscat_mono = mono_model.scattering_cross_section(dscat_mono)
        xr.testing.assert_allclose(cscat, cscat_mono)

        # now finite volume fraction, low polydispersity
        # note that we also test vectorization over volume fraction here
        volume_fraction = [0.4, 0.5, 0.6]
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)

        # structure factor should be almost the same as for a monodisperse
        # glass
        mono_model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                          index_matrix, index_medium)

        s = model.structure_factor(ql)
        s_mono = mono_model.structure_factor(ql)
        # tolerance would be 1e-7 but error at 444 nm is about a factor of 10
        # higher, for some reason
        # (note: position of VOLFRAC dim is different, so we ignore order)
        xr.testing.assert_allclose(s, s_mono, rtol=1e-5, check_dim_order=False)

        # and the same as would be calculated from creating a structure factor
        # directly
        structure_factor = sc.structure.Polydisperse(volume_fraction, dist)
        s_poly = structure_factor(ql)
        xr.testing.assert_equal(s, s_poly)

    @pytest.mark.parametrize("volume_fraction", [0.01, 0.3, 0.6])
    def test_formstructure_with_data(self, volume_fraction):
        """tests that FormStructure model with interpolated structure factor
        (generated from Percus Yevick) gives same results as HardSpheres model
        with PY structure factor

        """
        # all calculations here should work with an array of wavelengths
        wavelen = self.wavelen

        radius = Quantity('0.5 um')
        index_particle = sc.index.fused_silica
        index_matrix = sc.index.vacuum
        index_medium = sc.index.vacuum
        thickness = Quantity('50 um')

        # generate structure factor "data" from Percus-Yevick model
        ql_data = np.arange(0, 75, 0.1)
        structure_factor = sc.structure.PercusYevick(volume_fraction)
        s_data = structure_factor(ql_data)

        # make interpolation function.  Cubic interpolation (rather than the
        # default linear) reduces the number of data points required
        structure_factor_interp = sc.structure.Interpolated(s_data,
                                                            ql_data,
                                                            method="cubic")

        sphere = sc.Sphere(index_particle, radius)
        # need to explicitly specify effective index in FormStructureModel
        # because it doesn't know anything about particles or volume fractions.
        index_external = sc.EffectiveIndex.from_particle(sphere,
                                                         volume_fraction,
                                                         index_matrix)

        fs_model = sc.model.FormStructureModel(sphere.form_factor,
                                               structure_factor_interp,
                                               radius,
                                               index_external,
                                               index_medium)

        # for PY model we specify the matrix index and it will automatically
        # calculate the effective index
        py_model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                        index_medium)

        coords = py_model.make_input_coords(wavelen, self.angles)
        fs_dscat = fs_model.differential_cross_section(coords)
        py_dscat = py_model.differential_cross_section(coords)

        # with cubic interpolation, relative error is a little larger than
        # 1e-3 at 60% volume fraction and 750 data points.  It is higher at
        # some wavelengths than at others
        fs_dscat = fs_dscat.to_numpy().squeeze()
        py_dscat = py_dscat.to_numpy().squeeze()
        assert_allclose(fs_dscat[0], py_dscat[0], rtol=1e-2)
        assert_allclose(fs_dscat[1], py_dscat[1], rtol=1e-2)
        # TODO: test reflectance as well

    def test_number_density(self):
        """tests that the number_density property works as expected for Model
        objects.

        """
        index_matrix = sc.index.water
        index_medium = sc.index.vacuum

        # set density to 1 particle per cubic micrometer and calculate
        # volume fraction
        rho_expected = sc.Quantity(1, 'um^(-3)')
        volume_fraction = (4/3)*np.pi*self.ps_radius**3 * rho_expected

        # start with formstructure model
        fs_model = sc.model.FormStructureModel(None, None, self.ps_radius,
                                               sc.index.vacuum,
                                               sc.index.vacuum,
                                               particle = self.ps_sphere,
                                               volume_fraction =
                                               volume_fraction)
        rho_fs = fs_model.number_density
        assert isinstance(rho_fs, sc.Quantity)
        assert rho_fs.magnitude == rho_expected.magnitude

        # now for hard sphere model
        hs_model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                        index_matrix, index_medium)
        rho_hs = hs_model.number_density
        assert rho_hs.magnitude == rho_fs.magnitude

        # now for polydisperse model (single species)
        pdi = 0.15
        sphere_dist = sc.SphereDistribution(self.ps_sphere, 1.0, pdi)
        poly_model = sc.model.PolydisperseHardSpheres(sphere_dist,
                                                      volume_fraction,
                                                      index_matrix,
                                                      index_medium)
        rho_poly = poly_model.number_density
        assert rho_poly.magnitude == rho_hs.magnitude

    def test_differential_cross_section(self):
        """Test that the differential cross sections for non-core-shell
        particles and core-shells are the same at low volume fractions,
        assuming that the particle diameter of the non-core-shells is the same
        as the core diameter in the core-shells (shell of the core-shells is
        vacuum)

        """

        wavelen = self.wavelen
        index_matrix = sc.Index.constant(1.0)
        index_medium = sc.index.vacuum
        angles = Quantity(np.linspace(np.pi/2, np.pi, 200), 'rad')

        # Differential cross section for non-core-shells
        radius = Quantity('100.0 nm')
        index_particle = sc.Index.constant(1.5)
        sphere = sc.Sphere(index_particle, radius)
        volume_fraction = 1e-5

        model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                     index_medium)
        coords = model.make_input_coords(wavelen, angles)
        diff = model.differential_cross_section(coords)

        # Differential cross section for core-shells. Core is equal to
        # non-core-shell particle, and shell is made of vacuum
        radius_cs = Quantity(np.array([100.0, 110.0]), 'nm')
        index_cs = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
        sphere_cs = sc.Sphere(index_cs, radius_cs)
        n_particle_cs = sphere_cs.n(wavelen)

        # adjust volume fraction of core-shells so that volume fraction of
        # cores is same as that of non-core-shells
        vf_core = sphere_cs.volume_fraction()[0].to_numpy()
        volume_fraction_cs = volume_fraction/vf_core

        model_cs = sc.model.HardSpheres(sphere_cs, volume_fraction_cs,
                                        index_matrix, index_medium)
        diff_cs = model_cs.differential_cross_section(coords)

        assert_allclose(diff[0], diff_cs[0], rtol=1e-4)
        assert_allclose(diff[1], diff_cs[1], rtol=1e-4)

    @pytest.mark.parametrize("index_matrix", [sc.index.water,
                                              sc.Index.constant(1.59+0.001j)])
    def test_scattering_cross_section(self, index_matrix):
        """Test that the scattering_cross_section() method returns reasonable
        values (the above tests mostly focus on the
        differential_cross_section() method)

        """
        # scattering_cross_section() method is vectorized over wavelength
        wavelen = self.wavelen

        # test that cross section for vanishingly small volume fraction is the
        # same as calculated directly from Mie theory (structure factor should
        # be negligible here)
        volume_fraction = 1e-10
        index_medium = sc.index.vacuum
        model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                     index_matrix, index_medium)

        # use a lot of angles to get better precision in numerical integration
        angles = sc.Quantity(np.linspace(0, np.pi, 1000), 'rad')

        # for Mie calculations
        n_particle = self.index_particle(wavelen)
        n_matrix = index_matrix(wavelen)
        m = sc.index.ratio(n_particle, n_matrix).to_numpy()
        x = sc.size_parameter(n_matrix, self.ps_radius).to_numpy()

        # do the calculation using method from Model object
        ff_kwargs = {}
        coords = model.make_input_coords(wavelen, angles)
        dscat = model.differential_cross_section(coords, **ff_kwargs)
        cscat = model.scattering_cross_section(dscat)

        # make sure we have correct coords and shape
        assert cscat.shape == (3, len(np.atleast_1d(wavelen)), 1)
        assert sc.Coord.VOLFRAC in cscat.coords
        assert sc.Coord.POL in cscat.coords

        # now do calculation using Mie theory, using appropriate function for
        # the cross-section
        wavelen_media = wavelen/n_matrix.to_numpy()
        if np.any(n_matrix.imag > 0):
            nstop = mie._nstop(np.array(x).max())
            albl = mie._scatcoeffs(m, x, nstop)

            radius = self.ps_radius
            cscat_sud = mie._cross_sections_complex_medium_sudiarta(*albl, x,
                                                                    radius)
            # Fu cross sections
            cldl = mie._internal_coeffs(m, x, nstop)
            x_med = sc.size_parameter(n_matrix, radius).to_numpy()
            # fu calculation expects indexes as Quantity objects
            n_particle = sc.Quantity(n_particle.to_numpy(), '')
            n_medium = sc.Quantity(n_matrix.to_numpy(), '')

            # for the moment, Fu calculations have to be looped over wavelength
            # (this is because they are not yet set to take n_medium as an
            # array)
            c_fu_sca = np.zeros(len(wavelen))
            c_fu_abs = np.zeros_like(c_fu_sca)
            c_fu_ext = np.zeros_like(c_fu_sca)
            for i in range(len(wavelen)):
                c_fu_loop = \
                    mie._cross_sections_complex_medium_fu(albl[0][i],
                                                          albl[1][i],
                                                          cldl[0][i],
                                                          cldl[1][i],
                                                          radius,
                                                          n_particle[i],
                                                          n_medium[i],
                                                          x[i], x_med[i],
                                                          wavelen[i])
                # squeeze out the singleton wavelength dimension
                c_fu_sca[i] = c_fu_loop[0].magnitude.squeeze()
                c_fu_abs[i] = c_fu_loop[1].magnitude.squeeze()
                c_fu_ext[i] = c_fu_loop[2].magnitude.squeeze()

            # first check that the Fu and Sudiarta calculations agree (note:
            # absorption cross-sections do not agree)
            assert_allclose(c_fu_sca, cscat_sud[0].magnitude)

        else:
            cscat_sud = mie.calc_cross_sections(m, x)
            k = 2*np.pi/wavelen_media
            cscat_sud = cscat_sud/k**2

        # Now check that the Mie calculation and Model method calculations
        # agree.  We drop VOLFRAC dim, which is not in cscat_sud
        cscat_avg = cscat.loc["avg"].isel({sc.Coord.VOLFRAC: 0}, drop=True)
        cscat_sud0 = cscat_sud[0].to_preferred().magnitude
        assert_allclose(cscat_avg, cscat_sud0, rtol=1e-2)
        # Agreement is to within 1e-2 relative error for absorbing media
        # (around 1e-3 but a little higher for certain wavelengths, and 1e-5
        # for non-absorbing. The discrepancy in absorbing media doesn't seem to
        # improve with more integration points, but gets worse with
        # increasingly large imaginary component of the refractive index. Fu
        # and Sudiarta calculations agree at n.imag = 1j, though absorption
        # cross-sections do not agree.
        #
        # TODO: add more testing of Fu, Sudiarta cross sections in pymie, along
        # with more tests of integrate_intensity_complex_medium()

    @pytest.mark.parametrize("index_matrix", [sc.index.water,
                                              sc.Index.constant(1.59+0.001j)])
    def test_scattering_cross_section_polydisperse(self, index_matrix):
        """Test the scattering_cross_section() method for the
        PolydisperseHardSpheres model
        """
        # both the polydisperse form and structure factors are vectorized over
        # wavelength
        wavelen = self.wavelen
        volume_fraction = 0.5
        index_medium = sc.index.vacuum

        # avoid division by zero error by starting at finite angle
        angles = sc.Quantity(np.linspace(0.01, np.pi, 20), 'rad')

        # check that a binary polydisperse system with the same diameters for
        # the two components produces the same differential and total cross
        # sections as a single-component polydisperse system
        pdi = 0.15

        dist = sc.SphereDistribution(self.ps_sphere, 1.0, pdi)
        single_model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                        index_matrix,
                                                        index_medium)
        dist = sc.SphereDistribution([self.ps_sphere, self.ps_sphere],
                                     [0.5, 0.5], [pdi, pdi])
        binary_model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                        index_matrix,
                                                        index_medium)

        # do the calculation using single-species polydisperse model
        ff_kwargs = {}
        coords = single_model.make_input_coords(wavelen, angles)
        dscat_1 = single_model.differential_cross_section(coords, **ff_kwargs)
        cscat_1 = single_model.scattering_cross_section(dscat_1)

        # do the calculation using bidisperse polydisperse model
        ff_kwargs = {}

        dscat_2 = binary_model.differential_cross_section(coords, **ff_kwargs)
        cscat_2 = binary_model.scattering_cross_section(dscat_2)

        xr.testing.assert_equal(dscat_2, dscat_1)
        xr.testing.assert_allclose(cscat_2, cscat_1, rtol=1e-15)

    @pytest.mark.parametrize("volume_fraction", [1e-7, 1e-4, 1e-1, 0.30, 0.50])
    def test_scattering_cross_section_polarization(self, volume_fraction):
        """Test that the total cross-sections for unpolarized light are the
        same in the lab frame (cartesian coordinates) and in scattering plane
        coordinates.
        """
        wavelen = self.wavelen

        index_matrix = sc.index.water
        index_medium = sc.index.vacuum
        model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                index_matrix, index_medium)

        thetas = sc.Quantity(np.linspace(0, np.pi, 10), 'rad')
        phis = sc.Quantity(np.linspace(0, 2*np.pi, 20), 'rad')

        # do scattering plane calculation first.  Specifying incident vector
        # here forces the calculation to go through
        # mie.integrate_intensity_complex_medium(), which can handle an
        # incident vector.  If we don't specify it, the calculation would go
        # through mie.calc_ang_scat().
        #
        # For scattering plane coordinates, parallel and perpendicular
        # polarizations rotate with phi, so that the scattering is azimuthally
        # symmetric. To get all the light, we need to look at both the parallel
        # and perpendicular components, which we do by specifying (1,1) for the
        # incident vector:
        ff_kwargs = {"incident_vector": (1, 1)}
        coords = model.make_input_coords(wavelen, thetas)
        dscat = model.differential_cross_section(coords, **ff_kwargs)
        cscat = model.scattering_cross_section(dscat)

        # to show that incident vector was passed through, we pass an incorrect
        # vector here.  Should get a different result
        ff_kwargs["incident_vector"] = (1, 0)
        dscat = model.differential_cross_section(coords, **ff_kwargs)
        cscat_wrong = model.scattering_cross_section(dscat)

        assert not np.allclose(cscat.to_numpy(), cscat_wrong.to_numpy())

        # now do scattering plane calculation going through
        # mie.calc_ang_scat(), without specifying incident vector
        del ff_kwargs["incident_vector"]
        dscat = model.differential_cross_section(coords, **ff_kwargs)
        cscat_no_vector = model.scattering_cross_section(dscat)

        assert_allclose(cscat.to_numpy(), cscat_no_vector.to_numpy())

        # now do cartesian.  Incident vector is polarized, but remember we're
        # integrating over both polarizations at each detector position, so we
        # should get all the scattered light.
        ff_kwargs.update({"incident_vector": (1, 0)})
        coords = model.make_input_coords(wavelen, thetas, phis=phis)
        dscat_cart = model.differential_cross_section(coords, **ff_kwargs)
        cscat_cart = model.scattering_cross_section(dscat_cart)

        # only the total cross section should be the same.  The x- and y-
        # components should not be equal to the par and perp components.
        assert_allclose(cscat_cart.loc["avg"].to_numpy(),
                        cscat.loc["avg"].to_numpy())

    @pytest.mark.parametrize("index_matrix", [sc.index.water,
                                              sc.Index.constant(1.59 + 0.001j),
                                              sc.Index.constant(1.59 + 0.1j)])
    def test_phase_function(self, index_matrix):
        """Test that the phase functions for polydisperse and monodisperse
        systems are approximately equal when the polydispersity is small.
        """
        wavelen = self.wavelen
        volume_fraction = 0.5
        index_medium = sc.index.vacuum
        model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                     index_matrix, index_medium)

        # start at a few degrees to avoid division by zero error
        angles = sc.Quantity(np.linspace(2, 180., 20), 'deg')

        # monodisperse calculation
        ff_kwargs = {}
        coords = model.make_input_coords(wavelen, angles)
        dscat = model.differential_cross_section(coords, **ff_kwargs)
        cscat_mono = model.scattering_cross_section(dscat)
        phase_func_mono = model.phase_function(dscat)

        # polydisperse (single species) calculation
        concentration = 1.0
        pdi = 1e-5
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)

        dscat = model.differential_cross_section(coords, **ff_kwargs)
        cscat_poly = model.scattering_cross_section(dscat)
        phase_func_poly = model.phase_function(dscat)

        xr.testing.assert_allclose(cscat_poly, cscat_mono, rtol=1e-5)
        np.testing.assert_allclose(phase_func_poly, phase_func_mono, rtol=1e-3)
        # note: rtol would be 1e-5 but error at 444 nm is much larger for
        # some reason

    @pytest.mark.parametrize("maxwell_garnett", [True, False])
    def test_vectorization(self, maxwell_garnett):
        """Test that scattering methods and functions are vectorized over
        wavelength and other parameters

        """
        wavelen = self.wavelen
        # choose a matrix with dispersion
        index_matrix = sc.index.water
        angles = sc.Quantity(np.linspace(0, 180, 20), "deg")
        angles = angles.to("rad").magnitude
        angles = xr.DataArray(angles,
                              coords={sc.Coord.THETAIDX: range(len(angles))})
        volume_fraction = 0.6
        coords = sc._make_input_coords(wavelen, angles)

        # check that form factor is vectorized over wavelength by checking
        # against loop values
        ff = self.ps_sphere.form_factor(coords, index_matrix)
        ff_loop = []
        for i in range(len(wavelen)):
            coords = sc._make_input_coords(wavelen[i], angles)
            ff_loop.append(self.ps_sphere.form_factor(coords, index_matrix))
        ff_loop = xr.concat(ff_loop, sc.Coord.WAVELEN)
        xr.testing.assert_allclose(ff, ff_loop)

        # check that cross-sections are vectorized over wavelength
        model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                     index_matrix, sc.index.vacuum)
        coords = model.make_input_coords(wavelen, angles)
        dscat = model.differential_cross_section(coords)
        cscat = model.scattering_cross_section(dscat)
        dscat_loop = []
        cscat_loop = []
        for i in range(len(wavelen)):
            coords = model.make_input_coords(wavelen[i], angles)
            dscat_loop.append(model.differential_cross_section(coords))
            cscat_loop.append(model.scattering_cross_section(dscat_loop[i]))
        dscat_loop = xr.concat(dscat_loop, sc.Coord.WAVELEN)
        cscat_loop = xr.concat(cscat_loop, sc.Coord.WAVELEN)
        xr.testing.assert_allclose(dscat, dscat_loop)
        xr.testing.assert_allclose(cscat, cscat_loop)

        # check that structure factor is vectorized over volume fraction
        volume_fraction = np.array([0.05, 0.25, 0.35, 0.5, 0.6])
        structure_factor = sc.structure.PercusYevick(volume_fraction)
        ql = sc.ql(index_matrix(wavelen), self.ps_sphere.radius_q, angles)
        s = structure_factor(ql)
        assert np.ndim(s) == 3
        assert sc.Coord.WAVELEN in s.coords
        assert sc.Coord.THETAIDX in s.coords
        assert sc.Coord.VOLFRAC in s.coords

        # check that model will also take an array of volume fractions, both
        # with Maxwell-Garnett and Bruggeman.
        model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                     index_matrix, sc.index.vacuum,
                                     maxwell_garnett=maxwell_garnett)
        coords = model.make_input_coords(wavelen, angles)
        dscat = model.differential_cross_section(coords)
        cscat = model.scattering_cross_section(dscat)
        dscat_loop = []
        cscat_loop = []
        for i, vf in enumerate(volume_fraction):
            model = sc.model.HardSpheres(self.ps_sphere, vf,
                                         index_matrix, sc.index.vacuum,
                                         maxwell_garnett=maxwell_garnett)
            coords = model.make_input_coords(wavelen, angles)
            dscat_loop.append(model.differential_cross_section(coords))
            cscat_loop.append(model.scattering_cross_section(dscat_loop[i]))
        dscat_loop = xr.concat(dscat_loop, sc.Coord.VOLFRAC)
        dscat_loop = dscat_loop.transpose(*dscat.dims)
        cscat_loop = xr.concat(cscat_loop, sc.Coord.VOLFRAC)
        cscat_loop = cscat_loop.transpose(*cscat.dims)
        xr.testing.assert_allclose(dscat, dscat_loop)
        xr.testing.assert_allclose(cscat, cscat_loop)


class TestDetector():
    """Tests for the Detector class and derived classes.
    """
    def test_detector(self):
        """Test standard Detector object"""

        # make sure angles are accepted and stored correctly
        theta_min, theta_max = sc.Quantity('90 deg'), sc.Quantity('180 deg')
        phi_min, phi_max = sc.Quantity('0 deg'), sc.Quantity('360 deg')
        detector = sc.model.Detector(theta_min, theta_max, phi_min, phi_max)

        assert detector.theta_min == theta_min.to('rad').magnitude
        assert detector.theta_max == theta_max.to('rad').magnitude
        assert detector.phi_min == phi_min.to('rad').magnitude
        assert detector.phi_max == phi_max.to('rad').magnitude

        # make sure stored angles have no units
        for param in [detector.theta_min, detector.theta_max,
                      detector.phi_min, detector.phi_max]:
            assert not isinstance(param, sc.Quantity)

        # specifying no dimensions should give radians
        theta_min, theta_max = sc.Quantity(np.pi/2), sc.Quantity(np.pi)
        phi_min, phi_max = sc.Quantity(0), sc.Quantity(np.pi)
        detector = sc.model.Detector(theta_min, theta_max, phi_min, phi_max)
        assert detector.theta_min == theta_min.to('rad').magnitude
        assert detector.theta_max == theta_max.to('rad').magnitude
        assert detector.phi_min == phi_min.to('rad').magnitude
        assert detector.phi_max == phi_max.to('rad').magnitude

        # specifying mix of dimensions should work
        theta_min, theta_max = sc.Quantity(np.pi/2), sc.Quantity(np.pi, 'rad')
        phi_min, phi_max = sc.Quantity(0, 'deg'), sc.Quantity(np.pi, '')
        detector = sc.model.Detector(theta_min, theta_max, phi_min, phi_max)
        assert detector.theta_min == theta_min.to('rad').magnitude
        assert detector.theta_max == theta_max.to('rad').magnitude
        assert detector.phi_min == phi_min.to('rad').magnitude
        assert detector.phi_max == phi_max.to('rad').magnitude

        # not specifying dimensions should fail
        theta_min, theta_max = np.pi/2, np.pi
        phi_min, phi_max = 0, np.pi
        with pytest.raises(AttributeError):
            detector = sc.model.Detector(theta_min, theta_max, phi_min,
                                         phi_max)

        # when only theta is specified, phi should be set to 0 to 360 degrees
        theta_min, theta_max = sc.Quantity('90 deg'), sc.Quantity('180 deg')
        phi_min, phi_max = sc.Quantity('0 deg'), sc.Quantity('360 deg')
        detector = sc.model.Detector(theta_min, theta_max)
        assert detector.phi_min == phi_min.to('rad').magnitude
        assert detector.phi_max == phi_max.to('rad').magnitude

        # when no parameters are specified, detector should be equivalent to a
        # hemispherical reflectance detector
        detector = sc.model.Detector()
        assert detector.theta_min == theta_min.to('rad').magnitude
        assert detector.theta_max == theta_max.to('rad').magnitude
        assert detector.phi_min == phi_min.to('rad').magnitude
        assert detector.phi_max == phi_max.to('rad').magnitude

    def test_hemispherical_reflectance_detector(self):
        """Test the integrating sphere-type detector"""
        detector = sc.model.HemisphericalReflectanceDetector()
        assert detector.theta_min == sc.Quantity('90 deg').to('rad').magnitude
        assert detector.theta_max == sc.Quantity('180 deg').to('rad').magnitude
        assert detector.phi_min == sc.Quantity('0 deg').to('rad').magnitude
        assert detector.phi_max == sc.Quantity('360 deg').to('rad').magnitude


def test_fresnel():
    # test the fresnel reflection and transmission coefficients
    wavelen = sc.Quantity(400, 'nm')
    n1 = sc.Index.constant(1.00)(wavelen)
    n2 = sc.Index.constant(1.5)(wavelen)

    # quantities calculated from
    # http://www.calctool.org/CALC/phys/optics/reflec_refrac
    r, t = sc.model.fresnel_coeffs(n1, n2, Quantity('0.0 deg'))
    assert_almost_equal(r.loc["par"], 0.04)
    assert_almost_equal(r.loc["perp"], 0.04)
    r, t = sc.model.fresnel_coeffs(n1, n2, Quantity('45.0 deg'))
    assert_almost_equal(r.loc["par"], 0.00846646)
    assert_almost_equal(r.loc["perp"], 0.0920134)

    # test total internal reflection
    r, t = sc.model.fresnel_coeffs(n2, n1, Quantity('45.0 deg'))
    assert_equal(r.loc["par"].item(), 1.0)
    assert_equal(r.loc["perp"].item(), 1.0)

    # test no total internal reflection (just below critical angle)
    r, t = sc.model.fresnel_coeffs(n2, n1, Quantity('41.810 deg'))
    assert_almost_equal(r.loc["par"], 0.972175, decimal=6)
    assert_almost_equal(r.loc["perp"], 0.987536, decimal=6)

    # test vectorized computation over angles
    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # check for value error (can't go beyond 90 degree angle of incidence)
    with pytest.raises(ValueError):
        sc.model.fresnel_coeffs(n2, n1, angles)
    angles = Quantity(np.linspace(0, 90., 10), 'deg').to("rad").magnitude
    angles = xr.DataArray(angles, coords={sc.Coord.INCIDENT: angles})
    r, t = sc.model.fresnel_coeffs(n2, n1, angles)
    rpar_std = np.array([0.04, 0.0362780, 0.0243938, 0.00460754, 0.100064, 1.0,
                         1.0, 1.0, 1.0, 1])
    rperp_std = np.array([0.04, 0.0438879, 0.0590632, 0.105773, 0.390518, 1.0,
                         1.0, 1.0, 1.0, 1.0])
    assert_array_almost_equal(r.loc["par"].squeeze(), rpar_std)
    assert_array_almost_equal(r.loc["perp"].squeeze(), rperp_std)

    # test transmission
    tpar_std = 1.0-rpar_std
    tperp_std = 1.0-rperp_std
    assert_array_almost_equal(t.loc["par"].squeeze(), tpar_std)
    assert_array_almost_equal(t.loc["perp"].squeeze(), tperp_std)

    # test vectorized computation over wavelength (check that results match
    # those of loop).  We'll test a situation in which there is TIR.
    wavelen = sc.Quantity(np.linspace(400, 800, 10), 'nm')
    index_low = sc.index.vacuum
    index_high = sc.index.polystyrene
    n_low = index_low(wavelen)
    n_high = index_high(wavelen)
    angles = Quantity(np.linspace(0, 90., 10), 'deg').to("rad").magnitude
    angles = xr.DataArray(angles, coords={sc.Coord.INCIDENT: angles})
    # vectorized version
    rt = sc.model.fresnel_coeffs(n_high, n_low, angles)
    # check that dimensions are correct
    assert rt.dims == (sc.Coord.FRESNEL, sc.Coord.POL, sc.Coord.WAVELEN,
                       sc.Coord.INCIDENT)
    # loop-based version
    rt_list = []
    for wl in wavelen:
        n_low = index_low(wl)
        n_high = index_high(wl)
        rt_list.append(sc.model.fresnel_coeffs(n_high, n_low, angles))
    rt_loop = xr.concat(rt_list, dim=sc.Coord.WAVELEN)
    xr.testing.assert_equal(rt, rt_loop)


def test_theta_refraction():
    # test that the detection angles theta are refracted correctly at the
    # medium-sample interface. When n_sample < n_medium, the scattered angles
    # in the reflection hemisphere (90-180 deg) are refracted at the interface
    # into a smaller range of angles (>90-180 deg). This test checks that the
    # the reflectance is 0 when the angles between theta_min and theta_max are
    # outside the range of refracted scattered angles.
    incident_angle = sc.Quantity('0.0 deg')
    wavelength = sc.Quantity(np.linspace(400, 800, 11), "nm")
    radius = sc.Quantity('100.0 nm')
    volume_fraction = 0.5
    index_particle = sc.Index.constant(1.0)
    particle = sc.Sphere(index_particle, radius)
    index_matrix =  sc.Index.constant(1.0)
    index_medium = sc.Index.constant(2.0)
    n_medium = index_medium(wavelength)
    theta_min = sc.Quantity(np.pi/2, "rad")

    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix,
                                 index_medium)

    # when theta_max is pi, the detector captures the specular light. Since
    # there is no scattering from the sample (particles are index matched
    # here), specular reflection is the only contribution to the reflectance.
    theta_max = sc.Quantity(np.pi, "rad")
    detector = sc.model.Detector(theta_min, theta_max)
    # note that we have to specify a finite thickness here.  Otherwise
    # thickness is infinite, and all light is scattered, even under
    # index-matching conditions
    refl = sc.model.reflection(model, wavelength, detector=detector,
                               thickness=sc.Quantity(10, 'um'))[0]

    # make sure the reflectance is equal to fresnel
    index_sample = sc.EffectiveIndex.from_particle(particle, volume_fraction,
                                                   index_matrix)
    n_sample = index_sample(wavelength)
    r_fresnel, _ = sc.model.fresnel_coeffs(n_medium, n_sample, incident_angle)
    r_fresnel_avg = (r_fresnel[0] + r_fresnel[1]) / 2
    xr.testing.assert_equal(refl, r_fresnel_avg.drop_vars(sc.Coord.FRESNEL))

    # set theta_max to be slightly smaller than the theta at which light
    # scattered at pi/2 is refracted (= pi - arcsin(1/2) = 2.61799388, where
    # 1/2 is the ratio of the refractive indices)
    theta_max = sc.Quantity(2.617, "rad")
    detector = sc.model.Detector(theta_min, theta_max)
    refl1, _, _, _, _ = sc.model.reflection(model, wavelength,
                                            detector=detector)

    # try a different range of thetas (but keeping theta_max < pi-arcsin(1/2))
    theta_max = sc.Quantity(2., "rad")
    detector = sc.model.Detector(theta_min, theta_max)
    refl2, _, _, _, _ = sc.model.reflection(model, wavelength,
                                            detector=detector)

    # reflectance should be zero in both cases
    assert_equal(refl1.to_numpy(), 0)
    xr.testing.assert_allclose(refl1, refl2, rtol=1e-5)


def test_vectorized_reflection():
    """Test that model.reflection() vectorizes over wavelength and volume
    fraction.

    """
    # choose a small number of wavelengths and volume fractions because looping
    # over both is slow
    wavelength = sc.Quantity(np.linspace(400, 800, 5), "nm")
    volume_fraction = np.linspace(0.2, 0.6, 3)
    radius = sc.Quantity("0.125 um")
    index_particle = sc.index.polystyrene
    sphere = sc.Sphere(index_particle, radius)

    # choose a matrix with dispersion
    index_matrix = sc.index.water
    index_medium = sc.index.vacuum

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    refl = sc.model.reflection(model, wavelength)[0]

    # test that loop gives same values
    refl_loop = []
    for phi in volume_fraction:
        refl_loop_wl = []
        model = sc.model.HardSpheres(sphere, phi, index_matrix, index_medium)
        for wavelen in wavelength:
            refl_loop_wl.append(sc.model.reflection(model, wavelen)[0])
        refl_loop.append(xr.concat(refl_loop_wl, sc.Coord.WAVELEN))
    refl_loop = xr.concat(refl_loop, sc.Coord.VOLFRAC)

    xr.testing.assert_allclose(refl, refl_loop, rtol=1e-9)


def test_reflection_core_shell():
    # Test reflection, anisotropy factor, and transport length calculations to
    # make sure the values for refl, g, and lstar remain the same after adding
    # core-shell capability into the model
    wavelength = Quantity(500.0, 'nm')
    thickness = Quantity(15.0, 'um')
    small_angle = sc.Quantity("5.0 deg")

    # Non core-shell particles with Maxwell-Garnett effective index
    volume_fraction = 0.5
    radius = Quantity('120.0 nm')
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.Index.constant(1.0)
    index_medium = index_matrix

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium, maxwell_garnett=True)

    detector = sc.model.Detector(theta_min=Quantity('90.0 deg'))
    refl1, _, _, g1, lstar1 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness,
                                                  detector=detector,
                                                  small_angle=small_angle)

    # Non core-shell particles with Bruggeman effective index
    volume_fraction2 = 0.00001
    model = sc.model.HardSpheres(sphere, volume_fraction2, index_matrix,
                                 index_medium, maxwell_garnett=False)
    refl2, _, _, g2, lstar2 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness,
                                                  detector=detector,
                                                  small_angle=small_angle)


    # Core-shell particles of core diameter equal to non core shell particles,
    # and shell index of air. With Bruggeman effective index
    radius3 = Quantity(np.array([120.0, 130.0]), 'nm')
    index3 = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
    sphere_cs = sc.Sphere(index3, radius3)
    volume_fraction3 = volume_fraction2 * (radius3[1]**3 / radius3[0]**3)
    model = sc.model.HardSpheres(sphere_cs, volume_fraction3, index_matrix,
                                 index_medium)

    refl3, _, _, g3, lstar3 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness,
                                                  detector=detector,
                                                  small_angle=small_angle)

    # Outputs for refl, g, and lstar before adding core-shell capability
    refl = Quantity(0.20772170840902376, '')
    g = Quantity(-0.18931942267032678, '')
    lstar = Quantity(10810.088573316663, 'nm')

    # Compare old outputs (before adding core-shell capability) and new outputs
    # for a non-core-shell using Maxwell-Garnett
    assert_allclose(refl1.to_numpy(), refl.magnitude)
    assert_allclose(g1.magnitude, g.magnitude)
    assert_allclose(lstar1.to('nm').magnitude, lstar.magnitude)

    # Compare a non-core-shell and a core-shell with shell index of air using
    # Bruggeman.
    #
    # first show that the volume fractions are NOT equal for the two
    # calculations: the non-core-shell will have a lower volume fraction than
    # the core-shell, even though the shell is air, because we don't account
    # for the material of the shell when calculating volume fraction
    volfrac_refl2 = refl2.coords[sc.Coord.VOLFRAC][0]
    volfrac_refl3 = refl3.coords[sc.Coord.VOLFRAC][0]
    assert volfrac_refl2 != volfrac_refl3
    # next do numpy comparison on values
    assert_allclose(refl2, refl3, rtol=1e-5)
    assert_allclose(g2.magnitude, g3.magnitude, rtol=1e-5)
    assert_allclose(lstar2.to('mm').magnitude, lstar3.to('mm').magnitude,
                    rtol=1e-5)

    # Test that the reflectance is the same for a core-shell that absorbs (with
    # the same refractive indices for all layers) and a non-core-shell that
    # absorbs with the same index

    # Absorbing non-core-shell
    radius4 = Quantity('120.0 nm')
    index_particle4 = sc.Index.constant(1.5+0.001j)
    sphere = sc.Sphere(index_particle4, radius4)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    refl4 = sc.model.reflection(model, wavelength, thickness=thickness)[0]

    # Absorbing core-shell
    radius5 = Quantity(np.array([110.0, 120.0]), 'nm')
    index5 = [sc.Index.constant(1.5+0.001j), sc.Index.constant(1.5+0.001j)]
    sphere_cs = sc.Sphere(index5, radius5)
    model = sc.model.HardSpheres(sphere_cs, volume_fraction, index_matrix,
                                 index_medium)
    refl5 = sc.model.reflection(model, wavelength, thickness=thickness)[0]

    xr.testing.assert_allclose(refl4, refl5)

    # Same as previous test but with absorbing matrix
    # Non-core-shell
    radius6 = Quantity('120.0 nm')
    index_particle6 = sc.Index.constant(1.5+0.001j)
    sphere = sc.Sphere(index_particle6, radius6)
    index_matrix6 = sc.Index.constant(1.0+0.001j)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix6,
                                 index_medium)
    refl6 = sc.model.reflection(model, wavelength, thickness=thickness)[0]

    # Core-shell
    index7 = [sc.Index.constant(1.5+0.001j), sc.Index.constant(1.5+0.001j)]
    radius7 = Quantity(np.array([110.0, 120.0]), 'nm')
    sphere_cs = sc.Sphere(index7, radius7)
    index_matrix7 = sc.Index.constant(1.0+0.001j)
    model = sc.model.HardSpheres(sphere_cs, volume_fraction, index_matrix7,
                                 index_medium)
    refl7 = sc.model.reflection(model, wavelength, thickness=thickness)[0]

    xr.testing.assert_allclose(refl6, refl7)


def test_reflection_absorbing_particle():
    # test that the reflections with a real n_particle and with a complex
    # n_particle with a 0 imaginary component are the same
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = 0.5
    radius = Quantity('120.0 nm')
    index_matrix = sc.Index.constant(1.0)
    index_medium = index_matrix
    index_particle_real = sc.Index.constant(1.5)
    sphere_real = sc.Sphere(index_particle_real, radius)
    index_particle_complex = sc.Index.constant(1.5 + 0j)
    sphere_complex = sc.Sphere(index_particle_complex, radius)

    # With Maxwell-Garnett
    model = sc.model.HardSpheres(sphere_real, volume_fraction, index_matrix,
                                 index_medium, maxwell_garnett=True)
    refl_mg1, _, _, g_mg1, lstar_mg1 = sc.model.reflection(model, wavelength)
    model = sc.model.HardSpheres(sphere_complex, volume_fraction, index_matrix,
                                 index_medium, maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = sc.model.reflection(model, wavelength)

    # these should be pretty close
    rtol = 1e-13
    xr.testing.assert_allclose(refl_mg1, refl_mg2, rtol=rtol)
    assert_allclose(g_mg1.magnitude, g_mg2.magnitude, rtol=rtol)
    assert_allclose(lstar_mg1.magnitude, lstar_mg2.magnitude, rtol=rtol)

    # Outputs before refactoring structcol
    refl_mg1_before = 0.2963964709617333
    refl_mg2_before = 0.29639647096173255
    g_mg1_before = -0.18774057969370997
    g_mg2_before = -0.18774057969370903
    # this is in nm
    lstar_mg1_before = 10810.069633192961
    # lstar_mg2_before = 10810.069633193001
    # lstar_mg2 and lstar_mg1 are now equal, so we don't need to compare to
    # lstar_mg2_before

    assert_allclose(refl_mg1.to_numpy(), refl_mg1_before)
    assert_allclose(refl_mg2.to_numpy(), refl_mg2_before)
    assert_allclose(g_mg1.magnitude, g_mg1_before)
    assert_allclose(g_mg2.magnitude, g_mg2_before)
    assert_allclose(lstar_mg1.to('nm').magnitude, lstar_mg1_before)
    assert_allclose(lstar_mg1.magnitude, lstar_mg2.magnitude)

    # With Bruggeman
    model = sc.model.HardSpheres(sphere_real, volume_fraction, index_matrix,
                                 index_medium, maxwell_garnett=False)
    refl_bg1, _, _, g_bg1, lstar_bg1 = sc.model.reflection(model, wavelength)
    model = sc.model.HardSpheres(sphere_complex, volume_fraction, index_matrix,
                                 index_medium, maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = sc.model.reflection(model, wavelength)

    rtol = 1e-13
    xr.testing.assert_allclose(refl_bg1, refl_bg2, rtol=rtol)
    assert_allclose(g_bg1.magnitude, g_bg2.magnitude, rtol=rtol)
    assert_allclose(lstar_bg1.magnitude, lstar_bg2.magnitude, rtol=rtol)

    # Outputs before refactoring structcol
    refl_bg1_before = 0.2685710414987676
    refl_bg2_before = 0.2685710414987676
    g_bg1_before = -0.17681566915117486
    g_bg2_before = -0.17681566915117486
    # these are in nm
    lstar_bg1_before = 11593.280877304634
    lstar_bg2_before = 11593.280877304634

    assert_allclose(refl_bg1.to_numpy(), refl_bg1_before)
    assert_allclose(refl_bg2.to_numpy(), refl_bg2_before)
    assert_allclose(g_bg1.magnitude, g_bg1_before)
    assert_allclose(g_bg2.magnitude, g_bg2_before)
    assert_allclose(lstar_bg1.to('nm').magnitude, lstar_bg1_before)
    assert_allclose(lstar_bg2.to('nm').magnitude, lstar_bg2_before)

    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    index_particle_complex2 = sc.Index.constant(1.5+1e-8j)
    sphere_complex2 = sc.Sphere(index_particle_complex2, radius)

    thickness = Quantity('100.0 um')

    # With Bruggeman
    model = sc.model.HardSpheres(sphere_complex2, volume_fraction,
                                 index_matrix, index_medium,
                                 maxwell_garnett=False)
    refl_bg3, _, _, g_bg3, lstar_bg3 = sc.model.reflection(model, wavelength,
                                                           thickness=thickness)

    rtol = 1e-3
    xr.testing.assert_allclose(refl_bg1, refl_bg3, rtol=rtol)
    assert_allclose(g_bg1.magnitude, g_bg3.magnitude, rtol=rtol)
    assert_allclose(lstar_bg1.to('mm').magnitude, lstar_bg3.to('mm').magnitude,
                    rtol=rtol)


def test_calc_g():
    # test that the anisotropy factor for multilayer spheres are the same when
    # using calc_g from mie.py in pymie and using the model
    wavelength = Quantity(500.0, 'nm')

    # calculate g using the model
    radius = Quantity(np.array([120.0, 130.0]), 'nm')
    index = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
    sphere = sc.Sphere(index, radius)
    n_particle = sphere.n(wavelength)

    volume_fraction = 0.01
    index_matrix = sc.Index.constant(1.0)
    index_medium = index_matrix
    index_sample = sc.EffectiveIndex.from_particle(sphere, volume_fraction,
                                                   index_matrix)

    # need to specify particle and volume_fraction to calculate transport
    # length from FormStructureModel
    model = sc.model.FormStructureModel(sphere.form_factor, None,
                                        sphere.radius_q, index_sample,
                                        index_medium, particle=sphere,
                                        volume_fraction=volume_fraction)

    _, _, _, g1, _= sc.model.reflection(model, wavelength,
                                        small_angle=Quantity('0.01 deg'),
                                        num_angles=1000)

    # calculate g using calc_g in pymie
    n_sample = index_sample(wavelength)
    m = sc.index.ratio(n_particle, n_sample).to_numpy()
    x = mie.size_parameter(wavelength, n_sample.to_numpy().squeeze(), radius)
    g2 = mie.calc_g(m,x)

    assert_array_almost_equal(g1.magnitude, g2)

    # Outputs before refactoring structcol
    g1_before = 0.5064750277811477
    g2_before = 0.5064757158664487

    assert_allclose(g1.magnitude, g1_before)
    assert_allclose(g2, g2_before)

def test_transport_length_dilute():
    # test that the transport length for a dilute system matches the transport
    # length calculated from Mie theory

    # transport length from single scattering model for a dilute system
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = 0.0000001
    radius = Quantity('120.0 nm')
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.Index.constant(1.0)
    index_medium = index_matrix

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    _, _, _, _, lstar_model = sc.model.reflection(model, wavelength)

    # transport length from Mie theory
    index_sample = sc.index.EffectiveIndex.from_particle(sphere,
                                                         volume_fraction,
                                                         index_matrix)
    n_sample = index_sample(wavelength)
    n_particle = sphere.n(wavelength)
    m = sc.index.ratio(n_particle, n_sample).to_numpy()
    x = mie.size_parameter(wavelength, n_sample.to_numpy().squeeze(), radius)
    g = mie.calc_g(m,x)

    number_density = sphere.number_density(volume_fraction)
    cscat = mie.calc_cross_sections(m, x)[0]
    k = 2*np.pi*(n_sample.to_numpy())/wavelength
    cscat = cscat/k**2

    lstar_mie = 1 / (number_density * cscat * (1-g))

    assert_allclose(lstar_model.to('m').magnitude, lstar_mie.to('m').magnitude,
                    rtol=1e-5)

def test_reflection_absorbing_matrix():
    # test that the reflections with a real n_matrix and with a complex
    # n_matrix with a 0 imaginary component are the same
    wavelength = sc.Quantity(np.linspace(400, 800, 11), "nm")
    volume_fraction = 0.5
    radius = sc.Quantity("120.0 nm")
    index_matrix_real = sc.Index.constant(1.0)
    index_matrix_imag = sc.Index.constant(1.0 + 0j)
    index_medium = sc.Index.constant(1.0)
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)

    # With Maxwell-Garnett
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix_real,
                                 index_medium, maxwell_garnett=True)
    refl_mg1, _, _, g_mg1, lstar_mg1 = sc.model.reflection(model, wavelength)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix_imag,
                                 index_medium, maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = sc.model.reflection(model, wavelength)

    # should be very close
    rtol = 1e-13
    xr.testing.assert_allclose(refl_mg1, refl_mg2, rtol=rtol)
    assert_allclose(g_mg1.magnitude, g_mg2.magnitude, rtol=rtol)
    assert_allclose(lstar_mg1.magnitude, lstar_mg2.magnitude, rtol=rtol)

    # With Bruggeman
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix_real,
                                 index_medium, maxwell_garnett=False)
    refl_bg1, _, _, g_bg1, lstar_bg1 = sc.model.reflection(model, wavelength)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix_imag,
                                 index_medium, maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = sc.model.reflection(model, wavelength)

    xr.testing.assert_allclose(refl_bg1, refl_bg2, rtol=rtol)
    assert_allclose(g_bg1.magnitude, g_bg2.magnitude, rtol=rtol)
    assert_allclose(lstar_bg1.magnitude, lstar_bg2.magnitude, rtol=rtol)

    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    index_matrix_imag2 = sc.Index.constant(1.0 + 1e-8j)

    # With Bruggeman
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix_imag2,
                                 index_medium, maxwell_garnett=False)
    refl_bg3, _, _, g_bg3, lstar_bg3 = sc.model.reflection(model, wavelength)

    rtol=1e-5
    xr.testing.assert_allclose(refl_bg1, refl_bg3, rtol=rtol)
    assert_allclose(g_bg1.magnitude, g_bg3.magnitude, rtol=rtol)
    assert_allclose(lstar_bg1.to('mm').magnitude, lstar_bg3.to('mm').magnitude,
                    rtol=rtol)


def test_reflection_polydispersity():
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = 0.5
    radius = Quantity('120.0 nm')
    index_matrix = sc.Index.constant(1.0)
    index_medium = sc.Index.constant(1.0)
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)
    radius2 = Quantity('120.0 nm')
    sphere2 = sc.Sphere(index_particle, radius2)
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit
    sphere_dist = sc.SphereDistribution([sphere, sphere2], concentration, pdi)

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    index_effective = sc.EffectiveIndex.from_particle(sphere, volume_fraction,
                                                      index_matrix)
    # first check that form factors agree
    coords = sc._make_input_coords(wavelength, np.linspace(0, np.pi, 10))
    sphere_ff = sphere.form_factor(coords, index_effective)
    sphere_dist_ff = sphere_dist.form_factor(coords, index_effective)
    assert_allclose(sphere_ff.to_numpy(), sphere_dist_ff.to_numpy(), rtol=1e-6)

    # monodisperse Mie case: sphere form factor, no structure factor (need to
    # specify particle and volume_fraction to FormStructureModel to calculate
    # transport length
    model = sc.model.FormStructureModel(sphere.form_factor, None,
                                        sphere.radius_q,
                                        index_effective, index_medium,
                                        particle=sphere,
                                        volume_fraction=volume_fraction)
    refl, _, _, g, lstar = sc.model.reflection(model, wavelength)

    # polydisperse model: sphere_dist form factor, no structure factor
    model = sc.model.FormStructureModel(sphere_dist.form_factor, None,
                                        sphere.radius_q,
                                        index_effective, index_medium,
                                        particle=sphere_dist,
                                        volume_fraction=volume_fraction)
    refl2, _, _, g2, lstar2 = sc.model.reflection(model, wavelength)

    xr.testing.assert_allclose(refl, refl2)
    assert_allclose(g.magnitude, g2.magnitude)
    assert_allclose(lstar.to('mm').magnitude, lstar2.to('mm').magnitude)

    # Outputs before refactoring structcol
    refl_before = 0.021202873774022364
    refl2_before = 0.0212028737585751
    g_before = 0.6149959692900278
    g2_before = 0.6149959696365628 # A: 0.6149959692900626
    lstar_before = 0.0037795694345017063
    lstar2_before = 0.0037795694345017063 # V: 0.0037899271938978255, A: 0.0037899271967178523

    rtol = 1e-13
    assert_allclose(refl.to_numpy(), refl_before, rtol=rtol)
    assert_allclose(refl2.to_numpy(), refl2_before, rtol=rtol)
    assert_allclose(g.magnitude, g_before, rtol=rtol)
    assert_allclose(g2.magnitude, g2_before, rtol=rtol)
    # lstar results aren't quite as close
    assert_allclose(lstar.to('mm').magnitude, lstar_before)
    assert_allclose(lstar2.to('mm').magnitude, lstar2_before)

    # test that the reflectance using only the structure factor is the same
    # using the polydisperse formula vs using Percus-Yevick in the limit of
    # monodispersity
    py_structure = sc.structure.PercusYevick(volume_fraction)
    model = sc.model.FormStructureModel(None, py_structure, sphere.radius_q,
                                        index_effective, index_medium,
                                        particle=sphere,
                                        volume_fraction=volume_fraction)
    refl3, _, _, g3, lstar3 = sc.model.reflection(model, wavelength)

    poly_structure = sc.structure.Polydisperse(volume_fraction, sphere_dist)
    model = sc.model.FormStructureModel(None, poly_structure,
                                        sphere_dist.spheres[0].radius_q,
                                        index_effective, index_medium,
                                        particle=sphere_dist,
                                        volume_fraction=volume_fraction)
    refl4, _, _, g4, lstar4 = sc.model.reflection(model, wavelength)


    xr.testing.assert_allclose(refl3, refl4)
    assert_allclose(g3.magnitude, g4.magnitude)
    assert_array_almost_equal(lstar3.to('mm').magnitude,
                              lstar4.to('mm').magnitude)

    # Outputs before refactoring structcol
    refl3_before= 0.6310965269823348
    refl4_before = 0.6310965259195878
    g3_before = -0.635630839621477
    g4_before = -0.6356308390717892
    lstar3_before = 0.0002005604473366244
    lstar4_before = 0.00020056044751316733

    rtol = 1e-13
    assert_allclose(refl3.to_numpy(), refl3_before)
    assert_allclose(refl4.to_numpy(), refl4_before)
    assert_allclose(g3.magnitude, g3_before)
    assert_allclose(g4.magnitude, g4_before)
    assert_allclose(lstar3.to('mm').magnitude, lstar3_before, rtol=rtol)
    assert_allclose(lstar4.to('mm').magnitude, lstar4_before, rtol=rtol)

    # test that the reflectance using both the structure and form factors is
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in
    # the limit of monodispersity
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    refl5, _, _, g5, lstar5 = sc.model.reflection(model, wavelength)
    model = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                             index_matrix, index_medium)
    refl6, _, _, g6, lstar6 = sc.model.reflection(model, wavelength)

    xr.testing.assert_allclose(refl5, refl6)
    assert_allclose(g5.magnitude, g6.magnitude)
    assert_allclose(lstar5.to('mm').magnitude, lstar6.to('mm').magnitude)

    # Outputs before refactoring structcol
    refl5_before = 0.2685710414987676
    refl6_before = 0.2685710407296461
    g5_before = -0.17681566915117486
    g6_before = -0.1768156684026972
    lstar5_before = 0.011593280877304636
    lstar6_before = 0.011593280876210265 # A/V: 0.011625051809100308

    assert_allclose(refl5.to_numpy(), refl5_before)
    assert_allclose(refl6.to_numpy(), refl6_before)
    assert_allclose(g5.magnitude, g5_before)
    assert_allclose(g6.magnitude, g6_before)
    assert_allclose(lstar5.to('mm').magnitude, lstar5_before)
    assert_allclose(lstar6.to('mm').magnitude, lstar6_before)

    # test that the reflectance is the same for a polydisperse monospecies
    # and a bispecies with equal types of particles
    concentration_mono = Quantity(1., '')
    concentration_bi = Quantity(np.array([0.3,0.7]), '')
    pdi_mono = 1e-1
    pdi_bi = Quantity(np.array([1e-1, 1e-1]), '')
    dist_mono = sc.SphereDistribution(sphere, concentration_mono, pdi_mono)
    dist_bi = sc.SphereDistribution([sphere, sphere2], concentration_bi,
                                    pdi_bi)

    model_mono = sc.model.PolydisperseHardSpheres(dist_mono, volume_fraction,
                                                  index_matrix, index_medium)
    model_bi = sc.model.PolydisperseHardSpheres(dist_bi, volume_fraction,
                                                index_matrix, index_medium)
    refl7, _, _, g7, lstar7 = sc.model.reflection(model_mono, wavelength)
    refl8, _, _, g8, lstar8 = sc.model.reflection(model_bi, wavelength)

    # these should be almost exactly the same
    rtol = 1e-14
    xr.testing.assert_allclose(refl7, refl8, rtol=rtol)
    assert_allclose(g7.magnitude, g8.magnitude, rtol=rtol)
    assert_allclose(lstar7.to('mm').magnitude, lstar8.to('mm').magnitude,
                    rtol=rtol)

    # test that the reflectance is the same regardless of the order in which
    # the radii are specified
    radius3 = Quantity('90.0 nm')
    sphere3 = sc.Sphere(index_particle, radius3)
    concentration3 = Quantity(np.array([0.5,0.5]), '')
    dist_13 = sc.SphereDistribution([sphere, sphere3], concentration3, pdi_bi)
    dist_31 = sc.SphereDistribution([sphere3, sphere], concentration3, pdi_bi)

    model_13 = sc.model.PolydisperseHardSpheres(dist_13, volume_fraction,
                                                index_matrix, index_medium)
    refl9, _, _, g9, lstar9 = sc.model.reflection(model_13, wavelength)
    model_31 = sc.model.PolydisperseHardSpheres(dist_31, volume_fraction,
                                                index_matrix, index_medium)
    refl10, _, _, g10, lstar10 = sc.model.reflection(model_31, wavelength)

    # these should be almost exactly the same
    rtol = 1e-13
    xr.testing.assert_allclose(refl9, refl10, rtol=rtol)
    assert_allclose(g9.magnitude, g10.magnitude, rtol=rtol)
    assert_allclose(lstar9.to('mm').magnitude, lstar10.to('mm').magnitude,
                    rtol=rtol)

# many tests are repeated here from test_reflection_polydispersity, but some
# have variations (for example, finite thickness). Also tolerances on the
# asserts are different. So we keep as a separate test instead of using
# pytest.mark.parametrize, even though there is some duplication of previous
# tests.
def test_reflection_polydispersity_with_absorption():
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = 0.5
    radius = Quantity('120.0 nm')
    index_matrix = sc.Index.constant(1.0+0.0003j)
    index_medium = sc.Index.constant(1.0)
    index_particle = sc.Index.constant(1.5+0.0005j)
    radius2 = Quantity('120.0 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit
    thickness = Quantity('10.0 um')
    sphere = sc.Sphere(index_particle, radius)
    sphere2 = sc.Sphere(index_particle, radius2)
    sphere_dist = sc.SphereDistribution([sphere, sphere2], concentration, pdi)

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    index_effective = sc.EffectiveIndex.from_particle(sphere, volume_fraction,
                                                      index_matrix)

    # monodisperse Mie case: sphere form factor, no structure factor (need to
    # specify particle and volume_fraction to FormStructureModel for transport
    # length calculation
    model = sc.model.FormStructureModel(sphere.form_factor, None,
                                        sphere.radius_q,
                                        index_effective, index_medium,
                                        particle=sphere,
                                        volume_fraction=volume_fraction)
    refl, _, _, g, lstar = sc.model.reflection(model, wavelength)

    # polydisperse model: sphere_dist form factor, no structure factor
    model = sc.model.FormStructureModel(sphere_dist.form_factor, None,
                                        sphere.radius_q,
                                        index_effective, index_medium,
                                        particle=sphere_dist,
                                        volume_fraction=volume_fraction)
    refl2, _, _, g2, lstar2 = sc.model.reflection(model, wavelength)

    xr.testing.assert_allclose(refl, refl2)
    assert_allclose(g.magnitude, g2.magnitude)
    assert_allclose(lstar.to('mm').magnitude, lstar2.to('mm').magnitude)

    # Outputs before refactoring structcol
    refl_before = 0.020910087489548684 # A/V:0.020791487299024698
    refl2_before = 0.020909855930303707 # A:0.020909855944662756 # A/V:0.02079125872215926
    g_before = 0.6150771860765984 # A/V:0.61562921974002 # A/V:726274264.1349005
    g2_before = 0.6150771864230516# A:0.6150771860766332 #A/V:0.6156292197400548 #A/V:726274264.1349416
    lstar_before = 0.0037892294836040373 #Before updating absorption in single scat:0.0044653875445681166 #A/V:0.0044717814146885779 #A/V:0.006279358811781641
    lstar2_before = 0.0037996137159816796 #Before updating absorption in single scat: 0.00447762476116312 #A:0.0044776247644925321 #A/V:0.0044840361567639936 #A/V:0.006296567149019748

    # rtols here are based on decimal precisions of assert_array_almost_equal
    # tests in previous revision
    assert_allclose(refl.to_numpy(), refl_before, rtol=1e-2)
    assert_allclose(refl2.to_numpy(), refl2_before, rtol=1e-2)
    assert_allclose(g.magnitude, g_before)
    assert_allclose(g2.magnitude, g2_before)
    assert_allclose(lstar.to('mm').magnitude, lstar_before, rtol=1e-2)
    assert_allclose(lstar2.to('mm').magnitude, lstar2_before, rtol=1e-2)

    # test that the reflectance using only the structure factor is the same
    # using the polydisperse formula vs using Percus-Yevick in the limit of
    # monodispersity
    py_structure = sc.structure.PercusYevick(volume_fraction)
    model = sc.model.FormStructureModel(None, py_structure, sphere.radius_q,
                                        index_effective, index_medium,
                                        particle=sphere,
                                        volume_fraction=volume_fraction)
    refl3, _, _, g3, lstar3 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)
    poly_structure = sc.structure.Polydisperse(volume_fraction, sphere_dist)
    model = sc.model.FormStructureModel(None, poly_structure,
                                        sphere_dist.spheres[0].radius_q,
                                        index_effective, index_medium,
                                        particle=sphere_dist,
                                        volume_fraction=volume_fraction)
    refl4, _, _, g4, lstar4 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)

    xr.testing.assert_allclose(refl3, refl4)
    assert_allclose(g3.magnitude, g4.magnitude)
    assert_allclose(lstar3.to('mm').magnitude, lstar4.to('mm').magnitude)

    # Outputs before refactoring structcol. Changed a couple values after
    # re-implementing absorption into model.reflection() (now uses n_sample.imag
    # to calculate the absorption cross section, in the same way as montecarlo.py)
    g3_before = -0.6356307606571816 #A/V:-27901.50120849103
    g4_before = -0.6356307601051542 #A/V:-27901.50118425936

    # we don't compare the reflection or transport length to what we had
    # before, since without a form factor the scattering cross-section isn't
    # well defined (the magnitude depends on the convention we use to assign a
    # form factor when form_factor=None).  However, the asymmetry parameter
    # does not depend on the magnitude of the cross-section, so can be compared
    # to previous results.
    rtol = 1e-11
    assert_allclose(g3.magnitude, g3_before, rtol=rtol)
    assert_allclose(g4.magnitude, g4_before, rtol=rtol)

    # test that the reflectance using both the structure and form factors is
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in
    # the limit of monodispersity
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    refl5, _, _, g5, lstar5 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)
    model = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                             index_matrix, index_medium)
    refl6, _, _, g6, lstar6 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)

    xr.testing.assert_allclose(refl5, refl6)
    assert_allclose(g5.magnitude, g6.magnitude)
    assert_allclose(lstar5.to('mm').magnitude, lstar6.to('mm').magnitude)

    # Outputs before refactoring structcol
    refl5_before = 0.11395667616828457 # A/V:0.11277597784758357
    refl6_before = 0.11377420192668616 #A/V:0.11259532698024184
    g5_before = -0.176272600668118 # A/V:-0.17376384100464944 #A/V:-209.15733480514967
    g6_before = -0.1762725998533963 # A/V:-0.17376384019461683 #A/V:-209.1573338372998
    lstar5_before = 0.01163694691 #Before updating absorption in single scat: A/V:0.013809880819376879 #A/V:0.013405648948885825
    lstar6_before = 0.011668837507 #Before updating absorption in single scat: A/V:0.013847726256293521 #A/V:0.013442386605693767

    # output values above for reflectances are off by almost 10%. rtols based
    # on previous revision's (decimal) tolerances for
    # assert_array_almost_equal()
    assert_allclose(refl5.to_numpy(), refl5_before, rtol=1e-1)
    assert_allclose(refl6.to_numpy(), refl6_before, rtol=1e-1)
    assert_allclose(g5.magnitude, g5_before)
    assert_allclose(g6.magnitude, g6_before)
    assert_allclose(lstar5.to('mm').magnitude, lstar5_before, rtol=1e-2)
    assert_allclose(lstar6.to('mm').magnitude, lstar6_before, rtol=1e-2)

    # test that the reflectances are (almost) the same when using an
    # almost-non-absorbing vs an non-absorbing polydisperse system
    ## When there is 1 mean diameter
    index_matrix2 = sc.Index.constant(1.0+1e-20j)
    index_matrix2_real = sc.Index.constant(1.0)
    index_particle2 = sc.Index.constant(1.5+1e-20j)
    index_particle2_real = sc.Index.constant(1.5)
    radius2 = Quantity('150.0 nm')
    pdi2 = Quantity(np.array([0.33, 0.33]), '')

    sphere = sc.Sphere(index_particle2, radius)
    sphere_real = sc.Sphere(index_particle2_real, radius)
    dist_real = sc.SphereDistribution(sphere_real, concentrations=1.0,
                                      polydispersities=pdi2[0])
    model = sc.model.PolydisperseHardSpheres(dist_real, volume_fraction,
                                             index_matrix2_real, index_medium)
    refl7, _, _, g7, lstar7 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)

    dist = sc.SphereDistribution(sphere, concentrations=1.0,
                                 polydispersities=pdi2[0])
    model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                             index_matrix2, index_medium)
    refl8, _, _, g8, lstar8 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)

    rtol = 1e-13
    xr.testing.assert_allclose(refl7, refl8, rtol=rtol)
    assert_allclose(g7.magnitude, g8.magnitude, rtol=rtol)
    assert_allclose(lstar7.to('mm').magnitude, lstar8.to('mm').magnitude,
                    rtol=rtol)

    ## When there are 2 mean diameters
    sphere2_real = sc.Sphere(index_particle2_real, radius2)
    dist2_real = sc.SphereDistribution([sphere_real, sphere2_real],
                                       concentration, pdi2)
    model = sc.model.PolydisperseHardSpheres(dist2_real, volume_fraction,
                                             index_matrix2_real, index_medium)
    refl9, _, _, g9, lstar9 = sc.model.reflection(model, wavelength,
                                                  thickness=thickness)
    sphere2 = sc.Sphere(index_particle2, radius2)
    dist2 = sc.SphereDistribution([sphere, sphere2],
                                  concentration, pdi2)
    model = sc.model.PolydisperseHardSpheres(dist2, volume_fraction,
                                             index_matrix2, index_medium)
    refl10, _, _, g10, lstar10 = sc.model.reflection(model, wavelength,
                                                     thickness=thickness)

    xr.testing.assert_allclose(refl9, refl10, rtol=1e-2)
    assert_allclose(g9.magnitude, g10.magnitude, rtol=1e-1)
    assert_allclose(lstar9.to('mm').magnitude, lstar10.to('mm').magnitude,
                    rtol=1e-2)
    # TODO: we should be careful with this last test. Interestingly, the values
    # for refl9 and refl10 become incrasingly closer to each other when the pdi
    # becomes large (~33%). No bugs were found after a careful examination, so
    # this behavior might be related to how polydispersity is implemented for
    # binary mixtures. Currently in model.py we calculate the form factor using
    # distance = mean radii and then we integrate the differential cross section
    # at said mean radii. We then average the cross sections from each radius.
    # Potentially, using the mean radii to find the average polydisperse form
    # factor and cross section might be a better approximation to the real form
    # factor and cross section when the size distribution is closer to uniform
    # (less narrow).

def test_g_transport_length():
    # test that the g and transport length do not depend on the thickness in the
    # presence of absorption
    wavelength = Quantity(600.0, 'nm')
    volume_fraction = 0.55
    radius = Quantity('100.0 nm')
    index_matrix = sc.Index.constant(1.0+0.0004j)
    index_medium = sc.Index.constant(1.0)
    index_particle = sc.Index.constant(1.5+0.0006j)
    sphere = sc.Sphere(index_particle, radius)
    thickness1 = Quantity('10.0 um')
    thickness2 = Quantity('100.0 um')

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    _, _, _, g, lstar = sc.model.reflection(model, wavelength,
                                            thickness=thickness1)
    _, _, _, g2, lstar2 = sc.model.reflection(model, wavelength,
                                              thickness=thickness2)

    assert_equal(g.magnitude, g2.magnitude)
    assert_equal(lstar.to('mm').magnitude, lstar2.to('mm').magnitude)

def test_reflection_throws_warnings_for_unspecified_parameters():
    # test that warnings are thrown when trying to calculate values for
    # transport and absorption length for FormStructureModel without specifying
    # number density and/or particle index
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity(np.array([110.0, 120.0]), 'nm')
    index_particle = [sc.Index.constant(1.5), sc.Index.constant(1.5)]
    index_matrix = sc.Index.constant(1.0)
    index_medium = sc.Index.constant(1.0)
    thickness = sc.Quantity(10, 'um')

    sphere = sc.Sphere(index_particle, radius)

    model1 = sc.model.FormStructureModel(sphere.form_factor, None,
                                         sphere.radius_q, index_matrix,
                                         index_medium)
    model2 = sc.model.FormStructureModel(sphere.form_factor, None,
                                         sphere.radius_q, index_matrix,
                                         index_medium,
                                         volume_fraction=volume_fraction)

    # should get two warnings when volume_fraction and particle are not
    # specified, or only volume_fraction is specified
    for model in (model1, model2):
        with pytest.warns(UserWarning) as record:
            _ = sc.model.reflection(model, wavelength)

            assert len(record) == 2
            assert "Number density cannot be" in str(record[0].message)
            assert "Absorption cross-section cannot" in str(record[1].message)

    # for finite thickness, should get a third warning
    with pytest.warns(UserWarning) as record:
        _ = sc.model.reflection(model1, wavelength, thickness=thickness)

        assert len(record) == 3
        assert "infinite thickness" in str(record[2].message)
