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
from pytest import raises
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_allclose)
import pytest
import structcol as sc
from structcol import montecarlo
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
    # for now, test against a single volume fraction
    # TODO: vectorize with volume fraction as commented below:
    # phi = np.array([0.15, 0.3, 0.45])
    phi = 0.45
    my_units = sc.ureg.millimeter
    thickness = 0.050 * sc.ureg.millimeter

    angles = sc.Quantity(np.linspace(0, np.pi, 100), 'rad')

    def test_base_model(self):
        """tests for Model base class"""
        model = sc.model.Model(sc.index.vacuum)
        with pytest.raises(NotImplementedError):
            model.differential_cross_section(self.wavelen, self.angles)

    def test_formstructure_model(self):
        """tests for the FormStructureModel"""
        # if form factor is None and structure factor is a constant, should
        # have a constant differential scattering cross section
        const = 1.0
        model = sc.model.FormStructureModel(None, sc.structure.Constant(const),
                                            self.ps_radius,
                                            sc.index.vacuum,
                                            sc.index.vacuum)
        dscat = model.differential_cross_section(self.wavelen[0], self.angles)

        xr.testing.assert_equal(dscat, xr.ones_like(dscat)*const)

        # Test that constant structure factor yields the same results as form
        # factor.  There is no effective index for a FormStructureModel, unless
        # one is explicitly given
        model = sc.model.FormStructureModel(self.ps_sphere.form_factor,
                                            sc.structure.Constant(const),
                                            self.ps_radius,
                                            sc.index.vacuum,
                                            sc.index.vacuum)
        dscat = model.differential_cross_section(self.wavelen[0],
                                                     self.angles)
        ff = self.ps_sphere.form_factor(self.wavelen[0], self.angles,
                                        sc.index.vacuum)

        xr.testing.assert_equal(dscat, ff)

        # Test that constant form factor yields the same results as structure
        # factor.
        structure_factor = sc.structure.PercusYevick(0.5)
        index_matrix = sc.index.water
        model = sc.model.FormStructureModel(None,
                                            structure_factor,
                                            self.ps_radius,
                                            index_matrix,
                                            sc.index.vacuum)
        dscat = model.differential_cross_section(self.wavelen[0],
                                                 self.angles)

        x = sc.size_parameter(index_matrix(self.wavelen[0]), self.ps_radius)
        x = x.to_numpy().squeeze()
        ql = (4*np.abs(x)*np.sin(self.angles/2)).to('').magnitude
        s = structure_factor(ql)

        # test numpy versions because DataArrays will have different coords
        # dscat for both polarizations should be equal to s
        assert_equal(dscat[0].to_numpy().squeeze(), s.to_numpy().squeeze())
        assert_equal(dscat[1].to_numpy().squeeze(), s.to_numpy().squeeze())

    def test_hardsphere_model(self):
        """tests that HardSphere model construction and differential cross
        section method work

        """
        index_matrix = sc.index.water
        glass = sc.model.HardSpheres(self.ps_sphere, self.phi, sc.index.water,
                                     sc.index.vacuum)

        # make sure form factor is calculated correctly
        angles = sc.Quantity(np.linspace(0, 180., 19), 'deg')
        form_model = glass.form_factor(self.wavelen, angles,
                                               index_matrix)
        form_sphere = glass.sphere.form_factor(self.wavelen, angles,
                                               index_matrix)
        xr.testing.assert_equal(form_model, form_sphere)

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
        index_particle = sc.index.polystyrene
        index_medium = sc.index.vacuum

        # single particle species, low volume fraction
        volume_fraction = 1e-8
        pdi = 1e-5
        concentration = 1.0
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)

        # for this low volume fraction, form factor should dominate
        # (note that polydisperse functions are not yet vectorized, so wavelen
        # must be a scalar)
        wavelen = sc.Quantity(400, 'nm')
        # start at a few degrees to avoid division by zero error
        angles = sc.Quantity(np.linspace(2, 180., 19), 'deg')
        form_model = model.form_factor(wavelen, angles, index_matrix)
        form_sphere = dist.spheres[0].form_factor(wavelen, angles,
                                                  index_matrix)
        # monodisperse and polydisperse form factors should be equal at low
        # polydispersity
        xr.testing.assert_allclose(form_model, form_sphere)

        # differential scattering cross sections should be very close for
        # monodisperse and polydisperse models in the limit of low
        # polydispersity
        mono_model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                          index_matrix, index_medium)
        dscat = model.differential_cross_section(wavelen, angles)
        dscat_mono = mono_model.differential_cross_section(wavelen, angles)
        # dscat mono will have an extra "volfrac" scalar dimension because the
        # PY structure factor is a function of volume fraction.  The
        # interpolated structure factor is not
        xr.testing.assert_allclose(dscat, dscat_mono.drop_vars("volfrac"))

        # and structure factor should be close to 1
        n_ext = index_matrix(wavelen)
        lengthscale = dist.spheres[0].radius_q
        ql = sc.ql(n_ext, lengthscale, angles)
        s = model.structure_factor(ql)
        xr.testing.assert_allclose(s, xr.ones_like(s))

        # we check also that the scattering cross sections are the same for the
        # monodisperse and polydisperse models
        cscat = model.scattering_cross_section(wavelen, angles)
        cscat_mono = mono_model.scattering_cross_section(wavelen, angles)
        assert_allclose(cscat.to_preferred().magnitude,
                        cscat_mono.to_preferred().magnitude)

        # now finite volume fraction, low polydispersity
        volume_fraction = 0.5
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)

        # structure factor should be almost the same as for a monodisperse
        # glass
        mono_model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                          index_matrix, index_medium)

        s = model.structure_factor(ql)
        s_mono = mono_model.structure_factor(ql)
        assert_allclose(s, s_mono, rtol=1e-7, atol=1e-7)

        # and the same as would be calculated from creating a structure factor
        # directly
        diameters = self.ps_sphere.diameter_q
        structure_factor = sc.structure.Polydisperse(volume_fraction, dist)
        s_poly = structure_factor(ql)
        xr.testing.assert_equal(s, s_poly)

    @pytest.mark.parametrize("volume_fraction", [0.01, 0.3, 0.6])
    def test_formstructure_with_data(self, volume_fraction):
        """tests that FormStructure model with interpolated structure factor
        (generated from Percus Yevick) gives same results as HardSpheres model
        with PY structure factor

        """
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

        # TODO test vectorization; for now this is single-wavelength
        fs_dscat = fs_model.differential_cross_section(self.wavelen[0],
                                                       self.angles)
        py_dscat = py_model.differential_cross_section(self.wavelen[0],
                                                       self.angles)

        # with cubic interpolation, relative error is a little larger than
        # 1e-4 at 60% volume fraction and 750 data points.
        fs_dscat = fs_dscat.to_numpy().squeeze()
        py_dscat = py_dscat.to_numpy().squeeze()
        assert_allclose(fs_dscat[0], py_dscat[0], rtol=1e-3)
        assert_allclose(fs_dscat[1], py_dscat[1], rtol=1e-3)
        # TODO: test reflectance as well

    @pytest.mark.parametrize("index_matrix", [sc.index.water,
                                              sc.Index.constant(1.59+0.001j)])
    def test_scattering_cross_section(self, index_matrix):
        """Test that the scattering_cross_section() method returns reasonable
        values (the above tests mostly focus on the
        differential_cross_section() method)

        """
        # TODO: test vectorization after changing _integrate_cross_section
        wavelen = self.wavelen[0]

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
        m = sc.index.ratio(n_particle, n_matrix)
        x = sc.size_parameter(n_matrix, self.ps_radius).to_numpy()

        # do the calculation using method from Model object, setting distance
        # appropriately if there is absorption in the matrix
        ff_kwargs = {}
        if np.any(n_matrix.imag > 0):
            ff_kwargs["kd"] = sc.wavevector(n_matrix) * model.lengthscale
            ff_kwargs["kd"] = ff_kwargs["kd"].to('').magnitude
        cscat = model.scattering_cross_section(wavelen, angles, **ff_kwargs)

        # now do calculation using Mie theory, using appropriate function for
        # the cross-section
        wavelen_media = wavelen/n_matrix.to_numpy().squeeze()
        if np.any(n_matrix.imag > 0):
            lmax = mie._nstop(np.array(x).max())
            albl = mie._scatcoeffs(m, x, lmax)

            radius = self.ps_radius
            cscat_mie = mie._cross_sections_complex_medium_sudiarta(*albl, x,
                                                                    radius)
            # Fu cross sections
            nstop = mie._nstop(x)
            albl = mie._scatcoeffs(m, x, nstop)
            cldl = mie._internal_coeffs(m, x, nstop)
            x_med = sc.size_parameter(n_matrix, radius).to_numpy()
            # fu calculation expects indexes as Quantity objects
            n_particle = sc.Quantity(n_particle.to_numpy(), '')
            n_medium = sc.Quantity(n_matrix.to_numpy(), '')

            cscat_fu = mie._cross_sections_complex_medium_fu(*albl, *cldl,
                                                             radius, n_particle,
                                                             n_medium, x,
                                                             x_med, wavelen)

            # first check that the Fu and Sudiarta calculations agree
            assert_allclose(cscat_fu[0].magnitude, cscat_mie[0].magnitude)

        else:
            cscat_mie = mie.calc_cross_sections(m, x, wavelen_media)

        # Now check that the Mie calculation and Model method calculations
        # agree.
        assert_allclose(cscat.to_preferred().magnitude,
                        cscat_mie[0].to_preferred().magnitude,
                        rtol=1e-3)
        # Agreement is to within 1e-3 relative error for absorbing media, and
        # 1e-5 for non-absorbing.  The discrepancy in absorbing media doesn't
        # seem to improve with more integration points, but gets worse with
        # increasingly large imaginary component of the refractive index.  Fu
        # and Sudiarta calculations agree even at n.imag = 1j, so there may be
        # an issue in integrate_intensity_complex_medium()
        #
        # TODO: add more testing of Fu, Sudiarta cross sections in pymie, along
        # with more tests of integrate_intensity_complex_medium()

    @pytest.mark.parametrize("index_matrix", [sc.index.water,
                                              sc.Index.constant(1.59+0.001j)])
    def test_scattering_cross_section_polydisperse(self, index_matrix):
        """Test the scattering_cross_section() method for the
        PolydisperseHardSpheres model
        """
        # TODO: test vectorization after changing _integrate_cross_section
        wavelen = self.wavelen[0]
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
        n_matrix = single_model.index_matrix(wavelen)
        ff_kwargs = {}
        if np.any(n_matrix.imag > 0):
            ff_kwargs["kd"] = sc.wavevector(n_matrix) * self.ps_radius
            ff_kwargs["kd"] = ff_kwargs["kd"].to('').magnitude
        dscat_1 = single_model.differential_cross_section(wavelen, angles,
                                                          **ff_kwargs)
        cscat_1 = single_model.scattering_cross_section(wavelen, angles,
                                                        **ff_kwargs)

        # do the calculation using bidisperse polydisperse model
        n_matrix = binary_model.index_matrix(wavelen)
        ff_kwargs = {}
        if np.any(n_matrix.imag > 0):
            ff_kwargs["kd"] = (sc.wavevector(n_matrix) *
                               binary_model.sphere_dist.diameters_q/2)
            ff_kwargs["kd"] = ff_kwargs["kd"].to('').magnitude

        dscat_2 = binary_model.differential_cross_section(wavelen, angles,
                                                               **ff_kwargs)
        cscat_2 = binary_model.scattering_cross_section(wavelen, angles,
                                                        **ff_kwargs)
        xr.testing.assert_equal(dscat_2, dscat_1)
        assert_equal(cscat_2.to_preferred().magnitude,
                     cscat_1.to_preferred().magnitude)

    @pytest.mark.parametrize("index_matrix", [sc.index.water,
                                              sc.Index.constant(1.59 + 0.001j),
                                              sc.Index.constant(1.59 + 0.1j)])
    def test_scattering_against_phase_function_method(self, index_matrix):
        """Test that the scattering_cross_section() method rom a Model object
        returns exactly the same results as montecarlo.phase_function() method.
        This test exists only for refactoring.  Can remove when
        montecarlo.phase_function() is removed
        """
        wavelen = self.wavelen[0]
        volume_fraction = 0.5
        index_medium = sc.index.vacuum
        model = sc.model.HardSpheres(self.ps_sphere, volume_fraction,
                                     index_matrix, index_medium)

        # start at a few degrees to avoid division by zero error
        angles = sc.Quantity(np.linspace(2, 180., 19), 'deg')

        # Need to use effective index to get perfect agreement between the two.
        # Make sure that kd uses the effective index.
        index_external = sc.EffectiveIndex.from_particle(self.ps_sphere,
                                                         volume_fraction,
                                                         index_matrix)
        n_ext = index_external(wavelen)
        n_particle = self.index_particle(wavelen)
        k = sc.wavevector(n_ext)

        ff_kwargs = {}
        if np.any(n_ext.imag > 0):
            ff_kwargs['kd'] = (k * model.lengthscale).to('').magnitude
        cscat = model.scattering_cross_section(wavelen, angles,
                                               **ff_kwargs)

        m = sc.index.ratio(n_particle, n_ext)
        x = sc.size_parameter(n_ext, self.ps_radius).to_numpy()
        diameters = sc.Quantity(np.array(self.ps_radius.magnitude),
                                self.ps_radius.units) * 2
        _, cscat_mc = montecarlo.phase_function(m, x, angles,
                                                volume_fraction, k, None,
                                                diameters = diameters,
                                                n_sample=n_ext,
                                                wavelen=wavelen)

        # should be exactly equal
        assert_equal(cscat.to_preferred().magnitude,
                     cscat_mc.to_preferred().magnitude)

        # Now test for polydisperse system with single component, low
        # polydispersity.  Should give very close results to monodisperse
        concentration = 1.0
        pdi = 1e-5
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)

        if np.any(n_ext.imag > 0):
            ff_kwargs['kd'] = (k * model.lengthscale).to('').magnitude
        else:
            ff_kwargs = {}
        cscat = model.scattering_cross_section(wavelen, angles,
                                               **ff_kwargs)
        assert_allclose(cscat.to_preferred().magnitude,
                        cscat_mc.to_preferred().magnitude, rtol=1e-5)

        # check for polydisperse system with finite polydispersity.  We
        # compare against the analogous computation with the phase_function()
        # function.  Should give exactly the same results.
        pdi = 0.15
        dist = sc.SphereDistribution(self.ps_sphere, concentration, pdi)
        model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                 index_matrix, index_medium)
        cscat = model.scattering_cross_section(wavelen, angles, **ff_kwargs)

        diameters = sc.Quantity(np.atleast_1d(diameters.magnitude),
                                diameters.units)
        concentration = np.atleast_1d(1.0)
        _, cscat_mc = montecarlo.phase_function(m, x, angles,
                                                volume_fraction, k, None,
                                                concentration=concentration,
                                                pdi=pdi,
                                                diameters = diameters,
                                                form_type="polydisperse",
                                                structure_type="polydisperse",
                                                n_sample=n_ext,
                                                wavelen=wavelen)

        assert_equal(cscat.to_preferred().magnitude,
                     cscat_mc.to_preferred().magnitude)

        # Now binary system with finite polydispersity, compared to the
        # analogous computation with the phase_function() function. Should give
        # exactly the same results.
        sphere1 = sc.Sphere(self.index_particle, sc.Quantity(0.15, 'um'))
        sphere2 = sc.Sphere(self.index_particle, sc.Quantity(0.25, 'um'))
        concentration = np.array([0.1, 0.9])
        pdi = np.array([0.15, 0.15])
        dist = sc.SphereDistribution([sphere1, sphere2], concentration, pdi)
        binary_model = sc.model.PolydisperseHardSpheres(dist, volume_fraction,
                                                        index_matrix,
                                                        index_medium)
        ff_kwargs = {}
        diameters = binary_model.sphere_dist.diameters_q

        if np.any(n_ext.imag > 0):
            ff_kwargs["kd"] = (sc.wavevector(n_ext) * diameters/2)
            ff_kwargs["kd"] = ff_kwargs["kd"].to('').magnitude

        cscat = binary_model.scattering_cross_section(wavelen, angles,
                                                      **ff_kwargs)

        m = sc.index.ratio(n_particle, n_ext)
        x = sc.size_parameter(n_ext, sphere1.radius_q).to_numpy()
        _, cscat_mc = montecarlo.phase_function(m, x, angles,
                                                volume_fraction, k, None,
                                                concentration=concentration,
                                                pdi=pdi,
                                                diameters = diameters,
                                                form_type="polydisperse",
                                                structure_type="polydisperse",
                                                n_sample=n_ext,
                                                wavelen=wavelen)

        assert_equal(cscat.to_preferred().magnitude,
                     cscat_mc.to_preferred().magnitude)


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
    n1 = 1.00
    n2 = 1.5

    # quantities calculated from
    # http://www.calctool.org/CALC/phys/optics/reflec_refrac
    rpar, rperp = sc.model.fresnel_reflection(n1, n2, Quantity('0.0 deg'))
    assert_almost_equal(rpar, 0.04)
    assert_almost_equal(rperp, 0.04)
    rpar, rperp = sc.model.fresnel_reflection(n1, n2, Quantity('45.0 deg'))
    assert_almost_equal(rpar, 0.00846646)
    assert_almost_equal(rperp, 0.0920134)

    # test total internal reflection
    rpar, rperp = sc.model.fresnel_reflection(n2, n1, Quantity('45.0 deg'))
    assert_equal(rpar, 1.0)
    assert_equal(rperp, 1.0)

    # test no total internal reflection (just below critical angle)
    rpar, rperp = sc.model.fresnel_reflection(n2, n1, Quantity('41.810 deg'))
    assert_almost_equal(rpar, 0.972175, decimal=6)
    assert_almost_equal(rperp, 0.987536, decimal=6)

    # test vectorized computation
    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # check for value error
    raises(ValueError, sc.model.fresnel_reflection, n2, n1, angles)
    angles = Quantity(np.linspace(0, 90., 10), 'deg')
    rpar, rperp = sc.model.fresnel_reflection(n2, n1, angles)
    rpar_std = np.array([0.04, 0.0362780, 0.0243938, 0.00460754, 0.100064, 1.0,
                         1.0, 1.0, 1.0, 1])
    rperp_std = np.array([0.04, 0.0438879, 0.0590632, 0.105773, 0.390518, 1.0,
                         1.0, 1.0, 1.0, 1.0])
    assert_array_almost_equal(rpar, rpar_std)
    assert_array_almost_equal(rperp, rperp_std)

    # test transmission
    tpar, tperp = sc.model.fresnel_transmission(n2, n1, angles)
    tpar_std = 1.0-rpar_std
    tperp_std = 1.0-rperp_std
    assert_array_almost_equal(tpar, tpar_std)
    assert_array_almost_equal(tperp, tperp_std)


def test_theta_refraction():
    # test that the detection angles theta are refracted correctly at the
    # medium-sample interface. When n_sample < n_medium, the scattered angles
    # in the reflection hemisphere (90-180 deg) are refracted at the interface
    # into a smaller range of angles (>90-180 deg). This test checks that the
    # the reflectance is close to 0 when the angles between theta_min and
    # theta_max are outside the range of refracted scattered angles.
    incident_angle = Quantity('0.0 deg')
    wavelength = Quantity(500.0, 'nm')
    radius = Quantity('100.0 nm')
    volume_fraction = 0.5
    index_particle = sc.Index.constant(1.0)
    particle = sc.Sphere(index_particle, radius)
    vf_array = particle.volume_fraction(volume_fraction)
    index_matrix =  sc.Index.constant(1.0)
    index_medium = sc.Index.constant(2.0)
    n_medium = index_medium(wavelength)
    theta_min = Quantity(np.pi/2, 'deg')

    # set theta_max to be slightly smaller than the theta corresponding to
    # total internal reflection (calculated manually to be 2.61799388)
    theta_max = Quantity(2.617, 'deg')
    detector = sc.model.Detector(theta_min, theta_max)
    refl1, _, _, _, _ = sc.model.reflection(index_particle, index_matrix,
                                         index_medium,
                                         wavelength, radius, volume_fraction,
                                         detector=detector,
                                         structure_type=None)
    # try a different range of thetas (but keeping theta_max < total internal
    # reflection angle)
    theta_max = Quantity(2., 'deg')
    detector = sc.model.Detector(theta_min, theta_max)
    refl2, _, _, _, _ = sc.model.reflection(index_particle, index_matrix,
                                         index_medium,
                                         wavelength, radius, volume_fraction,
                                         detector=detector,
                                         structure_type=None)

    # the reflection should be zero plus the fresnel reflection term
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        vf_array, wavelength)
    r_fresnel = sc.model.fresnel_reflection(n_medium.to_numpy(),
                                         n_sample.to_numpy(), incident_angle)
    r_fresnel_avg = (r_fresnel[0] + r_fresnel[1]) / 2
    assert_almost_equal(refl1.magnitude, r_fresnel_avg)
    assert_almost_equal(refl2.magnitude, r_fresnel_avg)
    assert_almost_equal(refl1.magnitude, refl2.magnitude)


def test_differential_cross_section():
    # Test that the differential cross sections for non-core-shell particles
    # and core-shells are the same at low volume fractions, assuming that the
    # particle diameter of the non-core-shells is the same as the core
    # diameter in the core-shells (shell of the core-shells is vacuum)

    wavelen = Quantity('500.0 nm')
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
    diff = model.differential_cross_section(wavelen, angles)

    # Differential cross section for core-shells. Core is equal to
    # non-core-shell particle, and shell is made of vacuum
    radius_cs = Quantity(np.array([100.0, 110.0]), 'nm')
    index_cs = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
    sphere_cs = sc.Sphere(index_cs, radius_cs)
    n_particle_cs = sphere_cs.n(wavelen)

    # adjust volume fraction of core-shells so that volume fraction of cores is
    # same as that of non-core-shells
    vf_core = sphere_cs.volume_fraction()[0].to_numpy()
    volume_fraction_cs = volume_fraction/vf_core

    model_cs = sc.model.HardSpheres(sphere_cs, volume_fraction_cs,
                                    index_matrix, index_medium)
    diff_cs = model_cs.differential_cross_section(wavelen, angles)

    assert_allclose(diff[0], diff_cs[0], rtol=1e-4)
    assert_allclose(diff[1], diff_cs[1], rtol=1e-4)


def test_reflection_core_shell():
    # Test reflection, anisotropy factor, and transport length calculations to
    # make sure the values for refl, g, and lstar remain the same after adding
    # core-shell capability into the model
    wavelength = Quantity(500.0, 'nm')
    thickness = Quantity(15.0, 'um')

    # Non core-shell particles with Maxwell-Garnett effective index
    volume_fraction = 0.5
    radius = Quantity('120.0 nm')
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.Index.constant(1.0)
    index_medium = index_matrix

    detector = sc.model.Detector(theta_min=Quantity('90.0 deg'))
    refl1, _, _, g1, lstar1 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength, radius,
                                               volume_fraction, thickness =
                                               Quantity('15000.0 nm'),
                                               detector=detector,
                                               small_angle=Quantity('5.0 deg'),
                                               maxwell_garnett=True)

    # Non core-shell particles with Bruggeman effective index
    volume_fraction2 = 0.00001
    refl2, _, _, g2, lstar2 = sc.model.reflection(index_particle, index_matrix, index_medium,
                                               wavelength, radius,
                                               volume_fraction2,
                                               thickness =
                                               Quantity('15000.0 nm'),
                                               detector=detector,
                                               small_angle=Quantity('5.0 deg'),
                                               maxwell_garnett=False)

    # Core-shell particles of core diameter equal to non core shell particles,
    # and shell index of air. With Bruggeman effective index
    radius3 = Quantity(np.array([120.0, 130.0]), 'nm')
    index3 = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
    sphere_cs = sc.Sphere(index3, radius3)
    volume_fraction3 = volume_fraction2 * (radius3[1]**3 / radius3[0]**3)

    refl3, _, _, g3, lstar3 = sc.model.reflection(index3, index_matrix,
                                               index_medium,
                                               wavelength, radius3,
                                               volume_fraction3,
                                               thickness =
                                               Quantity('15000.0 nm'),
                                               small_angle=Quantity('5.0 deg'),
                                               detector=detector,
                                               maxwell_garnett=False)

    # Outputs for refl, g, and lstar before adding core-shell capability
    refl = Quantity(0.20772170840902376, '')
    g = Quantity(-0.18931942267032678, '')
    lstar = Quantity(10810.088573316663, 'nm')

    # Compare old outputs (before adding core-shell capability) and new outputs
    # for a non-core-shell using Maxwell-Garnett
    assert_array_almost_equal(refl1.magnitude, refl.magnitude)
    assert_array_almost_equal(g1.magnitude, g.magnitude)
    assert_array_almost_equal(lstar1.to('nm').magnitude, lstar.magnitude)

    # Compare a non-core-shell and a core-shell with shell index of air using
    # Bruggeman
    assert_array_almost_equal(refl2.magnitude, refl3.magnitude)
    assert_array_almost_equal(g2.magnitude, g3.magnitude, decimal=5)
    assert_array_almost_equal(lstar2.to('mm').magnitude, lstar3.to('mm').magnitude, decimal=4)


    # Test that the reflectance is the same for a core-shell that absorbs (with
    # the same refractive indices for all layers) and a non-core-shell that
    # absorbs with the same index

    # Absorbing non-core-shell
    radius4 = Quantity('120.0 nm')
    index_particle4 = sc.Index.constant(1.5+0.001j)
    sphere = sc.Sphere(index_particle4, radius4)
    refl4 = sc.model.reflection(index_particle4, index_matrix, index_medium,
                             wavelength, radius4, volume_fraction,
                             thickness=thickness)[0]

    # Absorbing core-shell
    radius5 = Quantity(np.array([110.0, 120.0]), 'nm')
    index5 = [sc.Index.constant(1.5+0.001j), sc.Index.constant(1.5+0.001j)]
    sphere_cs = sc.Sphere(index5, radius5)
    refl5 = sc.model.reflection(index5, index_matrix, index_medium, wavelength,
                             radius5, volume_fraction, thickness=thickness)[0]

    assert_array_almost_equal(refl4.magnitude, refl5.magnitude, decimal=3)

    # Same as previous test but with absorbing matrix
    # Non-core-shell
    radius6 = Quantity('120.0 nm')
    index_particle6 = sc.Index.constant(1.5+0.001j)
    sphere = sc.Sphere(index_particle6, radius6)
    index_matrix6 = sc.Index.constant(1.0+0.001j)
    refl6 = sc.model.reflection(index_particle6, index_matrix6, index_medium,
                             wavelength, radius6, volume_fraction,
                             thickness=thickness)[0]

    # Core-shell
    index7 = [sc.Index.constant(1.5+0.001j), sc.Index.constant(1.5+0.001j)]
    radius7 = Quantity(np.array([110.0, 120.0]), 'nm')
    sphere_cs = sc.Sphere(index7, radius7)
    index_matrix7 = sc.Index.constant(1.0+0.001j)
    refl7 = sc.model.reflection(index7, index_matrix7, index_medium, wavelength,
                             radius7, volume_fraction, thickness=thickness)[0]

    assert_array_almost_equal(refl6.magnitude, refl7.magnitude, decimal=3)


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
    n_particle_real = sphere_real.n(wavelength)
    n_particle_complex = sphere_complex.n(wavelength)

    # With Maxwell-Garnett
    refl_mg1, _, _, g_mg1, lstar_mg1 = sc.model.reflection(index_particle_real,
                                                        index_matrix,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = sc.model.reflection(index_particle_complex,
                                                        index_matrix,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=True)

    assert_array_almost_equal(refl_mg1.magnitude, refl_mg2.magnitude)
    assert_array_almost_equal(g_mg1.magnitude, g_mg2.magnitude)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg2.magnitude)

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

    assert_array_almost_equal(refl_mg1.magnitude, refl_mg1_before, decimal=9)
    assert_array_almost_equal(refl_mg2.magnitude, refl_mg2_before, decimal=9)
    assert_array_almost_equal(g_mg1.magnitude, g_mg1_before, decimal=10)
    assert_array_almost_equal(g_mg2.magnitude, g_mg2_before, decimal=10)
    assert_array_almost_equal(lstar_mg1.to('nm').magnitude,
                              lstar_mg1_before, decimal=10)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg2.magnitude, decimal=10)

    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = sc.model.reflection(index_particle_real,
                                                        index_matrix,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = sc.model.reflection(index_particle_complex,
                                                        index_matrix,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=False)

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg2.magnitude)
    assert_array_almost_equal(g_bg1.magnitude, g_bg2.magnitude)
    assert_array_almost_equal(lstar_bg1.magnitude, lstar_bg2.magnitude)

    # Outputs before refactoring structcol
    refl_bg1_before = 0.2685710414987676
    refl_bg2_before = 0.2685710414987676
    g_bg1_before = -0.17681566915117486
    g_bg2_before = -0.17681566915117486
    # these are in nm
    lstar_bg1_before = 11593.280877304634
    lstar_bg2_before = 11593.280877304634

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg1_before, decimal=10)
    assert_array_almost_equal(refl_bg2.magnitude, refl_bg2_before, decimal=10)
    assert_array_almost_equal(g_bg1.magnitude, g_bg1_before, decimal=10)
    assert_array_almost_equal(g_bg2.magnitude, g_bg2_before, decimal=10)
    assert_array_almost_equal(lstar_bg1.to('nm').magnitude,
                              lstar_bg1_before, decimal=10)
    assert_array_almost_equal(lstar_bg2.to('nm').magnitude,
                              lstar_bg2_before, decimal=10)

    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    index_particle_complex2 = sc.Index.constant(1.5+1e-8j)
    sphere_complex2 = sc.Sphere(index_particle_complex2, radius)

    thickness = Quantity('100.0 um')

    # With Bruggeman
    refl_bg3, _, _, g_bg3, lstar_bg3 = sc.model.reflection(index_particle_complex2, index_matrix,
                                                        index_medium, wavelength,
                                                        radius, volume_fraction,
                                                        thickness=thickness,
                                                        maxwell_garnett=False)
    assert_array_almost_equal(refl_bg1.magnitude, refl_bg3.magnitude, decimal=3)
    assert_array_almost_equal(g_bg1.magnitude, g_bg3.magnitude, decimal=3)
    assert_array_almost_equal(lstar_bg1.to('mm').magnitude, lstar_bg3.to('mm').magnitude, decimal=4)


def test_calc_g():
    # test that the anisotropy factor for multilayer spheres are the same when
    # using calc_g from mie.py in pymie and using the model
    wavelength = Quantity(500.0, 'nm')

    # calculate g using the model
    radius = Quantity(np.array([120.0, 130.0]), 'nm')
    index = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
    sphere = sc.Sphere(index, radius)
    n_particle = sphere.n(wavelength)

    volume_fraction = Quantity(0.01, '')
    index_matrix = sc.Index.constant(1.0)
    index_medium = index_matrix

    _, _, _, g1, _= sc.model.reflection(index, index_matrix, index_medium,
                                     wavelength, radius, volume_fraction,
                                     small_angle=Quantity('0.01 deg'),
                                     num_angles=1000, structure_type=None)

    # calculate g using calc_g in pymie
    vf_array = sphere.volume_fraction(volume_fraction)
    n_sample = sc.index.effective_index(index + [index_matrix], vf_array,
                                        wavelength)
    m = sc.index.ratio(n_particle, n_sample)
    x = mie.size_parameter(wavelength, n_sample.to_numpy().squeeze(), radius)
    qscat, qext, qback = mie.calc_efficiencies(m, x)
    g2 = mie.calc_g(m,x)

    assert_array_almost_equal(g1.magnitude, g2)

    # Outputs before refactoring structcol
    g1_before = 0.5064750277811477
    g2_before = 0.5064757158664487

    assert_almost_equal(g1.magnitude, g1_before)
    assert_almost_equal(g2, g2_before)

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

    _, _, _, _, lstar_model = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength, radius,
                                               volume_fraction,
                                               maxwell_garnett=False)

    # transport length from Mie theory
    vf_array = sphere.volume_fraction(volume_fraction)
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        vf_array, wavelength)
    n_particle = sphere.n(wavelength)
    m = sc.index.ratio(n_particle, n_sample)
    x = mie.size_parameter(wavelength, n_sample.to_numpy().squeeze(), radius)
    g = mie.calc_g(m,x)

    number_density = sphere.number_density(volume_fraction)
    cscat = mie.calc_cross_sections(m, x, wavelength)[0]

    lstar_mie = 1 / (number_density * cscat * (1-g))

    assert_array_almost_equal(lstar_model.to('m').magnitude, lstar_mie.to('m').magnitude, decimal=4)

def test_reflection_absorbing_matrix():
    # test that the reflections with a real n_matrix and with a complex
    # n_matrix with a 0 imaginary component are the same
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = 0.5
    radius = Quantity('120.0 nm')
    index_matrix_real = sc.Index.constant(1.0)
    index_matrix_imag = sc.Index.constant(1.0 + 0j)
    index_medium = sc.Index.constant(1.0)
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)

    # With Maxwell-Garnett
    refl_mg1, _, _, g_mg1, lstar_mg1 = sc.model.reflection(index_particle,
                                                        index_matrix_real,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = sc.model.reflection(index_particle,
                                                        index_matrix_imag,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=True)

    assert_array_almost_equal(refl_mg1.magnitude, refl_mg2.magnitude)
    assert_array_almost_equal(g_mg1.magnitude, g_mg2.magnitude)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg2.magnitude)

    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = sc.model.reflection(index_particle,
                                                        index_matrix_real,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = sc.model.reflection(index_particle,
                                                        index_matrix_imag,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        maxwell_garnett=False)

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg2.magnitude)
    assert_array_almost_equal(g_bg1.magnitude, g_bg2.magnitude)
    assert_array_almost_equal(lstar_bg1.magnitude, lstar_bg2.magnitude)

    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    thickness = Quantity('100.0 um')
    index_matrix_imag2 = sc.Index.constant(1.0 + 1e-8j)

    # With Bruggeman
    refl_bg3, _, _, g_bg3, lstar_bg3 = sc.model.reflection(index_particle,
                                                        index_matrix_imag2,
                                                        index_medium,
                                                        wavelength, radius,
                                                        volume_fraction,
                                                        thickness=thickness,
                                                        maxwell_garnett=False)

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg3.magnitude, decimal=3)
    assert_array_almost_equal(g_bg1.magnitude, g_bg3.magnitude, decimal=3)
    assert_array_almost_equal(lstar_bg1.to('mm').magnitude, lstar_bg3.to('mm').magnitude, decimal=4)


def test_reflection_polydispersity():
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120.0 nm')
    index_matrix = sc.Index.constant(1.0)
    index_medium = sc.Index.constant(1.0)
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)
    radius2 = Quantity('120.0 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    refl, _, _, g, lstar = sc.model.reflection(index_particle, index_matrix,
                                            index_medium, wavelength, radius,
                                            volume_fraction,
                                            structure_type=None,
                                            form_type='sphere')
    refl2, _, _, g2, lstar2 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction,
                                               radius2 = radius2,
                                               concentration =
                                               concentration, pdi = pdi,
                                               structure_type=None,
                                               form_type='polydisperse')
    assert_array_almost_equal(refl.magnitude, refl2.magnitude)
    assert_array_almost_equal(g.magnitude, g2.magnitude)
    assert_array_almost_equal(lstar.to('mm').magnitude, lstar2.to('mm').magnitude, decimal=4)

    # Outputs before refactoring structcol
    refl_before = 0.021202873774022364
    refl2_before = 0.0212028737585751
    g_before = 0.6149959692900278
    g2_before = 0.6149959696365628 # A: 0.6149959692900626
    lstar_before = 0.0037795694345017063
    lstar2_before = 0.0037795694345017063 # V: 0.0037899271938978255, A: 0.0037899271967178523

    assert_array_almost_equal(refl.magnitude, refl_before, decimal=10)
    assert_array_almost_equal(refl2.magnitude, refl2_before, decimal=10)
    assert_array_almost_equal(g.magnitude, g_before, decimal=10)
    assert_array_almost_equal(g2.magnitude, g2_before, decimal=10)
    assert_array_almost_equal(lstar.to('mm').magnitude, lstar_before, decimal=10)
    assert_array_almost_equal(lstar2.to('mm').magnitude, lstar2_before, decimal=11)

    # test that the reflectance using only the structure factor is the same
    # using the polydisperse formula vs using Percus-Yevick in the limit of
    # monodispersity

    refl3, _, _, g3, lstar3 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction,
                                               structure_type='glass',
                                               form_type=None)

    refl4, _, _, g4, lstar4 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius2, concentration =
                                               concentration, pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type=None)

    assert_array_almost_equal(refl3.magnitude, refl4.magnitude)
    assert_array_almost_equal(g3.magnitude, g4.magnitude)
    assert_array_almost_equal(lstar3.to('mm').magnitude, lstar4.to('mm').magnitude, decimal=4)

    # Outputs before refactoring structcol
    refl3_before= 0.6310965269823348
    refl4_before = 0.6310965259195878
    g3_before = -0.635630839621477
    g4_before = -0.6356308390717892
    lstar3_before = 0.0002005604473366244
    lstar4_before = 0.00020056044751316733

    assert_array_almost_equal(refl3.magnitude, refl3_before, decimal=10)
    assert_array_almost_equal(refl4.magnitude, refl4_before, decimal=10)
    assert_array_almost_equal(g3.magnitude, g3_before, decimal=10)
    assert_array_almost_equal(g4.magnitude, g4_before, decimal=10)
    assert_array_almost_equal(lstar3.to('mm').magnitude, lstar3_before, decimal=10)
    assert_array_almost_equal(lstar4.to('mm').magnitude, lstar4_before, decimal=10)

    # test that the reflectance using both the structure and form factors is
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in
    # the limit of monodispersity

    refl5, _, _, g5, lstar5 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction,
                                               structure_type='glass',
                                               form_type='sphere')
    refl6, _, _, g6, lstar6 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration,
                                               pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')

    assert_array_almost_equal(refl5.magnitude, refl6.magnitude)
    assert_array_almost_equal(g5.magnitude, g6.magnitude)
    assert_array_almost_equal(lstar5.to('mm').magnitude, lstar6.to('mm').magnitude, decimal=4)

    # Outputs before refactoring structcol
    refl5_before = 0.2685710414987676
    refl6_before = 0.2685710407296461
    g5_before = -0.17681566915117486
    g6_before = -0.1768156684026972
    lstar5_before = 0.011593280877304636
    lstar6_before = 0.011593280876210265 # A/V: 0.011625051809100308

    assert_array_almost_equal(refl5.magnitude, refl5_before, decimal=10)
    assert_array_almost_equal(refl6.magnitude, refl6_before, decimal=10)
    assert_array_almost_equal(g5.magnitude, g5_before, decimal=10)
    assert_array_almost_equal(g6.magnitude, g6_before, decimal=10)
    assert_array_almost_equal(lstar5.to('mm').magnitude, lstar5_before, decimal=10)
    assert_array_almost_equal(lstar6.to('mm').magnitude, lstar6_before, decimal=10)

    # test that the reflectance is the same for a polydisperse monospecies
    # and a bispecies with equal types of particles
    concentration_mono = Quantity(np.array([0.,1.]), '')
    concentration_bi = Quantity(np.array([0.3,0.7]), '')
    pdi = Quantity(np.array([1e-1, 1e-1]), '')

    refl7, _, _, g7, lstar7 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius2, concentration =
                                               concentration_mono, pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')
    refl8, _, _, g8, lstar8 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius2, concentration =
                                               concentration_bi, pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')

    assert_array_almost_equal(refl7.magnitude, refl8.magnitude, decimal=10)
    assert_array_almost_equal(g7.magnitude, g8.magnitude, decimal=10)
    assert_array_almost_equal(lstar7.to('mm').magnitude, lstar8.to('mm').magnitude, decimal=10)

    # test that the reflectance is the same regardless of the order in which
    # the radii are specified
    radius3 = Quantity('90.0 nm')
    concentration3 = Quantity(np.array([0.5,0.5]), '')

    refl9, _, _, g9, lstar9 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius3, concentration =
                                               concentration3, pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')
    refl10, _, _, g10, lstar10 = sc.model.reflection(index_particle, index_matrix,
                                                  index_medium, wavelength,
                                                  radius3, volume_fraction,
                                                  radius2 = radius,
                                                  concentration =
                                                  concentration3, pdi = pdi,
                                                  structure_type='polydisperse',
                                                  form_type='polydisperse')

    assert_array_almost_equal(refl9.magnitude, refl10.magnitude, decimal=10)
    assert_array_almost_equal(g9.magnitude, g10.magnitude, decimal=10)
    assert_array_almost_equal(lstar9.to('mm').magnitude, lstar10.to('mm').magnitude, decimal=10)


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

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    refl, _, _, g, lstar = sc.model.reflection(index_particle, index_matrix,
                                            index_medium, wavelength, radius,
                                            volume_fraction,
                                            structure_type=None,
                                            form_type='sphere',
                                            thickness=thickness)
    refl2, _, _, g2, lstar2 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction,
                                               radius2=radius2,
                                               concentration=concentration,
                                               pdi=pdi, structure_type=None,
                                               form_type='polydisperse',
                                               thickness=thickness)

    assert_array_almost_equal(refl.magnitude, refl2.magnitude, decimal=9)
    assert_array_almost_equal(g.magnitude, g2.magnitude, decimal=9)
    assert_array_almost_equal(lstar.to('mm').magnitude, lstar2.to('mm').magnitude, decimal=9)

    # Outputs before refactoring structcol
    refl_before = 0.020910087489548684 # A/V:0.020791487299024698
    refl2_before = 0.020909855930303707 # A:0.020909855944662756 # A/V:0.02079125872215926
    g_before = 0.6150771860765984 # A/V:0.61562921974002 # A/V:726274264.1349005
    g2_before = 0.6150771864230516# A:0.6150771860766332 #A/V:0.6156292197400548 #A/V:726274264.1349416
    lstar_before = 0.0037892294836040373 #Before updating absorption in single scat:0.0044653875445681166 #A/V:0.0044717814146885779 #A/V:0.006279358811781641
    lstar2_before = 0.0037996137159816796 #Before updating absorption in single scat: 0.00447762476116312 #A:0.0044776247644925321 #A/V:0.0044840361567639936 #A/V:0.006296567149019748

    assert_array_almost_equal(refl.magnitude, refl_before, decimal=4)
    assert_array_almost_equal(refl2.magnitude, refl2_before, decimal=4)
    assert_array_almost_equal(g.magnitude, g_before, decimal=10)
    assert_array_almost_equal(g2.magnitude, g2_before, decimal=10)
    assert_array_almost_equal(lstar.to('mm').magnitude, lstar_before, decimal=5)
    assert_array_almost_equal(lstar2.to('mm').magnitude, lstar2_before, decimal=5)

    # test that the reflectance using only the structure factor is the same
    # using the polydisperse formula vs using Percus-Yevick in the limit of
    # monodispersity
    refl3, _, _, g3, lstar3 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction,
                                               structure_type='glass',
                                               form_type=None,
                                               thickness=thickness)
    refl4, _, _, g4, lstar4 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius2, concentration =
                                               concentration, pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type=None,
                                               thickness=thickness)

    assert_array_almost_equal(refl3.magnitude, refl4.magnitude)
    assert_array_almost_equal(g3.magnitude, g4.magnitude, decimal=4)
    assert_array_almost_equal(lstar3.to('mm').magnitude, lstar4.to('mm').magnitude, decimal=4)

    # Outputs before refactoring structcol. Changed a couple values after
    # re-implementing absorption into model.reflection() (now uses n_sample.imag
    # to calculate the absorption cross section, in the same way as montecarlo.py)
    refl3_before = 0.629949154268635 #0.6311022445010561 #changed with new absorption implementation
    refl4_before = 0.629949153206364 #0.6311022434374303 #changed with new absorption implementation
    g3_before = -0.6356307606571816 #A/V:-27901.50120849103
    g4_before = -0.6356307601051542 #A/V:-27901.50118425936
    lstar3_before = 5.7241468935761515e-05 #Before updating absorption in single scat: 8.8037552221780592e-09 #A/V:1.4399291088853016e-08
    lstar4_before = 5.72414689861482e-05 #Before updating absorption in single scat: 8.8037552299275471e-09 #A/V:1.4399291096668534e-08

    assert_array_almost_equal(refl3.magnitude, refl3_before, decimal=10)
    assert_array_almost_equal(refl4.magnitude, refl4_before, decimal=10)
    assert_array_almost_equal(g3.magnitude, g3_before, decimal=10)
    assert_array_almost_equal(g4.magnitude, g4_before, decimal=10)
    assert_array_almost_equal(lstar3.to('mm').magnitude, lstar3_before, decimal=10)
    assert_array_almost_equal(lstar4.to('mm').magnitude, lstar4_before, decimal=10)

    # test that the reflectance using both the structure and form factors is
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in
    # the limit of monodispersity
    refl5, _, _, g5, lstar5 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength,
                                               radius, volume_fraction,
                                               structure_type='glass',
                                               form_type='sphere',
                                               thickness=thickness)
    refl6, _, _, g6, lstar6 = sc.model.reflection(index_particle, index_matrix,
                                               index_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration,
                                               pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse',
                                               thickness=thickness)

    assert_array_almost_equal(refl5.magnitude, refl6.magnitude, decimal=8)
    assert_array_almost_equal(g5.magnitude, g6.magnitude, decimal=8)
    assert_array_almost_equal(lstar5.to('mm').magnitude, lstar6.to('mm').magnitude, decimal=8)

    # Outputs before refactoring structcol
    refl5_before = 0.11395667616828457 # A/V:0.11277597784758357
    refl6_before = 0.11377420192668616 #A/V:0.11259532698024184
    g5_before = -0.176272600668118 # A/V:-0.17376384100464944 #A/V:-209.15733480514967
    g6_before = -0.1762725998533963 # A/V:-0.17376384019461683 #A/V:-209.1573338372998
    lstar5_before = 0.01163694691 #Before updating absorption in single scat: A/V:0.013809880819376879 #A/V:0.013405648948885825
    lstar6_before = 0.011668837507 #Before updating absorption in single scat: A/V:0.013847726256293521 #A/V:0.013442386605693767

    assert_array_almost_equal(refl5.magnitude, refl5_before, decimal=1)
    assert_array_almost_equal(refl6.magnitude, refl6_before, decimal=1)
    assert_array_almost_equal(g5.magnitude, g5_before, decimal=10)
    assert_array_almost_equal(g6.magnitude, g6_before, decimal=10)
    assert_array_almost_equal(lstar5.to('mm').magnitude, lstar5_before, decimal=4)
    assert_array_almost_equal(lstar6.to('mm').magnitude, lstar6_before, decimal=4)

    # test that the reflectances are (almost) the same when using an
    # almost-non-absorbing vs an non-absorbing polydisperse system
    ## When there is 1 mean diameter
    index_matrix2 = sc.Index.constant(1.0+1e-20j)
    index_matrix2_real = sc.Index.constant(1.0)
    index_particle2 = sc.Index.constant(1.5+1e-20j)
    index_particle2_real = sc.Index.constant(1.5)
    radius2 = Quantity('150.0 nm')
    pdi2 = Quantity(np.array([0.33, 0.33]), '')
    refl7, _, _, g7, lstar7 = sc.model.reflection(index_particle2_real,
                                               index_matrix2_real,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius, concentration =
                                               concentration, pdi = pdi2,
                                               structure_type='polydisperse',
                                               form_type='polydisperse',
                                               thickness=thickness)
    refl8, _, _, g8, lstar8 = sc.model.reflection(index_particle2, index_matrix2,
                                               index_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius,
                                               concentration = concentration,
                                               pdi = pdi2,
                                               structure_type='polydisperse',
                                               form_type='polydisperse',
                                               thickness=thickness)
    assert_array_almost_equal(refl7.magnitude, refl8.magnitude, decimal=10)
    assert_array_almost_equal(g7.magnitude, g8.magnitude, decimal=10)
    assert_array_almost_equal(lstar7.to('mm').magnitude, lstar8.to('mm').magnitude, decimal=10)

    ## When there are 2 mean diameters
    refl9, _, _, g9, lstar9 = sc.model.reflection(index_particle2_real,
                                               index_matrix2_real,
                                               index_medium, wavelength,
                                               radius, volume_fraction, radius2
                                               = radius2, concentration =
                                               concentration, pdi = pdi2,
                                               structure_type='polydisperse',
                                               form_type='polydisperse',
                                               thickness=thickness)
    refl10, _, _, g10, lstar10 = sc.model.reflection(index_particle2,
                                                  index_matrix2, index_medium,
                                                  wavelength, radius,
                                                  volume_fraction, radius2 =
                                                  radius2, concentration =
                                                  concentration, pdi = pdi2,
                                                  structure_type='polydisperse',
                                                  form_type='polydisperse',
                                                  thickness=thickness)
    assert_array_almost_equal(refl9.magnitude, refl10.magnitude, decimal=3)
    assert_array_almost_equal(g9.magnitude, g10.magnitude, decimal=2)
    assert_array_almost_equal(lstar9.to('mm').magnitude, lstar10.to('mm').magnitude, decimal=4)
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

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    _, _, _, g, lstar = sc.model.reflection(index_particle, index_matrix,
                                         index_medium, wavelength, radius,
                                         volume_fraction,
                                         thickness=thickness1)
    _, _, _, g2, lstar2 = sc.model.reflection(index_particle, index_matrix,
                                           index_medium, wavelength,
                                           radius, volume_fraction,
                                           thickness=thickness2)

    assert_equal(g.magnitude, g2.magnitude)
    assert_equal(lstar.to('mm').magnitude, lstar2.to('mm').magnitude)

def test_reflection_throws_valueerror_for_polydisperse_core_shells():
    # test that a valueerror is raised when trying to run polydisperse core-shells
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = 0.5
    radius = Quantity(np.array([110.0, 120.0]), 'nm')
    index = [sc.Index.constant(1.5), sc.Index.constant(1.5)]
    sphere = sc.Sphere(index, radius)
    index_matrix = sc.Index.constant(1.0)
    index_medium = sc.Index.constant(1.0)
    volume_fraction2 = volume_fraction * (radius[1]**3 / radius[0]**3)
    thickness = Quantity('10.0 um')

    radius2 = Quantity('120.0 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')

    msg_regex = r"cannot handle polydispersity"

    with pytest.raises(ValueError, match=msg_regex):
        # when running polydisperse core-shells, without absorption
        refl, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius,
                                            volume_fraction2, radius2 =
                                            radius2, concentration =
                                            concentration, pdi = pdi,
                                            structure_type='polydisperse',
                                            form_type='polydisperse')
    with pytest.raises(ValueError, match=msg_regex):
        refl2, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                             wavelength, radius,
                                             volume_fraction2, radius2 =
                                             radius2, concentration =
                                             concentration, pdi = pdi,
                                             structure_type='glass',
                                             form_type='polydisperse')
    with pytest.raises(ValueError, match=msg_regex):
        refl3, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                             wavelength, radius,
                                             volume_fraction2, radius2 =
                                             radius2, concentration =
                                             concentration, pdi = pdi,
                                             structure_type=None,
                                             form_type='polydisperse')
    with pytest.raises(ValueError, match=msg_regex):
        refl4, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='sphere')
    with pytest.raises(ValueError, match=msg_regex):
        refl5, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                             wavelength, radius,
                                             volume_fraction2, radius2 =
                                             radius2, concentration =
                                             concentration, pdi = pdi,
                                             structure_type='polydisperse',
                                             form_type=None)

    with pytest.raises(ValueError, match=msg_regex):
        # when running polydisperse core-shells, with absorption
        refl6, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse',
                                            thickness=thickness)
    with pytest.raises(ValueError, match=msg_regex):
        refl7, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='glass',
                                            form_type='polydisperse',
                                            thickness=thickness)
    with pytest.raises(ValueError, match=msg_regex):
        refl8, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type=None,
                                            form_type='polydisperse',
                                            thickness=thickness)
    with pytest.raises(ValueError, match=msg_regex):
        refl9, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='sphere', thickness=thickness)
    with pytest.raises(ValueError, match=msg_regex):
        refl10, _, _, _, _ = sc.model.reflection(index, index_matrix, index_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type=None, thickness=thickness)

def test_reflection_throws_valueerror_for_polydisperse_unspecified_parameters():
    # test that a valueerror is raised when trying to run polydisperse core-shells
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity(np.array([110.0, 120.0]), 'nm')
    index_particle = [sc.Index.constant(1.5), sc.Index.constant(1.5)]
    index_matrix = sc.Index.constant(1.0)
    index_medium = sc.Index.constant(1.0)

    volume_fraction2 = Quantity(volume_fraction * (radius[1]**3 / radius[0]**3), '')

    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')

    with pytest.raises(ValueError):
        # when running polydisperse core-shells, without absorption,
        # and unspecified radius2
        refl, _, _, _, _ = sc.model.reflection(index_particle, index_matrix,
                                            index_medium, wavelength, radius,
                                            volume_fraction2, concentration =
                                            concentration, pdi = pdi,
                                            structure_type='polydisperse',
                                            form_type='polydisperse')


    index_particle = [sc.Index.constant(1.5)+0.01j,
                      sc.Index.constant(1.5)+0.01j]
    index_matrix = sc.Index.constant(1.0)+0.01j
    with pytest.raises(ValueError):
        # when running polydisperse core-shells, with absorption,
        # and unspecified radius2
        refl, _, _, _, _ = sc.model.reflection(index_particle,
                                            index_matrix, index_medium,
                                            wavelength, radius,
                                            volume_fraction2, concentration =
                                            concentration, pdi = pdi,
                                            structure_type='polydisperse',
                                            form_type='polydisperse')
