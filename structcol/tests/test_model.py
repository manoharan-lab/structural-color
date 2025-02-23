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

from .. import Quantity, size_parameter, np, mie, model
from pytest import raises
from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal
import pytest
import structcol as sc
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
            my_particle = sc.model.Particle(sc.Index.constant(index),
                                                0.15)
        with pytest.raises(DimensionalityError):
            my_particle = sc.model.Particle(sc.Index.constant(index),
                                                sc.Quantity(0.15, 'kg'))

        my_particle = sc.model.Particle(sc.Index.constant(index), size)
        assert_equal(my_particle.n(self.wavelen),
                     np.ones_like(self.wavelen)*index)

        # make sure units are correct
        assert_equal(size.to_preferred(), my_particle.size_q)

    def test_sphere_construction(self):
        radius = sc.Quantity(150, 'nm')

        # test that both index and radius must be specified:
        with pytest.raises(KeyError):
            my_sphere = sc.model.Sphere(sc.index.polystyrene)
        with pytest.raises(DimensionalityError):
            my_sphere = sc.model.Sphere(sc.index.polystyrene, 0.15)

        my_sphere = sc.model.Sphere(sc.index.polystyrene, radius)

        # test that index works as expected
        assert_equal(my_sphere.n(self.wavelen),
                     sc.index.polystyrene(self.wavelen))

        # make sure diameter is correct
        assert_equal(radius.to_preferred() * 2, my_sphere.diameter_q)
        assert_equal(radius.to_preferred(), my_sphere.radius_q)
        assert_equal(radius.to_preferred().magnitude * 2, my_sphere.diameter)
        assert_equal(radius.to_preferred().magnitude, my_sphere.radius)
        assert not my_sphere.layered

    def test_layered_sphere(self):
        index = [sc.index.vacuum, sc.index.polystyrene, sc.index.water]
        radii = sc.Quantity([0.15, 0.16, 0.18], 'um')
        radii_wrong_order = sc.Quantity([0.15, 0.17, 0.14], 'um')
        radii_too_many = sc.Quantity([0.15, 0.16, 0.17, 0.18], 'um')
        radii_too_few = sc.Quantity(0.15, 'um')

        with pytest.raises(ValueError):
            my_layered_sphere = sc.model.Sphere(index, radii_wrong_order)
        with pytest.raises(ValueError):
            my_layered_sphere = sc.model.Sphere(index, radii_too_many)
        with pytest.raises(ValueError):
            my_layered_sphere = sc.model.Sphere(index, radii_too_few)

        my_layered_sphere = sc.model.Sphere(index, radii)
        assert_equal(radii.to_preferred().magnitude, my_layered_sphere.size)
        assert_equal((radii.to_preferred() * 2).magnitude,
                     my_layered_sphere.diameter_q.magnitude)
        assert_equal((radii.to_preferred() * 2).units,
                     my_layered_sphere.diameter_q.units)
        assert my_layered_sphere.layered

        # check indexes of refraction
        for layer, n in enumerate(my_layered_sphere.n):
            assert_equal(n(self.wavelen), index[layer](self.wavelen))


class TestModel():
    """Tests for the Model class and derived classes.
    """
    ps_sphere = sc.model.Sphere(sc.index.polystyrene,
                                    sc.Quantity('0.125 um'))
    hollow_sphere = sc.model.Sphere([sc.index.vacuum,
                                         sc.index.polystyrene],
                                        sc.Quantity([125, 135], 'nm'))
    qd = np.arange(0.1, 20, 0.01)
    phi = np.array([0.15, 0.3, 0.45])
    my_units = sc.ureg.millimeter
    thickness = 0.050 * sc.ureg.millimeter

    def test_hardsphere_structure(self):
        glass = sc.model.HardSpheres(self.ps_sphere, self.phi, sc.index.water,
                                     sc.index.vacuum, self.thickness)

        # make sure thickness is correctly stored
        assert_equal(glass.thickness_q, self.thickness.to_preferred())

        # make sure original units are correctly stored
        assert glass.original_units == self.my_units

        s_ps = glass.structure_factor(self.qd)
        structure_factor = sc.structure.PercusYevick(self.phi)

        # make sure structure factor is calculated correctly
        assert_equal(s_ps.to_numpy(), structure_factor(self.qd).to_numpy())

        # make sure structure factor is the same for layered spheres as for
        # solid spheres
        glass = sc.model.HardSpheres(self.hollow_sphere, self.phi,
                                     sc.index.water, sc.index.vacuum,
                                     self.thickness)
        s_hollow = glass.structure_factor(self.qd)
        assert_equal(s_hollow.to_numpy(), structure_factor(self.qd).to_numpy())


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
    rpar, rperp = model.fresnel_reflection(n1, n2, Quantity('0.0 deg'))
    assert_almost_equal(rpar, 0.04)
    assert_almost_equal(rperp, 0.04)
    rpar, rperp = model.fresnel_reflection(n1, n2, Quantity('45.0 deg'))
    assert_almost_equal(rpar, 0.00846646)
    assert_almost_equal(rperp, 0.0920134)

    # test total internal reflection
    rpar, rperp = model.fresnel_reflection(n2, n1, Quantity('45.0 deg'))
    assert_equal(rpar, 1.0)
    assert_equal(rperp, 1.0)

    # test no total internal reflection (just below critical angle)
    rpar, rperp = model.fresnel_reflection(n2, n1, Quantity('41.810 deg'))
    assert_almost_equal(rpar, 0.972175, decimal=6)
    assert_almost_equal(rperp, 0.987536, decimal=6)

    # test vectorized computation
    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # check for value error
    raises(ValueError, model.fresnel_reflection, n2, n1, angles)
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
    volume_fraction = Quantity(0.5, '')
    n_particle = 1.0
    n_matrix = 1.0
    n_medium = 2.0
    theta_min = Quantity(np.pi/2, 'deg')

    # set theta_max to be slightly smaller than the theta corresponding to
    # total internal reflection (calculated manually to be 2.61799388)
    theta_max = Quantity(2.617, 'deg')
    refl1, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                         wavelength, radius, volume_fraction,
                                         theta_min=theta_min,
                                         theta_max=theta_max,
                                         structure_type=None)
    # try a different range of thetas (but keeping theta_max < total internal
    # reflection angle)
    theta_max = Quantity(2., 'deg')
    refl2, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                         wavelength, radius, volume_fraction,
                                         theta_min=theta_min,
                                         theta_max=theta_max,
                                         structure_type=None)

    # the reflection should be zero plus the fresnel reflection term
    n_sample = sc.index.n_eff(n_particle, n_matrix, volume_fraction)
    r_fresnel = model.fresnel_reflection(n_medium, n_sample, incident_angle)
    r_fresnel_avg = (r_fresnel[0] + r_fresnel[1]) / 2
    assert_almost_equal(refl1.magnitude, r_fresnel_avg)
    assert_almost_equal(refl2.magnitude, r_fresnel_avg)
    assert_almost_equal(refl1.magnitude, refl2.magnitude)


def test_differential_cross_section():
    # Test that the differential cross sections for non-core-shell particles
    # and core-shells are the same at low volume fractions, assuming that the
    # particle diameter of the non-core-shells is the same as the core
    # diameter in the core-shells

    #n_sample = Quantity(1.5, '')
    n_matrix = 1.0
    wavelen = Quantity('500.0 nm')
    angles = Quantity(np.linspace(np.pi/2, np.pi, 200), 'rad')

    # Differential cross section for non-core-shells
    radius = Quantity('100.0 nm')
    n_particle = 1.5
    volume_fraction = Quantity(0.0001, '')              # IS VF TOO LOW?
    n_sample = sc.index.n_eff(n_particle, n_matrix, volume_fraction)
    m = n_particle/n_sample
    x = size_parameter(wavelen, n_sample, radius)
    diff = model.differential_cross_section(m, x, angles, volume_fraction)

    # Differential cross section for core-shells. Core is equal to
    # non-core-shell particle, and shell is made of vacuum
    radius_cs = Quantity(np.array([100.0, 110.0]), 'nm')
    n_particle_cs = np.array([1.5, 1.0])

    volume_fraction_shell = volume_fraction * (radius_cs[1]**3 / radius_cs[0]**3-1)
    volume_fraction_cs = Quantity(np.array([volume_fraction.magnitude,
                                            volume_fraction_shell.magnitude]), '')

    n_sample_cs = sc.index.n_eff(n_particle_cs, n_matrix, volume_fraction_cs)
    m_cs = (n_particle_cs/n_sample_cs).flatten()
    x_cs = size_parameter(wavelen, n_sample_cs, radius_cs).flatten()
    diff_cs = model.differential_cross_section(m_cs, x_cs, angles,
                                               np.sum(volume_fraction_cs))

    assert_array_almost_equal(diff[0], diff_cs[0], decimal=5)
    assert_array_almost_equal(diff[1], diff_cs[1], decimal=5)


def test_reflection_core_shell():
    # Test reflection, anisotropy factor, and transport length calculations to
    # make sure the values for refl, g, and lstar remain the same after adding
    # core-shell capability into the model
    wavelength = Quantity(500.0, 'nm')
    thickness = Quantity(15.0, 'um')

    # Non core-shell particles with Maxwell-Garnett effective index
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120.0 nm')
    n_particle = 1.5
    n_matrix = 1.0
    n_medium = n_matrix

    refl1, _, _, g1, lstar1 = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius,
                                            volume_fraction,
                                            thickness=Quantity('15000.0 nm'),
                                            theta_min=Quantity('90.0 deg'),
                                            small_angle=Quantity('5.0 deg'),
                                            maxwell_garnett=True)

    # Non core-shell particles with Bruggeman effective index
    volume_fraction2 = Quantity(0.00001, '')
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius,
                                               volume_fraction2,
                                               thickness=Quantity('15000.0 nm'),
                                               theta_min=Quantity('90.0 deg'),
                                               small_angle=Quantity('5.0 deg'),
                                               maxwell_garnett=False)

    # Core-shell particles of core diameter equal to non core shell particles,
    # and shell index of air. With Bruggeman effective index
    n_particle3 = np.array([1.5, 1.0])
    radius3 = Quantity(np.array([120.0, 130.0]), 'nm')
    volume_fraction3 = volume_fraction2 * (radius3[1]**3 / radius3[0]**3)

    refl3, _, _, g3, lstar3 = model.reflection(n_particle3, n_matrix, n_medium,
                                               wavelength, radius3,
                                               volume_fraction3,
                                               thickness=Quantity('15000.0 nm'),
                                               small_angle=Quantity('5.0 deg'),
                                               theta_min=Quantity('90.0 deg'),
                                               maxwell_garnett=False)

    # Outputs for refl, g, and lstar before adding core-shell capability
    refl = Quantity(0.20772170840902376, '')
    g = Quantity(-0.18931942267032678, '')
    lstar = Quantity(10810.088573316663, 'nm')

    # Compare old outputs (before adding core-shell capability) and new outputs
    # for a non-core-shell using Maxwell-Garnett
    assert_array_almost_equal(refl1.magnitude, refl.magnitude)
    assert_array_almost_equal(g1.magnitude, g.magnitude)
    assert_array_almost_equal(lstar1.magnitude, lstar.magnitude)

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
    n_particle4 = 1.5+0.001j
    refl4 = model.reflection(n_particle4, n_matrix, n_medium, wavelength,
                             radius4, volume_fraction, thickness=thickness)[0]

    # Absorbing core-shell
    n_particle5 = np.array([1.5+0.001j, 1.5+0.001j])
    radius5 = Quantity(np.array([110.0, 120.0]), 'nm')
    refl5 = model.reflection(n_particle5, n_matrix, n_medium, wavelength,
                             radius5, volume_fraction, thickness=thickness)[0]

    assert_array_almost_equal(refl4.magnitude, refl5.magnitude, decimal=3)

    # Same as previous test but with absorbing matrix
    # Non-core-shell
    radius6 = Quantity('120.0 nm')
    n_particle6 = 1.5+0.001j
    n_matrix6 = 1.0+0.001j
    refl6 = model.reflection(n_particle6, n_matrix6, n_medium, wavelength,
                             radius6, volume_fraction, thickness=thickness)[0]

    # Core-shell
    n_particle7 = np.array([1.5+0.001j, 1.5+0.001j])
    radius7 = Quantity(np.array([110.0, 120.0]), 'nm')
    n_matrix7 = 1.0+0.001j
    refl7 = model.reflection(n_particle7, n_matrix7, n_medium, wavelength,
                             radius7, volume_fraction, thickness=thickness)[0]

    assert_array_almost_equal(refl6.magnitude, refl7.magnitude, decimal=3)


def test_reflection_absorbing_particle():
    # test that the reflections with a real n_particle and with a complex
    # n_particle with a 0 imaginary component are the same
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120.0 nm')
    n_matrix = 1.0
    n_medium = n_matrix
    n_particle_real = 1.5
    n_particle_imag = 1.5 + 0j

    # With Maxwell-Garnett
    refl_mg1, _, _, g_mg1, lstar_mg1 = model.reflection(n_particle_real, n_matrix,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = model.reflection(n_particle_imag, n_matrix,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=True)

    assert_array_almost_equal(refl_mg1.magnitude, refl_mg2.magnitude)
    assert_array_almost_equal(g_mg1.magnitude, g_mg2.magnitude)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg2.magnitude)

    # Outputs before refactoring structcol
    refl_mg1_before = 0.2963964709617333
    refl_mg2_before = 0.29639647096173255
    g_mg1_before = -0.18774057969370997
    g_mg2_before = -0.18774057969370903
    lstar_mg1_before = 10810.069633192961
    # lstar_mg2_before = 10810.069633193001
    # lstar_mg2 and lstar_mg1 are now equal, so we don't need to compare to
    # lstar_mg2_before

    assert_array_almost_equal(refl_mg1.magnitude, refl_mg1_before, decimal=9)
    assert_array_almost_equal(refl_mg2.magnitude, refl_mg2_before, decimal=9)
    assert_array_almost_equal(g_mg1.magnitude, g_mg1_before, decimal=10)
    assert_array_almost_equal(g_mg2.magnitude, g_mg2_before, decimal=10)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg1_before, decimal=10)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg2.magnitude, decimal=10)

    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = model.reflection(n_particle_real, n_matrix,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = model.reflection(n_particle_imag, n_matrix,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=False)

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg2.magnitude)
    assert_array_almost_equal(g_bg1.magnitude, g_bg2.magnitude)
    assert_array_almost_equal(lstar_bg1.magnitude, lstar_bg2.magnitude)

    # Outputs before refactoring structcol
    refl_bg1_before = 0.2685710414987676
    refl_bg2_before = 0.2685710414987676
    g_bg1_before = -0.17681566915117486
    g_bg2_before = -0.17681566915117486
    lstar_bg1_before = 11593.280877304634
    lstar_bg2_before = 11593.280877304634

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg1_before, decimal=10)
    assert_array_almost_equal(refl_bg2.magnitude, refl_bg2_before, decimal=10)
    assert_array_almost_equal(g_bg1.magnitude, g_bg1_before, decimal=10)
    assert_array_almost_equal(g_bg2.magnitude, g_bg2_before, decimal=10)
    assert_array_almost_equal(lstar_bg1.magnitude, lstar_bg1_before, decimal=10)
    assert_array_almost_equal(lstar_bg2.magnitude, lstar_bg2_before, decimal=10)

    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    n_particle_imag2 = 1.5+1e-8j
    thickness = Quantity('100.0 um')

    # With Bruggeman
    refl_bg3, _, _, g_bg3, lstar_bg3 = model.reflection(n_particle_imag2, n_matrix,
                                                        n_medium, wavelength,
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
    n_particle = np.array([1.5, 1.0])
    radius = Quantity(np.array([120.0, 130.0]), 'nm')
    volume_fraction = Quantity(0.01, '')
    n_matrix = 1.0
    n_medium = n_matrix

    _, _, _, g1, _= model.reflection(n_particle, n_matrix, n_medium,
                                     wavelength, radius, volume_fraction,
                                     small_angle=Quantity('0.01 deg'),
                                     num_angles=1000, structure_type=None)

    # calculate g using calc_g in pymie
    vf_array = np.empty(len(np.atleast_1d(radius)))
    #r_array = np.array([0] + np.atleast_1d(radius).tolist())
    r_array = np.array([0] + np.atleast_1d(radius.magnitude).tolist())
    for r in np.arange(len(r_array)-1):
        vf_array[r] = ((r_array[r+1]**3-r_array[r]**3) / (r_array[-1]**3) *
                       volume_fraction.magnitude)

    n_sample = sc.index.n_eff(n_particle, n_matrix, vf_array)
    m = (n_particle/n_sample).flatten()
    x = size_parameter(wavelength, n_sample, radius).flatten()
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
    volume_fraction = Quantity(0.0000001, '')
    radius = Quantity('120.0 nm')
    n_matrix = 1.0
    n_medium = n_matrix
    n_particle = 1.5
    _, _, _, _, lstar_model = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction,
                                            maxwell_garnett=False)

    # transport length from Mie theory
    n_sample = sc.index.n_eff(n_particle, n_matrix, volume_fraction)
    m = n_particle/n_sample
    x = size_parameter(wavelength, n_sample, radius)
    g = mie.calc_g(m,x)

    number_density = model._number_density(volume_fraction, radius)
    cscat = mie.calc_cross_sections(m, x, wavelength)[0]

    lstar_mie = 1 / (number_density * cscat * (1-g))

    assert_array_almost_equal(lstar_model.to('m').magnitude, lstar_mie.to('m').magnitude, decimal=4)

def test_reflection_absorbing_matrix():
    # test that the reflections with a real n_matrix and with a complex
    # n_matrix with a 0 imaginary component are the same
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120.0 nm')
    n_matrix_real = 1.0
    n_matrix_imag = 1.0 + 0j
    n_medium = 1.0
    n_particle = 1.5

    # With Maxwell-Garnett
    refl_mg1, _, _, g_mg1, lstar_mg1 = model.reflection(n_particle, n_matrix_real,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = model.reflection(n_particle, n_matrix_imag,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=True)

    assert_array_almost_equal(refl_mg1.magnitude, refl_mg2.magnitude)
    assert_array_almost_equal(g_mg1.magnitude, g_mg2.magnitude)
    assert_array_almost_equal(lstar_mg1.magnitude, lstar_mg2.magnitude)

    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = model.reflection(n_particle, n_matrix_real,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = model.reflection(n_particle, n_matrix_imag,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        maxwell_garnett=False)

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg2.magnitude)
    assert_array_almost_equal(g_bg1.magnitude, g_bg2.magnitude)
    assert_array_almost_equal(lstar_bg1.magnitude, lstar_bg2.magnitude)

    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    thickness = Quantity('100.0 um')
    n_matrix_imag2 = 1.0 + 1e-8j

    # With Bruggeman
    refl_bg3, _, _, g_bg3, lstar_bg3 = model.reflection(n_particle, n_matrix_imag2,
                                                        n_medium, wavelength,
                                                        radius, volume_fraction,
                                                        thickness=thickness,
                                                        maxwell_garnett=False)

    assert_array_almost_equal(refl_bg1.magnitude, refl_bg3.magnitude, decimal=3)
    assert_array_almost_equal(g_bg1.magnitude, g_bg3.magnitude, decimal=3)
    assert_array_almost_equal(lstar_bg1.to('mm').magnitude, lstar_bg3.to('mm').magnitude, decimal=4)


def test_reflection_polydispersity():
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120.0 nm')
    n_matrix = 1.0
    n_medium = 1.0
    n_particle = 1.5
    radius2 = Quantity('120.0 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    refl, _, _, g, lstar = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction,
                                            structure_type=None,
                                            form_type='sphere')
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix,
                                                  n_medium, wavelength, radius,
                                                  volume_fraction,
                                                  radius2 = radius2,
                                                  concentration = concentration,
                                                  pdi = pdi, structure_type=None,
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

    refl3, _, _, g3, lstar3 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type=None)

    refl4, _, _, g4, lstar4 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration,
                                               pdi = pdi,
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

    refl5, _, _, g5, lstar5 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type='sphere')
    refl6, _, _, g6, lstar6 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius,
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

    refl7, _, _, g7, lstar7 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius, volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration_mono,
                                               pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')
    refl8, _, _, g8, lstar8 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration_bi,
                                               pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')

    assert_array_almost_equal(refl7.magnitude, refl8.magnitude, decimal=10)
    assert_array_almost_equal(g7.magnitude, g8.magnitude, decimal=10)
    assert_array_almost_equal(lstar7.to('mm').magnitude, lstar8.to('mm').magnitude, decimal=10)

    # test that the reflectance is the same regardless of the order in which
    # the radii are specified
    radius3 = Quantity('90.0 nm')
    concentration3 = Quantity(np.array([0.5,0.5]), '')

    refl9, _, _, g9, lstar9 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius, volume_fraction,
                                               radius2 = radius3,
                                               concentration = concentration3,
                                               pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')
    refl10, _, _, g10, lstar10 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius3,
                                               volume_fraction,
                                               radius2 = radius,
                                               concentration = concentration3,
                                               pdi = pdi,
                                               structure_type='polydisperse',
                                               form_type='polydisperse')

    assert_array_almost_equal(refl9.magnitude, refl10.magnitude, decimal=10)
    assert_array_almost_equal(g9.magnitude, g10.magnitude, decimal=10)
    assert_array_almost_equal(lstar9.to('mm').magnitude, lstar10.to('mm').magnitude, decimal=10)


def test_reflection_polydispersity_with_absorption():
    wavelength = Quantity(500.0, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120.0 nm')
    n_matrix = 1.0+0.0003j
    n_medium = 1.0
    n_particle = 1.5+0.0005j
    radius2 = Quantity('120.0 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit
    thickness = Quantity('10.0 um')

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    refl, _, _, g, lstar = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction,
                                            structure_type=None,
                                            form_type='sphere',
                                            thickness=thickness)
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius,
                                               volume_fraction, radius2=radius2,
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
    refl3, _, _, g3, lstar3 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type=None,
                                               thickness=thickness)
    refl4, _, _, g4, lstar4 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration,
                                               pdi = pdi,
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
    refl5, _, _, g5, lstar5 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type='sphere',
                                               thickness=thickness)
    refl6, _, _, g6, lstar6 = model.reflection(n_particle, n_matrix,
                                               n_medium, wavelength, radius,
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
    n_matrix2 = 1.0+1e-20j
    n_particle2 = 1.5+1e-20j
    radius2 = Quantity('150.0 nm')
    pdi2 = Quantity(np.array([0.33, 0.33]), '')
    refl7, _, _, g7, lstar7 = model.reflection(n_particle2.real, n_matrix2.real,
                                               n_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius,
                                               concentration = concentration,
                                               pdi = pdi2,
                                               structure_type='polydisperse',
                                               form_type='polydisperse',
                                               thickness=thickness)
    refl8, _, _, g8, lstar8 = model.reflection(n_particle2, n_matrix2,
                                               n_medium, wavelength, radius,
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
    refl9, _, _, g9, lstar9 = model.reflection(n_particle2.real, n_matrix2.real,
                                               n_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration,
                                               pdi = pdi2,
                                               structure_type='polydisperse',
                                               form_type='polydisperse',
                                               thickness=thickness)
    refl10, _, _, g10, lstar10 = model.reflection(n_particle2, n_matrix2,
                                               n_medium, wavelength, radius,
                                               volume_fraction,
                                               radius2 = radius2,
                                               concentration = concentration,
                                               pdi = pdi2,
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
    volume_fraction = Quantity(0.55, '')
    radius = Quantity('100.0 nm')
    n_matrix = 1.0+0.0004j
    n_medium = 1.0
    n_particle = 1.5+0.0006j
    thickness1 = Quantity('10.0 um')
    thickness2 = Quantity('100.0 um')

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    _, _, _, g, lstar = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction,
                                            thickness=thickness1)
    _, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix, n_medium,
                                               wavelength, radius,
                                               volume_fraction,
                                               thickness=thickness2)

    assert_equal(g.magnitude, g2.magnitude)
    assert_equal(lstar.to('mm').magnitude, lstar2.to('mm').magnitude)

def test_reflection_throws_valueerror_for_polydisperse_core_shells():
    # test that a valueerror is raised when trying to run polydisperse core-shells
    with pytest.raises(ValueError):
        wavelength = Quantity(500.0, 'nm')
        volume_fraction = Quantity(0.5, '')
        radius = Quantity(np.array([110.0, 120.0]), 'nm')
        n_matrix = 1.0
        n_medium = 1.0
        n_particle = np.array([1.5,1.5])
        volume_fraction2 = Quantity(volume_fraction * (radius[1]**3 / radius[0]**3), '')
        thickness = Quantity('10.0 um')

        radius2 = Quantity('120.0 nm')
        concentration = Quantity(np.array([0.9,0.1]), '')
        pdi = Quantity(np.array([1e-7, 1e-7]), '')

        # when running polydisperse core-shells, without absorption
        refl, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse')
        refl2, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='glass',
                                            form_type='polydisperse')
        refl3, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type=None,
                                            form_type='polydisperse')
        refl4, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='sphere')
        refl5, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type=None)

        # when running polydisperse core-shells, with absorption
        refl6, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse',
                                            thickness=thickness)
        refl7, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='glass',
                                            form_type='polydisperse',
                                            thickness=thickness)
        refl8, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type=None,
                                            form_type='polydisperse',
                                            thickness=thickness)
        refl9, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='sphere', thickness=thickness)
        refl10, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            radius2 = radius2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type=None, thickness=thickness)

def test_reflection_throws_valueerror_for_polydisperse_unspecified_parameters():
    # test that a valueerror is raised when trying to run polydisperse core-shells
    with pytest.raises(ValueError):
        wavelength = Quantity(500.0, 'nm')
        volume_fraction = Quantity(0.5, '')
        radius = Quantity(np.array([110.0, 120.0]), 'nm')
        n_matrix = 1.0
        n_medium = 1.0
        n_particle = np.array([1.5,1.5])
        volume_fraction2 = Quantity(volume_fraction * (radius[1]**3 / radius[0]**3), '')

        concentration = Quantity(np.array([0.9,0.1]), '')
        pdi = Quantity(np.array([1e-7, 1e-7]), '')

        # when running polydisperse core-shells, without absorption,
        # and unspecified radius2
        refl, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse')

        # when running polydisperse core-shells, with absorption,
        # and unspecified radius2
        refl, _, _, _, _ = model.reflection(n_particle+0.01j, n_matrix+0.01j, n_medium,
                                            wavelength, radius, volume_fraction2,
                                            concentration = concentration,
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse')
