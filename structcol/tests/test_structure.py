# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang, Annie Stephenson
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
Tests for the structure module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
"""

import structcol as sc
from structcol import montecarlo as mc
from structcol import detector as det
from .. import Quantity, np
from .. import size_parameter
import xarray as xr

from numpy.testing import assert_equal, assert_almost_equal
import pytest
from pint.errors import UnitStrippedWarning, DimensionalityError

class TestParticle():
    """Tests for the Particle class and derived classes
    """
    wavelen = sc.Quantity(np.linspace(400, 800, 100), 'nm')
    def test_particle_construction(self):
        index = 1.445
        size = sc.Quantity(150, 'nm')

        # particle construction without units or with wrong units should fail
        with pytest.raises(DimensionalityError):
            my_particle = sc.structure.Particle(sc.Index.constant(index),
                                                0.15)
        with pytest.raises(DimensionalityError):
            my_particle = sc.structure.Particle(sc.Index.constant(index),
                                                sc.Quantity(0.15, 'kg'))

        my_particle = sc.structure.Particle(sc.Index.constant(index), size)
        assert_equal(my_particle.n(self.wavelen),
                     np.ones_like(self.wavelen)*index)

        # make sure units are correct
        assert_equal(size.to_preferred(), my_particle.size_q)

    def test_sphere_construction(self):
        radius = sc.Quantity(150, 'nm')

        # test that both index and radius must be specified:
        with pytest.raises(KeyError):
            my_sphere = sc.structure.Sphere(sc.index.polystyrene)
        with pytest.raises(DimensionalityError):
            my_sphere = sc.structure.Sphere(sc.index.polystyrene, 0.15)

        my_sphere = sc.structure.Sphere(sc.index.polystyrene, radius)

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
            my_layered_sphere = sc.structure.Sphere(index, radii_wrong_order)
        with pytest.raises(ValueError):
            my_layered_sphere = sc.structure.Sphere(index, radii_too_many)
        with pytest.raises(ValueError):
            my_layered_sphere = sc.structure.Sphere(index, radii_too_few)

        my_layered_sphere = sc.structure.Sphere(index, radii)
        assert_equal(radii.to_preferred().magnitude, my_layered_sphere.size)
        assert_equal((radii.to_preferred() * 2).magnitude,
                     my_layered_sphere.diameter_q.magnitude)
        assert_equal((radii.to_preferred() * 2).units,
                     my_layered_sphere.diameter_q.units)
        assert my_layered_sphere.layered

        # check indexes of refraction
        for layer, n in enumerate(my_layered_sphere.n):
            assert_equal(n(self.wavelen), index[layer](self.wavelen))

class TestStructureFactor():
    """Tests for the StructureFactor class and derived classes.
    """
    qd = np.arange(0.1, 20, 0.01)
    phi = np.array([0.15, 0.3, 0.45])

    def test_structure_factor_base_class(self):
        structure_factor = sc.structure.StructureFactor()
        with pytest.raises(NotImplementedError):
            structure_factor.calculate(self.qd)
        # test that __call__ method works
        with pytest.raises(NotImplementedError):
            structure_factor(self.qd)

    def test_percus_yevick(self):
        """Tests the object version of the Percus-Yevick structure factor
        """

        # test how function handles dimensionless arguments
        with pytest.warns(UnitStrippedWarning):
            structure_factor = sc.structure.PercusYevick(Quantity('0.4'))
        # specifying quantities is not allowed when calculating
        with pytest.raises(AttributeError):
            s = structure_factor(Quantity('0.1'))
        # but scalars should work
        structure_factor = sc.structure.PercusYevick(0.4)
        s = structure_factor(0.1)

        # now calculate for arrays of phi and qd
        structure_factor = sc.structure.PercusYevick(self.phi)
        s = structure_factor(self.qd)

        # ensure that we are broadcasting correctly
        assert s.shape == (self.qd.shape[0], self.phi.shape[0])

        # make sure that calculation works with qd specified as DataArray
        qd = xr.DataArray(self.qd, coords = {"qd": self.qd})
        s = structure_factor(qd)
        assert s.shape == (self.qd.shape[0], self.phi.shape[0])

        # compare to values from Cipelletti, Trappe, and Pine, "Scattering
        # Techniques", in "Fluids, Colloids and Soft Materials: An Introduction
        # to Soft Matter Physics", 2016 (plot on page 137) (I extracted values
        # from the plot using a digitizer
        # (http://arohatgi.info/WebPlotDigitizer/app/). They are probably good
        # to only one decimal place, so this is a fairly crude test.)

        # max values of S(qd) at different phi
        max_vals = s.max(dim="qd")
        # values of qd at which S(qd) has max
        max_qds = s.idxmax(dim="qd")
        assert_almost_equal(max_vals[0], 1.17, decimal=1)
        assert_almost_equal(max_vals[1], 1.52, decimal=1)
        assert_almost_equal(max_vals[2], 2.52, decimal=1)
        assert_almost_equal(max_qds[0], 6.00, decimal=1)
        assert_almost_equal(max_qds[1], 6.37, decimal=1)
        assert_almost_equal(max_qds[2], 6.84, decimal=1)

        # compare to values from before refactoring
        qd = np.linspace(0.1, 20, 20)
        phi = 0.6

        structure_factor = sc.structure.PercusYevick(phi)
        s = structure_factor(qd)

        s_expected = [0.005292811521054822, 0.005782178319680089,
                      0.007377121990596123, 0.01122581884358338,
                      0.02133597659701994, 0.056320860137596386,
                      0.2842109080529606, 6.548734687715668, 0.655686067757609,
                      0.34078069018365464, 0.37117597252527373,
                      0.6743611932043811, 1.6508200842297407,
                      1.5730675915964172, 0.8423480630021749,
                      0.6431921078170443, 0.706599972588693,
                      1.0009844486818578, 1.3548268525788216,
                      1.2077694027197434]

        assert_almost_equal(s.squeeze(), s_expected)

        # test that structure factor converges to low-qd approximation at small
        # qd (but not so small that solution becomes numerically unstable).
        # Use a variety of volume fractions to test
        phi = np.linspace(0.05, 0.6, 10)
        structure_factor = sc.structure.PercusYevick(phi, qd_cutoff=0.01)
        s = structure_factor(qd)

        # generate structure factor with higher cutoff, so that below
        # qd_cutoff, the structure factor should be using the approximate
        # solution to the direct correlation function
        structure_factor_approx = sc.structure.PercusYevick(phi, qd_cutoff=0.2)
        s_approx = structure_factor_approx(qd)
        assert_almost_equal(s.sel(qd=0.1).to_numpy(),
                            s_approx.sel(qd=0.1).to_numpy())

    def test_paracrystal(self):
        """Tests the object version of the paracrystalline structure factor
        """
        volfrac = np.arange(0.1, 0.7, 0.1)
        sigma = np.arange(0, 0.5, 0.05)

        structure_factor = sc.structure.Paracrystal(volfrac, sigma=sigma)
        qd = np.linspace(0.05, 0.6, 10)
        s = structure_factor(qd)
        # TODO add tests to check that we are getting the right values

    def test_percus_yevick_polydisperse(self):
        """Tests the object version of the polydisperse structure factor.
        """
        # first test that the analytical structure factor for polydisperse
        # systems matches Percus-Yevick in the monodisperse limit
        qd = 5.0
        phi = 0.5

        # Percus-Yevick monodisperse
        monodisperse_structure_factor = sc.structure.PercusYevick(phi)
        s_py = monodisperse_structure_factor(qd)

        # Polydisperse Percus-Yevick
        d = Quantity('100.0 nm')
        c = 1.0
        pdi = 1e-5
        q2 = qd / d

        polydisperse_structure_factor = sc.structure.Polydisperse(phi, d, c,
                                                                  pdi)
        s_poly = polydisperse_structure_factor(q2)

        assert_almost_equal(s_py, s_poly.magnitude)

        # test that structure factor matches the calculations in Figure 1 of
        # Ginoza and Yasutomi, Journal of the Physical Society of Japan 1999.

        # Figure 1 curve A peaks and valleys, from digitized figure 1 (includes
        # the spurious peaks)
        qd = np.array([0.2909090909090909, 3.2, 5.03030303030303,
                       5.721212121212122, 6.484848484848484, 7.430303030303031,
                       8.824242424242424, 8.993939393939394, 9.078787878787878,
                       9.163636363636364, 9.515151515151516,
                       12.533333333333333, 15.078787878787878,
                       15.454545454545455, 15.89090909090909,
                       18.921212121212122])
        s_expected = np.array([0.09974380871050384, 0.20905209222886423,
                               0.7405636208368915, 1.2064901793339027,
                               1.5125533731853118, 1.2092228864218617,
                               0.869000853970965, 0.9209222886421862,
                               0.9960717335610589, 0.9209222886421862,
                               0.8321093082835184, 1.1081127241673783,
                               0.9468830059777967, 0.9947053800170794,
                               0.9359521776259607, 1.042527754056362])

        # results shouldn't depend on d but we need to input anyway
        d = Quantity('100.0 nm')
        c = 1.0
        phi = 0.3
        # in the paper, D_sigma is the square of the relative deviation, so
        # D_sigma=1e-4 corresponds to pdi=1e-2
        pdi = 1e-2
        q2 = qd / d

        polydisperse_structure_factor = sc.structure.Polydisperse(phi, d, c,
                                                                  pdi)
        s_poly = polydisperse_structure_factor(q2)
        # since plot is digitized, we expect agreement only to 1 decimal place
        assert_almost_equal(s_poly.magnitude, s_expected, decimal=1)

    def test_structure_factor_percus_yevick_core_shell(self):
        """Test that the structure factor is the same for core-shell particles
        and non-core-shell particles at low volume fraction (assuming the core
        diameter is the same as the particle diameter for the non-core-shell
        case)

        """
        wavelen = Quantity('400.0 nm')
        angles = Quantity(np.pi, 'rad')
        n_matrix = 1.0

        # Structure factor for non-core-shell particles
        radius = Quantity('100.0 nm')
        n_particle = 1.5
        volume_fraction = Quantity(0.0001, '')         # IS VF TOO LOW?
        n_sample = sc.index.n_eff(n_particle, n_matrix, volume_fraction)
        x = size_parameter(wavelen, n_sample, radius)
        qd = 4*x*np.sin(angles/2)
        with pytest.warns(UnitStrippedWarning):
            structure_factor = sc.structure.PercusYevick(volume_fraction)
        #s = structure.factor_py(qd, volume_fraction)
        s = structure_factor(qd)

        # Structure factor for core-shell particles with core size equal to
        # radius of non-core-shell particle
        radius_cs = Quantity(np.array([100.0, 105.0]), 'nm')
        n_particle_cs = np.array([1.5, 1.0])
        volume_fraction_shell = volume_fraction * (radius_cs[1]**3 / radius_cs[0]**3 -1)
        volume_fraction_cs = Quantity(np.array([volume_fraction.magnitude, volume_fraction_shell.magnitude]), '')

        n_sample_cs = sc.index.n_eff(n_particle_cs, n_matrix, volume_fraction_cs)
        x_cs = size_parameter(wavelen, n_sample_cs, radius_cs[1]).flatten()
        qd_cs = 4*x_cs*np.sin(angles/2)
        with pytest.warns(UnitStrippedWarning):
            structure_factor_cs = sc.structure.PercusYevick(
                                            np.sum(volume_fraction_cs))
            s_cs = structure_factor_cs(qd_cs)
        #s_cs = structure.factor_py(qd_cs, np.sum(volume_fraction_cs))

        assert_almost_equal(s, s_cs, decimal=5)

    def test_structure_factor_interpolated(self):
        qd = np.array([1, 2])
        qd_data = np.array([0.5, 2.5])
        s_data = np.array([1, 1])
        structure_factor = sc.structure.Interpolated(s_data, qd_data)
        s = structure_factor(qd)
        assert_equal(s[0], 1)

@pytest.mark.slow
def test_structure_factor_data_reflectances():
    """
    Tests that the reflectance (calculated from single-scattering model and
    Monte Carlo model) match expected values for a structure factor that comes
    from "data" (in this case, data generated from the PY function). The
    parameters, setup, and expected values come from the
    structure_factor_data_tutorial notebook.

    """

    wavelengths = Quantity(np.arange(400, 800, 20), 'nm')
    radius = Quantity('0.5 um')
    volume_fraction = 0.5
    n_particle = sc.index.fused_silica(wavelengths)
    n_matrix = sc.index.vacuum(wavelengths)
    n_medium = sc.index.vacuum(wavelengths)
    thickness = Quantity('50 um')

    # generate structure factor "data" from Percus-Yevick model
    qd_data = np.arange(0.001, 75, 0.1)
    structure_factor = sc.structure.PercusYevick(volume_fraction)
    s_data = structure_factor(qd_data)

    # make interpolation function
    qd = np.arange(0, 70, 0.1)
    structure_factor_interpolated = sc.structure.Interpolated(s_data, qd_data)
    s = structure_factor_interpolated(qd)

    # calculate reflectance from single-scattering model
    reflectance = np.zeros(len(wavelengths))
    for i in range(len(wavelengths)):
        reflectance[i],_,_,_,_ = \
            sc.model.reflection(n_particle[i],
                                n_matrix[i],
                                n_medium[i],
                                wavelengths[i],
                                radius,
                                volume_fraction,
                                thickness=thickness,
                                structure_type='data',
                                structure_s_data=s_data,
                                structure_qd_data=qd_data)

    reflectance_expected = [0.02776632370015263, 0.025862410582306178,
                            0.02804132579281817, 0.029567824927529483,
                            0.02883201740020668, 0.02489322299891402,
                            0.025529207080894668, 0.0313209850678431,
                            0.034468516429692606, 0.031091754929536804,
                            0.026184798431381887, 0.025578270586409414,
                            0.029741968447070125, 0.03553369792759859,
                            0.039927975180883306, 0.04145348535377456,
                            0.040166711576661615, 0.037305165165199786,
                            0.03432092904706069, 0.03218808662896649]

    assert_almost_equal(reflectance, reflectance_expected)

    # calculate reflectance from Monte Carlo model
    seed = 1
    rng = np.random.RandomState([seed])
    ntrajectories = 500
    nevents = 500
    wavelengths = sc.Quantity(np.arange(400, 800, 20), 'nm')
    radius = sc.Quantity('0.5 um')
    volume_fraction = 0.5
    n_particle = sc.index.fused_silica(wavelengths)
    n_matrix = sc.index.vacuum(wavelengths)
    n_medium = sc.index.vacuum(wavelengths)
    boundary = 'film'
    thickness = sc.Quantity('50 um')

    qd_data = np.arange(0.001, 75, 0.1)
    structure_factor = sc.structure.PercusYevick(volume_fraction)
    s_data = structure_factor(qd_data)

    reflectance = np.zeros(wavelengths.size)
    for i in range(wavelengths.size):
        n_sample = sc.index.n_eff(n_particle[i], n_matrix[i], volume_fraction)

        p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle[i], n_sample,
                                          volume_fraction,
                                          wavelengths[i],
                                          structure_type = 'data',
                                          structure_s_data = s_data,
                                          structure_qd_data = qd_data)

        r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium[i],
                                   n_sample, boundary, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')

        sintheta, costheta, sinphi, cosphi, _, _ = \
            mc.sample_angles(nevents, ntrajectories, p, rng=rng)
        step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)
        trajectories = mc.Trajectory(r0, k0, W0)

        trajectories.absorb(mu_abs, step)
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        with pytest.warns(UserWarning):
            reflectance[i], _ = det.calc_refl_trans(trajectories, thickness,
                                                    n_medium[i],
                                                    n_sample, boundary)
    reflectance_expected = [0.8095144529605994, 0.7708351929683783,
                            0.7683968574771831, 0.7731988230034157,
                            0.7926600420894914, 0.7581023055101348,
                            0.749330074570443, 0.7446221926705844,
                            0.7195657001870523, 0.7358665716261953,
                            0.7339394788533995, 0.7133017796237899,
                            0.6927204896275403, 0.6997635142273888,
                            0.6823068533385318, 0.6705433382082067,
                            0.6722019134466689, 0.6727772099454623,
                            0.650529567039051, 0.6410007651289527]

    assert_almost_equal(reflectance, reflectance_expected)
