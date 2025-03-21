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
import xarray as xr

from numpy.testing import assert_equal, assert_almost_equal
import pytest


class TestStructureFactor():
    """Tests for the StructureFactor class and derived classes.
    """
    ql = np.arange(0.1, 20, 0.01)
    phi = np.array([0.15, 0.3, 0.45])

    def test_structure_factor_base_class(self):
        structure_factor = sc.structure.StructureFactor()
        with pytest.raises(NotImplementedError):
            structure_factor.calculate(self.ql)
        # test that __call__ method works
        with pytest.raises(NotImplementedError):
            structure_factor(self.ql)

    def test_percus_yevick(self):
        """Tests the object version of the Percus-Yevick structure factor
        """

        # test how function handles dimensionless arguments
        structure_factor = sc.structure.PercusYevick(Quantity('0.4'))
        # specifying quantities is not allowed when calculating
        with pytest.raises(AttributeError):
            s = structure_factor(Quantity('0.1'))
        # but scalars should work
        structure_factor = sc.structure.PercusYevick(0.4)
        s = structure_factor(0.1)

        # now calculate for arrays of phi and ql
        structure_factor = sc.structure.PercusYevick(self.phi)
        s = structure_factor(self.ql)

        # ensure that we are broadcasting correctly
        assert s.shape == (self.ql.shape[0], self.phi.shape[0])

        # make sure that calculation works with ql specified as DataArray
        ql = xr.DataArray(self.ql, coords = {"ql": self.ql})
        s = structure_factor(ql)
        assert s.shape == (self.ql.shape[0], self.phi.shape[0])

        # compare to values from Cipelletti, Trappe, and Pine, "Scattering
        # Techniques", in "Fluids, Colloids and Soft Materials: An Introduction
        # to Soft Matter Physics", 2016 (plot on page 137) (I extracted values
        # from the plot using a digitizer
        # (http://arohatgi.info/WebPlotDigitizer/app/). They are probably good
        # to only one decimal place, so this is a fairly crude test.)

        # max values of S(ql) at different phi
        max_vals = s.max(dim="ql")
        # values of ql at which S(ql) has max
        max_qls = s.idxmax(dim="ql")
        assert_almost_equal(max_vals[0], 1.17, decimal=1)
        assert_almost_equal(max_vals[1], 1.52, decimal=1)
        assert_almost_equal(max_vals[2], 2.52, decimal=1)
        assert_almost_equal(max_qls[0], 6.00, decimal=1)
        assert_almost_equal(max_qls[1], 6.37, decimal=1)
        assert_almost_equal(max_qls[2], 6.84, decimal=1)

        # compare to values from before refactoring
        ql = np.linspace(0.1, 20, 20)
        phi = 0.6

        structure_factor = sc.structure.PercusYevick(phi)
        s = structure_factor(ql)

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

        # test that structure factor converges to low-ql approximation at small
        # ql (but not so small that solution becomes numerically unstable).
        # Use a variety of volume fractions to test
        phi = np.linspace(0.05, 0.6, 10)
        structure_factor = sc.structure.PercusYevick(phi, ql_cutoff=0.01)
        s = structure_factor(ql)

        # generate structure factor with higher cutoff, so that below
        # ql_cutoff, the structure factor should be using the approximate
        # solution to the direct correlation function
        structure_factor_approx = sc.structure.PercusYevick(phi, ql_cutoff=0.2)
        s_approx = structure_factor_approx(ql)
        assert_almost_equal(s.sel(ql=0.1).to_numpy(),
                            s_approx.sel(ql=0.1).to_numpy())

    def test_paracrystal(self):
        """Tests the object version of the paracrystalline structure factor
        """
        volfrac = np.arange(0.1, 0.7, 0.1)
        sigma = np.arange(0, 0.5, 0.05)

        structure_factor = sc.structure.Paracrystal(volfrac, sigma=sigma)
        ql = np.linspace(0.05, 0.6, 10)
        s = structure_factor(ql)
        # TODO add tests to check that we are getting the right values

    def test_percus_yevick_polydisperse(self):
        """Tests the object version of the polydisperse structure factor.
        """
        # first test that the analytical structure factor for polydisperse
        # systems matches Percus-Yevick in the monodisperse limit
        ql = 5.0
        phi = 0.5

        # Percus-Yevick monodisperse
        monodisperse_structure_factor = sc.structure.PercusYevick(phi)
        s_py = monodisperse_structure_factor(ql)

        # Polydisperse Percus-Yevick
        d = Quantity('100.0 nm')
        c = 1.0
        pdi = 1e-5
        q2 = ql / d

        polydisperse_structure_factor = sc.structure.Polydisperse(phi, d, c,
                                                                  pdi)
        s_poly = polydisperse_structure_factor(q2)

        assert_almost_equal(s_py, s_poly.magnitude)

        # test that structure factor matches the calculations in Figure 1 of
        # Ginoza and Yasutomi, Journal of the Physical Society of Japan 1999.

        # Figure 1 curve A peaks and valleys, from digitized figure 1 (includes
        # the spurious peaks)
        ql = np.array([0.2909090909090909, 3.2, 5.03030303030303,
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
        q2 = ql / d

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
        index_matrix = sc.Index.constant(1.0)

        # Structure factor for non-core-shell particles
        radius = Quantity('100.0 nm')
        index_particle = sc.Index.constant(1.5)
        sphere = sc.model.Sphere(index_particle, radius)
        volume_fraction = 0.0001         # IS VF TOO LOW?
        volume_fraction_da = sphere.volume_fraction(total_volume_fraction =
                                                    volume_fraction)
        n_sample = sc.index.effective_index([index_particle, index_matrix],
                                            volume_fraction_da, wavelen)
        x = sc.size_parameter(n_sample, radius)
        qa = 4*x*np.sin(angles/2)
        structure_factor = sc.structure.PercusYevick(volume_fraction)
        s = structure_factor(qa)

        # Structure factor for core-shell particles with core size equal to
        # radius of non-core-shell particle
        radius_cs = Quantity(np.array([100.0, 105.0]), 'nm')
        index_particle = [sc.Index.constant(1.5), sc.Index.constant(1.0)]
        sphere_cs = sc.model.Sphere(index_particle, radius_cs)
        volume_fraction_da = sphere_cs.volume_fraction(total_volume_fraction =
                                                       volume_fraction)

        n_sample_cs = sc.index.effective_index(index_particle + [index_matrix],
                                               volume_fraction_da, wavelen)
        x_cs = sc.size_parameter(n_sample_cs, radius_cs[1])
        qa_cs = 4*x_cs*np.sin(angles/2)
        structure_factor_cs = sc.structure.PercusYevick(volume_fraction)
        s_cs = structure_factor_cs(qa_cs)

        assert_almost_equal(s, s_cs, decimal=5)

    def test_structure_factor_interpolated(self):
        ql = np.array([1, 2])
        ql_data = np.array([0.5, 2.5])
        s_data = np.array([1, 1])
        structure_factor = sc.structure.Interpolated(s_data, ql_data)
        s = structure_factor(ql)
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
    index_particle = sc.index.fused_silica
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    thickness = Quantity('50 um')

    # generate structure factor "data" from Percus-Yevick model
    ql_data = np.arange(0.001, 75, 0.1)
    structure_factor = sc.structure.PercusYevick(volume_fraction)
    s_data = structure_factor(ql_data)

    # make interpolation function
    ql = np.arange(0, 70, 0.1)
    structure_factor_interpolated = sc.structure.Interpolated(s_data, ql_data)
    s = structure_factor_interpolated(ql)

    # calculate reflectance from single-scattering model
    reflectance = np.zeros(len(wavelengths))
    for i in range(len(wavelengths)):
        reflectance[i],_,_,_,_ = \
            sc.model.reflection(index_particle,
                                index_matrix,
                                index_medium,
                                wavelengths[i],
                                radius,
                                volume_fraction,
                                thickness=thickness,
                                structure_type='data',
                                structure_s_data=s_data,
                                structure_qd_data=ql_data)

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
    sphere = sc.model.Sphere(index_particle, radius)
    n_particle = sphere.n(wavelengths)
    n_medium = index_medium(wavelengths)
    boundary = 'film'
    thickness = sc.Quantity('50 um')

    vf_array = sphere.volume_fraction(volume_fraction)

    n_sample_eff = sc.index.effective_index([index_particle, index_matrix],
                                        vf_array, wavelengths)


    ql_data = np.arange(0.001, 75, 0.1)
    structure_factor = sc.structure.PercusYevick(volume_fraction)
    s_data = structure_factor(ql_data)

    reflectance = np.zeros(wavelengths.size)
    for i in range(wavelengths.size):
        n_sample = n_sample_eff[i]
        p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle[i], n_sample,
                                          volume_fraction,
                                          wavelengths[i],
                                          structure_type = 'data',
                                          structure_s_data = s_data,
                                          structure_qd_data = ql_data)

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
