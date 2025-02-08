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
from .. import Quantity, np, structure
from .. import size_parameter
from .. import refractive_index as ri
from numpy.testing import assert_equal, assert_almost_equal
import pytest
from pint.errors import DimensionalityError


def test_structure_factor_percus_yevick():
    # Test structure factor as calculated by solution of Ornstein-Zernike
    # integral equation and Percus-Yevick closure approximation

    # test that function handles dimensionless arguments, and only
    # dimensionless arguments
    structure.factor_py(Quantity('0.1'), Quantity('0.4'))
    structure.factor_py(0.1, 0.4)
    pytest.raises(DimensionalityError, structure.factor_py, Quantity('0.1'),
                  Quantity('0.1 m'))
    pytest.raises(DimensionalityError, structure.factor_py, Quantity('0.1 m'),
                  Quantity('0.1'))

    # test vectorization by doing calculation over range of qd and phi
    qd = np.arange(0.1, 20, 0.01)
    phi = np.array([0.15, 0.3, 0.45])
    # this little trick allows us to calculate the structure factor on a 2d
    # grid of points (turns qd into a column vector and phi into a row vector).
    # Could also use np.ogrid
    s = structure.factor_py(qd.reshape(-1,1), phi.reshape(1,-1))

    # compare to values from Cipelletti, Trappe, and Pine, "Scattering
    # Techniques", in "Fluids, Colloids and Soft Materials: An Introduction to
    # Soft Matter Physics", 2016 (plot on page 137)
    # (I extracted values from the plot using a digitizer
    # (http://arohatgi.info/WebPlotDigitizer/app/). They are probably good to
    # only one decimal place, so this is a fairly crude test.)
    max_vals = np.max(s, axis=0)    # max values of S(qd) at different phi
    max_qds = qd[np.argmax(s, axis=0)]  # values of qd at which S(qd) has max
    assert_almost_equal(max_vals[0], 1.17, decimal=1)
    assert_almost_equal(max_vals[1], 1.52, decimal=1)
    assert_almost_equal(max_vals[2], 2.52, decimal=1)
    assert_almost_equal(max_qds[0], 6.00, decimal=1)
    assert_almost_equal(max_qds[1], 6.37, decimal=1)
    assert_almost_equal(max_qds[2], 6.84, decimal=1)

def test_structure_factor_percus_yevick_core_shell():
    # Test that the structure factor is the same for core-shell particles and
    # non-core-shell particles at low volume fraction (assuming the core diameter
    # is the same as the particle diameter for the non-core-shell case)

    wavelen = Quantity('400.0 nm')
    angles = Quantity(np.pi, 'rad')
    n_matrix = Quantity(1.0, '')

    # Structure factor for non-core-shell particles
    radius = Quantity('100.0 nm')
    n_particle = Quantity(1.5, '')
    volume_fraction = Quantity(0.0001, '')         # IS VF TOO LOW?
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
    x = size_parameter(wavelen, n_sample, radius)
    qd = 4*x*np.sin(angles/2)
    s = structure.factor_py(qd, volume_fraction)

    # Structure factor for core-shell particles with core size equal to radius
    # of non-core-shell particle
    radius_cs = Quantity(np.array([100.0, 105.0]), 'nm')
    n_particle_cs = Quantity(np.array([1.5, 1.0]), '')
    volume_fraction_shell = volume_fraction * (radius_cs[1]**3 / radius_cs[0]**3 -1)
    volume_fraction_cs = Quantity(np.array([volume_fraction.magnitude, volume_fraction_shell.magnitude]), '')

    n_sample_cs = ri.n_eff(n_particle_cs, n_matrix, volume_fraction_cs)
    x_cs = size_parameter(wavelen, n_sample_cs, radius_cs[1]).flatten()
    qd_cs = 4*x_cs*np.sin(angles/2)
    s_cs = structure.factor_py(qd_cs, np.sum(volume_fraction_cs))

    assert_almost_equal(s.magnitude, s_cs.magnitude, decimal=5)


def test_structure_factor_polydisperse():
    # test that the analytical structure factor for polydisperse systems matches
    # Percus-Yevick in the monodisperse limit

    # Percus-Yevick
    qd = Quantity(5.0, '')
    phi = Quantity(0.5, '')
    S_py = structure.factor_py(qd, phi)

    # Polydisperse S
    d = Quantity('100.0 nm')
    c = Quantity(1.0, '')
    pdi = Quantity(1e-5, '')
    q2 = qd / d

    S_poly = structure.factor_poly(q2, phi, d, c, pdi)

    assert_almost_equal(S_py.magnitude, S_poly.magnitude)


def test_structure_factor_data():
    qd = np.array([1, 2])
    qd_data = np.array([0.5, 2.5])
    s_data = np.array([1, 1])
    s = structure.factor_data(qd, s_data, qd_data)
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
    volume_fraction = Quantity(0.5, '')
    n_particle = ri.n('fused silica', wavelengths)
    n_matrix = ri.n('vacuum', wavelengths)
    n_medium = ri.n('vacuum', wavelengths)
    thickness = Quantity('50 um')

    # generate structure factor "data" from Percus-Yevick model
    qd_data = np.arange(0.001, 75, 0.1)
    s_data = structure.factor_py(qd_data, volume_fraction.magnitude)

    # make interpolation function
    qd = np.arange(0, 70, 0.1)
    s = structure.factor_data(qd, s_data, qd_data)

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
    volume_fraction = sc.Quantity(0.5, '')
    n_particle = ri.n('fused silica', wavelengths)
    n_matrix = ri.n('vacuum', wavelengths)
    n_medium = ri.n('vacuum', wavelengths)
    boundary = 'film'
    thickness = sc.Quantity('50 um')

    qd_data = np.arange(0.001, 75, 0.1)
    s_data = structure.factor_py(qd_data, volume_fraction.magnitude)

    reflectance = np.zeros(wavelengths.size)
    for i in range(wavelengths.size):
        n_sample = ri.n_eff(n_particle[i], n_matrix[i], volume_fraction)

        p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle[i], n_sample,
                                          volume_fraction.magnitude,
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
