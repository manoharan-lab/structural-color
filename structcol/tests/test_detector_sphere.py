# Copyright 2018, Vinothan N. Manoharan, Annie Stephenson, Victoria Hwang,
# Solomon Barkley
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
Tests for the montecarlo model for sphere geometry (in structcol/montecarlo.py)
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleathor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

TODO: either delete this file or delete tests repeated in montecarlo.py
"""

import structcol as sc
from .. import montecarlo as mc
from .. import detector as det
from .. import refractive_index as ri
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
assembly_radius = 5
volume_fraction = 0.5
n_particle = sc.Quantity(1.5, '')
n_matrix = sc.Quantity(1.0, '')
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')
wavelen = sc.Quantity('400.0 nm')

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_calc_refl_trans():
    # this test should give deterministic results
    small_n = sc.Quantity(1.0,'')
    large_n = sc.Quantity(2.0,'')

    # test absoprtion and stuck without fresnel
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    x_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    y_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    ntrajectories = z_pos.shape[1]
    kx = np.zeros((3,4))
    ky = np.zeros((3,4))
    kz = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])
    weights = np.array([[.8, .8, .9, .8],[.7, .3, .7, 0],[.1, .1, .5, 0]])
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], weights)
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, small_n, volume_fraction, wavelen)
    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans = det.calc_refl_trans(trajectories, assembly_radius,
                                          small_n, small_n, 'sphere')
    expected_trans_array = np.array([0., .3, 0.25, 0])/ntrajectories #calculated manually
    expected_refl_array = np.array([.7, 0., .25, 0.])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test fresnel as well
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        refl, trans = det.calc_refl_trans(trajectories, assembly_radius,
                                          small_n, large_n, 'sphere')
    expected_trans_array = np.array([0.0345679, .25185185, 0.22222222, 0.])/ntrajectories #calculated manually
    expected_refl_array = np.array([.69876543, 0.12592593, 0.33333333, 0.11111111])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test steps in z longer than sample thickness
    z_pos = np.array([[0,0,0,0],[1,1,14,12],[-1,11,2,11],[-2,12,4,12]])
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], weights)
    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans= det.calc_refl_trans(trajectories, assembly_radius,
                                         small_n, small_n, 'sphere')
    expected_trans_array = np.array([0., .3, .9, .8])/ntrajectories #calculated manually
    expected_refl_array = np.array([.7, 0., 0., 0.])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test tir
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    weights = np.ones((3,4))
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], weights)
    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans = det.calc_refl_trans(trajectories, assembly_radius,
                                          small_n, small_n, 'sphere', p=p,
                                          mu_abs=mu_abs, mu_scat=mu_scat,
                                          run_fresnel_traj=True)
    # since the tir=True reruns the stuck trajectory, we don't know whether it will end up reflected or transmitted
    # all we can know is that the end refl + trans > 0.99
    assert_almost_equal(refl + trans, 1.)

def test_get_angles_sphere():
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    x_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,-0,0,0]])
    y_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    kx = np.zeros((3,4))
    ky = np.zeros((3,4))
    kz = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], None)

    indices = np.array([1,1,1,1])
    thetas, _ = det.get_angles(indices, 'sphere', trajectories, assembly_radius, init_dir = 1)
    assert_almost_equal(np.sum(thetas.magnitude), 0.)

def test_index_match():
    ntrajectories = 2
    nevents = 3
    wavelen = sc.Quantity('600.0 nm')
    radius = sc.Quantity('0.140 um')
    microsphere_radius = sc.Quantity('10.0 um')
    volume_fraction = sc.Quantity(0.55,'')
    n_particle = sc.Quantity(1.6,'')
    n_matrix = sc.Quantity(1.6,'')
    n_sample = n_matrix
    n_medium = sc.Quantity(1.0,'')

    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen)

    # initialize all at center top edge of the sphere going down
    r0_sphere = np.zeros((3,nevents+1,ntrajectories))
    k0_sphere = np.zeros((3,nevents,ntrajectories))
    k0_sphere[2,0,:] = 1
    W0_sphere = np.ones((nevents, ntrajectories))

    # make into quantities with units
    r0_sphere = sc.Quantity(r0_sphere, 'um')
    k0_sphere = sc.Quantity(k0_sphere, '')
    W0_sphere = sc.Quantity(W0_sphere, '')

    # Generate a matrix of all the randomly sampled angles first
    sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents, ntrajectories, p)

    # Create step size distribution
    step = mc.sample_step(nevents, ntrajectories, mu_scat)

    # make trajectories object
    trajectories_sphere = mc.Trajectory(r0_sphere, k0_sphere, W0_sphere)
    trajectories_sphere.absorb(mu_abs, step)
    trajectories_sphere.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories_sphere.move(step)

    # calculate reflectance
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        refl_sphere, trans = det.calc_refl_trans(trajectories_sphere,
                                                 microsphere_radius,
                                                 n_medium, n_sample,
                                                 'sphere', p=p,
                                                 mu_abs=mu_abs,
                                                 mu_scat=mu_scat,
                                                 run_fresnel_traj = True,
                                                 max_stuck = 0.0001)

    # calculated by hand from fresnel infinite sum
    refl_fresnel_int = 0.053 # calculated by hand
    refl_exact = refl_fresnel_int + (1-refl_fresnel_int)**2*refl_fresnel_int/(1-refl_fresnel_int**2)

    # under index-matched conditions, the step sizes are huge (bigger than the
    # sample size), and the light is scattered into the forward direction. As a
    # result, the reflectance is essentially deterministic, even though the
    # seed is not set for the random number generator.
    assert_almost_equal(refl_sphere, refl_exact, decimal=3)

def test_reflection_sphere_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a
    sphere containing particles. The parameters, setup, and expected values
    come from the montecarlo_tutorial notebook (might need to set the seed in
    the notebook to get these values).
    """

    seed = 1
    rng = np.random.RandomState([seed])
    ntrajectories = 100
    nevents = 100
    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity('0.125 um')
    assembly_diameter = sc.Quantity('10 um')
    volume_fraction = sc.Quantity(0.5, '')
    n_particle = sc.Quantity(1.54, '')
    n_matrix = ri.n('vacuum', wavelen)
    n_medium = ri.n('vacuum', wavelen)
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
    boundary = 'sphere'

    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelen)

    # Initialize the trajectories for a sphere
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                               boundary, plot_initial = False,
                               sample_diameter = assembly_diameter,
                               spot_size = assembly_diameter, rng=rng)

    # make positions, directions, and weights into quantities with units
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    # Generate a matrix of all the randomly sampled angles first
    sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents,
                                                                ntrajectories,
                                                                p, rng=rng)

    # Create step size distribution
    step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

    # Create trajectories object
    trajectories = mc.Trajectory(r0, k0, W0)

    # Run photons
    trajectories.absorb(mu_abs, step)
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.move(step)

    # Calculate reflectance and transmittance
    # The default value of run_tir is True, so you must set it to False to
    # exclude the fresnel reflected trajectories.
    with pytest.warns(UserWarning):
        R, T = det.calc_refl_trans(trajectories, assembly_diameter, n_medium,
                                   n_sample, boundary, plot_exits = False)

    R_expected = 0.24878084752516244
    T_expected = 0.7512191524748375

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

    # test with Fresnel reflections
    # Calculate reflectance and transmittance
    with pytest.warns(UserWarning):
        R, T = det.calc_refl_trans(trajectories, assembly_diameter, n_medium,
                                   n_sample, boundary, run_fresnel_traj = True,
                                   mu_abs=mu_abs, mu_scat=mu_scat, p=p,
                                   rng=rng)

    R_expected = 0.2508833560792594
    T_expected = 0.7491166439207406

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

