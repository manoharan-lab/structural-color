# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang, Solomon Barkley,
# Annie Stephenson
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
Tests for the montecarlo model (in structcol/montecarlo.py)

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>

"""

import structcol as sc
from .. import montecarlo as mc
from .. import detector as det
import numpy as np
import xarray as xr
import warnings
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
import pytest

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
volume_fraction = 0.5
angles = sc.Quantity(np.linspace(0.01, np.pi, 200), 'rad')
wavelen = sc.Quantity('400.0 nm')
index_particle = sc.ConstantIndex(1.5)
index_matrix = sc.ConstantIndex(1.0)
index_medium = sc.ConstantIndex(1.0)
particle = sc.Sphere(index_particle, radius)
model = sc.model.HardSpheres(particle, volume_fraction, index_matrix,
                             index_medium)
boundary = "film"

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

seed_list = list(range(229, 262))
# this will fail at seeds 230 and 261 because trajectories exit at the same
# event as total internal reflection
xfail_reason = "total internal reflection at same event as exit"
@pytest.mark.parametrize("seed", seed_list)
def test_exit_detection(seed):
    """Tests whether exit detection routines in detector.py correctly detects
    transmitted and reflected trajectories. We do this by comparing to results
    from looping through all the trajectories, accounting for total internal
    reflection.  Currently tests only film geometry.

    """
    # use modern random number generator.  We don't use default_rng here because
    # the default generator could change in new versions of numpy
    rng = np.random.Generator(np.random.PCG64([seed]))

    # run simulation
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary,
                        rng=rng)
    sim.run()

    thickness = 10
    n_sample = model.index_external(wavelen).to_numpy().squeeze()
    n_medium = index_medium(wavelen).to_numpy().squeeze()
    refl_indices_expected = np.zeros(ntrajectories, dtype=int)
    trans_indices_expected = np.zeros_like(refl_indices_expected)
    stuck_indices_expected = np.zeros_like(refl_indices_expected)
    tir_refl_expected = np.zeros((nevents, ntrajectories), dtype=bool)

    # need to do deep copy to avoid modifying the original
    traj = sim.traj.copy(deep=True)
    # use loop to figure out where each trajectory exits
    for j in range(ntrajectories):
        for i in range(1, nevents+1):
            # end position after event i-1 is the start position of event i
            # (again, need to use copy here to avoid modifying)
            z = traj.position.sel(component="z", event=i, trajectory=j).copy()
            # direction that took us to this position happened after last event
            kz_prev = traj.direction.sel(component="z", event=i-1,
                                         trajectory=j).copy()
            # condition on magnitude of z-direction for TIR
            tir = np.abs(kz_prev) <= np.cos(np.arcsin(n_medium / n_sample))
            if (z < 0) and not tir:
                # trajectory j exited backward (reflection) at event i
                refl_indices_expected[j] = i
                break
            elif (z > thickness) and not tir:
                # trajectory j exited forward (transmission) at event i
                trans_indices_expected[j] = i
                break
            elif i==nevents:
                stuck_indices_expected[j] = i
            if tir:
                # trajectory j met criteria for totally internal reflection
                # after event i-1 but we first need to check if it actually hit
                # an interface
                if ((z < 0) or (z > thickness)):
                    # need to reverse the z-component of the direction of all
                    # trajectories after the reflection
                    el = {"component": "z",
                          "trajectory": j,
                          "event": slice(i, None)}
                    traj.direction.loc[el] = -traj.direction.loc[el]
                    # and flip the z-component about the boundary for this and
                    # all subsequent postions
                    if z < 0:
                        traj.position.loc[el] = -traj.position.loc[el]
                    if z > thickness:
                        traj.position.loc[el] = (2*thickness
                                                 - traj.position.loc[el])
                    # check if direction was in reflection dir; if so, we count
                    # toward tir_refl_bool
                    if z < 0:
                        tir_refl_expected[i-1,j] = True
                    # check if exit happened in tir event. This is rare, but
                    # can happen when scattering length is comparable to
                    # thickness.
                    el = {"component": "z",
                          "trajectory": j,
                          "event": i}
                    if (traj.position.loc[el] < 0):
                        refl_indices_expected[j] = i
                        pytest.xfail(reason=xfail_reason)
                        break
                    if (traj.position.loc[el] > thickness):
                        trans_indices_expected[j] = i
                        pytest.xfail(reason=xfail_reason)
                        break

    tir_indices_expected = np.argmax(np.vstack([np.zeros(ntrajectories),
                                                tir_refl_expected]),
                                     axis=0)

    exit_tuple = det.find_exits(n_sample, n_medium, thickness, 0, boundary,
                                sim.traj)
    refl_indices, trans_indices, stuck_indices, tir_indices = exit_tuple

    assert_equal(refl_indices, refl_indices_expected)
    assert_equal(trans_indices, trans_indices_expected)
    assert_equal(stuck_indices, stuck_indices_expected)
    assert_equal(tir_indices, tir_indices_expected)

    # check to make sure also that find_exits() returns unambiguous
    # transmitted, reflected, and stuck indices (meaning that each trajectory
    # can have a non-zero entry in only one of the three arrays)
    assert not np.any(trans_indices & refl_indices)
    assert not np.any(stuck_indices & refl_indices)
    assert not np.any(stuck_indices & trans_indices)


def test_calc_refl_trans():
    # this test is deterministic; no rng is involved
    high_thresh = 10
    index_medium = sc.ConstantIndex(1)
    index_matrix_small = sc.ConstantIndex(1)
    index_matrix_large = sc.ConstantIndex(2)
    large_n = index_matrix_large(wavelen)

    # index match particle to matrix so that effective index is same as matrix
    particle = sc.Sphere(index_matrix, radius)

    # test absoprtion and stuck without fresnel
    z_pos = np.array([[0,   0,  0,  0],
                      [1,   1,  1,  1],
                      [-1, 11,  2,  11],
                      [-2, 12,  4,  12]])
    # looking at the array (and recalling that thickness=10), we can see that:
    # - trajectory 1 (column 1) has exited in -z dir (reflection) at event 2
    # - trajectory 2 (column 2) has exited in +z dir (transmission) at event 2
    # - trajectory 3 has not exited the sample
    # - trajectory 4 (column 3) has exited in +z dir (transmission) at event 2
    nevents = z_pos.shape[0]-1
    ntrajectories = z_pos.shape[1]
    pos_coords = {"component": ["x", "y", "z"],
                  "event": range(nevents+1),
                  "trajectory": range(ntrajectories)}
    r0 = xr.DataArray(np.zeros((3, nevents+1, ntrajectories)),
                      coords=pos_coords)
    r0.loc["z"] = z_pos
    k0 = xr.zeros_like(r0.isel(event=slice(0, -1)))
    k0.loc["z"] = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])
    weights = xr.DataArray([[1., 1., 1., 1.],
                            [.8, .8, .9, .8],
                            [.7, .3, .7, 0],
                            [.1, .1, .5, 0]],
                           coords=r0.sel(component="x", drop=True).coords)
    # now we can see that the weights at exit are as follows
    # - trajectory 1: 0.7
    # - trajectory 2: 0.3
    # - trajectory 3: did not exit, so weight is set to final weight (0.5)
    # - trajectory 4: 0

    trajectories = xr.Dataset({"position": r0,
                               "direction": k0,
                               "weight": weights})

    # set up a dummy simulation and insert the trajectories
    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix_small,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary)
    sim.traj = trajectories

    refl, trans = det.calc_refl_trans(sim, high_thresh)
    # calculated manually
    # this array contains the weight at exit for each trajectory, depending on
    # whether it is transmitted or reflected. Trajectory 1 was reflected with
    # weight 0.7, which gives the first element of expected_refl_array.
    # Trajectories 2 and 4 were transmitted with weights 0.3 and 0., which
    # gives the second and fourth elements of expected_trans_array. Trajectory
    # 3 is stuck with weight 0.5, so det.distribute_ambig_traj_weights()
    # distributes it equally between reflected and transmitted, both with
    # weight 0.25
    expected_trans_array = np.array([0, .3, .25, 0]) / ntrajectories
    # calculated manually
    expected_refl_array = np.array([.7, 0, .25, 0]) / ntrajectories
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test above but with covers on front and back
    refl, trans = det.calc_refl_trans(sim, high_thresh,
                                      n_front=large_n.to_numpy().squeeze(),
                                      n_back=large_n.to_numpy().squeeze())
    # calculated manually
    expected_trans_array = (np.array([0.00814545, 0.20014545, 0.2, 0.])
                            / ntrajectories)
    # calculated manually
    expected_refl_array = (np.array([0.66700606, 0.20349091, 0.4, 0.2])
                           / ntrajectories)
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test fresnel as well
    z_pos = np.array([[ 0,   0,  0,  0],
                      [ 5,   5,  5,  5],
                      [-5,  -5, 15, 15],
                      [ 5, -15,  5, 25],
                      [-5, -25,  6, 35]])
    nevents = z_pos.shape[0] - 1
    ntrajectories = z_pos.shape[1]
    pos_coords = {"component": ["x", "y", "z"],
                  "event": range(nevents+1),
                  "trajectory": range(ntrajectories)}
    r0 = xr.DataArray(np.zeros((3, nevents+1, ntrajectories)),
                      coords=pos_coords)
    r0.loc["z"] = z_pos
    kz = np.array([[ 1.0,  1.0,  1.0,  0.86746757864487367],
                   [-0.1, -0.1,  0.1,  0.1],
                   [ 0.1, -0.1, -0.1,  0.1],
                   [-1.0, -0.9,  1.0,  1.0]])
    k0 = xr.zeros_like(r0.isel(event=slice(0, -1)))
    k0.loc["z"] = kz
    weights = xr.DataArray([[1., 1., 1., 1.],
                            [.8, .8, .9, .8],
                            [.7, .3, .7, .5],
                            [.6, .2, .6, .4],
                            [.4, .1, .5, .3]],
                           coords=r0.sel(component="x", drop=True).coords)

    trajectories = xr.Dataset({"position": r0,
                               "direction": k0,
                               "weight": weights})

    # set up a dummy simulation and insert the trajectories
    particle = sc.Sphere(index_matrix_large, radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix_large,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary)
    sim.traj = trajectories

    refl, trans = det.calc_refl_trans(sim, high_thresh)
    # calculated manually
    expected_trans_array = (np.array([ .00167588, .00062052, .22222222,
                                       .11075425]) / ntrajectories)
    # calculated manually
    expected_refl_array = (np.array([ .43317894, .18760061, .33333333,
                                      .59300905]) / ntrajectories)
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test refraction and detection_angle
    refl, trans= det.calc_refl_trans(sim, high_thresh,
                                     detection_angle=0.1)
    # calculated manually
    expected_trans_array = (np.array([ .00167588, .00062052, .22222222,
                                       .11075425]) / ntrajectories)
    # calculated manually
    expected_refl_array = (np.array([ .43203386, .11291556, .29105299,
                                      .00046666]) / ntrajectories)
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test steps in z longer than sample thickness
    z_pos = np.array([[0,0,0,0,0,0,0], [1.1,2.1,3.1,0.6,0.6,0.6,0.1],
                      [1.2,2.2,3.2,1.6,0.7,0.7,-0.6],
                      [1.3,2.3,3.3,3.3,-2.1,-1.1,-2.1]])
    nevents = z_pos.shape[0] - 1
    ntrajectories = z_pos.shape[1]
    pos_coords = {"component": ["x", "y", "z"],
                  "event": range(nevents+1),
                  "trajectory": range(ntrajectories)}
    r0 = xr.DataArray(np.zeros((3, nevents+1, ntrajectories)),
                      coords=pos_coords)
    r0.loc["z"] = z_pos
    kz = np.array([[1,1,1,1,1,1,1], [1,1,1,0.1,1,1,-0.1], [1,1,1,1,-1,-1,-1]])
    k0 = xr.zeros_like(r0.isel(event=slice(0, -1)))
    k0.loc["z"] = kz
    weights = xr.DataArray([[1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1]],
                           coords=r0.sel(component="x", drop=True).coords)
    thin_sample_thickness = 1

    trajectories = xr.Dataset({"position": r0,
                               "direction": k0,
                               "weight": weights})

    # set up a dummy simulation and insert the trajectories
    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix_large,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary)
    sim.traj = trajectories

    refl, trans= det.calc_refl_trans(sim, thin_sample_thickness)

    # calculated manually
    expected_trans_array = (np.array([.8324515, .8324515, .8324515, .05643739,
                                     .05643739, .05643739, .8324515]) /
                            ntrajectories)
    # calculated manually
    expected_refl_array = (np.array([.1675485, .1675485, .1675485, .94356261,
                                     .94356261, .94356261, .1675485]) /
                           ntrajectories)
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

def test_reflection_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a film
    of particles. The parameters, setup, and expected values come from the
    montecarlo_tutorial notebook.
    """

    seed = 1
    ntrajectories = 100
    nevents = 100
    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity('0.125 um')
    volume_fraction = 0.5
    index_particle = sc.ConstantIndex(1.54)
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    R_expected = 0.564374409013182
    T_expected = 0.4356255909868179

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

def test_surface_roughness_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a film
    of particles with coarse and fine surface roughness. The parameters, setup,
    and expected values come from the montecarlo_tutorial notebook (might need
    to set the seed in the notebook to get these values)
    """

    seed = 1
    rng = np.random.RandomState([seed])

    # Properties of system
    ntrajectories = 100
    nevents = 100
    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity('0.125 um')
    volume_fraction = 0.5
    index_particle = sc.ConstantIndex(1.54)
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    boundary = 'film'

    incidence_theta_min = sc.Quantity(0, 'rad')
    incidence_theta_max = sc.Quantity(0, 'rad')
    incidence_phi_min = sc.Quantity(0, 'rad')
    incidence_phi_max = sc.Quantity(2 * np.pi, 'rad')

    # Need to specify fine_roughness and coarse_roughness
    fine_roughness = sc.Quantity(0.6, '')
    coarse_roughness = sc.Quantity(1.1, '')

    # Need to specify fine roughness parameter in this function
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary,
                        rng=rng,
                        incidence_theta_min = incidence_theta_min,
                        incidence_theta_max = incidence_theta_max,
                        incidence_phi_min = incidence_phi_min,
                        incidence_phi_max = incidence_phi_max,
                        coarse_roughness = coarse_roughness,
                        fine_roughness = fine_roughness)
    sim.run()

    cutoff = sc.Quantity('50 um')

    R, T = det.calc_refl_trans(sim, cutoff)

    # previous values were based on using n_tir = n_sample in detector.py.
    # Updated values to handle case when n_matrix and n_particle are used to
    # calculate n_tir:
    R_expected = 0.6868088783398588
    T_expected = 0.25255636332566694

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

def test_reflection_core_shell():
    # test that the reflection of a non-core-shell system is the same as that
    # of a core-shell with a shell index matched with the core
    seed = 1
    nevents = 60
    ntrajectories = 30
    radius = sc.Quantity('150.0 nm')
    volume_fraction = 0.5
    sphere = sc.Sphere(index_particle, radius)

    # Reflection using a non-core-shell system
    ## ignore the "not enough events" warning
    warnings.filterwarnings("ignore", category=UserWarning)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    # Reflection using core-shells with the shell index-matched to the core
    ## specify the radii from innermost to outermost layer
    radius_cs = sc.Quantity(np.array([100.0, 150.0]), 'nm')
    ## specify the index from innermost to outermost layer
    index_cs = [sc.ConstantIndex(1.5), sc.ConstantIndex(1.5)]
    sphere_cs = sc.Sphere(index_cs, radius_cs)
    model_cs = sc.model.HardSpheres(sphere_cs, volume_fraction, index_matrix,
                                    index_medium)

    R_cs, T_cs = calc_montecarlo(model_cs, nevents, ntrajectories, wavelen,
                                 seed)

    assert_almost_equal(R, R_cs)
    assert_almost_equal(T, T_cs)

    # Expected outputs, consistent with results expected from before refactoring
    R_before = 0.7862152377246211 #before correcting nevents in sample_angles:: 0.81382378303119451
    R_cs_before = 0.7862152377246211 #before correcting nevents in sample_angles: 0.81382378303119451
    T_before = 0.21378476227537888 #before correcting nevents in sample_angles: 0.1861762169688054
    T_cs_before = 0.21378476227537888 #before correcting nevents in sample_angles: 0.1861762169688054

    assert_almost_equal(R, R_before)
    assert_almost_equal(R_cs, R_cs_before)
    assert_almost_equal(T, T_before)
    assert_almost_equal(T_cs, T_cs_before)

    # Test that the reflectance is the same for a core-shell that absorbs (with
    # the same refractive indices for all layers) and a non-core-shell that
    # absorbs with the same index
    # Reflection using a non-core-shell absorbing system
    index_particle_abs = sc.ConstantIndex(1.5+0.001j)
    radius = sc.Quantity(150.0, 'nm')
    particle_abs = sc.Sphere(index_particle_abs, radius)
    model_abs = sc.model.HardSpheres(particle_abs, volume_fraction,
                                     index_matrix, index_medium)
    R_abs, T_abs = calc_montecarlo(model_abs, nevents, ntrajectories, wavelen,
                                   seed)

    # Reflection using core-shells with the shell index-matched to the core
    index_cs_abs = [sc.ConstantIndex(1.5+0.001j),
                    sc.ConstantIndex(1.5+0.001j)]
    sphere_cs_abs = sc.Sphere(index_cs_abs, radius_cs)
    model_cs_abs = sc.model.HardSpheres(sphere_cs_abs, volume_fraction,
                                        index_matrix, index_medium)
    R_cs_abs, T_cs_abs = calc_montecarlo(model_cs_abs, nevents, ntrajectories,
                                         wavelen, seed)

    assert_almost_equal(R_cs_abs, R_abs, decimal=6)
    assert_almost_equal(T_cs_abs, T_abs, decimal=6)

    # Expected outputs, consistent with results expected from before refactoring
    #
    # (note: values below may be off at the 10th decimal place as of scipy
    # 1.15.0; since the goal is to ensure that we get the same results for
    # homogeneous spheres as for index-matched core-shell spheres, this
    # difference should not concern us too much)
    R_abs_before = 0.3079106226852705 #before correcting nevents in sample_angles: 0.3956821177047554
    R_cs_abs_before = 0.3079106226846794 #before correcting nevents in sample_angles: 0.39568211770416667
    T_abs_before = 0.02335228504958959 #before correcting nevents in sample_angles: 0.009944245822685388
    T_cs_abs_before = 0.023352285049450985 #before correcting nevents in sample_angles: 0.009944245822595715

    # increased acceptable error after fixing complex index handling in
    # detector.py (used to use assert_almost_equal with decimal=10)
    assert_allclose(R_abs, R_abs_before, rtol=1e-6)
    assert_allclose(R_cs_abs, R_cs_abs_before, rtol=1e-6)
    assert_allclose(T_abs, T_abs_before, rtol=1e-6)
    assert_allclose(T_cs_abs, T_cs_abs_before, rtol=1e-6)

    # Same as previous test but with absorbing matrix as well
    # Reflection using a non-core-shell absorbing system
    index_particle_abs = sc.ConstantIndex(1.5+0.001j)
    particle_abs = sc.Sphere(index_particle_abs, radius)
    index_matrix_abs = sc.ConstantIndex(1.+0.001j)
    model_abs_mat = sc.model.HardSpheres(particle_abs, volume_fraction,
                                         index_matrix_abs, index_medium)
    R_abs, T_abs = calc_montecarlo(model_abs_mat, nevents, ntrajectories,
                                   wavelen, seed)

    # Reflection using core-shells with the shell index-matched to the core
    index_cs_abs = [sc.ConstantIndex(1.5+0.001j),
                    sc.ConstantIndex(1.5+0.001j)]
    sphere_cs_abs = sc.Sphere(index_cs_abs, radius_cs)
    model_cs_abs_match = sc.model.HardSpheres(sphere_cs_abs, volume_fraction,
                                              index_matrix_abs, index_medium)
    R_cs_abs, T_cs_abs = calc_montecarlo(model_cs_abs_match, nevents,
                                         ntrajectories, wavelen, seed)

    assert_almost_equal(R_cs_abs, R_abs, decimal=6)
    assert_almost_equal(T_cs_abs, T_abs, decimal=6)

    # Expected outputs, consistent with results expected from before refactoring
    R_abs_before = 0.19121902522926137 #before correcting nevents in sample_angles: 0.27087005070007175
    R_cs_abs_before = 0.19121902522926137 #before correcting nevents in sample_angles: 0.27087005070007175
    T_abs_before = 0.0038425936376528256 #before correcting nevents in sample_angles: 0.0006391960305096798
    T_cs_abs_before = 0.0038425936376528256 #before correcting nevents in sample_angles: 0.0006391960305096798

    # increased acceptable error after fixing complex index handling in
    # detector.py (used to use assert_almost_equal with default tolerance)
    assert_allclose(R_abs, R_abs_before, rtol=1e-5)
    assert_allclose(R_cs_abs, R_cs_abs_before, rtol=1e-5)
    assert_allclose(T_abs, T_abs_before, rtol=1e-5)
    assert_allclose(T_cs_abs, T_cs_abs_before, rtol=1e-5)

def test_reflection_core_shell_mc():
    # Tests whether the reflectance is what we expect from a simulation on a
    # film of core-shell particles. The parameters, setup, and expected values
    # come from the montecarlo_tutorial notebook (might need to set the seed in
    # the notebook to get these values). The setup is slightly different from
    # that in test_reflection_core_shell()
    seed = 1
    ntrajectories = 100
    nevents = 100

    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity(np.array([0.125, 0.13]), 'um')
    index_particle = [sc.ConstantIndex(1.54), sc.ConstantIndex(1.33)]
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    volume_fraction = 0.5

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    R_expected = 0.6236144236194011
    T_expected = 0.37638557638059883

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)


def test_reflection_absorbing_particle_or_matrix():
    # test that the reflections with a real n_particle and with a complex
    # n_particle with a 0 imaginary component are the same
    seed = 1
    nevents = 60
    ntrajectories = 30

    # Reflection using non-absorbing particle
    sphere = sc.Sphere(index_particle, radius)
    warnings.filterwarnings("ignore", category=UserWarning) # ignore the "not enough events" warning
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    # Reflection using particle with an imaginary component of 0
    index_particle_abs = sc.ConstantIndex(1.5 + 0j)
    sphere_abs = sc.Sphere(index_particle_abs, radius)
    model_abs = sc.model.HardSpheres(sphere_abs, volume_fraction, index_matrix,
                                     index_medium)
    R_abs, T_abs = calc_montecarlo(model_abs, nevents, ntrajectories, wavelen,
                                   seed)

    assert_equal(R, R_abs)
    assert_equal(T, T_abs)

    # Expected outputs, consistent with results expected from before refactoring
    R_before = 0.7862152377246211#before correcting nevents in sample_angles: 0.81382378303119451
    R_abs_before = 0.7862152377246211#before correcting nevents in sample_angles: 0.81382378303119451
    T_before = 0.21378476227537888#before correcting nevents in sample_angles: 0.1861762169688054
    T_abs_before = 0.21378476227537888#before correcting nevents in sample_angles: 0.1861762169688054

    assert_almost_equal(R, R_before)
    assert_almost_equal(R_abs, R_abs_before)
    assert_almost_equal(T, T_before)
    assert_almost_equal(T_abs, T_abs_before)

    # Same as previous test but with absorbing matrix
    # Reflection using matrix with an imaginary component of 0
    index_matrix_abs = sc.ConstantIndex(1. + 0j)
    sphere = sc.Sphere(index_particle, radius)

    model_abs_mat = sc.model.HardSpheres(sphere, volume_fraction,
                                         index_matrix_abs, index_medium)
    R_abs, T_abs = calc_montecarlo(model_abs_mat, nevents, ntrajectories,
                                   wavelen, seed)

    assert_equal(R, R_abs)
    assert_equal(T, T_abs)

    # Expected outputs, consistent with results expected from before refactoring
    R_before = 0.7862152377246211 #before correcting nevents in sample_angles: 0.81382378303119451
    R_abs_before = 0.7862152377246211 #before correcting nevents in sample_angles: 0.81382378303119451
    T_before = 0.21378476227537888 #before correcting nevents in sample_angles: 0.1861762169688054
    T_abs_before = 0.21378476227537888#before correcting nevents in sample_angles: 0.1861762169688054

    assert_almost_equal(R, R_before)
    assert_almost_equal(R_abs, R_abs_before)
    assert_almost_equal(T, T_before)
    assert_almost_equal(T_abs, T_abs_before)

    # test that the reflection is essentially the same when the imaginary
    # index is 0 or very close to 0
    index_matrix_abs = sc.ConstantIndex(1. + 1e-10j)

    model_abs = sc.model.HardSpheres(sphere, volume_fraction,
                                     index_matrix_abs, index_medium)
    R_abs, T_abs = calc_montecarlo(model_abs, nevents, ntrajectories, wavelen,
                                   seed)
    assert_almost_equal(R, R_abs, decimal=6)
    assert_almost_equal(T, T_abs, decimal=6)

def test_reflection_absorption_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a film
    of particles with absorption. The parameters, setup, and expected values
    come from the montecarlo_tutorial notebook (might need to set the seed in
    the notebook to get these values).  The setup is slightly different from
    that in test_reflection_absorbing_particle_or_matrix()
    """

    seed = 1
    ntrajectories = 100
    nevents = 100
    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity('0.125 um')
    volume_fraction = 0.5
    index_particle = sc.ConstantIndex(1.54 + 0.001j)
    index_matrix = sc.index.vacuum + 0.0001j

    sphere = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    R_expected = 0.17047479609558655
    T_expected = 0.0948230136065759

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

def test_reflection_polydispersity():
    seed = 1
    nevents = 60
    ntrajectories = 30

    sphere1 = sc.Sphere(index_particle, radius)
    radius2 = radius
    sphere2 = sc.Sphere(index_particle, radius2)
    concentration = np.array([0.9, 0.1])
    pdi = np.array([1e-7,1e-7])  # monodisperse limit

    sphere_dist = sc.SphereDistribution([sphere1, sphere2], concentration, pdi)

    # Without absorption: test that the reflectance using very small
    # polydispersity is the same as the monodisperse case
    warnings.filterwarnings("ignore", category=UserWarning) # ignore the "not enough events" warning
    model_mono = sc.model.HardSpheres(sphere1, volume_fraction, index_matrix,
                                      index_medium)
    R_mono, T_mono = calc_montecarlo(model_mono, nevents, ntrajectories,
                                     wavelen, seed)
    model_poly = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                                  index_matrix, index_medium)
    R_poly, T_poly = calc_montecarlo(model_poly, nevents, ntrajectories,
                                     wavelen, seed)
    assert_almost_equal(R_mono, R_poly)
    assert_almost_equal(T_mono, T_poly)

    # Outputs before refactoring structcol
    R_mono_before = 0.7862152377246211 #before correcting nevents in sample_angles: 0.81382378303119451
    R_poly_before = 0.7862152377246211 #before correcting nevents in sample_angles: 0.81382378303119451
    T_mono_before = 0.21378476227537888 #before correcting nevents in sample_angles: 0.1861762169688054
    T_poly_before = 0.21378476227537888 #before correcting nevents in sample_angles: 0.1861762169688054

    assert_almost_equal(R_mono, R_mono_before)
    assert_almost_equal(R_poly, R_poly_before)
    assert_almost_equal(T_mono, T_mono_before)
    assert_almost_equal(T_poly, T_poly_before)

    # With absorption: test that the reflectance using with very small
    # polydispersity is the same as the monodisperse case
    index_particle_abs = sc.ConstantIndex(1.5+0.0001j)
    index_matrix_abs = sc.ConstantIndex(1.+0.0001j)

    sphere_abs = sc.Sphere(index_particle_abs, radius)
    model_mono_abs = sc.model.HardSpheres(sphere_abs, volume_fraction,
                                          index_matrix_abs, index_medium)
    R_mono_abs, T_mono_abs = calc_montecarlo(model_mono_abs, nevents,
                                             ntrajectories, wavelen, seed)
    sphere_dist_abs = sc.SphereDistribution([sphere_abs, sphere_abs],
                                            concentration, pdi)
    model_poly_abs = sc.model.PolydisperseHardSpheres(sphere_dist_abs,
                                                      volume_fraction,
                                                      index_matrix_abs,
                                                      index_medium)
    R_poly_abs, T_poly_abs = calc_montecarlo(model_poly_abs, nevents,
                                             ntrajectories, wavelen, seed)

    assert_almost_equal(R_mono_abs, R_poly_abs, decimal=6)
    assert_almost_equal(T_mono_abs, T_poly_abs, decimal=6)

    # Outputs before refactoring structcol
    R_mono_abs_before = 0.5861304578863337  #before correcting nevents in sample_angles: 0.6480185516058052
    R_poly_abs_before = 0.5861304624420246  #before correcting nevents in sample_angles: 0.6476683654364985
    T_mono_abs_before = 0.11704096147886706 #before correcting nevents in sample_angles: 0.09473841417422774
    T_poly_abs_before = 0.11704096346317548 #before correcting nevents in sample_angles: 0.09456832138047852

    assert_almost_equal(R_mono_abs, R_mono_abs_before)
    assert_almost_equal(R_poly_abs, R_poly_abs_before)
    assert_almost_equal(T_mono_abs, T_mono_abs_before)
    assert_almost_equal(T_poly_abs, T_poly_abs_before)

    # test that the reflectance is the same for a polydisperse monospecies
    # and a bispecies with equal types of particles
    concentration_single = 1
    concentration_dual = np.array([0.3, 0.7])
    pdi2 = np.array([1e-1, 1e-1])

    sphere_dist_single = sc.SphereDistribution(sphere1, concentration_single,
                                               pdi2[0])
    model_single = sc.model.PolydisperseHardSpheres(sphere_dist_single,
                                                    volume_fraction,
                                                    index_matrix, index_medium)
    R_mono2, T_mono2 = calc_montecarlo(model_single, nevents, ntrajectories,
                                       wavelen, seed)
    sphere_dist_dual = sc.SphereDistribution([sphere1, sphere2],
                                             concentration_dual, pdi2)
    model_dual = sc.model.PolydisperseHardSpheres(sphere_dist_dual,
                                                  volume_fraction,
                                                  index_matrix, index_medium)
    R_bi, T_bi = calc_montecarlo(model_dual, nevents, ntrajectories, wavelen,
                                 seed)

    assert_equal(R_mono2, R_bi)
    assert_equal(T_mono2, T_bi)

    # test that the reflectance is the same regardless of the order in which
    # the species are specified
    radius1 = sc.Quantity("150.0 nm")
    sphere1 = sc.Sphere(index_particle, radius1)
    radius2 = sc.Quantity("70.0 nm")
    sphere2 = sc.Sphere(index_particle, radius2)
    concentration = np.array([0.5,0.5])

    sphere_dist = sc.SphereDistribution([sphere1, sphere2], concentration, pdi)
    model = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                             index_matrix, index_medium)
    R, T = calc_montecarlo(model, nevents, ntrajectories,  wavelen, seed)
    sphere_dist_rev = sc.SphereDistribution([sphere2, sphere1], concentration,
                                             pdi)
    model_rev = sc.model.PolydisperseHardSpheres(sphere_dist_rev,
                                                 volume_fraction, index_matrix,
                                                 index_medium)
    R2, T2 = calc_montecarlo(model_rev, nevents, ntrajectories, wavelen, seed)

    assert_almost_equal(R, R2)
    assert_almost_equal(T, T2)

    # test that the second size is ignored when its concentration is set to 0
    radius1 = sc.Quantity("150.0 nm")
    sphere1 = sc.Sphere(index_particle, radius1)
    radius2 = sc.Quantity("100.0 nm")
    sphere2 = sc.Sphere(index_particle, radius2)
    concentration = np.array([1, 0])
    pdi3 = np.array([0., 0.])

    sphere_dist = sc.SphereDistribution([sphere1, sphere2], concentration,
                                        pdi3)
    model = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                             index_matrix, index_medium)

    R3, T3 = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    assert_equal(R_mono, R3)
    assert_equal(T_mono, T3)

    # test that the reflection is essentially the same when the imaginary
    # index is 0 or very close to 0 in a polydisperse system
    ## When there's only 1 mean diameter
    radius1 = sc.Quantity("100.0 nm")
    radius2 = sc.Quantity("150.0 nm")
    index_matrix_noabs = sc.ConstantIndex(1.)
    index_matrix_abs = sc.ConstantIndex(1. + 1e-40*1j)

    sphere1 = sc.Sphere(index_particle, radius1)
    sphere2 = sc.Sphere(index_particle, radius2)
    concentration4 = sc.Quantity(np.array([0.1, 0.9]), '')
    pdi4 = sc.Quantity(np.array([0.2, 0.2]), '')
    sphere_dist = sc.SphereDistribution(sphere1, 1, pdi4[0])

    model_noabs = sc.model.PolydisperseHardSpheres(sphere_dist,
                                                   volume_fraction,
                                                   index_matrix_noabs,
                                                   index_medium)
    R_noabs1, T_noabs1 = calc_montecarlo(model_noabs, nevents, ntrajectories,
                                         wavelen, seed)

    model_abs1 = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                                  index_matrix_abs,
                                                  index_medium)

    R_abs1, T_abs1 = calc_montecarlo(model_abs1, nevents, ntrajectories,
                                     wavelen, seed)
    assert_almost_equal(R_noabs1, R_abs1)
    assert_almost_equal(T_noabs1, T_abs1)

    # When there are 2 mean diameters
    sphere_dist = sc.SphereDistribution([sphere1, sphere2], concentration4,
                                        pdi4)
    model_noabs2 = sc.model.PolydisperseHardSpheres(sphere_dist,
                                                    volume_fraction,
                                                    index_matrix_noabs,
                                                    index_medium)
    R_noabs2, T_noabs2 = calc_montecarlo(model_noabs2, nevents, ntrajectories,
                                         wavelen, seed)

    model_abs2 = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                                  index_matrix_abs,
                                                  index_medium)
    R_abs2, T_abs2 = calc_montecarlo(model_abs2, nevents, ntrajectories,
                                     wavelen, seed)

    # Note: Previously (before adding lines nevents=nevents-1 to sample_angles()),
    # this test yielded:
    # R_abs2 =   0.8682177456973259
    # R_noabs2 = 0.8682177456973241
    # making the results equal to 14 decimals. This superb agreement appears to be
    # a coincidence of the particular combination of events and trajectories,
    # as the results only matched to 1 or 2 decimals for other event and trajectory
    # numbers. We therefore change the required decimal agreement to only
    # one place.
    assert_almost_equal(R_noabs2, R_abs2, decimal=1)
    assert_almost_equal(T_noabs2, T_abs2, decimal=1)


def test_reflection_polydispersity_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a film
    of polydisperse particles. The parameters, setup, and expected values
    come from the montecarlo_tutorial notebook (might need to set the seed in
    the notebook to get these values).  The setup is slightly different from
    that in test_reflection_polydispersity()
    """

    seed = 1
    ntrajectories = 100
    nevents = 100
    wavelen = sc.Quantity("600 nm")
    volume_fraction = 0.5
    index_particle = sc.ConstantIndex(1.54)
    index_matrix = sc.index.vacuum

    # define the parameters for polydispersity
    radius1 = sc.Quantity("125 nm")
    radius2 = sc.Quantity("150 nm")
    sphere1 = sc.Sphere(index_particle, radius1)
    sphere2 = sc.Sphere(index_particle, radius2)
    concentration = np.array([0.9, 0.1])
    pdi = np.array([0.01, 0.01])
    sphere_dist = sc.SphereDistribution([sphere1, sphere2], concentration, pdi)

    model = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                             index_matrix, index_medium)
    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    R_expected = 0.5807373008878349
    T_expected = 0.41926269911216507

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)


def test_detectors_mc():
    """
    Tests whether the reflectances on the default detector (full reflection
    hemisphere), large aperture detector, and goniometer detector are what we
    expect from a simulation on a film of particles. The parameters, setup, and
    expected values come from the detector_tutorial notebook.
    """

    seed = 1
    rng = np.random.RandomState([seed])

    wavelength = sc.Quantity('600 nm')

    radius = sc.Quantity('0.140 um')
    volume_fraction = 0.55
    volume_fraction_da = xr.DataArray([[0.55, 1-0.55]],
                                      coords = {sc.Coord.VOLFRAC: [0.55],
                                                sc.Coord.MAT: range(2)})
    n_imag = 2.1e-4 * 1j
    index_particle = sc.index.polystyrene + sc.ConstantIndex(n_imag)
    sphere = sc.Sphere(index_particle, radius)

    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    thickness = sc.Quantity('80 um')
    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 300
    nevents = 200

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Create simulation object and run
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories, boundary,
                        rng=rng)
    sim.run()

    # test default detector (full reflection hemisphere)
    R, _ = det.calc_refl_trans(sim, thickness)

    R_expected = 0.42179454919817455

    assert_almost_equal(R, R_expected)

    # test with 80 degree large-aperture detector
    detection_angle = sc.Quantity('80 degrees')

    R, _ = det.calc_refl_trans(sim, thickness,
                               detection_angle = detection_angle)

    R_expected = 0.4130249995689382

    assert_almost_equal(R, R_expected)

    # test with goniometer detector
    detector = True
    det_theta = sc.Quantity('45 degrees')
    det_len = sc.Quantity('5 cm')
    det_dist = sc.Quantity('10 cm')

    # Calculate reflectance
    R, _ = det.calc_refl_trans(sim, thickness,
                               detector = detector,
                               det_theta = det_theta,
                               det_len = det_len,
                               det_dist = det_dist,
                               plot_detector = False)

    R_expected = 0.028071349010350494

    assert_almost_equal(R, R_expected)

    # test renormalized goniometer reflectance
    refl_renorm = det.normalize_refl_goniometer(R, det_dist, det_len,
                                                det_theta)

    refl_renorm_expected = 0.4988708702766998

    assert_almost_equal(refl_renorm.magnitude, refl_renorm_expected)


def test_throw_valueerror_for_polydisperse_core_shells():
# test that a valueerror is raised when trying to run polydisperse core-shells
    seed = 1
    nevents = 10
    ntrajectories = 5

    # specify the radii from innermost to outermost layer
    radius_cs = sc.Quantity(np.array([100.0, 150.0]), 'nm')
    # specify the index from innermost to outermost layer
    index_particle_cs = [sc.ConstantIndex(1.5), sc.ConstantIndex(1.5)]
    sphere_cs = sc.Sphere(index_particle_cs, radius_cs)
    radius2 = radius
    sphere_cs_2 = sc.Sphere(index_particle, radius2)
    concentration = sc.Quantity(np.array([0.9, 0.1]), '')
    # monodisperse limit
    pdi = sc.Quantity(np.array([1e-7, 1e-7]), '')

    sphere_dist = sc.SphereDistribution([sphere_cs, sphere_cs_2],
                                        concentration, pdi)
    model = sc.model.PolydisperseHardSpheres(sphere_dist, volume_fraction,
                                             index_matrix, index_medium)

    with pytest.raises(ValueError, match="Cannot handle polydispersity"):
        R_cs, T_cs = calc_montecarlo(model, nevents, ntrajectories, wavelen,
                                     seed)


def test_surface_roughness():
    # test that the reflectance with very small surface roughness is the same
    # as without any roughness
    seed = 1
    nevents = 100
    ntrajectories = 30

    sphere = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Reflection with no surface roughness
    R, T = calc_montecarlo(model, nevents, ntrajectories, wavelen, seed)

    # Reflection with very little fine surface roughness
    R_fine, T_fine = calc_montecarlo(model, nevents, ntrajectories, wavelen,
                                     seed, fine_roughness=1e-4)

    # Reflection with very little coarse surface roughness
    R_coarse, T_coarse = calc_montecarlo(model, nevents, ntrajectories,
                                         wavelen, seed, coarse_roughness=1e-5)

    # Reflection with very little fine and coarse surface roughness
    R_both, T_both = calc_montecarlo(model, nevents, ntrajectories, wavelen,
                                     seed, fine_roughness=1e-4,
                                     coarse_roughness=1e-5)

    # tolerances set to handle small differences due to roughness
    assert_allclose(R, R_fine, rtol=1e-5)
    assert_allclose(T, T_fine, rtol=1e-5)
    assert_allclose(R, R_coarse, rtol=1e-5)
    assert_allclose(T, T_coarse, rtol=1e-5)
    assert_allclose(R, R_both, rtol=1e-5)
    assert_allclose(T, T_both, rtol=1e-5)


def calc_montecarlo(model, nevents, ntrajectories, wavelen, seed,
                    fine_roughness=0., coarse_roughness=0.,
                    incidence_theta_min=0., incidence_theta_max=0.):
    # Function to run montecarlo for the tests

    # set up a seeded random number generator that will give consistent results
    # between numpy versions. This is to reproduce the gold values which are
    # hardcoded in the tests. Note that seed is in the form of a list. Setting
    # the seed without the list brackets yields a different set of random
    # numbers.
    rng = np.random.RandomState([seed])
    incidence_theta_min=sc.Quantity(incidence_theta_min,'rad')
    incidence_theta_max=sc.Quantity(incidence_theta_min,'rad')

    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, "film", rng=rng,
                        fine_roughness=fine_roughness,
                        coarse_roughness=coarse_roughness,
                        incidence_theta_min = incidence_theta_min,
                        incidence_theta_max = incidence_theta_max)
    sim.run()

    cutoff = sc.Quantity('50.0 um')

    # calculate R, T
    R, T = det.calc_refl_trans(sim, cutoff)

    return R, T

def test_goniometer_normalization():

    # test the goniometer renormalization function
    refl = 0.002
    det_distance = 13.
    det_len = 2.4
    det_theta = 0
    refl_renorm = det.normalize_refl_goniometer(refl, det_distance, det_len,
                                                det_theta)

    assert_almost_equal(refl_renorm, 0.368700804483) # calculated by hand

def test_goniometer_detector():
    # test that goniometer (fixed angle) detection works
    z_pos = np.array([[ 0,  0,  0,  0],
                      [ 1,  1,  1,  1],
                      [-1, -1,  2,  2],
                      [-2, -2, 20, -0.0000001]])
    ntrajectories = z_pos.shape[1]
    nevents = z_pos.shape[0] - 1
    x_pos = np.zeros((nevents+1, ntrajectories))
    y_pos = np.zeros((nevents+1, ntrajectories))
    # direction is set so that the last trajectory reflects at 45 degrees to the
    # normal
    ky = np.zeros((nevents, ntrajectories))
    kx = np.array([[ 0,  0,  0,  0],
                   [ 0,  0,  0,  0],
                   [ 0,  0,  0,  1/np.sqrt(2)]])
    kz = np.array([[ 1,  1,  1,  1],
                   [-1, -1,  1,  1],
                   [-1, -1,  1, -1/np.sqrt(2)]])
    positions = xr.DataArray(np.array([x_pos, y_pos, z_pos]),
                             coords = {"component": ["x", "y", "z"],
                                       "event": range(nevents + 1),
                                       "trajectory": range(ntrajectories)})
    directions = xr.DataArray(np.array([kx, ky, kz]),
                              coords =
                              positions.isel(event=slice(0, -1)).coords)

    weights = xr.DataArray(np.ones((nevents+1, ntrajectories)),
                           coords =
                           positions.sel(component="x", drop=True).coords)

    trajectories = xr.Dataset({"position": positions,
                               "direction": directions,
                               "weight": weights})

    # set up a dummy simulation and insert the trajectories
    index_medium = sc.ConstantIndex(1)
    index_matrix = sc.ConstantIndex(1)
    # particle is index matched to matrix so that effective index is 1 and there
    # is no refraction at the boundary
    particle = sc.Sphere(index_matrix, radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary)
    sim.traj = trajectories

    thickness = 10
    out = det.calc_refl_trans(sim, thickness,
                              detector=True,
                              det_theta=sc.Quantity('45.0 degrees'),
                              det_len=sc.Quantity('1.0 um'),
                              det_dist=sc.Quantity('10.0 cm'),
                              plot_detector=False, return_extra=True)

    R = out[13]

    # one out of the four trajectories should hit the detector
    assert_almost_equal(R, 0.25)

    # there should be no internal reflection
    tir_indices = out[3]
    assert_equal(tir_indices, np.zeros_like(tir_indices))

