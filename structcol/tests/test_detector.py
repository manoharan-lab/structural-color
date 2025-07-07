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
from numpy.testing import assert_equal, assert_almost_equal
import pytest

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
volume_fraction = 0.5
angles = sc.Quantity(np.linspace(0.01, np.pi, 200), 'rad')
wavelen = sc.Quantity('400.0 nm')
index_particle = sc.Index.constant(1.5)
index_matrix = sc.Index.constant(1.0)
index_medium = sc.Index.constant(1.0)
volume_fraction_da = xr.DataArray([0.5, 1-0.5],
                                  coords={sc.Coord.MAT: range(2)})
n_particle = index_particle(wavelen)
n_matrix = index_matrix(wavelen)
n_medium = index_medium(wavelen)
n_sample = sc.index.effective_index([index_particle, index_matrix],
                                    volume_fraction_da, wavelen)

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_calc_refl_trans():
    # this test is deterministic; no rng is involved
    high_thresh = 10
    small_n = 1
    large_n = 2

    # test absoprtion and stuck without fresnel
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    ntrajectories = z_pos.shape[1]
    kz = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])
    weights = np.array([[.8, .8, .9, .8],[.7, .3, .7, 0],[.1, .1, .5, 0]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz],
                                 weights)

    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans= det.calc_refl_trans(trajectories, high_thresh, small_n,
                                         small_n, 'film')
    # calculated manually
    expected_trans_array = np.array([0, .3, .25, 0]) / ntrajectories
    # calculated manually
    expected_refl_array = np.array([.7, 0, .25, 0]) / ntrajectories
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test above but with covers on front and back
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        refl, trans = det.calc_refl_trans(trajectories, high_thresh, small_n,
                                          small_n, 'film',n_front=large_n,
                                          n_back=large_n)
    # calculated manually
    expected_trans_array = (np.array([0.00814545, 0.20014545, 0.2, 0.])
                            / ntrajectories)
    # calculated manually
    expected_refl_array = (np.array([0.66700606, 0.20349091, 0.4, 0.2])
                           / ntrajectories)
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test fresnel as well
    z_pos = np.array([[0,0,0,0], [5,5,5,5], [-5,-5,15,15], [5,-15,5,25],
                      [-5,-25,6,35]])
    ntrajectories = z_pos.shape[1]
    kz = np.array([[1,1,1,0.86746757864487367], [-.1,-.1,.1,.1],
                   [0.1,-.1,-.1,0.1], [-1,-.9,1,1]])
    weights = np.array([[.8, .8, .9, .8], [.7, .3, .7, .5], [.6, .2, .6, .4],
                        [.4, .1, .5, .3]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos], [np.nan, np.nan, kz],
                                 weights)

    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans= det.calc_refl_trans(trajectories, high_thresh, small_n,
                                         large_n, 'film')
    # calculated manually
    expected_trans_array = (np.array([ .00167588, .00062052, .22222222,
                                       .11075425]) / ntrajectories)
    # calculated manually
    expected_refl_array = (np.array([ .43317894, .18760061, .33333333,
                                      .59300905]) / ntrajectories)
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test refraction and detection_angle
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        refl, trans= det.calc_refl_trans(trajectories, high_thresh, small_n,
                                         large_n, 'film', detection_angle=0.1)
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
    ntrajectories = z_pos.shape[1]
    kz = np.array([[1,1,1,1,1,1,1], [1,1,1,0.1,1,1,-0.1], [1,1,1,1,-1,-1,-1]])
    weights = np.array([[1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]])
    thin_sample_thickness = 1
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz],
                                 weights)
    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans= det.calc_refl_trans(trajectories, thin_sample_thickness,
                                         small_n, large_n, 'film')
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
    index_particle = sc.Index.constant(1.54)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_particle = index_particle(wavelen)
    n_medium = index_medium(wavelen)
    volume_fraction_da = xr.DataArray([0.5, 1-0.5],
                                      coords = {sc.Coord.MAT: range(2)})
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        volume_fraction_da, wavelen)

    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen,
                           seed)

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
    index_particle = sc.Index.constant(1.54)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_particle = index_particle(wavelen)
    n_medium = index_medium(wavelen)
    n_matrix = index_matrix(wavelen)
    volume_fraction_da = xr.DataArray([0.5, 1-0.5],
                                      coords = {sc.Coord.MAT: range(2)})
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        volume_fraction_da, wavelen)
    boundary = 'film'

    incidence_theta_min = sc.Quantity(0, 'rad')
    incidence_theta_max = sc.Quantity(0, 'rad')
    incidence_phi_min = sc.Quantity(0, 'rad')
    incidence_phi_max = sc.Quantity(2 * np.pi, 'rad')

    # Need to specify fine_roughness and coarse_roughness
    fine_roughness = sc.Quantity(0.6, '')
    coarse_roughness = sc.Quantity(1.1, '')

    # Need to specify fine roughness parameter in this function
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelen,
                                      fine_roughness=fine_roughness,
                                      n_matrix=n_matrix)

    r0, k0, W0, kz0_rotated, kz0_reflected = \
        mc.initialize(nevents, ntrajectories, n_medium, n_sample, boundary,
                      rng=rng, incidence_theta_min = incidence_theta_min,
                      incidence_theta_max = incidence_theta_max,
                      incidence_phi_min = incidence_phi_min,
                      incidence_phi_max = incidence_phi_max,
                      coarse_roughness=coarse_roughness)

    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents,
                                                                ntrajectories,
                                                                p, rng=rng)

    # Need to specify the fine roughness parameter in this function
    step = mc.sample_step(nevents, ntrajectories, mu_scat,
                          fine_roughness=fine_roughness, rng=rng)

    trajectories = mc.Trajectory(r0, k0, W0)
    trajectories.absorb(mu_abs, step)
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.move(step)

    cutoff = sc.Quantity('50 um')

    # If there is coarse roughness, need to specify kz0_rotated and
    # kz0_reflected.
    with pytest.warns(UserWarning):
        R, T = det.calc_refl_trans(trajectories, cutoff, n_medium, n_sample,
                                   boundary, kz0_rot=kz0_rotated,
                                   kz0_refl=kz0_reflected)

    R_expected = 0.7166421049108462
    T_expected = 0.24500285182641726

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

    # Reflection using a non-core-shell system
    ## ignore the "not enough events" warning
    warnings.filterwarnings("ignore", category=UserWarning)
    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen, seed)

    # Reflection using core-shells with the shell index-matched to the core
    ## specify the radii from innermost to outermost layer
    radius_cs = sc.Quantity(np.array([100.0, 150.0]), 'nm')
    ## specify the index from innermost to outermost layer
    index_cs = [sc.Index.constant(1.5), sc.Index.constant(1.5)]
    sphere_cs = sc.Sphere(index_cs, radius_cs)
    n_particle_cs = sphere_cs.n(wavelen)

    vf_array = sphere_cs.volume_fraction(total_volume_fraction=volume_fraction)
    n_sample_cs = sc.index.effective_index(index_cs + [index_matrix], vf_array,
                                           wavelen)
    R_cs, T_cs = calc_montecarlo(nevents, ntrajectories, radius_cs,
                                 n_particle_cs, n_sample_cs, n_medium,
                                 volume_fraction, wavelen, seed)

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
    index_particle_abs = sc.Index.constant(1.5+0.001j)
    radius = sc.Quantity(150.0, 'nm')
    particle_abs = sc.Sphere(index_particle_abs, radius)
    n_particle_abs = particle_abs.n(wavelen)
    vf_array = particle_abs.volume_fraction(volume_fraction)
    n_sample_abs = sc.index.effective_index([index_particle_abs, index_matrix],
                                            vf_array, wavelen)

    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius,
                                   n_particle_abs, n_sample_abs, n_medium,
                                   volume_fraction, wavelen, seed)

    # Reflection using core-shells with the shell index-matched to the core
    index_cs_abs = [sc.Index.constant(1.5+0.001j),
                    sc.Index.constant(1.5+0.001j)]
    sphere_cs_abs = sc.Sphere(index_cs_abs, radius_cs)
    n_particle_cs_abs = sphere_cs_abs.n(wavelen)
    vf_array = sphere_cs_abs.volume_fraction(volume_fraction)
    n_sample_cs_abs = sc.index.effective_index(index_cs_abs + [index_matrix],
                                               vf_array, wavelen)

    R_cs_abs, T_cs_abs = calc_montecarlo(nevents, ntrajectories, radius_cs,
                                         n_particle_cs_abs, n_sample_cs_abs,
                                         n_medium, volume_fraction, wavelen,
                                         seed)

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

    assert_almost_equal(R_abs, R_abs_before, decimal=10)
    assert_almost_equal(R_cs_abs, R_cs_abs_before, decimal=10)
    assert_almost_equal(T_abs, T_abs_before, decimal=10)
    assert_almost_equal(T_cs_abs, T_cs_abs_before, decimal=10)

    # Same as previous test but with absorbing matrix as well
    # Reflection using a non-core-shell absorbing system
    index_particle_abs = sc.Index.constant(1.5+0.001j)
    n_particle_abs = index_particle_abs(wavelen)
    particle_abs = sc.Sphere(index_particle_abs, radius)
    index_matrix_abs = sc.Index.constant(1.+0.001j)
    n_matrix_abs = index_matrix_abs(wavelen)
    vf_array = particle_abs.volume_fraction(volume_fraction)
    n_sample_abs = sc.index.effective_index([index_particle_abs,
                                             index_matrix_abs], vf_array,
                                            wavelen)

    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius,
                           n_particle_abs, n_sample_abs, n_medium,
                           volume_fraction, wavelen, seed)

    # Reflection using core-shells with the shell index-matched to the core
    index_cs_abs = [sc.Index.constant(1.5+0.001j),
                    sc.Index.constant(1.5+0.001j)]
    sphere_cs_abs = sc.Sphere(index_cs_abs, radius_cs)
    n_particle_cs_abs = sphere_cs_abs.n(wavelen)
    vf_array = sphere_cs_abs.volume_fraction(volume_fraction)
    n_sample_cs_abs = sc.index.effective_index(index_cs_abs +
                                                [index_matrix_abs], vf_array,
                                               wavelen)

    R_cs_abs, T_cs_abs = calc_montecarlo(nevents, ntrajectories, radius_cs,
                                 n_particle_cs_abs, n_sample_cs_abs, n_medium,
                                 volume_fraction, wavelen, seed)

    assert_almost_equal(R_cs_abs, R_abs, decimal=6)
    assert_almost_equal(T_cs_abs, T_abs, decimal=6)

    # Expected outputs, consistent with results expected from before refactoring
    R_abs_before = 0.19121902522926137 #before correcting nevents in sample_angles: 0.27087005070007175
    R_cs_abs_before = 0.19121902522926137 #before correcting nevents in sample_angles: 0.27087005070007175
    T_abs_before = 0.0038425936376528256 #before correcting nevents in sample_angles: 0.0006391960305096798
    T_cs_abs_before = 0.0038425936376528256 #before correcting nevents in sample_angles: 0.0006391960305096798

    assert_almost_equal(R_abs, R_abs_before)
    assert_almost_equal(R_cs_abs, R_cs_abs_before)
    assert_almost_equal(T_abs, T_abs_before)
    assert_almost_equal(T_cs_abs, T_cs_abs_before)

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
    index_particle = [sc.Index.constant(1.54), sc.Index.constant(1.33)]
    sphere = sc.Sphere(index_particle, radius)
    n_particle = sphere.n(wavelen)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelen)
    volume_fraction = 0.5
    volume_fraction_da = sphere.volume_fraction(volume_fraction)
    n_sample = sc.index.effective_index(index_particle + [index_matrix],
                                        volume_fraction_da, wavelen)

    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen, seed)

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
    n_particle = index_particle(wavelen)

    # Reflection using non-absorbing particle
    warnings.filterwarnings("ignore", category=UserWarning) # ignore the "not enough events" warning
    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen, seed)

    # Reflection using particle with an imaginary component of 0
    index_particle_abs = sc.Index.constant(1.5 + 0j)
    n_particle_abs = index_particle_abs(wavelen)
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius,
                                   n_particle_abs, n_sample, n_medium,
                                   volume_fraction, wavelen, seed)

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
    index_matrix_abs = sc.Index.constant(1. + 0j)
    sphere = sc.Sphere(index_particle, radius)
    n_particle = sphere.n(wavelen)
    vf_array = sphere.volume_fraction(volume_fraction)
    n_sample_abs = sc.index.effective_index([index_particle, index_matrix_abs],
                                            vf_array,
                                            wavelen)
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius,
                                   n_particle, n_sample_abs, n_medium,
                                   volume_fraction, wavelen, seed)

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
    index_matrix_abs = sc.Index.constant(1. + 1e-10j)

    n_sample_abs = sc.index.effective_index([index_particle, index_matrix_abs],
                                            vf_array, wavelen)
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius,
                                   n_particle, n_sample_abs, n_medium,
                                   volume_fraction, wavelen, seed)
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
    index_particle = sc.Index.constant(1.54 + 0.001j)
    index_matrix = sc.index.vacuum + 0.0001j

    sphere = sc.Sphere(index_particle, radius)
    n_particle = sphere.n(wavelen)
    vf_array = sphere.volume_fraction(volume_fraction)
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        vf_array, wavelen)

    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen,
                           seed)

    R_expected = 0.17023086537622875
    T_expected = 0.09485003836166318

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

def test_reflection_polydispersity():
    seed = 1
    nevents = 60
    ntrajectories = 30

    radius2 = radius
    concentration = sc.Quantity(np.array([0.9,0.1]), '')
    pdi = sc.Quantity(np.array([1e-7,1e-7]), '')  # monodisperse limit

    # Without absorption: test that the reflectance using very small
    # polydispersity is the same as the monodisperse case
    warnings.filterwarnings("ignore", category=UserWarning) # ignore the "not enough events" warning
    R_mono, T_mono = calc_montecarlo(nevents, ntrajectories, radius,
                                     n_particle, n_sample, n_medium,
                                     volume_fraction, wavelen, seed,
                                     polydisperse=False)
    R_poly, T_poly = calc_montecarlo(nevents, ntrajectories, radius,
                                     n_particle, n_sample, n_medium,
                                     volume_fraction, wavelen, seed,
                                     radius2 = radius2,
                                     concentration = concentration, pdi = pdi,
                                     polydisperse=True)
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
    index_particle_abs = sc.Index.constant(1.5+0.0001j)
    index_matrix_abs = sc.Index.constant(1.+0.0001j)
    n_particle_abs = index_particle_abs(wavelen)
    n_matrix_abs = index_matrix_abs(wavelen)
    n_sample_abs = sc.index.effective_index([index_particle_abs,
                                             index_matrix_abs],
                                            volume_fraction_da, wavelen)

    R_mono_abs, T_mono_abs = calc_montecarlo(nevents, ntrajectories, radius,
                                             n_particle_abs, n_sample_abs,
                                             n_medium, volume_fraction, wavelen,
                                             seed, polydisperse=False)
    R_poly_abs, T_poly_abs = calc_montecarlo(nevents, ntrajectories, radius,
                                             n_particle_abs, n_sample_abs,
                                             n_medium, volume_fraction, wavelen,
                                             seed, radius2 = radius2,
                                             concentration = concentration,
                                             pdi = pdi, polydisperse=True)

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
    concentration_mono = sc.Quantity(np.array([0.,1.]), '')
    concentration_bi = sc.Quantity(np.array([0.3,0.7]), '')
    pdi2 = sc.Quantity(np.array([1e-1, 1e-1]), '')

    R_mono2, T_mono2 = calc_montecarlo(nevents, ntrajectories, radius,
                                       n_particle, n_sample, n_medium,
                                       volume_fraction, wavelen, seed, radius2 =
                                       radius2, concentration =
                                       concentration_mono, pdi = pdi2,
                                       polydisperse=True)
    R_bi, T_bi = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                                 n_sample, n_medium, volume_fraction,
                                 wavelen, seed, radius2 = radius2,
                                 concentration = concentration_bi, pdi =
                                 pdi2, polydisperse=True)

    assert_equal(R_mono2, R_bi)
    assert_equal(T_mono2, T_bi)

    # test that the reflectance is the same regardless of the order in which
    # the radii are specified
    radius2 = sc.Quantity('70.0 nm')
    concentration2 = sc.Quantity(np.array([0.5,0.5]), '')

    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen, seed,
                           radius2 = radius2, concentration = concentration2,
                           pdi = pdi,polydisperse=True)
    R2, T2 = calc_montecarlo(nevents, ntrajectories, radius2, n_particle,
                             n_sample, n_medium, volume_fraction, wavelen, seed,
                             radius2 = radius, concentration = concentration2,
                             pdi = pdi, polydisperse=True)

    assert_almost_equal(R, R2)
    assert_almost_equal(T, T2)

    # test that the second size is ignored when its concentration is set to 0
    radius1 = sc.Quantity('150.0 nm')
    radius2 = sc.Quantity('100.0 nm')
    concentration3 = sc.Quantity(np.array([1,0]), '')
    pdi3 = sc.Quantity(np.array([0., 0.]), '')

    R3, T3 = calc_montecarlo(nevents, ntrajectories, radius1, n_particle,
                             n_sample, n_medium, volume_fraction, wavelen, seed,
                             radius2 = radius2, concentration = concentration3,
                             pdi = pdi3, polydisperse=True)

    assert_equal(R_mono, R3)
    assert_equal(T_mono, T3)

    # test that the reflection is essentially the same when the imaginary
    # index is 0 or very close to 0 in a polydisperse system
    ## When there's only 1 mean diameter
    radius1 = sc.Quantity('100.0 nm')
    radius2 = sc.Quantity('150.0 nm')
    index_matrix_abs = sc.Index.constant(1. + 1e-40*1j)
    n_matrix_abs = index_matrix_abs(wavelen)
    n_sample_abs = sc.index.effective_index([index_particle, index_matrix_abs],
                                            volume_fraction_da, wavelen)
    pdi4 = sc.Quantity(np.array([0.2, 0.2]), '')
    concentration2 = sc.Quantity(np.array([0.1,0.9]), '')

    R_noabs1, T_noabs1 = calc_montecarlo(nevents, ntrajectories, radius1,
                                   n_particle, n_sample_abs.real, n_medium,
                                   volume_fraction, wavelen, seed,
                                   radius2 = radius1,
                                   concentration = concentration2,
                                   pdi = pdi4, polydisperse=True)

    R_abs1, T_abs1 = calc_montecarlo(nevents, ntrajectories, radius1,
                                   n_particle, n_sample_abs, n_medium,
                                   volume_fraction, wavelen, seed,
                                   radius2 = radius1,
                                   concentration = concentration2,
                                   pdi = pdi4, polydisperse=True)
    assert_almost_equal(R_noabs1, R_abs1)
    assert_almost_equal(T_noabs1, T_abs1)

    # When there are 2 mean diameters
    R_noabs2, T_noabs2 = calc_montecarlo(nevents, ntrajectories, radius1,
                                   n_particle, n_sample_abs.real, n_medium,
                                   volume_fraction, wavelen, seed,
                                   radius2 = radius2,
                                   concentration = concentration2,
                                   pdi = pdi4, polydisperse=True)

    # something to do with the combination of absorber, 2 radii, and nevents-1
    R_abs2, T_abs2 = calc_montecarlo(nevents, ntrajectories, radius1,
                                   n_particle, n_sample_abs, n_medium,
                                   volume_fraction, wavelen, seed,
                                   radius2 = radius2,
                                   concentration = concentration2,
                                   pdi = pdi4, polydisperse=True)

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
    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity('0.125 um')
    volume_fraction = 0.5
    volume_fraction_da = xr.DataArray([0.5, 1-0.5],
                                      coords = {sc.Coord.MAT : range(2)})
    index_particle = sc.Index.constant(1.54)
    n_particle = index_particle(wavelen)
    index_matrix = sc.index.vacuum
    n_matrix = index_matrix(wavelen)
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        volume_fraction_da, wavelen)

    # define the parameters for polydispersity
    radius = sc.Quantity('125 nm')
    radius2 = sc.Quantity('150 nm')
    concentration = sc.Quantity(np.array([0.9,0.1]), '')
    pdi = sc.Quantity(np.array([0.01, 0.01]), '')

    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction,
                           wavelen, seed, radius2 = radius2,
                           concentration = concentration, pdi = pdi,
                           polydisperse=True)

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
    volume_fraction_da = xr.DataArray([0.55, 1-0.55],
                                      coords = {sc.Coord.MAT: range(2)})
    n_imag = 2.1e-4 * 1j
    index_particle = sc.index.polystyrene + sc.Index.constant(n_imag)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_particle = index_particle(wavelength)
    n_medium = index_medium(wavelength)
    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        volume_fraction_da, wavelength)
    thickness = sc.Quantity('80 um')
    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 300
    nevents = 200

    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelength)

    # Initialize the trajectories
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                               boundary, rng=rng)
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

    # test default detector (full reflection hemisphere)
    with pytest.warns(UserWarning):
        R, _ = det.calc_refl_trans(trajectories, thickness, n_medium,
                                   n_sample, boundary)

    R_expected = 0.42179454919817455

    assert_almost_equal(R, R_expected)

    # test with 80 degree large-aperture detector
    detection_angle = sc.Quantity('80 degrees')

    with pytest.warns(UserWarning):
        R, _ = det.calc_refl_trans(trajectories, thickness, n_medium, n_sample,
                                   boundary,
                                   detection_angle = detection_angle)

    R_expected = 0.4130249995689382

    assert_almost_equal(R, R_expected)

    # test with goniometer detector
    detector = True
    det_theta = sc.Quantity('45 degrees')
    det_len = sc.Quantity('5 cm')
    det_dist = sc.Quantity('10 cm')

    # Calculate reflectance
    with pytest.warns(UserWarning):
        R, _ = det.calc_refl_trans(trajectories, thickness, n_medium, n_sample,
                                   boundary,
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
    index_particle_cs = [sc.Index.constant(1.5), sc.Index.constant(1.5)]
    sphere_cs = sc.Sphere(index_particle_cs, radius_cs)
    n_particle_cs = sphere_cs.n(wavelen)
    radius2 = radius
    concentration = sc.Quantity(np.array([0.9,0.1]), '')
    # monodisperse limit
    pdi = sc.Quantity(np.array([1e-7, 1e-7]), '')

    # calculate the volume fractions of each layer
    vf_array = sphere_cs.volume_fraction(volume_fraction)

    n_sample_cs = sc.index.effective_index(index_particle_cs + [index_matrix],
                                           vf_array, wavelen)

    with pytest.raises(ValueError):
        R_cs, T_cs = calc_montecarlo(nevents, ntrajectories, radius_cs,
                                     n_particle_cs, n_sample_cs, n_medium,
                                     volume_fraction, wavelen, seed,
                                     radius2=radius2,
                                     concentration=concentration, pdi=pdi,
                                     polydisperse=True)

def test_throw_valueerror_for_polydisperse_unspecified_parameters():
# test that a valueerror is raised when the system is polydisperse and radius2
# concentration or pdi are not specified
    seed = 1
    nevents = 10
    ntrajectories = 5

    # specify the radii from innermost to outermost layer
    radius = sc.Quantity(150.0, 'nm')
    # specify the index from innermost to outermost layer
    index_particle = sc.Index.constant(1.5)
    sphere = sc.Sphere(index_particle, radius)
    n_particle = sphere.n(wavelen)
    concentration = sc.Quantity(np.array([0.9,0.1]), '')
    # monodisperse limit
    pdi = sc.Quantity(np.array([1e-7, 1e-7]), '')

    # calculate the volume fractions of each layer
    vf_array = sphere.volume_fraction(volume_fraction)

    n_sample = sc.index.effective_index([index_particle, index_matrix],
                                        vf_array, wavelen)

    with pytest.raises(ValueError):
        R_cs, T_cs = calc_montecarlo(nevents, ntrajectories, radius,
                                     n_particle, n_sample, n_medium,
                                     volume_fraction, wavelen, seed,
                                     concentration=concentration, pdi=pdi,
                                     polydisperse=True)  # unspecified radius2

def test_surface_roughness():
    # test that the reflectance with very small surface roughness is the same
    # as without any roughness
    seed = 1
    nevents = 100
    ntrajectories = 30

    # Reflection with no surface roughness
    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle,
                           n_sample, n_medium, volume_fraction, wavelen, seed)

    # Reflection with very little fine surface roughness
    R_fine, T_fine = calc_montecarlo(nevents, ntrajectories, radius,
                                     n_particle, n_sample, n_medium,
                                     volume_fraction, wavelen, seed,
                                     fine_roughness = 1e-4, n_matrix=n_matrix)

    # Reflection with very little coarse surface roughness
    R_coarse, T_coarse = calc_montecarlo(nevents, ntrajectories, radius,
                                         n_particle, n_sample, n_medium,
                                         volume_fraction, wavelen, seed,
                                         coarse_roughness = 1e-5)

    # Reflection with very little fine and coarse surface roughness
    R_both, T_both = calc_montecarlo(nevents, ntrajectories, radius,
                                     n_particle, n_sample, n_medium,
                                     volume_fraction, wavelen, seed,
                                     fine_roughness=1e-4, coarse_roughness =
                                     1e-5, n_matrix=n_matrix)

    assert_almost_equal(R, R_fine)
    assert_almost_equal(T, T_fine)
    assert_almost_equal(R, R_coarse)
    assert_almost_equal(T, T_coarse)
    assert_almost_equal(R, R_both)
    assert_almost_equal(T, T_both)

def calc_montecarlo(nevents, ntrajectories, radius, n_particle, n_sample,
                    n_medium, volume_fraction, wavelen, seed, radius2=None,
                    concentration=None, pdi=None, polydisperse=False,
                    fine_roughness=0., coarse_roughness=0., n_matrix=None,
                    incidence_theta_min=0., incidence_theta_max=0.):

    # set up a seeded random number generator that will give consistent results
    # between numpy versions. This is to reproduce the gold values which are
    # hardcoded in the tests. Note that seed is in the form of a list. Setting
    # the seed without the list brackets yields a different set of random
    # numbers.
    rng = np.random.RandomState([seed])
    incidence_theta_min=sc.Quantity(incidence_theta_min,'rad')
    incidence_theta_max=sc.Quantity(incidence_theta_min,'rad')

    # Function to run montecarlo for the tests
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelen,
                                      radius2=radius2,
                                      concentration=concentration, pdi=pdi,
                                      polydisperse=polydisperse,
                                      fine_roughness=fine_roughness,
                                      n_matrix=n_matrix)

    if coarse_roughness > 0.:
        r0, k0, W0, kz0_rotated, kz0_reflected = \
            mc.initialize(nevents, ntrajectories, n_medium, n_sample, 'film',
                          rng=rng, coarse_roughness=coarse_roughness,
                          incidence_theta_min=incidence_theta_min,
                          incidence_theta_max=incidence_theta_max)
    else:
        r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                                   'film', rng=rng,
                                   incidence_theta_min=incidence_theta_min,
                                   incidence_theta_max=incidence_theta_max)
        kz0_rotated = None
        kz0_reflected = None

    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    sintheta, costheta, sinphi, cosphi, _, _= mc.sample_angles(nevents,
                                                               ntrajectories,
                                                               p, rng=rng)
    step = mc.sample_step(nevents, ntrajectories, mu_scat,
                          fine_roughness=fine_roughness, rng=rng)

    trajectories = mc.Trajectory(r0, k0, W0)
    trajectories.absorb(mu_abs, step)
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.move(step)

    cutoff = sc.Quantity('50.0 um')

    # calculate R, T
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        R, T = det.calc_refl_trans(trajectories, cutoff, n_medium, n_sample,
                                   'film', kz0_rot=kz0_rotated,
                                   kz0_refl=kz0_reflected)

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
    # test
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,-1,2,2],[-2,-2,20,-0.0000001]])
    ntrajectories = z_pos.shape[1]
    nevents = z_pos.shape[0]
    x_pos = np.zeros((nevents, ntrajectories))
    y_pos = np.zeros((nevents, ntrajectories))
    ky = np.zeros((nevents-1, ntrajectories))
    kx = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1/np.sqrt(2)]])
    kz = np.array([[1,1,1,1],[-1,-1,1,1],[-1,-1,1,-1/np.sqrt(2)]])
    weights = np.ones((nevents, ntrajectories))
    trajectories = mc.Trajectory(np.array([x_pos, y_pos, z_pos]),
                                 np.array([kx, ky, kz]), weights)
    thickness = 10
    n_medium = 1
    n_sample = 1
    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        R, T = det.calc_refl_trans(trajectories, thickness, n_medium, n_sample,
                                   'film', detector=True,
                                   det_theta=sc.Quantity('45.0 degrees'),
                                   det_len=sc.Quantity('1.0 um'),
                                   det_dist=sc.Quantity('10.0 cm'),
                                   plot_detector=False)

    assert_almost_equal(R, 0.25)
