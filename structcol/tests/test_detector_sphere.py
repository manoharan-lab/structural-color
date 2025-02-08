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
from structcol import phase_func_sphere as pfs
from .. import montecarlo as mc
from .. import detector as det
from .. import refractive_index as ri
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from pint.errors import UnitStrippedWarning
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

@pytest.mark.slow
def test_multiscale_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a bulk
    collection of spheres containing particles. The parameters, setup, and
    expected values come from the multiscale_montecarlo_tutorial.ipynb notebook
    """

    seed = 1
    rng = np.random.RandomState([seed])

    wavelengths = sc.Quantity(np.arange(400., 801.,15),'nm')

    # Geometric properties of sample
    particle_radius = sc.Quantity('0.110 um')
    volume_fraction_particles = sc.Quantity(0.5, '')
    volume_fraction_bulk = sc.Quantity(0.55,'')
    sphere_boundary_diameter = sc.Quantity(10,'um')
    bulk_thickness = sc.Quantity('50 um')
    boundary = 'sphere'
    boundary_bulk = 'film'

    # Refractive indices
    n_particle = ri.n('vacuum', wavelengths)
    n_matrix = (ri.n('fused silica', wavelengths)
                + 9e-4 * ri.n('vacuum', wavelengths) * 1j)
    n_matrix_bulk = ri.n('vacuum', wavelengths)
    n_medium = ri.n('vacuum', wavelengths)

    # number of trajectories to run with a spherical boundary
    ntrajectories = 2000
    # number of scattering events for each trajectory in a spherical boundary
    nevents = 300
    # number of trajectories to run in the bulk film
    ntrajectories_bulk = 2000
    # number of events to run in the bulk film
    nevents_bulk = 300

    # initialize quantities we want to save as a function of wavelength
    reflectance_sphere = np.zeros(wavelengths.size)
    mu_scat_bulk = sc.Quantity(np.zeros(wavelengths.size),'1/um')
    mu_abs_bulk = sc.Quantity(np.zeros(wavelengths.size),'1/um')
    p_bulk = np.zeros((wavelengths.size, 200))

    # loop through wavelengths
    for i in range(wavelengths.size):
        # caculate the effective index of the sample
        n_sample = ri.n_eff(n_particle[i], n_matrix[i],
                            volume_fraction_particles)

        p, mu_scat, mu_abs = mc.calc_scat(particle_radius, n_particle[i],
                                          n_sample, volume_fraction_particles,
                                          wavelengths[i])

        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_matrix_bulk[i],
                                   n_sample, boundary, sample_diameter =
                                   sphere_boundary_diameter, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')

        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

        # Generate a matrix of all the randomly sampled angles first
        sintheta, costheta, sinphi, cosphi, _, _ = \
            mc.sample_angles(nevents, ntrajectories, p, rng=rng)

        # Create step size distribution
        step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

        # Run photons
        trajectories.absorb(mu_abs, step)
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        # Calculate reflection and transmission
        with pytest.warns(UserWarning):
            (refl_indices,
             trans_indices,
             _, _, _,
             refl_per_traj, trans_per_traj,
             _,_,_,_,
             reflectance_sphere[i],
             _,_, norm_refl, norm_trans) = \
                 det.calc_refl_trans(trajectories, sphere_boundary_diameter,
                                     n_matrix_bulk[i], n_sample, boundary, p=p,
                                     mu_abs=mu_abs, mu_scat=mu_scat,
                                     run_fresnel_traj = False, return_extra =
                                     True)


        ### Calculate phase function and lscat ###
        # use output of calc_refl_trans to calculate phase function, mu_scat,
        # and mu_abs for the bulk
        p_bulk[i,:], mu_scat_bulk[i], mu_abs_bulk[i] = \
            pfs.calc_scat_bulk(refl_per_traj, trans_per_traj, refl_indices,
                               trans_indices, norm_refl, norm_trans,
                               volume_fraction_bulk, sphere_boundary_diameter,
                               n_matrix_bulk[i], wavelengths[i], plot=False,
                               phi_dependent=False)

    # test that reflectance and phase function at backscattering angle are
    # as expected
    R_sphere_expected = [0.38123541767508723, 0.3835200359496135,
                         0.4094742443938745, 0.392447398856936,
                         0.38787629316233174, 0.32173386484914546,
                         0.29902165959552296, 0.2329953363061719,
                         0.17092553030675361, 0.12922719885919093,
                         0.1118153709864796, 0.08419077495592843,
                         0.07435025691548613, 0.054437428807312074,
                         0.046680792782201844, 0.04039405830712657,
                         0.040839999042456096, 0.03087860342358341,
                         0.028260613177474078, 0.02410139150538851,
                         0.02562060712817843, 0.022200859471872208,
                         0.020544288676286923, 0.017158143993391044,
                         0.018960512892166843, 0.016640174256390673,
                         0.015241452113027344]

    # phase function at backscattering
    pfb_expected = [0.0035377912407208944, 0.0034538749401316674,
                    0.002990668071679147, 0.002849170446354908,
                    0.0027947093116751473, 0.0025022176721912447,
                    0.002620898913840109, 0.00289280239320613,
                    0.0031233074380674205, 0.003746697953756506,
                    0.003321649084200047, 0.003701775378630818,
                    0.004022688852993721, 0.003858205590532855,
                    0.0038643794457639733, 0.004190303743579821,
                    0.00399478182087941, 0.004278656161938358,
                    0.004307156304755717, 0.004630096424806511,
                    0.004575501923096159, 0.004546256426287217,
                    0.004762624992343167, 0.004623080773544097,
                    0.0047806588190352104, 0.004824798406597221,
                    0.00477158130536513]

    assert_almost_equal(reflectance_sphere, R_sphere_expected)
    assert_almost_equal(p_bulk[:, 100], pfb_expected)

    # now look at bulk film
    # initialize some quantities we want to save as a function of wavelength
    reflectance_bulk = np.zeros(wavelengths.size)

    for i in range(wavelengths.size):
        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents_bulk, ntrajectories_bulk,
                                   n_medium[i], n_matrix_bulk[i],
                                   boundary_bulk, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        W0 = sc.Quantity(W0, '')
        k0 = sc.Quantity(k0, '')

        # Sample angles
        sintheta, costheta, sinphi, cosphi, _, _ = \
            mc.sample_angles(nevents_bulk, ntrajectories_bulk, p_bulk[i,:],
                             rng=rng)


        # Calculate step size note: in future versions, mu_abs will be removed
        # from step size sampling, so 0 is entered here
        step = mc.sample_step(nevents_bulk, ntrajectories_bulk,
                              mu_scat_bulk[i], rng=rng)

        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

        # Run photons
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        # calculate bulk reflectance
        with pytest.warns(UserWarning):
            reflectance_bulk[i], transmittance = \
                det.calc_refl_trans(trajectories, bulk_thickness, n_medium[i],
                                    n_matrix_bulk[i], boundary_bulk)

    # these numbers look a little strange (multiply them by the number of
    # trajectories, and they all become integers). That's because there's no
    # absorption in this simulation, so every trajectory has a weight of 1 when
    # it exits the sample.  The reflectance is then an integer number of
    # trajectories divided by the number of trajectories.

    R_bulk_expected = [0.74500000013005, 0.7645000001109206,
                       0.7780000000985678, 0.7505000001245004,
                       0.7450000001300501, 0.743000000132098,
                       0.7130000001647379, 0.642000000256328,
                       0.5565000003933844, 0.5090000004821619,
                       0.45400000059623213, 0.372000000788768,
                       0.34300000086329807, 0.27200000105996813,
                       0.22850000119042457, 0.22150000121212457,
                       0.2130000012387382, 0.15500000142805012,
                       0.14800000145180814, 0.13200000150684812,
                       0.1490000014484022, 0.11750000155761267,
                       0.1255000015295006, 0.08450000167628063,
                       0.10550000160026066, 0.08250000168361263,
                       0.08100000168912212]

    assert_almost_equal(reflectance_bulk, R_bulk_expected)

@pytest.mark.slow
def test_multiscale_polydispersity_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a
    polydisperse bulk collection of spheres containing particles. The
    parameters, setup, and expected values come from the
    multiscale_polydispersity_tutorial.ipynb notebook

    """
    seed = 1
    rng = np.random.RandomState([seed])

    # sphere simulation
    wavelengths = sc.Quantity(np.arange(400., 801.,10),'nm')

    # Geometric properties of the sample
    num_diams = 3

    sphere_boundary_diam_mean = sc.Quantity(10,'um')
    pdi = 0.2
    particle_radius = sc.Quantity(160,'nm')
    volume_fraction_bulk = sc.Quantity(0.63,'')
    volume_fraction_particles = sc.Quantity(0.55, '')
    bulk_thickness = sc.Quantity('50 um')
    boundary = 'sphere'
    boundary_bulk = 'film'

    n_particle = ri.n('vacuum', wavelengths)
    n_matrix = ri.n('polystyrene', wavelengths) + 2e-5*1j
    n_matrix_bulk = ri.n('vacuum', wavelengths)
    n_medium = ri.n('vacuum', wavelengths)

    ntrajectories = 500
    nevents = 300
    ntrajectories_bulk = 1000
    nevents_bulk = 300

    # calculate diameter list to sample from
    sphere_boundary_diameters = pfs.calc_diam_list(num_diams,
                                                   sphere_boundary_diam_mean,
                                                   pdi, plot = False,
                                                   equal_spacing = False)

    # test that sphere boundaries are what we expect
    sbd_expected = sc.Quantity(np.array([7.470784641068447, 9.595993322203672,
                                         12.101836393989982]), 'um')

    assert_almost_equal(sphere_boundary_diameters.magnitude,
                        sbd_expected.magnitude)

    reflectance_sphere = np.zeros(wavelengths.size)

    p_bulk = np.zeros((sphere_boundary_diameters.size, wavelengths.size, 200))
    mu_scat_bulk = sc.Quantity(np.zeros((sphere_boundary_diameters.size,
                                         wavelengths.size)),'1/um')
    mu_abs_bulk = sc.Quantity(np.zeros((sphere_boundary_diameters.size,
                                        wavelengths.size)),'1/um')

    for j in range(sphere_boundary_diameters.size):
        for i in range(wavelengths.size):

            n_sample = ri.n_eff(n_particle[i], n_matrix[i],
                                volume_fraction_particles)

            p, mu_scat, mu_abs = mc.calc_scat(particle_radius, n_particle[i],
                                              n_sample,
                                              volume_fraction_particles,
                                              wavelengths[i])

            # Initialize the trajectories
            r0, k0, W0 = mc.initialize(nevents, ntrajectories,
                                       n_matrix_bulk[i], n_sample,
                                       boundary, sample_diameter =
                                       sphere_boundary_diameters[j], rng=rng)
            r0 = sc.Quantity(r0, 'um')
            k0 = sc.Quantity(k0, '')
            W0 = sc.Quantity(W0, '')

            # Create trajectories object
            trajectories = mc.Trajectory(r0, k0, W0)

            # Generate a matrix of all the randomly sampled angles first
            sintheta, costheta, sinphi, cosphi, _, _ = \
                mc.sample_angles(nevents, ntrajectories, p, rng=rng)

            # Create step size distribution
            step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

            # Run photons
            trajectories.absorb(mu_abs, step)
            trajectories.scatter(sintheta, costheta, sinphi, cosphi)
            trajectories.move(step)

            # Calculate reflection and transmition
            with pytest.warns(UserWarning):
                (refl_indices,
                 trans_indices,
                 _, _, _,
                 refl_per_traj, trans_per_traj,
                 _,_,_,_,
                 reflectance_sphere[i],
                 _,_, norm_refl, norm_trans) = \
                     det.calc_refl_trans(trajectories,
                                         sphere_boundary_diameters[j],
                                         n_matrix_bulk[i], n_sample, boundary,
                                         run_fresnel_traj = False,
                                         return_extra = True)


            ### Calculate phase function and lscat ###
            p_bulk[j,i,:], mu_scat_bulk[j,i], mu_abs_bulk[j,i] = \
                pfs.calc_scat_bulk(refl_per_traj, trans_per_traj, refl_indices,
                                   trans_indices, norm_refl, norm_trans,
                                   volume_fraction_bulk,
                                   sphere_boundary_diameters[j],
                                   n_matrix_bulk[i], wavelengths[i],
                                   plot=False, phi_dependent=False)

    # sample
    # This will raise a warning from Pint -- need to refactor function
    with pytest.warns(UnitStrippedWarning):
        sphere_diams_sampled = pfs.sample_diams(pdi,
                                                sphere_boundary_diameters,
                                                sphere_boundary_diam_mean,
                                                ntrajectories_bulk,
                                                nevents_bulk, rng=rng)

    # test that the number of samples for each diameter matches what is
    # in the notebook
    num_samples = np.unique(sphere_diams_sampled,
                            return_counts=True)[1]
    num_samples_expected = np.array([74692, 150504, 74804])
    assert_equal(num_samples, num_samples_expected)

    reflectance_bulk_poly = np.zeros(wavelengths.size)
    for i in range(wavelengths.size):
        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents_bulk, ntrajectories_bulk,
                                   n_medium[i], n_matrix_bulk[i],
                                   boundary_bulk, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        W0 = sc.Quantity(W0, '')
        k0 = sc.Quantity(k0, '')

        # Sample angles and calculate step size based on sampled radii
        sintheta, costheta, sinphi, cosphi, step, _, _ = \
            pfs.sample_angles_step_poly(nevents_bulk, ntrajectories_bulk,
                                        p_bulk[:,i,:],
                                        sphere_diams_sampled,
                                        mu_scat_bulk[:,i],
                                        param_list =
                                        sphere_boundary_diameters, rng=rng)

        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

        # Run photons. Note: polydisperse absorption does not currently work in
        # the bulk so we arbitrarily use index 0, assuming that all scattering
        # events have the same amount of absorption
        trajectories.absorb(mu_abs_bulk[0,i], step)
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        # calculate reflectance
        with pytest.warns(UserWarning):
            reflectance_bulk_poly[i], transmittance = \
                det.calc_refl_trans(trajectories, bulk_thickness, n_medium[i],
                                    n_matrix_bulk[i], boundary_bulk)

    # test reflectance from the bulk polydisperse sample
    R_expected = [0.5896236932355958, 0.5960565958801791, 0.543160195730125,
                  0.5536441470775867, 0.6146460242667455, 0.5600786954281792,
                  0.5591015346345805, 0.537326061668949, 0.5343714085229028,
                  0.5556209426054656, 0.5573765922734981, 0.530893284940807,
                  0.5751906767082536, 0.5435708031809252, 0.5447576605975664,
                  0.6196425942793337, 0.5964867735583548, 0.5797280032336561,
                  0.579181518844517, 0.6207895333963239, 0.6464214348887908,
                  0.5881764642292608, 0.6696133846534973, 0.6702995614740612,
                  0.6661892161362656, 0.6803786045588573, 0.7288906889813367,
                  0.6867392056800635, 0.6979365357395395, 0.6865719401864127,
                  0.6328918782573599, 0.6460554946539726, 0.587383967418907,
                  0.571182343621333, 0.5673227206390983, 0.5310336137501601,
                  0.5257883007967756, 0.42045495158959834, 0.41978848175878514,
                  0.41228258651823824, 0.3873631350450162]
    assert_almost_equal(reflectance_bulk_poly, R_expected)

@pytest.mark.slow
def test_multiscale_color_mixing_mc():
    """
    Tests whether the reflectance is what we expect from a simulation on a bulk
    collection of two types of spheres with different internal particle sizes.
    The parameters, setup, and expected values come from the
    multiscale_color_mixing_tutorial.ipynb notebook

    """
    seed = 1
    rng = np.random.RandomState([seed])

    # Properties of the source
    wavelengths = sc.Quantity(np.arange(400., 801.,10),'nm')

    # Geometric properties of the sample
    # radii of the two species of particles
    particle_radii = sc.Quantity([130, 160],'nm')
    # volume fraction of the spheres in the bulk film
    volume_fraction_bulk = sc.Quantity(0.63,'')
    # volume fraction of the particles in the sphere boundary
    volume_fraction_particles = sc.Quantity(0.55, '')
    # diameter of sphere boundary in bulk film
    sphere_boundary_diameter = sc.Quantity('10 um')
    bulk_thickness = sc.Quantity('50 um')
    # geometry of sample
    boundary = 'sphere'
    # geometry of the bulk sample
    boundary_bulk = 'film'

    # Refractive indices
    n_particle = ri.n('vacuum', wavelengths)
    n_matrix = ri.n('polystyrene', wavelengths) + 2e-5*1j
    n_matrix_bulk = ri.n('vacuum', wavelengths)
    n_medium = ri.n('vacuum', wavelengths)

    # Monte Carlo parameters
    ntrajectories = 2000
    nevents = 300
    ntrajectories_bulk = 2000
    nevents_bulk = 300

    p_bulk = np.zeros((particle_radii.size, wavelengths.size, 200))

    reflectance_sphere = np.zeros(wavelengths.size)
    mu_scat_bulk = sc.Quantity(np.zeros((particle_radii.size,
                                         wavelengths.size)),'1/um')
    mu_abs_bulk = sc.Quantity(np.zeros((particle_radii.size,
                                        wavelengths.size)),'1/um')


    for j in range(particle_radii.size):
        for i in range(wavelengths.size):
            n_sample = ri.n_eff(n_particle[i], n_matrix[i],
                                volume_fraction_particles)

            p, mu_scat, mu_abs = mc.calc_scat(particle_radii[j], n_particle[i],
                                              n_sample,
                                              volume_fraction_particles,
                                              wavelengths[i])

            r0, k0, W0 = mc.initialize(nevents, ntrajectories,
                                       n_matrix_bulk[i], n_sample, boundary,
                                       sample_diameter =
                                       sphere_boundary_diameter, rng=rng)
            r0 = sc.Quantity(r0, 'um')
            k0 = sc.Quantity(k0, '')
            W0 = sc.Quantity(W0, '')

            trajectories = mc.Trajectory(r0, k0, W0)

            sintheta, costheta, sinphi, cosphi, _, _ = \
                mc.sample_angles(nevents, ntrajectories, p, rng=rng)


            step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

            trajectories.absorb(mu_abs, step)
            trajectories.scatter(sintheta, costheta, sinphi, cosphi)
            trajectories.move(step)

            with pytest.warns(UserWarning):
                (refl_indices,
                 trans_indices,
                 _, _, _,
                 refl_per_traj, trans_per_traj,
                 _,_,_,_,
                 reflectance_sphere[i],
                 _,_, norm_refl, norm_trans) = \
                     det.calc_refl_trans(trajectories,
                                      sphere_boundary_diameter,
                                      n_matrix_bulk[i], n_sample, boundary,
                                      run_fresnel_traj = False,
                                      return_extra = True)

            p_bulk[j,i,:], mu_scat_bulk[j,i], mu_abs_bulk[j,i] = \
                pfs.calc_scat_bulk(refl_per_traj, trans_per_traj, refl_indices,
                                   trans_indices, norm_refl, norm_trans,
                                   volume_fraction_bulk,
                                   sphere_boundary_diameter, n_matrix_bulk[i],
                                   wavelengths[i], plot=False,
                                   phi_dependent=False)

    # sample
    prob = np.array([0.5, 0.5]) # fraction of each sphere color type
    sphere_type_sampled = pfs.sample_concentration(prob, ntrajectories_bulk,
                                                   nevents_bulk, rng=rng)

    # test that the number of samples for each sphere type matches what is
    # in the notebook
    num_samples = np.unique(sphere_type_sampled,
                            return_counts=True)[1]
    num_samples_expected = np.array([299530, 300470])
    assert_equal(num_samples, num_samples_expected)

    # calculate reflectance of bulk film with spheres of two different colors
    reflectance_bulk_mix = np.zeros(wavelengths.size)
    for i in range(wavelengths.size):
        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents_bulk, ntrajectories_bulk,
                                   n_medium[i], n_matrix_bulk[i],
                                   boundary_bulk, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        W0 = sc.Quantity(W0, '')
        k0 = sc.Quantity(k0, '')

        (sintheta, costheta, sinphi, cosphi, step, _, _) = \
            pfs.sample_angles_step_poly(nevents_bulk, ntrajectories_bulk,
                                        p_bulk[:,i,:],
                                        sphere_type_sampled,
                                        mu_scat_bulk[:,i],
                                        rng=rng)


        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

        # Run photons
        # Note: we assume that all scattering events
        # have the same amount of absorption
        trajectories.absorb(mu_abs_bulk[0,i], step)
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        # calculate reflectance
        with pytest.warns(UserWarning):
            reflectance_bulk_mix[i], transmittance = \
                det.calc_refl_trans(trajectories, bulk_thickness, n_medium[i],
                                    n_matrix_bulk[i], boundary_bulk)

    R_expected = [0.5826801822412575, 0.5702215184018711, 0.5731687923054422,
                  0.5766088842163823, 0.6053588610189652, 0.5845773357414805,
                  0.5779789355691176, 0.6076395346359109, 0.5943424417671181,
                  0.6185563799423084, 0.637205901773559, 0.6530679940741657,
                  0.6804828781523293, 0.6865780901198721, 0.69671357688658,
                  0.7220475454635316, 0.7067427468589211, 0.6938059843106995,
                  0.6917673690258764, 0.6729278013509614, 0.6760325991355923,
                  0.6297464297708327, 0.6028277805948036, 0.5825058356257393,
                  0.5512018280412787, 0.5627401564798604, 0.5292611114058134,
                  0.5286803240372856, 0.5171994032545681, 0.4890223620686648,
                  0.45994954424484724, 0.4406385043576606, 0.4188602185212018,
                  0.4230511974137862, 0.35669602730162475, 0.3509835670828349,
                  0.3116737424739726, 0.28723857026233096, 0.27742488907293594,
                  0.251790453205595, 0.24318888896750582]

    assert_almost_equal(reflectance_bulk_mix, R_expected)
