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
import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from pint.errors import UnitStrippedWarning
import pytest

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
assembly_radius = 5
volume_fraction = 0.5
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')
wavelen = sc.Quantity('400.0 nm')
index_particle = sc.Index.constant(1.5)
index_matrix = sc.Index.constant(1.0)

sphere = sc.Sphere(index_particle, radius)

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_calc_refl_trans():
    # this test should give deterministic results
    index_small_n = sc.Index.constant(1.0)
    small_n = index_small_n(wavelen)
    index_large_n = sc.Index.constant(2.0)
    large_n = index_large_n(wavelen)
    index_medium = sc.index.vacuum

    # test absoprtion and stuck without fresnel
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]],
                     dtype=float)
    x_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
    y_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
    nevents = z_pos.shape[0] - 1
    ntrajectories = z_pos.shape[1]
    pos_coords = {"component": ["x", "y", "z"],
                  "event": range(nevents+1),
                  "trajectory": range(ntrajectories)}
    r0 = xr.DataArray([x_pos, y_pos, z_pos], coords=pos_coords)
    k0 = xr.zeros_like(r0.isel(event=slice(0, -1)))
    k0.loc["z"] = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])

    weights = xr.DataArray([[1., 1., 1., 1.],
                            [.8, .8, .9, .8],
                            [.7, .3, .7, 0],
                            [.1, .1, .5, 0]],
                           coords=r0.sel(component="x", drop=True).coords)
    trajectories = xr.Dataset({"position": r0,
                               "direction": k0,
                               "weight": weights})
    trajectories = mc.QtyTrajectory(trajectories)

    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, "sphere",
                        sample_diameter = assembly_radius*2)

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
    z_pos = np.array([[0,0,0,0],[1,1,14,12],[-1,11,2,11],[-2,12,4,12]],
                     dtype=float)
    r0 = xr.DataArray([x_pos, y_pos, z_pos], coords=pos_coords)
    trajectories = xr.Dataset({"position": r0,
                               "direction": k0,
                               "weight": weights})
    trajectories = mc.QtyTrajectory(trajectories)
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
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]],
                     dtype=float)
    r0 = xr.DataArray([x_pos, y_pos, z_pos], coords=pos_coords)
    weights = xr.ones_like(r0.sel(component="x", drop=True))
    trajectories = xr.Dataset({"position": r0,
                               "direction": k0,
                               "weight": weights})

    trajectories = mc.QtyTrajectory(trajectories)
    # Should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index
    with pytest.warns(UserWarning):
        refl, trans = det.calc_refl_trans(trajectories, assembly_radius,
                                          small_n, small_n, 'sphere', p=sim.p,
                                          mu_abs=sim.mu_abs,
                                          mu_scat=sim.mu_scat,
                                          run_fresnel_traj=True)
    # since the tir=True reruns the stuck trajectory, we don't know whether it will end up reflected or transmitted
    # all we can know is that the end refl + trans > 0.99
    assert_almost_equal(refl + trans, 1.)

def test_get_angles_sphere():
    nevents = 3
    ntrajectories = 4
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]],
                     dtype=float)
    x_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,-0,0,0]], dtype=float)
    y_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
    positions = xr.DataArray([x_pos, y_pos, z_pos],
                             coords = {"component": ["x", "y", "z"],
                                       "event": range(nevents + 1),
                                       "trajectory": range(ntrajectories)})

    kx = np.zeros((3,4), dtype=float)
    ky = np.zeros((3,4), dtype=float)
    kz = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]], dtype=float)
    directions = xr.DataArray([kx, ky, kz],
                              coords=positions.isel(event=slice(0, -1)).coords)

    weights = xr.DataArray(np.ones((4, 4)),
                           coords =
                           positions.sel(component="x", drop=True).coords)

    trajectories = xr.Dataset({"position": positions,
                               "direction": directions,
                               "weight": weights})
    trajectories = mc.QtyTrajectory(trajectories)

    indices = np.array([1,1,1,1], dtype=float)
    thetas, _ = det.get_angles(indices, 'sphere', trajectories, assembly_radius,
                               init_dir = 1)
    assert_almost_equal(np.sum(thetas.magnitude), 0.)

def test_index_match():
    ntrajectories = 2
    nevents = 3
    wavelen = sc.Quantity('600.0 nm')
    radius = sc.Quantity('0.140 um')
    microsphere_radius = sc.Quantity('10.0 um')
    volume_fraction = sc.Quantity(0.55,'')
    index_particle = sc.Index.constant(1.6)
    index_matrix = sc.Index.constant(1.6)
    index_sample = index_matrix
    n_sample = index_sample(wavelen)
    index_medium = sc.Index.constant(1.0)
    n_medium = index_medium(wavelen)

    sphere = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # initialize all at center top edge of the sphere going down
    r0_sphere = xr.DataArray(np.zeros((3, nevents+1, ntrajectories)),
                             coords = {"component": ["x", "y", "z"],
                                       "event": range(nevents+1),
                                       "trajectory": range(ntrajectories)})
    k0_sphere = xr.zeros_like(r0_sphere.isel(event=slice(0, -1)))
    k0_sphere[2,0,:] = 1
    W0_sphere = xr.ones_like(r0_sphere.sel(component="x", drop=True))

    # make dummy simulation object and replace trajectories in the object with
    # the ones that we've set up
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, "sphere",
                        sample_diameter = microsphere_radius*2)
    trajectories_sphere = xr.Dataset({"position": r0_sphere,
                                      "direction": k0_sphere,
                                      "weight": W0_sphere})
    sim.traj = trajectories_sphere

    # now run the simulation
    sim.run()

    trajectories_sphere = mc.QtyTrajectory(sim.traj)

    # calculate reflectance
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        refl_sphere, trans = det.calc_refl_trans(trajectories_sphere,
                                                 microsphere_radius,
                                                 n_medium, n_sample,
                                                 'sphere', p=sim.p,
                                                 mu_abs=sim.mu_abs,
                                                 mu_scat=sim.mu_scat,
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
    index_particle = sc.Index.constant(1.54)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelen)
    boundary = 'sphere'

    sphere = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    n_sample = model.index_external(wavelen)

    # Initialize and run the simulation for a sphere
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary,
                        plot_initial = False,
                        sample_diameter = assembly_diameter,
                        spot_size = assembly_diameter,
                        rng=rng)
    sim.run()

    # Calculate reflectance and transmittance
    # The default value of run_tir is True, so you must set it to False to
    # exclude the fresnel reflected trajectories.
    trajectories = mc.QtyTrajectory(sim.traj)
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
                                   mu_abs=sim.mu_abs, mu_scat=sim.mu_scat,
                                   p=sim.p, rng=rng)

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
    volume_fraction_particles = 0.5
    volume_fraction_bulk = 0.55
    sphere_boundary_diameter = sc.Quantity(10,'um')
    bulk_thickness = sc.Quantity('50 um')
    boundary = 'sphere'
    boundary_bulk = 'film'

    # Refractive indices
    index_particle = sc.index.vacuum
    index_matrix = sc.index.fused_silica + 9e-4*1j
    index_matrix_bulk = sc.index.vacuum
    n_matrix_bulk = index_matrix_bulk(wavelengths)
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelengths)

    particle = sc.Sphere(index_particle, particle_radius)

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

    # set up scattering model
    model = sc.model.HardSpheres(particle, volume_fraction_particles,
                                 index_matrix, index_medium)
    n_sample = model.index_external(wavelengths)

    # loop through wavelengths
    for i in range(wavelengths.size):
        n_s = n_sample.isel(wavelength=[i])
        n_m = n_matrix_bulk.isel(wavelength=[i])

        # Initialize and run the simulation
        sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                            boundary,
                            sample_diameter = sphere_boundary_diameter, rng=rng)
        sim.run()

        # Calculate reflection and transmission
        trajectories = mc.QtyTrajectory(sim.traj)
        with pytest.warns(UserWarning):
            (refl_indices,
             trans_indices,
             _, _, _,
             refl_per_traj, trans_per_traj,
             _,_,_,_,
             reflectance_sphere[i],
             _,_, norm_refl, norm_trans) = \
                 det.calc_refl_trans(trajectories, sphere_boundary_diameter,
                                     n_m, n_s, boundary, p=sim.p,
                                     mu_abs=sim.mu_abs, mu_scat=sim.mu_scat,
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
    R_sphere_expected = [0.3812401406637523, 0.38351119595635,
                         0.4094781139993421, 0.3924474851210314,
                         0.3878769163474509, 0.321740994446611,
                         0.2990222289692781, 0.2329957735928287,
                         0.1709254702427257, 0.1292272328045406,
                         0.1118154155458773, 0.0841900071877681,
                         0.0743503711231209, 0.0544419333918082,
                         0.046680858210779, 0.040394125718838,
                         0.0408400680512956, 0.0308786667172062,
                         0.0282601849477853, 0.0241014657525225,
                         0.0256206692964288, 0.0222009291224357,
                         0.0205443700789016, 0.0171582035877526,
                         0.0189605801412236, 0.0166402418487938,
                         0.0152415209035317]

    # phase function at backscattering
    pfb_expected = [0.0035377584848251, 0.0034539363376863,
                    0.0029906602771838, 0.0028491683710419,
                    0.0027946840935744, 0.0025021756009958,
                    0.0026208964409995, 0.0028927995320099,
                    0.0031233124855259, 0.003746697208146,
                    0.0033216501053297, 0.0037017486697866,
                    0.0040226876546305, 0.0038580857341182,
                    0.0038643794666177, 0.0041903033706962,
                    0.0039947817173421, 0.0042786565605497,
                    0.004307127023794,  0.0046300961527623,
                    0.0045755010377515, 0.004546256380485,
                    0.0047626254815983, 0.0046230800603827,
                    0.004780658532144,  0.0048247983220431,
                    0.0047715812804799]

    assert_allclose(reflectance_sphere, R_sphere_expected)
    assert_allclose(p_bulk[:, 100], pfb_expected)

    # now look at bulk film
    # initialize some quantities we want to save as a function of wavelength
    reflectance_bulk = np.zeros(wavelengths.size)
    # particle doesn't matter here but is needed to set up model object
    dummy_particle = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(dummy_particle, volume_fraction_particles,
                                 index_matrix_bulk, index_medium)
    for i in range(wavelengths.size):
        n_med = n_medium.isel(wavelength=[i])
        n_mat = n_matrix_bulk.isel(wavelength=[i])
        # Initialize the simulation
        sim = mc.Simulation(model, wavelengths[i], nevents_bulk,
                            ntrajectories_bulk, boundary_bulk, rng=rng)

        # insert scattering quantities calculated for bulk system into object
        sim.p = p_bulk[i, :]
        sim.mu_scat = mu_scat_bulk[i]
        sim.mu_abs = mu_abs_bulk[i]

        # TODO: change the below to include absorption (using sim.run()). Test
        # values will need to be updated

        # Sample angles
        sintheta, costheta, sinphi, cosphi, _, _= sim.sample_angles()

        # Calculate step size
        step = sim.sample_step()

        # Run photons
        sim.scatter(sintheta, costheta, sinphi, cosphi)
        sim.move(step)

        # calculate bulk reflectance
        trajectories = mc.QtyTrajectory(sim.traj)
        with pytest.warns(UserWarning):
            reflectance_bulk[i], transmittance = \
                det.calc_refl_trans(trajectories, bulk_thickness, n_med,
                                    n_mat, boundary_bulk)

    # these numbers look a little strange (multiply them by the number of
    # trajectories, and they all become integers). That's because there's no
    # absorption in this simulation, so every trajectory has a weight of 1 when
    # it exits the sample.  The reflectance is then an integer number of
    # trajectories divided by the number of trajectories.

    R_bulk_expected = [0.74500000013005, 0.7640000001113922,
                       0.7780000000985678, 0.7505000001245004,
                       0.7450000001300501, 0.743000000132098,
                       0.7130000001647379, 0.642000000256328,
                       0.5565000003933844, 0.5090000004821619,
                       0.4540000005962321, 0.372000000788768,
                       0.3430000008632981, 0.2720000010599681,
                       0.2285000011904246, 0.2215000012121246,
                       0.2130000012387382, 0.1550000014280501,
                       0.1480000014518081, 0.1320000015068481,
                       0.1490000014484022, 0.1175000015576127,
                       0.1255000015295006, 0.0845000016762806,
                       0.1055000016002607, 0.0825000016836126,
                       0.0810000016891221]

    assert_allclose(reflectance_bulk, R_bulk_expected)

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
    wavelengths = sc.Quantity(np.arange(400., 801., 10),'nm')

    # Geometric properties of the sample
    num_diams = 3

    sphere_boundary_diam_mean = sc.Quantity(10,'um')
    pdi = 0.2
    particle_radius = sc.Quantity(160,'nm')
    volume_fraction_bulk = 0.63
    volume_fraction_particles = 0.55
    bulk_thickness = sc.Quantity('50 um')
    boundary = 'sphere'
    boundary_bulk = 'film'

    index_particle = sc.index.vacuum
    index_matrix = sc.index.polystyrene + 2e-5*1j
    index_matrix_bulk = sc.index.vacuum
    n_matrix_bulk = index_matrix_bulk(wavelengths)
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelengths)

    particle = sc.Sphere(index_particle, particle_radius)

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

    # set up scattering model
    model = sc.model.HardSpheres(particle, volume_fraction_particles,
                                 index_matrix, index_medium)

    n_sample = model.index_external(wavelengths)

    for j in range(sphere_boundary_diameters.size):
        for i in range(wavelengths.size):
            n_m = n_matrix_bulk.isel(wavelength=[i])
            n_s = n_sample.isel(wavelength=[i])

            # Initialize and run the simulation
            sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                                boundary,
                                sample_diameter = sphere_boundary_diameters[j],
                                rng=rng)
            sim.run()

            # Calculate reflection and transmition
            trajectories = mc.QtyTrajectory(sim.traj)
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
                                         n_m, n_s, boundary,
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

    model = sc.model.HardSpheres(particle, volume_fraction_particles,
                                 index_matrix_bulk, index_medium)
    for i in range(wavelengths.size):
        n_med = n_medium.isel(wavelength=[i])
        n_mat = n_matrix_bulk.isel(wavelength=[i])
        # Initialize the simulation
        sim = mc.Simulation(model, wavelengths[i], nevents_bulk,
                            ntrajectories_bulk, boundary_bulk, rng=rng)

        # Sample angles and calculate step size based on sampled radii
        sintheta, costheta, sinphi, cosphi, step, _, _ = \
            pfs.sample_angles_step_poly(nevents_bulk, ntrajectories_bulk,
                                        p_bulk[:,i,:],
                                        sphere_diams_sampled,
                                        mu_scat_bulk[:,i],
                                        param_list =
                                        sphere_boundary_diameters, rng=rng)

        # insert scattering quantities calculated for bulk system into object
        sim.mu_abs = mu_abs_bulk[0, i]

        # Run photons. Note: polydisperse absorption does not currently work in
        # the bulk so we arbitrarily use index 0, assuming that all scattering
        # events have the same amount of absorption
        sim.absorb(step)
        sim.scatter(sintheta, costheta, sinphi, cosphi)
        sim.move(step)

        # calculate reflectance
        trajectories = mc.QtyTrajectory(sim.traj)
        with pytest.warns(UserWarning):
            reflectance_bulk_poly[i], transmittance = \
                det.calc_refl_trans(trajectories, bulk_thickness, n_med,
                                    n_mat, boundary_bulk)

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
    volume_fraction_bulk = 0.63
    # volume fraction of the particles in the sphere boundary
    volume_fraction_particles = 0.55
    # diameter of sphere boundary in bulk film
    sphere_boundary_diameter = sc.Quantity('10 um')
    bulk_thickness = sc.Quantity('50 um')
    # geometry of sample
    boundary = 'sphere'
    # geometry of the bulk sample
    boundary_bulk = 'film'

    # Refractive indices
    index_particle = sc.index.vacuum
    index_matrix = sc.index.polystyrene + 2e-5*1j
    index_matrix_bulk = sc.index.vacuum
    n_matrix_bulk = index_matrix_bulk(wavelengths)
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelengths)

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
        particle = sc.Sphere(index_particle, particle_radii[j])

        # set up scattering model
        model = sc.model.HardSpheres(particle, volume_fraction_particles,
                                     index_matrix, index_medium)
        n_sample_eff = model.index_external(wavelengths)

        for i in range(wavelengths.size):
            n_sample = n_sample_eff.isel(wavelength=[i])
            n_mat = n_matrix_bulk.isel(wavelength=[i])

            sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                                boundary,
                                sample_diameter = sphere_boundary_diameter,
                                rng=rng)

            sintheta, costheta, sinphi, cosphi, _, _ = sim.sample_angles()

            step = sim.sample_step()

            sim.absorb(step)
            sim.scatter(sintheta, costheta, sinphi, cosphi)
            sim.move(step)

            trajectories = mc.QtyTrajectory(sim.traj)
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
                                      n_mat, n_sample, boundary,
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
    # particle doesn't matter here but is needed to set up model object
    dummy_particle = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(dummy_particle, volume_fraction_particles,
                                 index_matrix_bulk, index_medium)
    for i in range(wavelengths.size):
        n_med = n_medium.isel(wavelength=[i])
        n_mat = n_matrix_bulk.isel(wavelength=[i])
        # Initialize the simulation
        sim = mc.Simulation(model, wavelengths[i], nevents_bulk,
                            ntrajectories_bulk, boundary_bulk, rng=rng)

        (sintheta, costheta, sinphi, cosphi, step, _, _) = \
            pfs.sample_angles_step_poly(nevents_bulk, ntrajectories_bulk,
                                        p_bulk[:,i,:],
                                        sphere_type_sampled,
                                        mu_scat_bulk[:,i],
                                        rng=rng)

        # insert scattering quantities calculated for bulk system into object
        sim.mu_abs = mu_abs_bulk[0, i]

        # Run photons
        # Note: we assume that all scattering events
        # have the same amount of absorption
        sim.absorb(step)
        sim.scatter(sintheta, costheta, sinphi, cosphi)
        sim.move(step)

        # calculate reflectance
        trajectories = mc.QtyTrajectory(sim.traj)
        with pytest.warns(UserWarning):
            reflectance_bulk_mix[i], transmittance = \
                det.calc_refl_trans(trajectories, bulk_thickness, n_med,
                                    n_mat, boundary_bulk)

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
