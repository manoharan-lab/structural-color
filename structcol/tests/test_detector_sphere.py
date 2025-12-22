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
import pytest

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
assembly_radius = 5
volume_fraction = 0.5
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')
wavelen = sc.Quantity('400.0 nm')
index_particle = sc.ConstantIndex(1.5)
index_matrix = sc.ConstantIndex(1.0)

sphere = sc.Sphere(index_particle, radius)

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_calc_refl_trans():
    # this test should give deterministic results
    index_small_n = sc.ConstantIndex(1.0)
    index_large_n = sc.ConstantIndex(2.0)
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
    # other dimensions needed to specify trajectories Dataset
    expanded_dims = {"wavelength": [wavelen.to_preferred().magnitude],
                     "volume_fraction": [volume_fraction]}
    trajectories = trajectories.expand_dims(expanded_dims)

    # set up a dummy simulation and insert the trajectories
    # (index match particle so that effective index is same as matrix)
    particle = sc.Sphere(index_small_n, radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_small_n,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, "sphere",
                        sample_diameter = assembly_radius*2)
    sim.traj = trajectories

    refl, trans = det.calc_refl_trans(sim, assembly_radius)
    expected_trans_array = np.array([0., .3, 0.25, 0])/ntrajectories #calculated manually
    expected_refl_array = np.array([.7, 0., .25, 0.])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test fresnel as well
    # (need to change index of sample to be higher than that of medium)
    particle = sc.Sphere(index_large_n, radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_large_n,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, "sphere",
                        sample_diameter = assembly_radius*2)
    sim.traj = trajectories

    refl, trans = det.calc_refl_trans(sim, assembly_radius)
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
    trajectories = trajectories.expand_dims(expanded_dims)

    # (go back to small index for matrix and particle)
    particle = sc.Sphere(index_small_n, radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_small_n,
                                 index_medium)
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, "sphere",
                        sample_diameter = assembly_radius*2)
    sim.traj = trajectories

    refl, trans= det.calc_refl_trans(sim, assembly_radius)
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
    trajectories = trajectories.expand_dims(expanded_dims)
    sim.traj = trajectories

    # ignore warning that too many trajectories did not exit the sample
    with pytest.warns(UserWarning, match = "Increase Nevents"):
        refl, trans = det.calc_refl_trans(sim, assembly_radius,
                                          run_fresnel_traj=True)
    # since the tir=True reruns the stuck trajectory, we don't know whether it will end up reflected or transmitted
    # all we can know is that the end refl + trans > 0.99
    assert_almost_equal(refl + trans, xr.ones_like(refl))

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

    indices = xr.DataArray(np.array([1,1,1,1], dtype=float),
                           coords = {"trajectory": range(ntrajectories)})
    thetas, _ = det.get_angles(indices, 'sphere', trajectories, assembly_radius,
                               init_dir = 1)

    # indices are all 1, meaning to look at the 0th values of the k array
    assert_almost_equal(np.sum(thetas), 0)

def test_index_match():
    ntrajectories = 2
    nevents = 3
    wavelen = sc.Quantity('600.0 nm')
    radius = sc.Quantity('0.140 um')
    microsphere_radius = sc.Quantity('10.0 um')
    volume_fraction = sc.Quantity(0.55,'')
    index_particle = sc.ConstantIndex(1.6)
    index_matrix = sc.ConstantIndex(1.6)
    index_medium = sc.ConstantIndex(1.0)

    seed = 1
    rng = np.random.RandomState([seed])
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
                        sample_diameter = microsphere_radius*2, rng=rng)
    trajectories_sphere = xr.Dataset({"position": r0_sphere,
                                      "direction": k0_sphere,
                                      "weight": W0_sphere})
    expanded_dims = {"wavelength": [wavelen.to_preferred().magnitude],
                     "volume_fraction": [volume_fraction]}
    trajectories_sphere = trajectories_sphere.expand_dims(expanded_dims)
    sim.traj = trajectories_sphere

    # now run the simulation
    sim.run()

    # calculate reflectance
    refl_sphere, trans = det.calc_refl_trans(sim, microsphere_radius,
                                             run_fresnel_traj = True,
                                             max_stuck = 0.0001)

    # calculated by hand from fresnel infinite sum
    refl_fresnel_int = 0.053 # calculated by hand
    refl_exact = refl_fresnel_int + (1-refl_fresnel_int)**2*refl_fresnel_int/(1-refl_fresnel_int**2)

    # under index-matched conditions, the step sizes are huge (bigger than the
    # sample size), and the light is scattered into the forward direction. As a
    # result, the reflectance is essentially deterministic, even though the
    # seed is not set for the random number generator.
    assert_almost_equal(refl_sphere, refl_exact*xr.ones_like(refl_sphere),
                        decimal=3)

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
    index_particle = sc.ConstantIndex(1.54)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    boundary = 'sphere'

    sphere = sc.Sphere(index_particle, radius)
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Initialize and run the simulation for a sphere
    sim = mc.Simulation(model, wavelen, nevents, ntrajectories, boundary,
                        plot_initial = False,
                        sample_diameter = assembly_diameter,
                        spot_size = assembly_diameter,
                        rng=rng)
    sim.run()

    # Calculate reflectance and transmittance without fresnel reflected
    # trajectories
    R, T = det.calc_refl_trans(sim, assembly_diameter, plot_exits = False)

    R_expected = 0.2679582782715561
    T_expected = 0.732041721728444

    assert_almost_equal(R, R_expected)
    assert_almost_equal(T, T_expected)

    # test with Fresnel reflections
    # Calculate reflectance and transmittance
    # (need to pass rng here because run_fresnel_traj=True will run additional
    # MC simulations)
    R, T = det.calc_refl_trans(sim, assembly_diameter, run_fresnel_traj = True,
                               rng=rng)

    R_expected = 0.2700452674098231
    T_expected = 0.7299547325901768

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

    # loop through wavelengths
    for i in range(wavelengths.size):
        # Initialize and run the simulation
        sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                            boundary,
                            sample_diameter = sphere_boundary_diameter, rng=rng)
        sim.run()

        # Calculate reflection and transmission
        (refl_indices, trans_indices, stuck_indices, tir_indices,
         _, _, _,
         refl_per_traj, trans_per_traj,
         _,_,_,_,
         reflectance,
         _, norm_refl, norm_trans) = \
             det.calc_refl_trans(sim, sphere_boundary_diameter,
                                 run_fresnel_traj = False,
                                 return_extra = True)

        # until phase_func_sphere is refactored to use xarray, convert to numpy
        reflectance_sphere[i] = reflectance.to_numpy().squeeze()
        refl_indices = refl_indices.sortby("trajectory").to_numpy().squeeze()
        trans_indices = trans_indices.sortby("trajectory").to_numpy().squeeze()
        refl_per_traj = refl_per_traj.sortby("trajectory").to_numpy().squeeze()
        trans_per_traj = (trans_per_traj.sortby("trajectory").to_numpy()
                          .squeeze())
        # need to expand norm_refl and norm_trans to have all trajectories as
        # coordinates
        alltraj_comp = xr.DataArray(np.ones((ntrajectories, 3)),
                                    coords={"trajectory": range(ntrajectories),
                                            "component": ["x", "y", "z"]})
        norm_refl = (norm_refl.reindex_like(alltraj_comp, fill_value=0.0)
                     .to_numpy().squeeze().transpose())
        norm_trans = (norm_trans.reindex_like(alltraj_comp, fill_value=0.0)
                      .to_numpy().squeeze().transpose())

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
    R_sphere_expected = [0.3850260204091323, 0.3848878202884299,
                         0.4104691302724734, 0.3948270677219652,
                         0.3905764366623931, 0.3235798183714219,
                         0.3015967879076175, 0.2329292966694058,
                         0.1692681065622285, 0.1306104224138474,
                         0.1118669239670618, 0.084484609902532,
                         0.0741408815754614, 0.0548696141062594,
                         0.0462914110098221, 0.0403999685290333,
                         0.0410868379102632, 0.0315069146852409,
                         0.0286038116595564, 0.0242746650071694,
                         0.0256664481709116, 0.0220462722034538,
                         0.0203787356830617, 0.0168983257233101,
                         0.0191883697174941, 0.0169402797755538,
                         0.0154173648586895]

    # phase function at backscattering
    pfb_expected = [0.0035379732380342, 0.0034540136167342,
                    0.002990476722742, 0.0028489844355928,
                    0.0027948912420616, 0.0025022909596577,
                    0.0026212345292497, 0.0028929761552317,
                    0.0031231656984502, 0.0037472713414806,
                    0.0033216155642621, 0.0037019785945371,
                    0.0040226567868681, 0.0038584403119186,
                    0.0038640289772288, 0.0041905715122369,
                    0.0039952025122601, 0.0042789931680442,
                    0.0043071709320683, 0.004630501585259,
                    0.0045758294841199, 0.004546116263575,
                    0.0047622282893478, 0.0046233307855247,
                    0.0047808140756161, 0.0048249082594418,
                    0.0047717541318914]

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
        reflectance, transmittance = \
            det.calc_refl_trans(sim, bulk_thickness)

        reflectance_bulk[i] = reflectance.to_numpy().squeeze()

    # these numbers look a little strange (multiply them by the number of
    # trajectories, and they all become integers). That's because there's no
    # absorption in this simulation, so every trajectory has a weight of 1 when
    # it exits the sample.  The reflectance is then an integer number of
    # trajectories divided by the number of trajectories.

    R_bulk_expected = [0.74500000013005, 0.7645000001109206,
                       0.7780000000985678, 0.7500000001249998,
                       0.7450000001300501, 0.743000000132098,
                       0.7130000001647379, 0.642000000256328,
                       0.5565000003933844, 0.5090000004821619,
                       0.4545000005951405, 0.372000000788768,
                       0.3435000008619846, 0.2720000010599681,
                       0.2285000011904246, 0.2220000012105681,
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

    for j in range(sphere_boundary_diameters.size):
        for i in range(wavelengths.size):
            # Initialize and run the simulation
            sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                                boundary,
                                sample_diameter = sphere_boundary_diameters[j],
                                rng=rng)
            sim.run()

            # Calculate reflection and transmition
            (refl_indices, trans_indices, stuck_indices, tir_indices,
             _, _, _,
             refl_per_traj, trans_per_traj,
             _,_,_,_,
             reflectance,
             _, norm_refl, norm_trans) = \
                 det.calc_refl_trans(sim,
                                     sphere_boundary_diameters[j],
                                     run_fresnel_traj = False,
                                     return_extra = True)

            # until phase_func_sphere is refactored to use xarray, convert to numpy
            reflectance_sphere[i] = reflectance.to_numpy().squeeze()
            refl_indices = (refl_indices.sortby("trajectory").to_numpy()
                            .squeeze())
            trans_indices = (trans_indices.sortby("trajectory").to_numpy()
                             .squeeze())
            refl_per_traj = (refl_per_traj.sortby("trajectory").to_numpy()
                             .squeeze())
            trans_per_traj = (trans_per_traj.sortby("trajectory").to_numpy()
                              .squeeze())
            # need to expand norm_refl and norm_trans to have all trajectories as
            # coordinates
            alltraj_comp = xr.DataArray(np.ones((ntrajectories, 3)),
                                        coords={"trajectory":
                                                range(ntrajectories),
                                                "component": ["x", "y", "z"]})
            norm_refl = (norm_refl.reindex_like(alltraj_comp, fill_value=0.0)
                         .to_numpy().squeeze().transpose())
            norm_trans = (norm_trans.reindex_like(alltraj_comp, fill_value=0.0)
                          .to_numpy().squeeze().transpose())

            ### Calculate phase function and lscat ###
            p_bulk[j,i,:], mu_scat_bulk[j,i], mu_abs_bulk[j,i] = \
                pfs.calc_scat_bulk(refl_per_traj, trans_per_traj, refl_indices,
                                   trans_indices, norm_refl, norm_trans,
                                   volume_fraction_bulk,
                                   sphere_boundary_diameters[j],
                                   n_matrix_bulk[i], wavelengths[i],
                                   plot=False, phi_dependent=False)

    # sample
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
        reflectance, transmittance = \
            det.calc_refl_trans(sim, bulk_thickness)

        reflectance_bulk_poly[i] = reflectance.to_numpy().squeeze()

    # test reflectance from the bulk polydisperse sample
    R_expected = [0.5896400063098672, 0.5954498381410573, 0.5429987792670864,
                  0.5541401132650923, 0.6142581186225504, 0.5592671340151376,
                  0.5587022021677739, 0.5370964948250692, 0.5342409702280295,
                  0.5555635128876976, 0.55749625997851, 0.5300307556496392,
                  0.5754126678926142, 0.5437177636189243, 0.5448688621886826,
                  0.619404912238551, 0.5963559186102096, 0.5797277478977132,
                  0.5791441212522996, 0.6206196722616002, 0.6464214348942805,
                  0.5879934565077676, 0.6698986634567492, 0.6697680274118676,
                  0.6661892161427485, 0.6803886165686105, 0.7288906889824045,
                  0.6866712197879697, 0.6979365357413201, 0.6865719401909671,
                  0.6328918797151462, 0.6460554946590632, 0.5873824729873645,
                  0.5716065374111554, 0.5673227179839966, 0.5310336115306016,
                  0.5257883037969457, 0.4210454026749897, 0.4197884817583484,
                  0.4122825885825516, 0.3873631350660365]

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

        for i in range(wavelengths.size):
            sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                                boundary,
                                sample_diameter = sphere_boundary_diameter,
                                rng=rng)

            sintheta, costheta, sinphi, cosphi, _, _ = sim.sample_angles()

            step = sim.sample_step()

            sim.absorb(step)
            sim.scatter(sintheta, costheta, sinphi, cosphi)
            sim.move(step)

            (refl_indices, trans_indices, stuck_indices, tir_indices,
             _, _, _,
             refl_per_traj, trans_per_traj,
             _,_,_,_,
             reflectance,
             _, norm_refl, norm_trans) = \
                 det.calc_refl_trans(sim,
                                     sphere_boundary_diameter,
                                     run_fresnel_traj = False,
                                     return_extra = True)

            # until phase_func_sphere is refactored to use xarray, convert to numpy
            reflectance_sphere[i] = reflectance.to_numpy().squeeze()
            refl_indices = (refl_indices.sortby("trajectory").to_numpy()
                            .squeeze())
            trans_indices = (trans_indices.sortby("trajectory").to_numpy()
                             .squeeze())
            refl_per_traj = (refl_per_traj.sortby("trajectory").to_numpy()
                             .squeeze())
            trans_per_traj = (trans_per_traj.sortby("trajectory").to_numpy()
                              .squeeze())
            # need to expand norm_refl and norm_trans to have all trajectories as
            # coordinates
            alltraj_comp = xr.DataArray(np.ones((ntrajectories, 3)),
                                        coords={"trajectory":
                                                range(ntrajectories),
                                                "component": ["x", "y", "z"]})
            norm_refl = (norm_refl.reindex_like(alltraj_comp, fill_value=0.0)
                         .to_numpy().squeeze().transpose())
            norm_trans = (norm_trans.reindex_like(alltraj_comp, fill_value=0.0)
                          .to_numpy().squeeze().transpose())

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
        reflectance, transmittance = \
            det.calc_refl_trans(sim, bulk_thickness)

        reflectance_bulk_mix[i] = reflectance.to_numpy().squeeze()

    R_expected = [0.5822243679017965, 0.570507985912688, 0.5732372435793517,
                  0.5766096689394413, 0.6050485178180293, 0.5851506936930788,
                  0.577949881486997,  0.6081106760514416, 0.5942373320927175,
                  0.618883396417119,  0.6377375480459256, 0.6532212151884353,
                  0.6803755780002321, 0.6867687272143803, 0.6968432858657334,
                  0.7222004292152797, 0.7068808284626948, 0.6939474253116436,
                  0.6917395062405838, 0.6729278007474482, 0.6762740578312749,
                  0.6297738158347119, 0.6027418526132862, 0.5825143108016013,
                  0.551201828053193,  0.5627228732396158, 0.5292611113954446,
                  0.5286802691090797, 0.5171994033166856, 0.4892354376074173,
                  0.4603244880878474, 0.4406385044116344, 0.4188602185447609,
                  0.4230196754357246, 0.3566960225934557, 0.3509835630226965,
                  0.3119104024632695, 0.2872385723431669, 0.2774248890098253,
                  0.2517904562212582, 0.2431888889355178]

    assert_almost_equal(reflectance_bulk_mix, R_expected)
