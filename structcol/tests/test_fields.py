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
Tests for the phase calculations in montecarlo model

.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>

"""

import structcol as sc
from .. import montecarlo as mc
from .. import detector as det
from .. import detector_polarization_phase as detp
import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal, assert_allclose
import pytest

def test_2pi_shift():
    # test that phase mod 2Pi is the same as phase.
    # This test should pass irrespective of the state of the random number
    # generator, so we do not need to explicitly specify a seed.

    # incident light wavelength
    wavelength = sc.Quantity('600.0 nm')

    # sample parameters
    radius = sc.Quantity('0.140 um')
    volume_fraction = 0.55
    n_imag = 2.1e-4
    index_particle = sc.index.polystyrene + n_imag*1j
    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelength)

    thickness = sc.Quantity('50.0 um')
    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 10
    nevents = 30

    # set up scattering model
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Initialize and run simulation
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories, boundary,
                        fields=True)
    sim.run(radius=radius, wavelength=wavelength)

    # calculate reflectance
    trajectories = sim.traj
    refl_trans_result = det.calc_refl_trans(sim, thickness,
                                            return_extra=True)

    refl_indices = refl_trans_result[0]
    refl_per_traj = refl_trans_result[3]
    reflectance_fields, _ = detp.calc_refl_phase_fields(trajectories,
                                                        refl_indices,
                                                        refl_per_traj)

    # now do mod 2pi
    trajectories["fields"] = trajectories.fields*np.exp(2*np.pi*1j)
    reflectance_fields_shift, _ = detp.calc_refl_phase_fields(trajectories,
                                                              refl_indices,
                                                              refl_per_traj)

    assert_almost_equal(reflectance_fields, reflectance_fields_shift,
                        decimal=15)


def test_intensity_coherent():
    # tests that the intensity of the summed fields correspond to the equation
    # for coherent light: Ix = E_x1^2 + E_x2^2 + 2E_x1*E_x2

    # this test isn't based on random values, so should produce deterministic
    # results.

    # construct 2 identical trajectories that exit at same event
    ntrajectories = 2
    nevents = 2
    z_pos = np.array([[0,0],[1,1],[-1,-1]])
    x_pos = np.zeros_like(z_pos)
    y_pos = np.zeros_like(z_pos)
    positions = xr.DataArray([x_pos, y_pos, z_pos],
                             coords = {"component": ["x", "y", "z"],
                                       "event": range(nevents+1),
                                       "trajectory": range(ntrajectories)})
    kz = np.array([[1,1],[-1,1]])
    directions = xr.DataArray([kz,kz,kz],
                              coords =
                              positions.isel(event=slice(0, -1)).coords)

    weights = xr.DataArray(np.array([[1, 1],[1, 1],[1, 1]]),
                           coords =
                           positions.sel(component="x", drop=True).coords)

    fields = xr.DataArray(np.zeros((3, nevents+1, ntrajectories),
                                   dtype = complex),
                          coords = positions.coords)
    fields[:,0,:] = 0.5
    fields[:,1,:] = 1
    fields[:,2,:] = 1.5

    trajectories = xr.Dataset({"position": positions,
                               "direction": directions,
                               "weight": weights,
                               "fields": fields})

    # calculate reflectance phase
    refl_per_traj = np.array([0.5, 0.5])
    refl_indices = np.array([2, 2])
    refl_phase, _ = detp.calc_refl_phase_fields(trajectories, refl_indices,
                                                refl_per_traj)
    intensity_incident = np.sum(trajectories.weight[0,:])
    intensity = refl_phase*intensity_incident

    # Calculate I = (E1 + E2)*(E1 + E2) = E1*E1 + E2*E2 + E1*E2 + E2*E1
    ev = 1
    field_x = np.sqrt(trajectories.weight[ev,:])*trajectories.fields[0,ev+1,:]
    field_y = np.sqrt(trajectories.weight[ev,:])*trajectories.fields[1,ev+1,:]
    field_z = np.sqrt(trajectories.weight[ev,:])*trajectories.fields[2,ev+1,:]
    intensity_x = (np.conj(field_x[0])*field_x[0]
                   + np.conj(field_x[1])*field_x[1]
                   + np.conj(field_x[0])*field_x[1]
                   + np.conj(field_x[1])*field_x[0])
    intensity_y = (np.conj(field_y[0])*field_y[0]
                   + np.conj(field_y[1])*field_y[1]
                   + np.conj(field_y[0])*field_y[1]
                   + np.conj(field_y[1])*field_y[0])
    intensity_z = (np.conj(field_z[0])*field_z[0]
                   + np.conj(field_z[1])*field_z[1]
                   + np.conj(field_z[0])*field_z[1]
                   + np.conj(field_z[1])*field_z[0])
    intensity_2 = intensity_x + intensity_y + intensity_z

    # compare values
    assert_almost_equal(intensity, intensity_2, decimal=15)

def test_pi_shift_zero():
    # tests if a pi shift leads to zero intensity. This test should produce a
    # deterministic result.

    # construct 2 trajectories with relative pi phase shift that exit at same
    # event
    ntrajectories = 2
    nevents = 2
    z_pos = np.array([[0,0],[1,1],[-1,-1]])
    x_pos = np.array([[0,0],[1,1],[-1,-1]])
    y_pos = np.zeros_like(x_pos)
    positions = xr.DataArray([x_pos, y_pos, z_pos],
                             coords = {"component": ["x", "y", "z"],
                                       "event": range(nevents+1),
                                       "trajectory": range(ntrajectories)})
    kz = np.array([[1,1],[-1,1]])
    directions = xr.DataArray([kz,kz,kz],
                              coords =
                              positions.isel(event=slice(0, -1)).coords)

    weights = xr.DataArray(np.array([[1, 1],[1, 1],[1, 1]]),
                           coords =
                           positions.sel(component="x", drop=True).coords)

    fields = xr.DataArray(np.zeros((3, nevents+1, ntrajectories),
                                   dtype = complex),
                          coords = positions.coords)

    fields[:,2,0] = 1
    fields[:,2,1] = np.exp(np.pi*1j)

    trajectories = xr.Dataset({"position": positions,
                               "direction": directions,
                               "weight": weights,
                               "fields": fields})

    # calculate reflectance phase
    refl_per_traj = np.array([0.5, 0.5])
    refl_indices = np.array([2, 2])
    refl_fields, _ = detp.calc_refl_phase_fields(trajectories, refl_indices,
                                                 refl_per_traj)

    # check whether reflectance phase is 0
    assert_almost_equal(refl_fields, 0, decimal=15)


def test_field_normalized():
    # calculate fields and directions

    # This test should pass regardless of the state of the random number
    # generator, so we do not need to specify an explicit seed.

    # incident light wavelength
    wavelength = sc.Quantity('600.0 nm')

    # sample parameters
    radius = sc.Quantity('0.140 um')
    volume_fraction = 0.55
    index_imag = sc.ConstantIndex(2.1e-4*1j)
    index_particle = sc.index.polystyrene + index_imag

    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum

    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 10
    nevents = 10

    # set up scattering model
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Initialize and run simulation
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories,
                        boundary, fields=True)
    sim.run(radius=radius, wavelength=wavelength)

    # take the dot product
    trajectories = sim.traj

    field_mag= np.sqrt(np.conj(trajectories.fields[0,:,:])
                       * trajectories.fields[0,:,:] +
                       np.conj(trajectories.fields[1,:,:])
                       * trajectories.fields[1,:,:] +
                       np.conj(trajectories.fields[2,:,:])
                       * trajectories.fields[2,:,:])

    assert_almost_equal(np.sum(field_mag)/(ntrajectories*(nevents+1)), 1,
                        decimal=15)

def test_field_perp_direction():
    # calculate fields and directions

    # This test should pass regardless of the state of the random number
    # generator, so we do not need to specify an explicit seed.

    # incident light wavelength
    wavelength = sc.Quantity('600.0 nm')

    # sample parameters
    radius = sc.Quantity('0.140 um')
    volume_fraction = 0.55
    index_imag = sc.ConstantIndex(2.1e-4*1j)
    index_particle = sc.index.polystyrene + index_imag

    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum

    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 10
    nevents = 10

    # set up scattering model
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Initialize and run simulation
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories, boundary,
                        fields=True)
    sim.run(radius=radius, wavelength=wavelength)

    # direction at event=n and field at event=n+1 should be orthogonal
    dot_prod = xr.dot(sim.traj.direction.shift(event=1, fill_value=0),
                      sim.traj.fields.sel(event=slice(1, None)),
                      dim="component")

    # equivalent code for numpy arrays
    # dot_prod = (trajectories.direction[0,:,:]*trajectories.fields[0,1:,:] +
    #            trajectories.direction[1,:,:]*trajectories.fields[1,1:,:] +
    #            trajectories.direction[2,:,:]*trajectories.fields[2,1:,:])

    assert_almost_equal(np.sum(dot_prod), 0., decimal=14)

@pytest.mark.slow
def test_field_reflectance_mc():
    """
    Tests whether the reflectance for the fields model is what we expect from a
    simulation on a film of particles. The parameters, setup, and
    expected values come from the fields_montecarlo_tutorial.ipynb notebook.
    """

    seed = 1
    rng = np.random.RandomState([seed])

    wavelength = sc.Quantity('600 nm')

    # sample parameters
    radius = sc.Quantity('0.140 um')
    volume_fraction = 0.55
    index_imag = sc.ConstantIndex(2.1e-4*1j)
    index_particle = sc.index.polystyrene + index_imag

    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum

    thickness = sc.Quantity('800 um')
    boundary = 'film'

    ntrajectories = 2000
    nevents = 300

    # set up scattering model
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    # Initialize and run simulation
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories, boundary,
                        coherent=False, fields=True, fine_roughness=0, rng=rng)
    sim.run(radius=radius, wavelength=wavelength)

    trajectories = sim.traj

    # calculate reflectance including phase
    refl_trans_result = det.calc_refl_trans(sim, thickness,
                                            return_extra=True)

    reflectance = refl_trans_result[11]
    refl_indices = refl_trans_result[0]
    refl_per_traj = refl_trans_result[3]

    refl_fields, _ = detp.calc_refl_phase_fields(trajectories,
                                                 refl_indices,
                                                 refl_per_traj)

    refl_fields_expected = 0.38488798713539296
    refl_intensity_expected = 0.42164540478888135

    assert_allclose(refl_fields, refl_fields_expected)
    assert_allclose(reflectance, refl_intensity_expected)

@pytest.mark.slow
def test_field_co_cross_mc():
    """
    Tests whether the co- and cross-polarized reflectances for the fields model
    match the results in the fields_montecarlo_tutorial.ipynb notebook.
    """

    seed = 1
    rng = np.random.RandomState([seed])

    wavelengths = sc.Quantity(np.arange(440, 780, 20), 'nm')

    radius = sc.Quantity('0.140 um')
    volume_fraction = 0.55
    index_imag = sc.ConstantIndex(2.1e-5*1j)
    index_particle = sc.index.polystyrene + index_imag

    sphere = sc.Sphere(index_particle, radius)
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelengths)

    thickness = sc.Quantity('80 um')
    boundary = 'film'

    ntrajectories = 500
    nevents = 150

    # polarization detector parameters
    det_theta = sc.Quantity('10 deg')

    reflectance = np.zeros(wavelengths.size)
    refl_tot = np.zeros(wavelengths.size)
    refl_co = np.zeros(wavelengths.size)
    refl_cr = np.zeros(wavelengths.size)
    refl_perp = np.zeros(wavelengths.size)
    refl_field = np.zeros(wavelengths.size)
    refl_intensity = np.zeros(wavelengths.size)

    # set up scattering model
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)

    for i in range(wavelengths.size):
        # Initialize and run simulation
        sim = mc.Simulation(model, wavelengths[i], nevents, ntrajectories,
                            boundary, fields=True, coherent=False, rng=rng)
        sim.run(radius=radius, wavelength=wavelengths[i])

        trajectories = sim.traj
        refl_trans_result = det.calc_refl_trans(sim, thickness,
                                                return_extra=True)

        reflectance[i] = refl_trans_result[11]
        refl_indices = refl_trans_result[0]
        refl_per_traj = refl_trans_result[3]

        # calculate reflectance including fields
        refl_fields, _ = detp.calc_refl_phase_fields(trajectories,
                                                     refl_indices,
                                                     refl_per_traj)

        # calculate reflectance contribution from each polarization component
        (refl_co[i],
         refl_cr[i],
         refl_perp[i],
         refl_field[i],
         refl_intensity[i]) = detp.calc_refl_co_cross_fields(trajectories,
                                                             refl_indices,
                                                             refl_per_traj,
                                                             det_theta)

    R_expected = [0.681824026794771, 0.6948436254210142, 0.6576767551090704,
                  0.6485419612486806, 0.6105017450658912, 0.6102481369185175,
                  0.5589356952841685, 0.5575609139175314, 0.5757231116604751,
                  0.6101266205743023, 0.5758022190004686, 0.5538054214886035,
                  0.5142860097093124, 0.4719620939391233, 0.4205562551205015,
                  0.3883739385932425, 0.3574811679229245]

    R_field_expected = [0.5979770839555806, 0.2009518658385173,
                        1.1676679907415521, 0.5477814037701005,
                        0.3348114066994026, 0.435234249904512,
                        0.5037181299542506, 0.9002597827984578,
                        0.6655094197859748, 0.5314301745170483,
                        0.3925955232748141, 0.9171678835775795,
                        0.5048045446051103, 0.4079200830599042,
                        0.4203689149917714, 0.3245322247300941,
                        0.1334153821305706]

    R_co_expected = [0.6055849927736964, 0.1317894835605825, 0.1936507952963292,
                     0.6282265410963711, 0.304811974233517,  0.6587975418369814,
                     0.3291062834640942, 0.5108196977088308, 1.,
                     0.1227778758597088, 0.0349428250060584, 0.1110410178589713,
                     0.0786121186900533, 0.309390455717338,  0.3040556216223184,
                     0.1602021554963812, 0.0244466445162816]

    R_cross_expected = [0.0106367345293578, 0.1145163630939446, 1.,
                        0.1845293635396514, 0.2050032845469504,
                        0.0945465161862712, 0.292784499944308,
                        0.629594953902804,  0.197893023775687,
                        0.2929026924710842, 0.3429393746808307,
                        0.8433712815740945, 0.5113729369590685,
                        0.1621279606148279, 0.2101056708217003,
                        0.1435147433999443, 0.0716347952278302]

    assert_allclose(refl_intensity, R_expected)
    assert_allclose(refl_field, R_field_expected)
    assert_allclose(refl_co/np.max(refl_co), R_co_expected)
    assert_allclose(refl_cr/np.max(refl_cr), R_cross_expected)
