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
from numpy.testing import assert_almost_equal
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
    n_particle = sc.index.polystyrene(wavelength) + n_imag
    n_matrix = sc.index.vacuum(wavelength)
    n_medium = sc.index.vacuum(wavelength)
    n_sample = sc.index.n_eff(n_particle,
                        n_matrix,
                        volume_fraction)
    thickness = sc.Quantity('50.0 um')
    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 10
    nevents = 30

    # Calculate scattering quantities
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelength, fields=True)

    # Initialize trajectories
    r0, k0, W0, E0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample, boundary,
                                   fields=True)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')
    E0 = sc.Quantity(E0, '')

    trajectories = mc.Trajectory(r0, k0, W0, fields=E0)

    # Sample trajectory angles
    sintheta, costheta, sinphi, cosphi, theta, phi = mc.sample_angles(nevents,
                                                                      ntrajectories, p)
    # Sample step sizes
    step = mc.sample_step(nevents, ntrajectories, mu_scat)

    # Update trajectories based on sampled values
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.move(step)
    trajectories.absorb(mu_abs, step)
    trajectories.calc_fields(theta, phi, sintheta, costheta, sinphi, cosphi,
                             n_particle, n_sample, radius, wavelength,
                             step, volume_fraction)

    # calculate reflectance
    # (should raise warning that n_matrix and n_particle are not set, so
    # tir correction is based only on sample index)
    with pytest.warns(UserWarning):
        refl_trans_result = det.calc_refl_trans(trajectories, thickness,
                                                n_medium, n_sample, boundary,
                                                return_extra=True)

    refl_indices = refl_trans_result[0]
    refl_per_traj = refl_trans_result[3]
    reflectance_fields, _ = detp.calc_refl_phase_fields(trajectories,
                                                        refl_indices,
                                                        refl_per_traj)

    # now do mod 2pi
    trajectories.fields = trajectories.fields*np.exp(2*np.pi*1j)
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
    nevents = 3
    z_pos = np.array([[0,0],[1,1],[-1,-1]])
    kz = np.array([[1,1],[-1,1],[-1,1]])
    directions = np.array([kz,kz,kz])
    weights = np.array([[1, 1],[1, 1],[1, 1]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos], directions, weights)
    trajectories.fields = np.zeros((3, nevents, ntrajectories))
    trajectories.fields[:,0,:] = 0.5
    trajectories.fields[:,1,:] = 1
    trajectories.fields[:,2,:] = 1.5

    # calculate reflectance phase
    refl_per_traj = np.array([0.5, 0.5])
    refl_indices = np.array([2, 2])
    refl_phase, _ = detp.calc_refl_phase_fields(trajectories, refl_indices,
                                                refl_per_traj)
    intensity_incident = np.sum(trajectories.weight[0,:])
    intensity = refl_phase*intensity_incident

    # Calculate I = (E1 + E2)*(E1 + E2) = E1*E1 + E2*E2 + E1*E2 + E2*E1
    ev = 2
    field_x = np.sqrt(trajectories.weight[ev,:])*trajectories.fields[0,ev,:]
    field_y = np.sqrt(trajectories.weight[ev,:])*trajectories.fields[1,ev,:]
    field_z = np.sqrt(trajectories.weight[ev,:])*trajectories.fields[2,ev,:]
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
    nevents = 3
    z_pos = np.array([[0,0],[1,1],[-1,-1]])
    x_pos = np.array([[0,0],[1,1],[-1,-1]])
    kz = np.array([[1,1],[-1,1],[-1,1]])
    directions = np.array([kz,kz,kz])
    weights = np.array([[1, 1],[1, 1],[1, 1]])
    trajectories = mc.Trajectory([x_pos, np.nan, z_pos],directions, weights)
    trajectories.fields = np.zeros((3, nevents, ntrajectories), dtype=complex)
    trajectories.fields[:,2,0] = 1
    trajectories.fields[:,2,1] = np.exp(np.pi*1j)

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
    n_imag = 2.1e-4
    n_particle = sc.index.polystyrene(wavelength) + n_imag*1j
    n_matrix = sc.index.vacuum(wavelength)
    n_medium = sc.index.vacuum(wavelength)
    n_sample = sc.index.n_eff(n_particle,
                        n_matrix,
                        volume_fraction)
    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 10
    nevents = 10

    # Calculate scattering quantities
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelength, fields=True)

    # Initialize trajectories
    r0, k0, W0, E0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                                   boundary, fields=True)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')
    E0 = sc.Quantity(E0,'')

    trajectories = mc.Trajectory(r0, k0, W0, fields=E0)


    # Sample trajectory angles
    sintheta, costheta, sinphi, cosphi, theta, phi= mc.sample_angles(nevents,
                                                               ntrajectories,p)
    # Sample step sizes
    step = mc.sample_step(nevents, ntrajectories, mu_scat)

    # Update trajectories based on sampled values
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.move(step)
    trajectories.calc_fields(theta, phi, sintheta, costheta, sinphi, cosphi,
                                 n_particle, n_sample, radius, wavelength,
                                 step, volume_fraction)
    trajectories.absorb(mu_abs, step)

    # take the dot product
    trajectories.fields = trajectories.fields.magnitude

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
    n_imag = 2.1e-4
    n_particle = sc.index.polystyrene(wavelength) + n_imag*1j
    n_matrix = sc.index.vacuum(wavelength)
    n_medium = sc.index.vacuum(wavelength)
    n_sample = sc.index.n_eff(n_particle,
                        n_matrix,
                        volume_fraction)
    boundary = 'film'

    # Monte Carlo parameters
    ntrajectories = 10
    nevents = 10

    # Calculate scattering quantities
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelength, fields=True)

    # Initialize trajectories
    r0, k0, W0, E0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                                   boundary, fields=True)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')
    E0 = sc.Quantity(E0,'')

    trajectories = mc.Trajectory(r0, k0, W0, fields = E0)


    # Sample trajectory angles
    sintheta, costheta, sinphi, cosphi, theta, phi =\
        mc.sample_angles(nevents, ntrajectories,p)

    step = mc.sample_step(nevents, ntrajectories, mu_scat)

    # Update trajectories based on sampled values
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.move(step)
    trajectories.calc_fields(theta, phi, sintheta, costheta, sinphi, cosphi,
                             n_particle, n_sample, radius, wavelength, step,
                             volume_fraction)
    trajectories.absorb(mu_abs, step)

    # take the dot product
    trajectories.direction = trajectories.direction.magnitude
    trajectories.fields = trajectories.fields.magnitude

    dot_prod = (trajectories.direction[0,:,:]*trajectories.fields[0,1:,:] +
               trajectories.direction[1,:,:]*trajectories.fields[1,1:,:] +
               trajectories.direction[2,:,:]*trajectories.fields[2,1:,:])

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
    n_imag = 2.1e-4
    n_particle = sc.index.polystyrene(wavelength) + n_imag*1j
    n_matrix = sc.index.vacuum(wavelength)
    n_medium = sc.index.vacuum(wavelength)
    n_sample = sc.index.n_eff(n_particle,
                        n_matrix,
                        volume_fraction)
    thickness = sc.Quantity('800 um')
    boundary = 'film'

    ntrajectories = 2000
    nevents = 300

    # Calculate scattering quantities
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,
                                      volume_fraction, wavelength,
                                      fields=True)

    # Initialize trajectories
    r0, k0, W0, E0 = mc.initialize(nevents, ntrajectories,
                                   n_medium, n_sample, boundary,
                                   coherent=False,
                                   fields=True, rng=rng)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')
    E0 = sc.Quantity(E0,'')

    trajectories = mc.Trajectory(r0, k0, W0, fields=E0)

    # Sample trajectory angles
    sintheta, costheta, sinphi, cosphi, theta, phi = \
        mc.sample_angles(nevents, ntrajectories, p, rng=rng)

    # Sample step sizes
    step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

    # Update trajectories based on sampled values
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.calc_fields(theta, phi, sintheta, costheta, sinphi, cosphi,
                             n_particle, n_sample, radius, wavelength, step,
                             volume_fraction,
                             fine_roughness=0, tir_refl_bool=None)
    trajectories.move(step)
    trajectories.absorb(mu_abs, step)

    with pytest.warns(UserWarning):
        refl_trans_result = det.calc_refl_trans(trajectories, thickness,
                                                n_medium, n_sample, boundary,
                                                return_extra=True)

    reflectance = refl_trans_result[11]
    refl_indices = refl_trans_result[0]
    refl_per_traj = refl_trans_result[3]

    # calculate reflectance including phase
    with pytest.warns(UserWarning):
        refl_trans_result = det.calc_refl_trans(trajectories, thickness,
                                                n_medium, n_sample, boundary,
                                                return_extra=True)

    refl_indices = refl_trans_result[0]
    refl_per_traj = refl_trans_result[3]
    refl_fields, _ = detp.calc_refl_phase_fields(trajectories,
                                                 refl_indices,
                                                 refl_per_traj)

    refl_fields_expected = 0.3848868020860198
    refl_intensity_expected = 0.4216450105698871

    assert_almost_equal(refl_fields, refl_fields_expected)
    assert_almost_equal(reflectance, refl_intensity_expected)

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
    n_imag = 2.1e-5
    n_particle = sc.index.polystyrene(wavelengths) + n_imag*1j
    n_matrix = sc.index.vacuum(wavelengths)
    n_medium = sc.index.vacuum(wavelengths)

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

    for i in range(wavelengths.size):
        # calculate n_sample
        n_sample = sc.index.n_eff(n_particle[i], n_matrix[i], volume_fraction)

        # Calculate scattering quantities
        p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle[i], n_sample,
                                          volume_fraction, wavelengths[i],
                                          fields=True)

        # Initialize trajectories
        r0, k0, W0, E0 = mc.initialize(nevents, ntrajectories, n_medium[i],
                                       n_sample, boundary, fields=True,
                                       coherent=False, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')
        E0 = sc.Quantity(E0,'')

        trajectories = mc.Trajectory(r0, k0, W0, E0)

        sintheta, costheta, sinphi, cosphi, theta, phi =\
            mc.sample_angles(nevents, ntrajectories, p, rng=rng)

        step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)
        trajectories.absorb(mu_abs, step)
        trajectories.calc_fields(theta, phi, sintheta, costheta, sinphi,
                                 cosphi, n_particle[i], n_sample, radius,
                                 wavelengths[i], step, volume_fraction,
                                 fine_roughness=0, tir_refl_bool=None)

        with pytest.warns(UserWarning):
            refl_trans_result = det.calc_refl_trans(trajectories,thickness,
                                                    n_medium[i], n_sample,
                                                    boundary,
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

    R_expected = [0.6818239027988798, 0.6948435902788378, 0.6576767363912293,
                  0.6485419490282506, 0.6105017312246305, 0.610248113448991,
                  0.5589356887128389, 0.5575609090530609, 0.5757231078357871,
                  0.6101265925414807, 0.5758020810429944, 0.5538053785009902,
                  0.5142860078764621, 0.47196209364896763, 0.4205562336078029,
                  0.38837230436216574, 0.3574811605636892]

    R_field_expected = [0.5979755068722215, 0.20095123221751834,
                        1.1676679521354882, 0.5477814385252179,
                        0.3348114079532808, 0.43523427289893646,
                        0.5037181358415458, 0.9002597022095241,
                        0.6655094198238951, 0.5314303249118808,
                        0.39259816407642845, 0.917167794491729,
                        0.5048045364499516, 0.4079200822892743,
                        0.42036863036736405, 0.3245279368044734,
                        0.13341537008710438]

    R_co_expected = [0.605587279561404, 0.13178875779291185,
                     0.1936508027764766, 0.6282263171707353,
                     0.30481199220342176, 0.6587976580578032,
                     0.32910629573759087, 0.5108196425031858, 1.0,
                     0.1227779716301733, 0.034942618242929886,
                     0.11104145426216996, 0.07861211507410897,
                     0.30939045564175116, 0.3040550419655162,
                     0.16016706545322174, 0.024446646332762535]

    R_cross_expected = [0.0106364701868872, 0.11451626694721864, 1.0,
                        0.18452942880221118, 0.20500327797022916,
                        0.0945464760088375, 0.29278450424549957,
                        0.6295948478851608, 0.1978930219113831,
                        0.29290279122003954, 0.3429407370602462,
                        0.8433708859800838, 0.5113729248909841,
                        0.16212795938328153, 0.21010564421109604,
                        0.14350179291386156, 0.07163478994233281]

    assert_almost_equal(refl_intensity, R_expected)
    assert_almost_equal(refl_field, R_field_expected)
    assert_almost_equal(refl_co/np.max(refl_co), R_co_expected)
    assert_almost_equal(refl_cr/np.max(refl_cr), R_cross_expected)
