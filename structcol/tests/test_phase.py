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
from .. import refractive_index as ri
from .. import index_ratio, size_parameter, model
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest

def test_mod():
# test that phase mod 2Pi is the same as phase    

    # incident light wavelength
    wavelength = sc.Quantity('600 nm')
    
    # sample parameters
    radius = sc.Quantity('0.140 um')
    volume_fraction = sc.Quantity(0.55, '')
    n_imag = 2.1e-4
    n_particle = ri.n('polystyrene', wavelength) + n_imag    # refractive indices can be specified as pint quantities or
    n_matrix = ri.n('vacuum', wavelength)      # called from the refractive_index module. n_matrix is the 
    n_medium = ri.n('vacuum', wavelength)      # space within sample. n_medium is outside the sample
    n_sample = ri.n_eff(n_particle,         # refractive index of sample, calculated using Bruggeman approximation
                        n_matrix, 
                        volume_fraction)
    thickness = sc.Quantity('800 um')
    boundary = 'film'
    
    # Monte Carlo parameters
    ntrajectories = 10                # number of trajectories
    nevents = 10                         # number of scattering events in each trajectory
    
    # Calculate scattering quantities
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample, 
                                      volume_fraction, wavelength, polarization= True)
    
    # Initialize trajectories
    r0, k0, W0, p0, ph0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample, boundary, 
                                   polarization=True, phase=True)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')
    p0 = sc.Quantity(p0,'')
    ph0 = sc.Quantity(ph0,'')
    
    trajectories = mc.Trajectory(r0, k0, W0, p0, ph0)
    
    # Sample trajectory angles
    sintheta, costheta, sinphi, cosphi, theta, phi= mc.sample_angles(nevents, 
                                                               ntrajectories,p)
    # Sample step sizes
    step = mc.sample_step(nevents, ntrajectories, mu_scat)
    
    # Update trajectories based on sampled values
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)
    trajectories.polarize(theta, phi, sintheta, costheta, sinphi,cosphi,
                          n_particle, n_sample, radius, wavelength, volume_fraction)
    trajectories.shift_phase(step, theta, phi, sintheta, costheta, sinphi, cosphi,
                             n_particle, n_sample, radius, wavelength, volume_fraction)
    trajectories.move(step)
    trajectories.absorb(mu_abs, step)  
    
    refl_trans_result = det.calc_refl_trans(trajectories, thickness, n_medium, n_sample, boundary, 
                                                 return_extra=True)

    refl_indices = refl_trans_result[0]
    refl_per_traj = refl_trans_result[3]
    reflectance_phase, _ = det.calc_refl_phase(trajectories, refl_indices, refl_per_traj)
    
    # now do mod 2pi
    trajectories.phase = np.mod(trajectories.phase, 2*np.pi)
    reflectance_phase_mod, _ = det.calc_refl_phase(trajectories, refl_indices, refl_per_traj)
    
    assert_almost_equal(reflectance_phase, reflectance_phase_mod, decimal=15)
    
def test_intensity_coherent():
# tests that the intensity of the summed fields correspond to the equation for
# coherent light: Ix = E_x1^2 + E_x2^2 + 2E_x1*E_x2

    # construct 2 identical trajectories that exit at same event
    ntrajectories = 2
    nevents = 3
    z_pos = np.array([[0,0],[1,1],[-1,-1]])
    kz = np.array([[1,1],[-1,1],[-1,1]])
    weights = np.array([[1, 1],[1, 1],[1, 1]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz], weights)
    trajectories.phase = np.zeros((3, nevents, ntrajectories))
    trajectories.phase[:,0,:] = 0.5
    trajectories.phase[:,1,:] = 1
    trajectories.phase[:,2,:] = 1.5
    trajectories.polarization = np.ones((3, nevents, ntrajectories))

    # calculate reflectance phase
    refl_per_traj = np.array([0.5, 0.5])
    refl_indices = np.array([2, 2])
    refl_phase, _ = det.calc_refl_phase(trajectories, refl_indices, refl_per_traj)
    intensity_incident = np.sum(trajectories.weight[0,:])
    intensity = refl_phase*intensity_incident
    
    # Calculate I = (E1 + E2)*(E1 + E2) = E1*E1 + E2*E2 + E1*E2 + E2*E1
    ev = 2
    field_x = np.sqrt(trajectories.weight[ev,:])*trajectories.polarization[0,ev,:]*np.exp(1j*trajectories.phase[0,ev,:])
    field_y = np.sqrt(trajectories.weight[ev,:])*trajectories.polarization[1,ev,:]*np.exp(1j*trajectories.phase[1,ev,:])
    field_z = np.sqrt(trajectories.weight[ev,:])*trajectories.polarization[2,ev,:]*np.exp(1j*trajectories.phase[2,ev,:])
    intensity_x = np.conj(field_x[0])*field_x[0] + np.conj(field_x[1])*field_x[1] + np.conj(field_x[0])*field_x[1] + np.conj(field_x[1])*field_x[0] 
    intensity_y = np.conj(field_y[0])*field_y[0] + np.conj(field_y[1])*field_y[1] + np.conj(field_y[0])*field_y[1] + np.conj(field_y[1])*field_y[0] 
    intensity_z = np.conj(field_z[0])*field_z[0] + np.conj(field_z[1])*field_z[1] + np.conj(field_z[0])*field_z[1] + np.conj(field_z[1])*field_z[0]
    intensity_2 = intensity_x + intensity_y + intensity_z

    # compare values
    assert_almost_equal(intensity, intensity_2, decimal=15)

def test_intensity_zero():
# tests if a pi shift leads to zero intensity    
    
    # construct 2 trajectories with relative pi phase shift that exit at same event
    ntrajectories = 2
    nevents = 3
    z_pos = np.array([[0,0],[1,1],[-1,-1]])
    kz = np.array([[1,1],[-1,1],[-1,1]])
    weights = np.array([[1, 1],[1, 1],[1, 1]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz], weights)
    trajectories.phase = np.zeros((3, nevents, ntrajectories))
    trajectories.phase[:,2,0] = 0
    trajectories.phase[:,2,1] = np.pi
    trajectories.polarization = np.ones((3, nevents, ntrajectories))

    # calculate reflectance phase
    refl_per_traj = np.array([0.5, 0.5])
    refl_indices = np.array([2, 2])
    refl_phase, _ = det.calc_refl_phase(trajectories, refl_indices, refl_per_traj)

    # check whether reflectance phase is 0
    assert_almost_equal(refl_phase, 0, decimal=15)
