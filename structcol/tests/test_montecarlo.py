# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang
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
"""

import structcol as sc
from .. import montecarlo as mc
from .. import refractive_index as ri
import os
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

# Define a system to be used for the tests
nevents = 2
ntrajectories = 3
radius = sc.Quantity('150 nm')
volume_fraction = 0.5
n_particle = sc.Quantity(1.5, '')
n_matrix = sc.Quantity(1.0, '')
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction) 
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')  
wavelen = sc.Quantity('400 nm')

# Define an array of z-positions and cutoff
z_pos = sc.Quantity(np.array([[0,0,0],[2,6,6],[-1,3,-1]]), 'um')     
z_low = sc.Quantity('0. um')
cutoff = sc.Quantity('5. um')

# Define an array of directions (trajectories travel straight down into the 
# sample in the initial and final step, and travel upward in the middle step)
kx = sc.Quantity(np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]), '')
ky = sc.Quantity(np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]), '')
kz = sc.Quantity(np.array([[1.,1.,1.],[-1.,-1.,-1.],[1.,1.,1.]]), '')

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_event = np.array([1,1])
refl_traj = np.array([0,2])

# Assign a weight of 1 to all the trajectories (no absorption)
weights = np.ones((nevents,ntrajectories))
    
    
def test_sampling():
    # Test that 'calc_scat' runs
    p, lscat, labs = mc.calc_scat(radius, n_particle, n_sample, volume_fraction, 
                                  angles, wavelen, phase_mie=False, lscat_mie=False)
    
    # Test that 'sample_angles' runs
    mc.sample_angles(nevents, ntrajectories, p, angles)
    
    # Test that 'sample_step' runs
    mc.sample_step(nevents, ntrajectories, 1/labs, 1/lscat)
    

def test_fresnel_refl():
    # Test that 'fresnel_refl' runs
    mc.fresnel_refl(n_sample, n_matrix, kz, refl_event, refl_traj, weights)

def test_trajectory_status():
    trajectories_z = np.array([[ 0, 0, 0, 0, 0, 0, 0],
                               [ 1, 1, 1,-1, 1, 3,-1],
                               [ 5, 7,-1, 1,11, 3,-1],
                               [-1,11, 4, 2, 3, 3,-1],
                               [ 2,14,12, 3, 7, 3,-1],
                               [ 4,11, 7, 4, 8, 3,-1],
                               [ 8, 8, 9, 5,-2, 3,-1]])

    refl_row_indices = np.array([3,2,1,1])
    refl_col_indices = np.array([0,2,3,6])
    trans_row_indices = np.array([3,2])
    trans_col_indices = np.array([1,4])
    stuck_indices = np.array([5])
    all_output = (refl_col_indices, refl_row_indices, trans_col_indices, trans_row_indices, stuck_indices)
    assert_equal(mc.trajectory_status(trajectories_z, 0, 10), all_output)

def test_calc_reflection():    
    # Test that it calculates the correct number of reflected trajectories
    R_fraction = mc.calc_reflection(z_pos, z_low, cutoff, ntrajectories, n_matrix, n_sample, kx, ky, kz)
    assert_equal(R_fraction, 0.33696688277707043)
    
def test_trajectories():
    # Initialize runs
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, seed=1)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    # Create a Trajectory object
    trajectories = mc.Trajectory(r0, k0, W0, nevents)
    
    # Test the absorb function
    mua = 1 / sc.Quantity(10, 'um')    
    step = sc.Quantity(np.array([[1,1,1],[1,1,1]]), 'um')    
    trajectories.absorb(mua, step)     
    assert_almost_equal(trajectories.weight, 
                 np.array([[ 0.90483742,  0.90483742,  0.90483742],
                           [ 0.81873075,  0.81873075,  0.81873075]]))
    
    # Make up some test theta and phi
    sintheta = np.array([[0.,0.,0.],[0.,0.,0.]])  
    costheta = np.array([[-1.,-1.,-1.],[1.,1.,1.]])  
    sinphi = np.array([[0.,0.,0.],[0.,0.,0.]])
    cosphi = np.array([[0.,0.,0.],[0.,0.,0.]])
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)       
    
    # Expected propagation directions
    kx = sc.Quantity(np.array([[0.,0.,0.],[0.,0.,0.]]), '')
    ky = sc.Quantity(np.array([[0.,0.,0.],[0.,0.,0.]]), '')
    kz = sc.Quantity(np.array([[1.,1.,1.],[-1.,-1.,-1.]]), '')
    
    # Test the scatter function
    assert_almost_equal(trajectories.direction[0], kx.magnitude)
    assert_almost_equal(trajectories.direction[1], ky.magnitude)
    assert_almost_equal(trajectories.direction[2], kz.magnitude)
    
    # Test the move function    
    trajectories.move(step)
    assert_equal(trajectories.position[2], np.array([[0,0,0],[1,1,1],[0,0,0]]))
