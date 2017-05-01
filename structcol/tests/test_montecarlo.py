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

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_sampling():
    # Test that 'calc_scat' runs
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample, volume_fraction, 
                                  wavelen, phase_mie=False, mu_scat_mie=False)
    
    # Test that 'sample_angles' runs
    mc.sample_angles(nevents, ntrajectories, p)
    
    # Test that 'sample_step' runs
    mc.sample_step(nevents, ntrajectories, mu_abs, mu_scat)

def test_trajectory_status():
    trajectories_z = np.array([[ 0, 0, 0, 0, 0, 0, 0],
                               [ 1, 1, 1,-1, 1, 3,-1],
                               [ 5, 7,-1, 1,11, 3,-1],
                               [-1,11, 4, 2, 3, 3,-1],
                               [ 2,14,12, 3, 7, 3,-1],
                               [ 4,11, 7, 4, 8, 3,-1],
                               [ 8, 8, 9, 5,-2, 3,-1]])

    refl_indices = np.array([3,0,2,1,0,0,1])
    trans_indices = np.array([0,3,0,0,2,0,0])
    stuck_indices = np.array([0,0,0,0,0,6,0])
    all_output = (refl_indices, trans_indices, stuck_indices)
    assert_equal(mc.trajectory_status(trajectories_z, 0, 10), all_output)

def test_calc_refl_trans():
    events=3
    dummy_array = np.zeros([3, 4])
    low_thresh = 0
    high_thresh = 5
    small_n = 1
    large_n = 2

    # test absoprtion and stuck without fresnel
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,6,2,6],[-1,6,2,6]])
    kz = np.array([[1,1,1,1],[-1,1,0,1],[0,0,0,0]])
    weights = np.array([[1,1,2,1],[1,0.3,1,0],[0.1,0.1,0.5,0]])
    trajectories = mc.Trajectory([dummy_array, dummy_array, z_pos],[dummy_array, dummy_array, kz], weights, events)
#    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, small_n)
    expected_trans_array = np.array([0, 0.3, 0.033333333, 0])/np.sum(weights[0]) #calculated manually
    expected_refl_array = np.array([1, 0, 0.1111111111, 0])/np.sum(weights[0]) #calculated manually
#    assert_almost_equal(refl, np.sum(expected_refl_array))
#    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test absorption and fresnel without stuck
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[1,1,1,1],[-1,-1,6,6]])
    kz = np.array([[1,1,1,0.86746757864487367],[1,1,1,1],[-0.8,-1,0.9,0.9]])
    weights = np.array([[1,1,1,1],[1,1,1,1],[0.9,0.8,0.7,0.6]])
    trajectories = mc.Trajectory([dummy_array, dummy_array, z_pos],[dummy_array, dummy_array, kz], weights, events)
#   refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, large_n)
    expected_trans_array = np.array([ 0.28222975, 0.02787454, 0.55595101, 0.21849502])/np.sum(weights[0]) #calculated manually
    expected_refl_array = np.array([ 0.3574732,  0.76754193, 0.14264385, 0.60482546])/np.sum(weights[0]) #calculated manually
#    assert_almost_equal(refl, np.sum(expected_refl_array))
#    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test fresnel and stuck without absorption
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,1,1,6],[-1,1,6,6]])
    kz = np.array([[0.86746757864487367,1,1,1],[-1,1,.95,1],[-0.9,1,.8,.9]])
    weights = np.array([[1,1,1,1],[1,1,1,1],[1.,1.,1.,1.]])
    trajectories = mc.Trajectory([dummy_array, dummy_array, z_pos],[dummy_array, dummy_array, kz], weights, events)
#    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, large_n)
    expected_trans_array = np.array([ 0.03104891, 0.60944866, 0.60944866, 0.85783997])/np.sum(weights[0]) #calculated manually
    expected_refl_array = np.array([ 0.96895109, 0.39055134, 0.39055134, 0.14216003])/np.sum(weights[0]) #calculated manually
#    assert_almost_equal(refl, np.sum(expected_refl_array))
#    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test refraction and detection_angle
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[1,1,1,1],[-1,-1,6,6]])
    kz = np.array([[1,1,1,0.86746757864487367],[1,1,1,1],[-0.8,-0.9,0.95,0.9]])
    weights = np.array([[1.,1.,1.,1.],[1,1,1,1],[1.,1.,1.,1.]])
    trajectories = mc.Trajectory([dummy_array, dummy_array, z_pos],[dummy_array, dummy_array, kz], weights, events)
    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, large_n, detection_angle=np.pi/4)
    expected_trans_array = np.array([ 0.37354458, 0.06147163, 0.8287071, 0.02818555])/np.sum(weights[0]) #calculated manually
    expected_refl_array = np.array([ 0.1111111,  0.11111111, 0.1111111, 0])/np.sum(weights[0]) #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))


def test_trajectories():
    # Initialize runs
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_matrix, n_sample, seed=1)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    # Create a Trajectory object
    trajectories = mc.Trajectory(r0, k0, W0, nevents)
    
    # Test the absorb function
    mu_abs = 1/sc.Quantity(10, 'um')    
    step = sc.Quantity(np.array([[1,1,1],[1,1,1]]), 'um')    
    trajectories.absorb(mu_abs, step)     
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
