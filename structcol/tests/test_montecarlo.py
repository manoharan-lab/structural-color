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
nevents = 3
ntrajectories = 4
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

def test_calc_refl_trans():
    low_thresh = 0
    high_thresh = 10
    small_n = 1
    large_n = 2

    # test absoprtion and stuck without fresnel
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    ntrajectories = z_pos.shape[1]
    kz = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])
    weights = np.array([[.8, .8, .9, .8],[.7, .3, .7, 0],[.1, .1, .5, 0]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz], weights)
    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, small_n)
    expected_trans_array = np.array([0, .3, .25, 0])/ntrajectories #calculated manually
    expected_refl_array = np.array([.7, 0, .25, 0])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test above but with covers on front and back
    refl, trans = mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, small_n, n_front=large_n, n_back=large_n)
    expected_trans_array = np.array([0.00814545, 0.20014545, 0.2, 0.])/ntrajectories #calculated manually
    expected_refl_array = np.array([0.66700606, 0.20349091, 0.4, 0.2])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test fresnel as well
    z_pos = np.array([[0,0,0,0],[5,5,5,5],[-5,-5,15,15],[5,-15,5,25],[-5,-25,6,35]])
    ntrajectories = z_pos.shape[1]
    kz = np.array([[1,1,1,0.86746757864487367],[-.1,-.1,.1,.1],[0.1,-.1,-.1,0.1],[-1,-.9,1,1]])
    weights = np.array([[.8, .8, .9, .8],[.7, .3, .7, .5],[.6, .2, .6, .4], [.4, .1, .5, .3]])
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz], weights)
    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, large_n)
    expected_trans_array = np.array([ .00167588, .00062052, .22222222, .11075425])/ntrajectories #calculated manually
    expected_refl_array = np.array([ .43317894, .18760061, .33333333, .59300905])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test refraction and detection_angle
    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, high_thresh, small_n, large_n, detection_angle=0.1)
    expected_trans_array = np.array([ .00167588, .00062052, .22222222,  .11075425])/ntrajectories #calculated manually
    expected_refl_array = np.array([  .43203386, .11291556, .29105299,  .00046666])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    # test steps in z longer than sample thickness
    z_pos = np.array([[0,0,0,0,0,0,0],[1.1,2.1,3.1,0.6,0.6,0.6,0.1],[1.2,2.2,3.2,1.6,0.7,0.7,-0.6],[1.3,2.3,3.3,3.3,-2.1,-1.1,-2.1]])
    ntrajectories = z_pos.shape[1]
    kz = np.array([[1,1,1,1,1,1,1],[1,1,1,0.1,1,1,-0.1],[1,1,1,1,-1,-1,-1]])
    weights = np.array([[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]])
    thin_sample_thickness = 1
    trajectories = mc.Trajectory([np.nan, np.nan, z_pos],[np.nan, np.nan, kz], weights)
    refl, trans= mc.calc_refl_trans(trajectories, low_thresh, thin_sample_thickness, small_n, large_n)
    expected_trans_array = np.array([.8324515, .8324515, .8324515, .05643739, .05643739, .05643739, .8324515])/ntrajectories #calculated manually
    expected_refl_array = np.array([.1675485, .1675485, .1675485, .94356261, .94356261, .94356261, .1675485])/ntrajectories #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

def test_trajectories():
    # Initialize runs
    nevents = 2
    ntrajectories = 3
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_matrix, n_sample, seed=1)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    # Create a Trajectory object
    trajectories = mc.Trajectory(r0, k0, W0)
    
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
