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
n_medium = sc.Quantity(1.0, '')
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction) 
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')  
wavelen = sc.Quantity('400 nm')

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_sampling():
    # Test that 'calc_scat' runs
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample,  
                                      volume_fraction, wavelen)
    
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

def test_reflection_core_shell():
    # test that the reflection of a non-core-shell system is the same as that
    # of a core-shell with a shell index matched with the core
    seed = 1
    nevents = 60
    ntrajectories = 30
    
    # Reflection using a non-core-shell system
    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle, 
                           n_sample, n_medium, volume_fraction, wavelen, seed)
    
    # Reflection using core-shells with the shell index-matched to the core
    radius_cs = sc.Quantity(np.array([100, 150]), 'nm')  # specify the radii from innermost to outermost layer
    n_particle_cs = sc.Quantity(np.array([1.5,1.5]), '')  # specify the index from innermost to outermost layer           
    
    # calculate the volume fractions of each layer
    vf_array = np.empty(len(radius_cs))
    r_array = np.array([0] + radius_cs.magnitude.tolist()) 
    for r in np.arange(len(r_array)-1):
        vf_array[r] = (r_array[r+1]**3-r_array[r]**3) / (r_array[-1:]**3) * volume_fraction

    n_sample_cs = ri.n_eff(n_particle_cs, n_matrix, vf_array) 
    R_cs, T_cs = calc_montecarlo(nevents, ntrajectories, radius_cs, 
                                 n_particle_cs, n_sample_cs, n_medium, 
                                 volume_fraction, wavelen, seed)

    assert_almost_equal(R, R_cs)
    assert_almost_equal(T, T_cs)
    
    
    # Test that the reflectance is the same for a core-shell that absorbs (with
    # the same refractive indices for all layers) and a non-core-shell that 
    # absorbs with the same index
    # Reflection using a non-core-shell absorbing system
    n_particle_abs = sc.Quantity(1.5+0.001j, '')  
    n_sample_abs = ri.n_eff(n_particle_abs, n_matrix, volume_fraction)
    
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius, n_particle_abs, 
                           n_sample_abs, n_medium, volume_fraction, wavelen, seed)
    
    # Reflection using core-shells with the shell index-matched to the core
    n_particle_cs_abs = sc.Quantity(np.array([1.5+0.001j,1.5+0.001j]), '')  
    n_sample_cs_abs = ri.n_eff(n_particle_cs_abs, n_matrix, vf_array) 
    
    R_cs_abs, T_cs_abs = calc_montecarlo(nevents, ntrajectories, radius_cs, 
                                 n_particle_cs_abs, n_sample_cs_abs, n_medium, 
                                 volume_fraction, wavelen, seed)

    assert_almost_equal(R_abs, R_cs_abs, decimal=3)
    assert_almost_equal(T_abs, T_cs_abs, decimal=3)


    # Same as previous test but with absorbing matrix as well
    # Reflection using a non-core-shell absorbing system
    n_particle_abs = sc.Quantity(1.5+0.001j, '')  
    n_matrix_abs = sc.Quantity(1.+0.001j, '')  
    n_sample_abs = ri.n_eff(n_particle_abs, n_matrix_abs, volume_fraction)
    
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius, n_particle_abs, 
                           n_sample_abs, n_medium, volume_fraction, wavelen, seed)
    
    # Reflection using core-shells with the shell index-matched to the core
    n_particle_cs_abs = sc.Quantity(np.array([1.5+0.001j,1.5+0.001j]), '')  
    n_sample_cs_abs = ri.n_eff(n_particle_cs_abs, n_matrix_abs, vf_array) 
    
    R_cs_abs, T_cs_abs = calc_montecarlo(nevents, ntrajectories, radius_cs, 
                                 n_particle_cs_abs, n_sample_cs_abs, n_medium, 
                                 volume_fraction, wavelen, seed)

    assert_almost_equal(R_abs, R_cs_abs, decimal=3)
    assert_almost_equal(T_abs, T_cs_abs, decimal=3)


def test_reflection_absorbing_particle_or_matrix():
    # test that the reflections with a real n_particle and with a complex
    # n_particle with a 0 imaginary component are the same 
    seed = 1
    nevents = 60
    ntrajectories = 30
    
    # Reflection using non-absorbing particle
    R, T = calc_montecarlo(nevents, ntrajectories, radius, n_particle, 
                           n_sample, n_medium, volume_fraction, wavelen, seed)

    # Reflection using particle with an imaginary component of 0
    n_particle_abs = sc.Quantity(1.5 + 0j, '')
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius, 
                                   n_particle_abs, n_sample, n_medium, 
                                   volume_fraction, wavelen, seed)
  
    assert_almost_equal(R, R_abs)
    assert_almost_equal(T, T_abs)
    
    # Same as previous test but with absorbing matrix
    # Reflection using matrix with an imaginary component of 0
    n_matrix_abs = sc.Quantity(1. + 0j, '')
    n_sample_abs = ri.n_eff(n_particle, n_matrix_abs, volume_fraction)
    R_abs, T_abs = calc_montecarlo(nevents, ntrajectories, radius, 
                                   n_particle, n_sample_abs, n_medium, 
                                   volume_fraction, wavelen, seed)
    
    assert_almost_equal(R, R_abs)
    assert_almost_equal(T, T_abs)


def test_reflection_polydispersity():
    seed = 1
    nevents = 60
    ntrajectories = 30
    
    radius2 = radius
    concentration = sc.Quantity(np.array([0.9,0.1]), '')
    pdi = sc.Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit

    # Without absorption: test that the reflectance using with very small 
    # polydispersity is the same as the monodisperse case
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

    # With absorption: test that the reflectance using with very small 
    # polydispersity is the same as the monodisperse case  
    n_particle_abs = sc.Quantity(1.5+0.0001j, '')  
    n_matrix_abs = sc.Quantity(1.+0.0001j, '')  
    n_sample_abs = ri.n_eff(n_particle_abs, n_matrix_abs, volume_fraction)
    
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

    assert_almost_equal(R_mono_abs, R_poly_abs, decimal=3)
    assert_almost_equal(T_mono_abs, T_poly_abs, decimal=3)

    
def calc_montecarlo(nevents, ntrajectories, radius, n_particle, n_sample, 
                    n_medium, volume_fraction, wavelen, seed, radius2=None, 
                    concentration=None, pdi=None, polydisperse=False):
    # Function to run montecarlo for the tests
    p, mu_scat, mu_abs = mc.calc_scat(radius, n_particle, n_sample, 
                                      volume_fraction, wavelen, radius2=radius2, 
                                      concentration=concentration, pdi=pdi, 
                                      polydisperse=polydisperse)
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample, 
                               seed=seed, incidence_angle = 0.)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')
    sintheta, costheta, sinphi, cosphi, _, _= mc.sample_angles(nevents, 
                                                               ntrajectories,p)
    step = mc.sample_step(nevents, ntrajectories, mu_abs, mu_scat)
    trajectories = mc.Trajectory(r0, k0, W0)
    trajectories.absorb(mu_abs, step)                         
    trajectories.scatter(sintheta, costheta, sinphi, cosphi)         
    trajectories.move(step)
    z_low = sc.Quantity('0.0 um')
    cutoff = sc.Quantity('50 um')
    R, T = mc.calc_refl_trans(trajectories, z_low, cutoff, n_medium, n_sample)

    return R, T
