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
"""

import structcol as sc
from .. import montecarlo as mc
from .. import refractive_index as ri
from .main import Source, DetectorMultScat, Spheres, StructuredSphere 
import numpy as np
from numpy.testing import assert_almost_equal


def test_sampling():
    '''
    Check that the scattering parameter calculations and sampling functions run
    '''
    # set params
    nevents = 3
    ntraj = 4
    radius = sc.Quantity('150 nm')
    volume_fraction = 0.5
    n_particle = sc.Quantity(1.5, '')
    n_matrix = sc.Quantity(1.0, '')
    n_medium = sc.Quantity(1.0,'')  
    microsphere_radius = 5
    wavelen = sc.Quantity('400 nm')
    
    # calculate n_sample according to Bruggeman
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction) 
    
    # Test that 'calc_scat' runs
    source = Source(wavelen, polarization=None, incidence_angle=0)
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = StructuredSphere(species, n_matrix, n_medium, microsphere_radius, structure='glass')
    phase_function, scat_coeff, abs_coeff = mc.calc_scat(system, source, n_sample)
    
    # Test that 'sample_angles' runs
    mc.sample_angles(nevents, ntraj, phase_function)
    
    # Test that 'sample_step' runs
    mc.sample_step(nevents, ntraj, abs_coeff, scat_coeff)

def test_calc_refl_trans():
    '''
    Test that calc_refl_trans() gives the correct results
    '''
    # set parameters
    radius = sc.Quantity('150 nm')
    volume_fraction = 0.5
    n_particle = sc.Quantity(1.5, '')
    n_matrix = sc.Quantity(1.0, '')
    wavelen = sc.Quantity('400 nm')
    
    # set optical properties
    n_eff = 'bruggeman'
    form = 'sphere'
    
    small_n = sc.Quantity(1,'') # for testing fresnel
    large_n = sc.Quantity(2,'') # for testing fresnel
    microsphere_radius = sc.Quantity('5 um')
    
    # create 4 trajectories for 3 events
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    x_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    y_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    kx = np.zeros((3,4))
    ky = np.zeros((3,4))
    kz = np.array([[1,1,1,1],[-1,1,1,1],[-1,1,1,1]])
    ntraj = z_pos.shape[1]
    weights = np.array([[.8, .8, .9, .8],[.7, .3, .7, 0],[.1, .1, .5, 0]])
    pol = None
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], weights, pol) 
    
    ########## test absorption and stuck without fresnel ##########
    
    # create simulation objects
    source = Source(wavelen, polarization=None, incidence_angle=0)
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = StructuredSphere(species, small_n, small_n, microsphere_radius, structure='glass')
    detector = DetectorMultScat()
    
    # calculate reflectance
    results = mc.Results(trajectories, system, source, n_eff=n_eff, form=form)
    refl, trans = results.detect(detector)

    # compare to expected result
    expected_trans_array = np.array([0., .3, 0.25, 0])/ntraj #calculated manually
    expected_refl_array = np.array([.7, 0., .25, 0.])/ntraj #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    ########## test fresnel as well ##########
    
    # create new simulation objects to test fresnel
    system = StructuredSphere(species, small_n, large_n, microsphere_radius, structure='glass')
    results = mc.Results(trajectories, system, source, n_eff=n_eff, form=form)
    
    # calculate reflectance
    refl, trans = results.detect(detector)
    
    # compare to expected result
    expected_trans_array = np.array([0.0345679, .25185185, 0.22222222, 0.])/ntraj #calculated manually
    expected_refl_array = np.array([.69876543, 0.12592593, 0.33333333, 0.11111111])/ntraj #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    ########## test steps in z longer than sample thickness ##########
    
    # create new simulation objects to test z longer than thickness
    z_pos = np.array([[0,0,0,0],[1,1,14,12],[-1,11,2,11],[-2,12,4,12]])
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], weights, pol)
    system = StructuredSphere(species, small_n, small_n, microsphere_radius, structure='glass')
    results = mc.Results(trajectories, system, source, n_eff=n_eff, form=form)
    
    # calculate reflectance
    refl, trans = results.detect(detector)
    
    # compare to exprected result
    expected_trans_array = np.array([0., .3, .9, .8])/ntraj #calculated manually
    expected_refl_array = np.array([.7, 0., 0., 0.])/ntraj #calculated manually
    assert_almost_equal(refl, np.sum(expected_refl_array))
    assert_almost_equal(trans, np.sum(expected_trans_array))

    ########## test tir ##########
    
    # create new simulation objects to test tir
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    weights = np.ones((3,4))
    trajectories = mc.Trajectory([x_pos, y_pos, z_pos],[kx, ky, kz], weights, pol)
    system = StructuredSphere(species, small_n, small_n, microsphere_radius, structure='glass')
    results = mc.Results(trajectories, system, source, n_eff=n_eff, form=form)
    
    # calculate reflectance
    refl, trans = results.detect(detector)
    
    # compare to expected result
    # since the tir=True reruns the stuck trajectory, we don't know whether it will end up reflected or transmitted
    # all we can know is that the end refl + trans > 0.99
    assert_almost_equal(refl + trans, 1.) 


def test_trajectories():
    '''
    Test trajectory initialization for structured sphere 
    '''
    
    # set parameters
    nevents = 2
    ntraj = 3
    microsphere_radius = 5
    radius = sc.Quantity('150 nm')
    n_particle = sc.Quantity(1.5, '')
    n_matrix = sc.Quantity(1.0, '')
    n_medium = sc.Quantity(1.0, '')
    volume_fraction = 0.5
    
    # calculate n_sample according to bruggeman
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction) 

    # set up the system
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = StructuredSphere(species, n_matrix, n_medium, microsphere_radius, structure='glass')
    
    # Initialize the attributes of the trajectories
    pos0, dir0, weight0, pol0 = mc.initialize(nevents, ntraj, system, n_sample, seed=1)
    pos0 = sc.Quantity(pos0, 'um')
    dir0 = sc.Quantity(dir0, '')
    weight0 = sc.Quantity(weight0, '')

    # Create a Trajectory object
    trajectories = mc.Trajectory(pos0, dir0, weight0, pol0)
    
    
def test_get_angles_sphere():
    '''
    Make sure get_angles_sphere() returns correct angle values
    
    TODO: get_angles_sphere() may be depricated in refactored version. Make sure
    to check this function after refactoring
    '''
    # set parameters
    microsphere_radius = 5
    
    z_pos = np.array([[0,0,0,0],[1,1,1,1],[-1,11,2,11],[-2,12,4,12]])
    x_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,-0,0,0]])
    y_pos = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    indices = np.array([1,1,1,1])
    _, _, thetas = mc.get_angles_sphere(x_pos, y_pos, z_pos, microsphere_radius, indices, incident = True)
    assert_almost_equal(np.sum(thetas), 0.) 

def test_index_match():
    '''
    Compare reflectance for sphere where particle is index-matched to matrix
    and incident direction is +Z to fresnel film with infinite reflections
    '''
    
    # set parameters
    ntraj = 2
    nevents = 3
    wavelen = sc.Quantity('600 nm')
    radius = sc.Quantity('0.140 um')
    microsphere_radius = sc.Quantity('10 um')
    volume_fraction = sc.Quantity(0.55,'')
    n_particle = sc.Quantity(1.6,'')
    n_matrix = sc.Quantity(1.6,'')
    n_medium = sc.Quantity(1,'')
    
    # set up simulation
    source = Source(wavelen, polarization=None, incidence_angle=0)
    detector = DetectorMultScat(angle=0, length=np.inf, distance=0)
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = StructuredSphere(species, n_matrix, n_medium, microsphere_radius, structure='glass')
    
    results = mc.run(system, source, ntraj, nevents, seed=None, n_eff='bruggeman', form = 'auto')
    refl_sphere = results.detect(detector)
    
    # calculated by hand from fresnel infinite sum
    refl_fresnel_int = 0.053 # calculated by hand
    refl_exact = refl_fresnel_int + (1-refl_fresnel_int)**2*refl_fresnel_int/(1-refl_fresnel_int**2)
    assert_almost_equal(refl_sphere, refl_exact, decimal=3) 
    
    