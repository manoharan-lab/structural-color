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
from .main import Spheres, Film, Source
import os
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest

# Define a system to be used for the tests
nevents = 3
ntraj = 4
radius = sc.Quantity('150 nm')
volume_fraction = 0.5
n_particle = sc.Quantity(1.5, '')
n_matrix = sc.Quantity(1.0, '')
n_medium = sc.Quantity(1.0, '')
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction) 
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')  
wavelen = sc.Quantity('400 nm')
thickness = sc.Quantity(50, 'um')

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_sampling():
    # Test that 'calc_scat' runs
    phase_function, scat_coeff, abs_coeff = mc.calc_scat(radius, n_particle, n_sample,  
                                            volume_fraction, wavelen)
    
    # Test that 'sample_angles' runs
    mc.sample_angles(nevents, ntraj, phase_function)
    
    # Test that 'sample_step' runs
    mc.sample_step(nevents, ntraj, abs_coeff, scat_coeff)

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
    ntraj = 3
    pos0, dir0, weight0 = mc.initialize(nevents, ntraj, n_matrix, n_sample, seed=1)
    pos0 = sc.Quantity(pos0, 'um')
    dir0 = sc.Quantity(dir0, '')
    weight0 = sc.Quantity(weight0, '')

    # Create a Trajectory object
    trajectories = mc.Trajectory(pos0, dir0, weight0)
    
    # Test the absorb function
    abs_coeff = 1/sc.Quantity(10, 'um')    
    step = sc.Quantity(np.array([[1,1,1],[1,1,1]]), 'um')    
    trajectories.absorb(abs_coeff, step)     
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
    dirx = sc.Quantity(np.array([[0.,0.,0.],[0.,0.,0.]]), '')
    diry = sc.Quantity(np.array([[0.,0.,0.],[0.,0.,0.]]), '')
    dirz = sc.Quantity(np.array([[1.,1.,1.],[-1.,-1.,-1.]]), '')
    
    # Test the scatter function
    assert_almost_equal(trajectories.direction[0], dirx.magnitude)
    assert_almost_equal(trajectories.direction[1], diry.magnitude)
    assert_almost_equal(trajectories.direction[2], dirz.magnitude)
    
    # Test the move function    
    trajectories.move(step)
    assert_equal(trajectories.position[2], np.array([[0,0,0],[1,1,1],[0,0,0]]))

def test_reflection_core_shell():
    # test that the reflection of a non-core-shell system is the same as that
    # of a core-shell with a shell index matched with the core
    seed = 1
    nevents = 60
    ntraj = 30
    source = Source(wavelen, polarization=None, incidence_angle=0)
    detector = mc.Detector(angle=0, length=np.inf, distance=0) 

    # Reflection using a non-core-shell system
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    R, T = calc_montecarlo(nevents, ntraj, system, source, detector, seed, form='auto') 
    
    
    # Reflection using core-shells with the shell index-matched to the core
    radius_cs = sc.Quantity(np.array([100, 150]), 'nm')  # specify the radii from innermost to outermost layer
    n_particle_cs = sc.Quantity(np.array([1.5,1.5]), '')  # specify the index from innermost to outermost layer  
    
    # calculate the volume fractions of each layer
    vf_array = np.empty(len(radius_cs))
    r_array = np.array([0] + radius_cs.magnitude.tolist()) 
    for r in np.arange(len(r_array)-1):
        vf_array[r] = (r_array[r+1]**3-r_array[r]**3) / (r_array[-1:]**3) * volume_fraction

    species1 = Spheres(n_particle_cs, radius_cs, vf_array, pdi=0)
    species2 = Spheres(n_particle_cs, radius_cs, vf_array, pdi=0)    
    system_cs = Film([species1, species2], n_matrix, n_medium, thickness, structure='glass')
    
    R_cs, T_cs = calc_montecarlo(nevents, ntraj, system_cs, source, detector, seed, form='auto')     

    assert_almost_equal(R, R_cs)
    assert_almost_equal(T, T_cs)

    # Outputs before refactoring structcol
    R_before = 0.81382378303119451
    R_cs_before = 0.81382378303119451
    T_before = 0.1861762169688054
    T_cs_before = 0.1861762169688054
    
    assert_equal(R_before, R)
    assert_equal(R_cs_before, R_cs)
    assert_equal(T_before, T)
    assert_equal(T_cs_before, T_cs)
    
    ###########################################################################
    # Test that the reflectance is the same for a core-shell that absorbs (with
    # the same refractive indices for all layers) and a non-core-shell that 
    # absorbs with the same index
    # Reflection using a non-core-shell absorbing system
    n_particle_abs = sc.Quantity(1.5+0.001j, '')  
    species_abs = Spheres(n_particle_abs, radius, volume_fraction, pdi=0)
    system_abs = Film(species_abs, n_matrix, n_medium, thickness, structure='glass')
    R_abs, T_abs = calc_montecarlo(nevents, ntraj, system_abs, source, detector, seed, form='auto') 

    
    # Reflection using core-shells with the shell index-matched to the core
    n_particle_cs_abs = sc.Quantity(np.array([1.5+0.001j,1.5+0.001j]), '')  
    species1_abs = Spheres(n_particle_cs_abs, radius_cs, vf_array, pdi=0)
    species2_abs = Spheres(n_particle_cs_abs, radius_cs, vf_array, pdi=0)    
    system_cs_abs = Film([species1_abs, species2_abs], n_matrix, n_medium, thickness, structure='glass')
    
    R_cs_abs, T_cs_abs = calc_montecarlo(nevents, ntraj, system_cs_abs, source, detector, seed, form='auto')     

    assert_almost_equal(R_abs, R_cs_abs, decimal=3)
    assert_almost_equal(T_abs, T_cs_abs, decimal=3)

    # Outputs before refactoring structcol
    R_abs_before = 0.50534237684703909
    R_cs_abs_before = 0.50534237684642402
    T_abs_before = 0.017215194324142709
    T_cs_abs_before = 0.017215194324029608

    assert_equal(R_abs_before, R_abs)
    assert_equal(R_cs_abs_before, R_cs_abs)
    assert_equal(T_abs_before, T_abs)
    assert_equal(T_cs_abs_before, T_cs_abs)
    
    ###########################################################################
    # Same as previous test but with absorbing matrix as well
    # Reflection using a non-core-shell absorbing system
    n_particle_abs = sc.Quantity(1.5+0.001j, '')  
    n_matrix_abs = sc.Quantity(1.+0.001j, '')  
    species_abs = Spheres(n_particle_abs, radius, volume_fraction, pdi=0)
    system_abs = Film(species_abs, n_matrix_abs, n_medium, thickness, structure='glass')
    R_abs, T_abs = calc_montecarlo(nevents, ntraj, system_abs, source, detector, seed, form='auto') 

    
    # Reflection using core-shells with the shell index-matched to the core
    n_particle_cs_abs = sc.Quantity(np.array([1.5+0.001j,1.5+0.001j]), '')  
    species1_abs = Spheres(n_particle_cs_abs, radius_cs, vf_array, pdi=0)
    species2_abs = Spheres(n_particle_cs_abs, radius_cs, vf_array, pdi=0)    
    system_cs_abs = Film([species1_abs, species2_abs], n_matrix_abs, n_medium, thickness, structure='glass')
    
    R_cs_abs, T_cs_abs = calc_montecarlo(nevents, ntraj, system_cs_abs, source, detector, seed, form='auto')  

    assert_almost_equal(R_abs, R_cs_abs, decimal=3)
    assert_almost_equal(T_abs, T_cs_abs, decimal=3)

    # Outputs before refactoring structcol
    R_abs_before = 0.37384878890851575
    R_cs_abs_before = 0.37384878890851575
    T_abs_before = 0.002180700021951509
    T_cs_abs_before = 0.002180700021951509

    assert_equal(R_abs_before, R_abs)
    assert_equal(R_cs_abs_before, R_cs_abs)
    assert_equal(T_abs_before, T_abs)
    assert_equal(T_cs_abs_before, T_cs_abs)
    
    
def test_reflection_absorbing_particle_or_matrix():
    # test that the reflections with a real n_particle and with a complex
    # n_particle with a 0 imaginary component are the same 
    seed = 1
    nevents = 60
    ntraj = 30
    source = Source(wavelen, polarization=None, incidence_angle=0)
    detector = mc.Detector(angle=0, length=np.inf, distance=0) 

    # Reflection using non-absorbing particle
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    R, T = calc_montecarlo(nevents, ntraj, system, source, detector, seed, form='auto')      

    # Reflection using particle with an imaginary component of 0
    n_particle_abs = sc.Quantity(1.5 + 0j, '')
    species_abs = Spheres(n_particle_abs, radius, volume_fraction, pdi=0)
    system_part_abs = Film(species_abs, n_matrix, n_medium, thickness, structure='glass')
    R_part_abs, T_part_abs = calc_montecarlo(nevents, ntraj, system_part_abs, 
                                             source, detector, seed, form='auto')   
  
    assert_almost_equal(R, R_part_abs)
    assert_almost_equal(T, T_part_abs)
    
    # Outputs before refactoring structcol
    R_before = 0.81382378303119451
    R_part_abs_before = 0.81382378303119451
    T_before = 0.1861762169688054
    T_part_abs_before = 0.1861762169688054
    
    assert_equal(R_before, R)
    assert_equal(R_part_abs_before, R_part_abs)
    assert_equal(T_before, T)
    assert_equal(T_part_abs_before, T_part_abs)

    ###########################################################################
    # Same as previous test but with absorbing matrix
    # Reflection using matrix with an imaginary component of 0
    n_matrix_abs = sc.Quantity(1. + 0j, '')
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system_matrix_abs = Film(species, n_matrix_abs, n_medium, thickness, structure='glass')
    R_matrix_abs, T_matrix_abs = calc_montecarlo(nevents, ntraj, system_matrix_abs, 
                                                 source, detector, seed, form='auto')      
    
    assert_almost_equal(R, R_matrix_abs)
    assert_almost_equal(T, T_matrix_abs)
    
    # Outputs before refactoring structcol
    R_matrix_abs_before = 0.81382378303119451
    T_matrix_abs_before = 0.1861762169688054

    assert_equal(R_matrix_abs_before, R_matrix_abs)
    assert_equal(T_matrix_abs_before, T_matrix_abs)
    

def test_reflection_polydispersity():
    seed = 1
    nevents = 60
    ntraj = 30
    source = Source(wavelen, polarization=None, incidence_angle=0)
    detector = mc.Detector(angle=0, length=np.inf, distance=0)  
    volume_fraction1 = 0.4
    volume_fraction2 = 0.1
    
    ###########################################################################
    # Without absorption: test that the reflectance using with very small 
    # polydispersity is the same as the monodisperse case
    pdi = sc.Quantity(1e-7, '')  # monodisperse limit

    # with very small polydispersity
    species1 = Spheres(n_particle, radius, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle, radius, volume_fraction2, pdi=pdi)    
    system = Film([species1, species2], n_matrix, n_medium, thickness, 
                     structure='glass')
                     
    # with no polydispersity                 
    species1_no_pdi = Spheres(n_particle, radius, volume_fraction1, pdi=0)
    species2_no_pdi = Spheres(n_particle, radius, volume_fraction2, pdi=0) 
    system_no_pdi = Film([species1_no_pdi, species2_no_pdi], n_matrix, n_medium, 
                             thickness, structure='glass')

    R_small_pdi, T_small_pdi = calc_montecarlo(nevents, ntraj, system, source, 
                                               detector, seed, form='auto')                          
    R_no_pdi, T_no_pdi = calc_montecarlo(nevents, ntraj, system_no_pdi, source, 
                                         detector, seed, form='auto')                                 

    assert_almost_equal(R_no_pdi, R_small_pdi)
    assert_almost_equal(T_no_pdi, T_small_pdi)

    # Outputs before refactoring structcol
    R_no_pdi_before = 0.81382378303119451
    R_small_pdi_before = 0.81382378303119451
    T_no_pdi_before = 0.1861762169688054
    T_small_pdi_before = 0.1861762169688054

    assert_equal(R_no_pdi_before, R_no_pdi)
    assert_equal(R_small_pdi_before, R_small_pdi)
    assert_equal(T_no_pdi_before, T_no_pdi)
    assert_equal(T_small_pdi_before, T_small_pdi)
    
    ###########################################################################
    # With absorption: test that the reflectance using with very small 
    # polydispersity is the same as the monodisperse case  
    n_particle_abs = sc.Quantity(1.5+0.0001j, '')  
    n_matrix_abs = sc.Quantity(1.+0.0001j, '')  
    
    # with very small polydispersity
    species1_abs = Spheres(n_particle_abs, radius, volume_fraction1, pdi=pdi)
    species2_abs = Spheres(n_particle_abs, radius, volume_fraction2, pdi=pdi)
    system_abs = Film([species1_abs, species2_abs], n_matrix_abs, n_medium, thickness, 
                          structure='glass')
                     
    # with no polydispersity                 
    species1_no_pdi_abs = Spheres(n_particle_abs, radius, volume_fraction1, pdi=0)
    species2_no_pdi_abs = Spheres(n_particle_abs, radius, volume_fraction2, pdi=0)
    system_no_pdi_abs = Film([species1_no_pdi_abs, species2_no_pdi_abs], n_matrix_abs, n_medium, 
                             thickness, structure='glass')

    R_small_pdi_abs, T_small_pdi_abs = calc_montecarlo(nevents, ntraj, system_abs, source, 
                                               detector, seed, form='auto') 
    R_no_pdi_abs, T_no_pdi_abs = calc_montecarlo(nevents, ntraj, system_no_pdi_abs, source, 
                                         detector, seed, form='auto')  

    assert_almost_equal(R_no_pdi_abs, R_small_pdi_abs, decimal=3)
    assert_almost_equal(T_no_pdi_abs, T_small_pdi_abs, decimal=3)
    
    # Outputs before refactoring structcol
    R_no_pdi_abs_before = 0.74182070115289855
    R_small_pdi_abs_before = 0.74153254583803685
    T_no_pdi_abs_before = 0.083823525277616467
    T_small_pdi_abs_before = 0.083720861809212316
    
    assert_equal(R_no_pdi_abs_before, R_no_pdi_abs)
    assert_equal(R_small_pdi_abs_before, R_small_pdi_abs)
    assert_equal(T_no_pdi_abs_before, T_no_pdi_abs)
    assert_equal(T_small_pdi_abs_before, T_small_pdi_abs)
    
    ###########################################################################
    # test that the reflectance is the same for a polydisperse monospecies
    # and a bispecies with equal types of particles
    pdi = sc.Quantity(1e-1, '')
    
    # monospecies            
    species = Spheres(n_particle, radius, volume_fraction, pdi=pdi)
    system_mono = Film(species, n_matrix, n_medium, thickness, structure='glass')
    
    # bispecies
    species1 = Spheres(n_particle, radius, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle, radius, volume_fraction2, pdi=pdi)
    system_bi = Film([species1, species2], n_matrix, n_medium, thickness, 
                     structure='glass')

    R_mono, T_mono = calc_montecarlo(nevents, ntraj, system_mono, source, 
                                     detector, seed, form='auto')                             
    R_bi, T_bi = calc_montecarlo(nevents, ntraj, system_bi, source, 
                                 detector, seed, form='auto')                                 

    assert_equal(R_mono, R_bi)
    assert_equal(T_mono, T_bi)
    
    ###########################################################################
    # test that the reflectance is the same regardless of the order in which
    # the species are specified
    radius2 = sc.Quantity('100 nm')
    
    # bispecies
    species1 = Spheres(n_particle, radius, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle, radius2, volume_fraction2, pdi=pdi)
    system1 = Film([species1, species2], n_matrix, n_medium, thickness, 
                        structure='glass')
    system2 = Film([species2, species1], n_matrix, n_medium, thickness, 
                        structure='glass')
                                     
    R, T = calc_montecarlo(nevents, ntraj, system1, source, detector, seed, 
                           form='auto')             
    R2, T2 = calc_montecarlo(nevents, ntraj, system2, source, detector, seed, 
                             form='auto')                               
                                   
    assert_almost_equal(R, R2)
    assert_almost_equal(T, T2)


def test_throw_valueerror_for_polydisperse_core_shells(): 
# test that a valueerror is raised when trying to run polydisperse core-shells                 
    with pytest.raises(ValueError):
        seed = 1
        nevents = 10
        ntraj = 5
        
        # define core-shell particles
        radius_cs = sc.Quantity(np.array([100, 150]), 'nm')  # specify the radii from innermost to outermost layer
        n_particle_cs = sc.Quantity(np.array([1.5,1.5]), '')  # specify the index from innermost to outermost layer  
        pdi = sc.Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit

        # calculate the volume fractions of each layer
        vf_array = np.empty(len(radius_cs))
        r_array = np.array([0] + radius_cs.magnitude.tolist()) 
        for r in np.arange(len(r_array)-1):
            vf_array[r] = (r_array[r+1]**3-r_array[r]**3) / (r_array[-1:]**3) * volume_fraction

        # define two species to make it a polydisperse core-shell system
        species1 = Spheres(n_particle_cs, radius_cs, vf_array, pdi=pdi)
        species2 = Spheres(n_particle_cs, radius_cs, vf_array, pdi=pdi)
        
        system = Film([species1, species2], n_matrix, n_medium, thickness, 
                         structure='glass')
        source = Source(wavelen, polarization=None, 
                           incidence_angle=0)
        detector = mc.Detector(angle=0, length=np.inf, distance=0)   

        R_cs, T_cs = calc_montecarlo(nevents, ntraj, system, source, detector, 
                                     seed, form='auto')  
                                     
                                    
'''
This test will no longer be relevant in the refactored version
def test_throw_valueerror_for_polydisperse_unspecified_parameters(): 
# test that a valueerror is raised when the system is polydisperse and radius2
# concentration or pdi are not specified                 
    with pytest.raises(ValueError):
        seed = 1
        nevents = 10
        ntraj = 5
        
        radius1 = sc.Quantity(100, 'nm') 
        radius2 = sc.Quantity(150, 'nm')
        n_particle = sc.Quantity(1.5, '') 
        pdi = sc.Quantity(1e-7, '')  # monodisperse limit
        
        # calculate the volume fractions of each layer
        radius_cs = np.array([radius1.magnitude, radius2.to(radius1.units).magnitude], radius1.units)    
        vf_array = np.empty(len(radius_cs))
        r_array = np.array([0] + radius_cs.tolist()) 
        for r in np.arange(len(r_array)-1):
            vf_array[r] = (r_array[r+1]**3-r_array[r]**3) / (r_array[-1:]**3) * volume_fraction

        species1 = Spheres(n_particle, radius1, vf_array[0], pdi=pdi)
        species2 = Spheres(n_particle, radius2, vf_array[1], pdi=pdi)
        
        system = Film([species1, species2], n_matrix, n_medium, thickness, 
                         structure='glass')
        source = Source(wavelen, polarization=None, 
                           incidence_angle=0)
        detector = mc.Detector(angle=0, length=np.inf, distance=0)   

        R_cs, T_cs = calc_montecarlo(nevents, ntraj, system, source, detector, 
                                     seed, form='auto')  
'''

def calc_montecarlo(nevents, ntraj, system, source, detector, seed, 
                    form='auto'):
                        
    results = mc.run(system, source, ntraj, nevents, seed=seed, form=form)
    normalized_intensity = results.calc_scattering(detector)
    
    return normalized_intensity


#def run():
#    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)  # for core-shells, volume_fraction must be array of vf 
#                                                                # from innermost to outermost layers
#    phase_function, scat_coeff, abs_coeff = mc.calc_scat(system, source)
#    pos0, dir0, weight0 , pol0 = mc.initialize(nevents, ntraj, n_medium, n_sample, 
#                                               seed, incidence_angle, polarization)
#    sintheta, costheta, sinphi, cosphi theta, phi = mc.sample_angles(nevents, 
#                                                                     ntraj, 
#                                                                     phase_function, 
#                                                                     polarization)
#    step = mc.sample_step(nevents, ntraj, abs_coeff, scat_coeff)
#
#    if polarization is not None:
#        singamma, cosgamma, pol_x_loc, pol_y_loc = mc.polarize(theta, phi, 
#                                                               system, source)
#    else:
#        singamma = None
#        cosgamma = None
#
#    trajectories = mc.Trajectory(pos0, dir0, weight0, pol0)
#    trajectories.absorb(abs_scat, step)
#    trajectories.scatter(sintheta, costheta, sinphi, cosphi, singamma, cosgamma)
#    trajectories.move(step)
#    
#    results = mc.Results(trajectories, system, source)
#    
#    return(results)