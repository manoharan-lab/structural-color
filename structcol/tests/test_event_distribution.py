# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang, Annie Stephenson
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
from structcol import montecarlo as mc
from structcol import refractive_index as ri
from structcol import event_distribution as ed
from structcol import detector as det
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

# Monte Carlo parameters
ntrajectories = 30 # number of trajectories
nevents = 300 # number of scattering events in each trajectory

# source/detector properties
wavelength = sc.Quantity(np.array(550),'nm') # wavelength at which to run simulation 

# sample properties
particle_radius = sc.Quantity('140 nm') # radius of the particles
volume_fraction = sc.Quantity(0.56, '') # volume fraction of particles
thickness = sc.Quantity('10 um')
particle = 'ps'
matrix = 'air'
boundary = 'film'

# indices of refraction
n_particle = ri.n('polystyrene', wavelength) # refractive indices can be specified as pint quantities or
n_matrix = ri.n('vacuum', wavelength)      # called from the refractive_index module. n_matrix is the 
n_medium = ri.n('vacuum', wavelength)      # space within sample. n_medium is outside the sample.

# Calculate the effective refractive index of the sample
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)

# Calculate the phase function and scattering and absorption coefficients from the single scattering model
# (this absorption coefficient is of the scatterer, not of an absorber added to the system)
p, mu_scat, mu_abs = mc.calc_scat(particle_radius, n_particle, n_sample, volume_fraction, wavelength)
lscat = 1/mu_scat.magnitude # microns

# Initialize the trajectories
r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample, boundary)
r0 = sc.Quantity(r0, 'um')
k0 = sc.Quantity(k0, '')
W0 = sc.Quantity(W0, '')

# Generate a matrix of all the randomly sampled angles first 
sintheta, costheta, sinphi, cosphi, theta, _ = mc.sample_angles(nevents, ntrajectories, p)
sintheta = np.sin(theta)
costheta = np.cos(theta)

# Create step size distribution
step = mc.sample_step(nevents, ntrajectories, mu_scat)

# Create trajectories object
trajectories = mc.Trajectory(r0, k0, W0)

# Run photons
trajectories.absorb(mu_abs, step)                         
trajectories.scatter(sintheta, costheta, sinphi, cosphi)         
trajectories.move(step)

refl_indices, trans_indices,\
inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
trans_frac, refl_frac,\
refl_fresnel,\
trans_fresnel,\
reflectance,\
transmittance,\
tir_refl_bool,_,_ = det.calc_refl_trans(trajectories, thickness, n_medium, 
                                   n_sample, boundary, return_extra = True)

refl_events, trans_events = ed.calc_refl_trans_event(refl_per_traj, 
                                                     inc_refl_per_traj, 
                                                     trans_per_traj, 
                                                     refl_indices, 
                                                     trans_indices, 
                                                     nevents)

def test_refl_events():
    '''
    Check that refl_events is consistent with reflectance
    '''
    
    # sum of refl_events should be less than reflectance because it doesn't
    # contain correction terms for fresnel (and stuck for cases where that matters)
    assert_array_less(np.sum(refl_events), reflectance)
    
    # trajectories always propagate into the sample for first event, so none
    # can be reflected
    assert_equal(refl_events[1],0)
    
    # trajectories cannot be transmitted at interface before first scattering
    # event
    assert_equal(trans_events[0],0)
    
def test_fresnel_events():
    '''
    Check that fresnel corrections make sense
    '''
    refl_events_fresnel_avg = ed.calc_refl_event_fresnel_avg(refl_events, 
                                                             refl_indices, 
                                                             trans_indices, 
                                                             refl_fresnel, 
                                                             trans_fresnel,
                                                             refl_frac, 
                                                             trans_frac, 
                                                             nevents)
    
    pdf_refl, pdf_trans = ed.calc_pdf_scat(refl_events, trans_events, nevents)

    refl_events_fresnel_samp = ed.calc_refl_event_fresnel_pdf(refl_events, 
                                                              pdf_refl, 
                                                              pdf_trans, 
                                                              refl_indices, 
                                                              trans_indices,
                                                              refl_fresnel, 
                                                              trans_fresnel,
                                                              refl_frac, 
                                                              trans_frac, 
                                                              nevents)
    
    # check that average and sampling give same total
    assert_almost_equal(np.sum(refl_events_fresnel_avg), np.sum(refl_events_fresnel_samp))
    
    # check that reflectance from monte carlo gives same as fresnel reflected
    # summed reflectance from event distribution
    # TODO these should be equal to more decimals. Need to look into this. 
    assert_almost_equal(reflectance, np.sum(refl_events_fresnel_avg), decimal=1)
    
def test_tir_events():
    '''
    Check that totally internally reflected trajectories make sense
    '''
    tir_all,\
    tir_all_refl,\
    tir_single,\
    tir_single_refl,\
    tir_indices_single = ed.calc_tir(tir_refl_bool, refl_indices, 
                                     trans_indices, inc_refl_per_traj, 
                                     n_sample, 
                                     n_medium,
                                     boundary, 
                                     trajectories,
                                     thickness)
    
    # the reflected tir's should always be less than total tir's
    assert_array_less(np.sum(tir_single_refl), np.sum(tir_single))
    assert_array_less(np.sum(tir_all_refl), np.sum(tir_all))
    
    
    
    