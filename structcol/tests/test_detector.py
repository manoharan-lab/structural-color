#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:50:55 2018

@author: stephenson
"""
# imports
import numpy as np
import structcol as sc
import structcol.refractive_index as ri
from structcol import detector as det

# Properties of system
ntrajectories = 100                     # number of trajectories
nevents = 50                           # number of scattering events in each trajectory
wavelen = sc.Quantity('600 nm') 
radius = sc.Quantity('0.125 um')
volume_fraction = sc.Quantity(0.5, '')
n_particle = sc.Quantity(1.54, '')      # refractive indices can be specified as pint quantities or
n_matrix = ri.n('vacuum', wavelen)      # called from the refractive_index module. n_matrix is the 
n_medium = ri.n('vacuum', wavelen)      # space within sample. n_medium is outside the sample. 
                                        # n_particle and n_matrix can have complex indices if absorption is desired
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)

# Calculate the phase function and scattering and absorption coefficients from the single scattering model
p, mu_scat, mu_abs = det.calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen, mie_theory=False)

# Initialize the trajectories
r0, k0, W0 = det.initialize(nevents, ntrajectories, n_medium, n_sample, incidence_angle = 0.)
r0 = sc.Quantity(r0, 'um')
k0 = sc.Quantity(k0, '')
W0 = sc.Quantity(W0, '')

# Generate a matrix of all the randomly sampled angles first 
sintheta, costheta, sinphi, cosphi, _, _ = det.sample_angles(nevents, ntrajectories, p)

# Create step size distribution
step = det.sample_step(nevents, ntrajectories, mu_abs, mu_scat)
    
# Create trajectories object
trajectories = det.Trajectory(r0, k0, W0)

# Run photons
trajectories.absorb(mu_abs, step)                         
trajectories.scatter(sintheta, costheta, sinphi, cosphi)         
trajectories.move(step)

z_low = sc.Quantity('0.0 um')
thickness = sc.Quantity('50 um')

#reflectance, transmittance = det.calc_refl_trans(trajectories, thickness, z_low, n_medium, n_sample, 'film')
reflectance, transmittance = det.calc_refl_trans(trajectories, thickness, z_low, n_medium, n_sample, 'sphere')