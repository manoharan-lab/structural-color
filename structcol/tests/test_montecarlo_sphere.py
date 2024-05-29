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
from .. import detector as det
from .. import refractive_index as ri
import numpy as np
from numpy.testing import assert_almost_equal

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
assembly_radius = 5
volume_fraction = 0.5
n_particle = sc.Quantity(1.5, '')
n_matrix = sc.Quantity(1.0, '')
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction) 
angles = sc.Quantity(np.linspace(0.01,np.pi, 200), 'rad')  
wavelen = sc.Quantity('400.0 nm')

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2,0,2])

def test_trajectories():
    # Initialize runs
    nevents = 2
    ntrajectories = 3
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_matrix, n_sample, 'sphere', seed=1, sample_diameter=sc.Quantity('1.0 um'))
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    # Create a Trajectory object
    trajectories = mc.Trajectory(r0, k0, W0)
    
    
    
    
