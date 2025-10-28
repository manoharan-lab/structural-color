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
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import structcol as sc
from .. import montecarlo as mc

# Define a system to be used for the tests
radius = sc.Quantity('150.0 nm')
volume_fraction = 0.5
wavelen = sc.Quantity('400.0 nm')
index_particle = sc.Index.constant(1.5)
index_matrix = sc.Index.constant(1.0)
index_medium = sc.index.vacuum
sphere = sc.Sphere(index_particle, radius)

model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                             index_medium)

def test_simulation():
    # Initialize runs. Since this test just checks to make sure a Simulation
    # object can be created, we don't need to give it a seeded random number
    # generator.
    nevents = 2
    ntrajectories = 3

    # Create a Simulation object
    trajectories = mc.Simulation(model, wavelen, nevents, ntrajectories,
                                 'sphere',
                                 sample_diameter = sc.Quantity('1.0 um'))
