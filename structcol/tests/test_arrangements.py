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
Tests for the sources module (in structcol/sources.py)

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import structcol as sc
from .. containers import Sphere
from .. arrangements import Glass, Paracrystal

def test_glass():
    # Test that the Glass object is correctly created
    radius = sc.Quantity(100, 'nm')
    index = sc.Quantity(1.5, '')
    species = Sphere(radius, index, filling=None, pdi=0)
    
    volume_fraction = sc.Quantity(0.6, '')
    Glass(species, volume_fraction)

def test_paracrystal():
    # Test that the Paracrystal object is correctly created
    radius = sc.Quantity(100, 'nm')
    index = sc.Quantity(1.5, '')
    species = Sphere(radius, index, filling=None, pdi=0)
    
    volume_fraction = sc.Quantity(0.6, '')
    sigma = sc.Quantity(0.15, '')
    Paracrystal(species, volume_fraction, sigma=sigma)