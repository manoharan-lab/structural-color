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
Tests for the containers module (in structcol/containers.py)

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import structcol as sc
from .. containers import Sphere, Film, LayeredSphere

def test_sphere():
    # Test that the Sphere object is correctly created
    radius = sc.Quantity(100, 'nm')
    index = sc.Quantity(1.5, '')
    
    Sphere(radius, index, filling=None, pdi=0)


def test_film():
    # Test that the Film object is correctly created
    thickness = sc.Quantity(10, 'um')
    index = sc.Quantity(1.5, '')
    
    Film(thickness, index, filling=None)
    

def test_layered_sphere():
    # Test that the LayeredSphere object is correctly created
    core_radius = sc.Quantity(50, 'nm')
    shell_radius = sc.Quantity(100, 'nm')
    core_index = sc.Quantity(1.0, '')
    shell_index = sc.Quantity(1.5, '')
    
    LayeredSphere([core_radius, shell_radius], [core_index, shell_index], 
                  filling=[None, None])