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
Tests for the arrangements module (in structcol/arrangements.py)

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import structcol as sc
from .. sources import Collimated

def test_collimated():
    # Test that the Glass object is correctly created
    wavelength = sc.Quantity(500, 'nm')
    medium_index = sc.Quantity(1.0, '')
    incidence_angle = sc.Quantity(0, 'deg')
    
    Collimated(wavelength, medium_index, pol=None, 
               incidence_angle=incidence_angle)