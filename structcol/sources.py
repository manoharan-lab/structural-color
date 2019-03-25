# -*- coding: utf-8 -*-
# Copyright 2016 Vinothan N. Manoharan, Victoria Hwang, Anna B. Stephenson, 
# Solomon Barkley.
#
# This file is part of the structural-color python package.
#
# This package is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this package.  If not, see <http://www.gnu.org/licenses/>.

"""
This module creates classes for different types of light sources for Monte 
Carlo simulations of multiple scattering. 

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import structcol as sc

class Collimated:
    """
    Class that describes a collimated light source. 
    
    Attributes
    ----------
    wavelength: float (structcol.Quantity [length])
        Wavelength of incident light from the source.
    medium_index: float (structcol.Quantity [dimensionless])
        Refractive index of the medium in which the light source is located. 
        For most practical applications, the medium will be air so medium_index 
        will be 1. 
    pol: list-like (structcol.Quantity [dimensionless]) or None
        Polarization of the incident light from the source. If not None, the 
        list requires (x,y,z) components of the polarization vector. Default is 
        set to None to indicate unpolarized light. 
    incidence_angle: float (structcol.Quantity [rad or deg])
        Angle at which light incides on the sample. Default is set to 0 to 
        indicate perpendicular incidence. 
        
    """
    def __init__(self, wavelength, medium_index=sc.Quantity(1,''), pol=None, 
                 incidence_angle=sc.Quantity(0,'rad')):
        """
        Constructor for Glass object.
    
        Attributes
        ---------
        species: see Class attributes.
        volume_fraction: see Class attributes.

        """
        self.wavelength = wavelength
        self.medium_index = medium_index
        self.pol = pol
        self.incidence_angle = incidence_angle        
        