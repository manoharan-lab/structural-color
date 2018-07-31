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
This module creates containers classes for Monte Carlo simulations of 
multiple scattering. The containers are different types of geometric 
boundary conditions for the samples.

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import structcol as sc

class Sphere:
    """
    Class that describes spherical boundary conditions for a sample. Depending
    on how it is utilized, the Sphere object can be either a solid particle or
    contain an arrangement of particles. 
    
    Attributes
    ----------
    radius: float (structcol.Quantity [length])
        Radius of the sphere.
    index: float (structcol.Quantity [dimensionless])
        Refractive index of the sphere. If the Sphere object contains an 
        arrangement of particles, then this refractive index corresponds to 
        the "matrix" in which the particles are embedded. 
    filling: None or object from arrangements module
        Interior of the Sphere object. If None, the Sphere object will be a 
        solid homogeneous sphere. Otherwise, it will be an arrangement (for 
        example, a Glass object consisting of a glassy assembly of 
        nanoparticles). Default is set to None.
    pdi: float (structcol.Quantity [dimensionless])
        Polydispersity index of the sphere. Default is set to 0. 
    
    """
    
    def __init__(self, radius, index, filling=None, pdi=sc.Quantity(0,'')):
        """
        Constructor for Sphere object.
    
        Attributes
        ---------
        radius: see Class attributes.
        index: see Class attributes.
        filling: see Class attributes.
        pdi: see Class attributes.

        """
        self.radius = radius
        self.index = index
        self.filling = filling
        self.pdi = pdi
    
    

class Film:
    """
    Class that describes parallel plane boundary conditions for a sample. 
    Depending on how it is utilized, the Film object can be either a solid 
    film or contain an arrangement of particles, though the latter option will 
    be used most of the time. 
    
    Attributes
    ----------
    thickness: float (structcol.Quantity [length])
        Thickness of the film.
    index: float (structcol.Quantity [dimensionless])
        Refractive index of the film. If the Film object contains an 
        arrangement of particles, then this refractive index corresponds to 
        the "matrix" in which the particles are embedded. 
    filling: None or object from arrangements module
        Interior of the Sphere object. If None, the Film object will be a 
        solid homogeneous film. Otherwise, it will be an arrangement (for 
        example, a Glass object consisting of a glassy assembly of 
        nanoparticles). Default is set to None.

    """
    
    def __init__(self, thickness, index, filling=None):
        """
        Constructor for Film object.
    
        Attributes
        ---------
        radius: see Class attributes.
        index: see Class attributes.
        filling: see Class attributes.

        """
        self.thickness = thickness
        self.index = index
        self.filling = filling


    
class LayeredSphere:
    """
    Class that describes multilayered boundary conditions for a sample, 
    consisting of concentric shells or layers of varying refractive index, with
    the center layer being the core. The simplest LayeredSphere object is a 
    core-shell particle. If the shells have the same refractive index as the 
    core, then the LayeredSphere object outputs the same result as the 
    equivalent Sphere object. 
    
    Attributes
    ----------
    radius: list-like (structcol.Quantity [length])
        Array of radii of each layer, starting from the innermost (the core) to 
        the outermost layer. 
    index: list-like (structcol.Quantity [dimensionless])
        Array of refractive indices of each layer, starting from the innermost 
        (the core) to the outermost layer. If a layer contains an arrangement
        of particles, then this refractive index corresponds to the "matrix" 
        in which the particles are embedded. 
    filling: list-like of None and/or objects from arrangements module
        Array of the arrangements of each layer, starting from innermost (the
        core) to the outermost layer. If a layer has a filling of None, the 
        layer will have a constant and homogeneous index. Otherwise, the layer
        will have an arrangement (for example, a Glass object consisting of a 
        glassy assembly of nanoparticles). Default is set to None.
      
    """
    
    def __init__(self, radius, index, filling=None):
        """
        Constructor for LayeredSphere object.
    
        Attributes
        ---------
        radius: see Class attributes.
        index: see Class attributes.
        filling: see Class attributes.

        """
        self.radius = radius
        self.index = index
        self.filling = filling   
