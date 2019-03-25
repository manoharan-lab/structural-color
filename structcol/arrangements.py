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
This module creates arrangement classes for Monte Carlo simulations of 
multiple scattering. The arrangements are different types of sample assemblies. 

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import structcol as sc

class Glass:
    """
    Class that describes an amorphous or glassy assembly of particles. 
    
    Attributes
    ----------
    species: list-like
        List-like of objects describing the species of the assembly. For example, 
        if the Glass object is an assembly of two species of nanoparticles, 
        the species list will be [nanoparticle1, nanoparticle2], where 
        each nanoparticle is a Sphere object.  
    volume_fraction: list-like (structcol.Quantity [dimensionless])
        List-like of partial volume fractions of each species in the total 
        assembly. The volume fractions must be specified in the same order as 
        the species. For a species list [nanoparticle1, nanoparticle2], 
        the volume_fraction list will be [volume_fraction1, volume_fraction2]. 
        The sum of all the partial volume fractions must not be larger than
        0.64, which is the random closed packed volume fraction for amorphous
        assemblies. 

    """
    def __init__(self, species, volume_fraction):
        """
        Constructor for Glass object.
    
        Attributes
        ---------
        species: see Class attributes.
        volume_fraction: see Class attributes.

        """
        self.species = species
        self.volume_fraction = volume_fraction        


class Paracrystal:
    """
    Class that describes a paracrystalline assembly of particles. 
    
    Attributes
    ----------
    species: list-like
        List-like of objects describing the species of the assembly. For example, 
        if the Paracrystal object is an assembly of two species of nanoparticles, 
        the species list will be [nanoparticle1, nanoparticle2], where 
        each nanoparticle is a Sphere object.  
    volume_fraction: list-like (structcol.Quantity [dimensionless])
        List-like of partial volume fractions of each species in the total 
        assembly. The volume fractions must be specified in the same order as 
        the species. For a species list [nanoparticle1, nanoparticle2], 
        the volume_fraction list will be [volume_fraction1, volume_fraction2]. 
    sigma: list-like (structcol.Quantity [dimensionless])
        The standard deviation of a Gaussian representing the distribution of 
        particle/void spacings in the structure. A larger sigma will give more 
        broad peaks, and a smaller sigma more sharp peaks. Default value is set 
        to 0.15.  
    """
    def __init__(self, species, volume_fraction, sigma=sc.Quantity(0.15, '')):
        """
        Constructor for Paracrystal object.
    
        Attributes
        ---------
        species: see Class attributes.
        volume_fraction: see Class attributes.
        sigma: see Class attributes.
        
        """
        self.species = species
        self.volume_fraction = volume_fraction             
        self.sigma = sigma