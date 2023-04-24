# -*- coding: utf-8 -*-
# Copyright 2016 Vinothan N. Manoharan, Victoria Hwang, Anna B. Stephenson, Solomon Barkley
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
This package uses a Monte Carlo approach to model multiple scattering of
photons in a medium.

References
----------
[1] K. Wood, B. Whitney, J. Bjorkman, M. Wolff. “Introduction to Monte Carlo
Radiation Transfer” (July 2013).
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import pymie as pm
from pymie import mie, size_parameter, index_ratio
from pymie import multilayer_sphere_lib as msl
from . import model
from . import refraction
from . import normalize
from . import select_events
from . import event_distribution as ed
import numpy as np
from numpy.random import random as random
import structcol as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy
import seaborn as sns
from . import structure
import copy

eps = 1.e-9

# some templates to use when refactoring later
class MCSimulation:
    """
    Input parameters and methods for running a Monte Carlo calculation.
    
    Attributes
    ----------
    
    Methods
    -------
    run()
    
    """
    def __init__(self):
        """
        Constructor for MCSimulation object.
        
        Parameters
        ----------
    
        """
        pass

    def run(self):
        """
        Run the simulation.
    
        Parameters
        ----------
        
        Returns
        -------
        
        MCResult object:
            results of simulation
        
        """
        pass

class MCResult:
    """
    Results from running Monte Carlo simulation.
    
    Attributes
    ----------
    Methods
    -------
    
    """
    def __init__(self):
        """
        Constructor for MCResult object.
        
        Parameters
        ----------
        
        """
        pass

class Trajectory:
    """
    Class that describes trajectories of photons packets in a scattering
    and/or absorbing medium.
    
    Attributes
    ----------
    position: ndarray (structcol.Quantity [length])
        array of position vectors in cartesian coordinates of n trajectories
    direction: ndarray (structcol.Quantity [dimensionless])
        array of direction of propagation vectors in cartesian coordinates
        of n trajectories after every scattering event
    weight: ndarray (structcol.Quantity [dimensionless])
        array of photon packet weights for absorption modeling of n
        trajectories
    field: ndarray (structcol.Quantity [dimensionless])
        electric fields of photon packets in cartesian coordinates
    nevents: int
        number of scattering events
    
    Methods
    -------
    absorb(mu_abs, step_size)
        calculate absorption at each scattering event with given absorption 
        coefficient and step size.
    scatter(sintheta, costheta, sinphi, cosphi)
        calculate directions of propagation after each scattering event with
        given randomly sampled scattering and azimuthal angles.
    move(mu_scat)
        calculate new positions of the trajectory with given scattering 
        coefficient, obtained from either Mie theory or the single scattering 
        model.
    calc_fields()
    plot_coord(ntraj, three_dim=False)
        plot positions of trajectories as a function of number scattering
        events.
    
    """

    def __init__(self, position, direction, weight,
                 fields=None):
        """
        # TODO: remove phase and polarization as they have been replaced by fields
        Constructor for Trajectory object.
        
        Attributes
        ----------
        position : see Class attributes
            Dimensions of (3, nevents+1, number of trajectories)
        direction : see Class attributes
            Dimensions of (3, nevents, number of trajectories)
        weight : see Class attributes
            Dimensions of (nevents, number of trajectories)
        
        """
      
        self.position = position
        self.direction = direction
        self.weight = weight
        self.fields = fields

    @property
    def nevents(self):
        return self.weight.shape[0]

    def absorb(self, mu_abs, step_size):
        """
        Calculates absorption of photon packet due to traveling the sample 
        between scattering events. Absorption is modeled as a reduction of a 
        photon packet's weight using Beer-Lambert's law. 
        
        Parameters
        ----------
        mu_abs: ndarray (structcol.Quantity [1/length])
            Absorption coefficient of the sample as an effective medium.
        step_size: ndarray (structcol.Quantity [length])
            Step size of packet (sampled from scattering lengths).
            
        """
        # beer lambert
        weight = self.weight*np.exp(-(mu_abs * np.cumsum(step_size[:,:], 
                                                         axis=0)).to(''))
        self.weight = sc.Quantity(weight)
        
    def scatter(self, sintheta, costheta, sinphi, cosphi):
        """
        Calculates the directions of propagation after scattering (for either
        'scattering plane' or 'cartesian' polarizations).
        
        At a scattering event, a photon packet adopts a new direction of
        propagation, which is randomly sampled from the phase function. The new
        direction of propagation also changes the polarization direction.
        
        Parameters
        ----------
        sintheta, costheta, sinphi, cosphi : array_like
            Sines and cosines of scattering (theta) and azimuthal (phi) angles
            sampled from the phase function. Theta and phi are angles that are
            defined with respect to the previous corresponding direction of
            propagation. Thus, they are defined in a local spherical coordinate
            system. All have dimensions of (nevents, ntrajectories).
        
        """
        kn = self.direction.magnitude

        # the 0th event is the inital direction which does not change when the
        # photon first enters the sample. It only changes after the first scattering
        # event. The trajectory steps first, then scatters. The reason that we don't
        # extend the arange to nevents + 1 is that we still count this original 
        # initialized direction as an event, since the step size must be sampled
        # once it enters the material.
        for n in np.arange(1, self.nevents):
            # Calculate the new x, y, z coordinates of the propagation direction
            # using the following equations, which can be derived by using matrix
            # operations to perform a rotation about the y-axis by angle theta
            # followed by a rotation about the z-axis by angle phi
            # see pg 105 in A.B. Stephenson lab notebook 1 for derivation and
            # notes
            kn[0, n, :] = ((kn[0, n - 1, :] * costheta[n - 1, :]
                            + kn[2, n - 1, :] * sintheta[n - 1, :]) *
                           cosphi[n - 1, :]) - kn[1, n - 1, :] * sinphi[n - 1, :]
            kn[1, n, :] = ((kn[0, n - 1, :] * costheta[n - 1, :]
                            + kn[2, n - 1, :] * sintheta[n - 1, :]) *
                           sinphi[n - 1, :]) + kn[1, n - 1, :] * cosphi[n - 1, :]
            kn[2, n, :] = -kn[0, n - 1, :] * sintheta[n - 1, :] + kn[2, n - 1, :] * costheta[n - 1, :]

        # Update all the directions of the trajectories
        self.direction = sc.Quantity(kn, self.direction.units)

    def calc_fields(self, theta, phi, sintheta, costheta, sinphi, cosphi,
                    n_particle, n_sample, radius, wavelen, step, volume_fraction, 
                    fine_roughness=0, tir_refl_bool=None):
        """
        Calculates local x and y polarization rotated in reference frame where 
        initial polarization is x-polarized. Assumes the incident light is in 
        +z direction

        Within one trajectory, fields is accounted for by calculating
        the form factor using Mie theory, which gives the scattered fields 
        and phase.
        
        To calculate the effects of interference between different trajectories, 
        we include the phase shift calculated from Mie theory, as well as the 
        phase shift due to the distances travelled. The structure factor contribution
        comes in through the phase shift due to the distances travelled. 
        
        Here is an outline of how it is implemented: 
        
        We start by calculating the amplitude scattering matrix in the
        parallel/perpendicular basis. We then multiply the matrix by the 
        initial fields. This gives the scattered fields purely due to 
        the form factor.
        
        Then we add these phase shifts to the phase shift incurred due to distance
        travelled, calculated as k*distance. 
        
        We then rotate these phase values into local x and y coordinates, 
        and after that, rotate them into global x, y, and z coordinates. 
        
        Parameters
        ----------
        theta: 2d array
            Theta angles.
        phi: 2d array
            Phi angles.
        sintheta, costheta, sinphi, cosphi : array_like
            Sines and cosines of scattering (theta) and azimuthal (phi) angles
            sampled from the phase function. Theta and phi are angles that are
            defined with respect to the previous corresponding direction of
            propagation. Thus, they are defined in a local spherical coordinate
            system. All have dimensions of (nevents, ntrajectories).
        n_particle: float
            Index of refraction of particle.
        n_sample: float
            Index of refraction of sample.
        radius: float
            Radius of particle.
        wavelen: float
            Wavelength.
        step: ndarray (structcol.Quantity [length])
            Step sizes of packets (sampled from scattering lengths).
        volume_fraction: float (structcol.Quantity [dimensionless])
            Volume fraction of the sample. 
        fine_roughness: float (structcol.Quantity [dimensionless])
            Fraction of the sample area that has fine roughness. Should be between 
            0 and 1. For ex, a value of 0.3 means that 30% of incident light will 
            hit fine surface roughness (e.g. will "see" a Mie scatterer first). The 
            rest of the light will see a smooth surface, which could be flat or 
            have coarse roughness (long in the lengthscale of light).
        tir_refl_bool: 2d array of booleans (shape: nevents, ntraj)
            Describes whether a trajectory gets totally internally reflected at any 
            event and also exits in the negative direction to contribute to reflectance
        
        Calculates:
        ----------
        En: ndarray, shape: (3, nevents, ntrajectories)
            Electric field vector for each trajectory and event 
            in global coordinates
        """
        m = index_ratio(n_particle, n_sample)
        x = size_parameter(wavelen, n_sample, radius)
        k = 2 * np.pi * n_sample.magnitude / wavelen.magnitude
        step = step.magnitude
        ntraj = theta.shape[1]
         
        # calculate the mie amplitude scattering matrix
        # we need to calculate the full matrix, rather than just the vector
        # scattering amplitude, because each matrix element contributes to 
        # the changes in E field
        S1, S2, S3, S4 = mie.amplitude_scattering_matrix(m, x, theta,
                                                         coordinate_system='cartesian', 
                                                         phis=phi)
        
        # mutliply the scat amp mats
        En = self.fields
        if isinstance(En, sc.Quantity):
            En = En.magnitude

        Ex = En[0, 0, :]
        Ey = En[1, 0, :]

        # Ex and Ey are the initialized as the incident field vectors. 
        # To get the Ex and Ey at each event, we have to multiply by the scattering
        # amplitude matrix, cumulatively for each event. 
        # this gives us the local Ex and Ey vectors
        # Reminder: there is one less sampled angle than event number, because
        # the first event propogates straight into the sample. 
        # Note: this basis assumes that 
        # the direction of propagation is the +z direction. 
        for n in np.arange(0, self.nevents - 1): 
            Ex = S2[n, :] * Ex + S3[n, :] * Ey
            Ey = S4[n, :] * Ex + S1[n, :] * Ey
            if n + 2 > self.nevents:
                break
            else:
                # 0th event is before sample, the 1st event has no rotation
                En[0, n + 2, :] = Ex 
                En[1, n + 2, :] = Ey
                
        # Deal with tir
        if tir_refl_bool is not None:
            # get indices for the first TIR event for each trajectory
            tir_indices = np.argmax(np.vstack([np.zeros(ntraj), tir_refl_bool]), axis=0)
            
            # select the tir event for each trajectory
            theta_1 = select_events(theta, tir_indices - 2)
            kz_tir = select_events(self.direction[2], tir_indices)
            theta_r = np.arccos(kz_tir)
            theta_tir = 2 * (np.pi / 2 - theta_r)
            costheta_tir = np.cos(theta_1 + theta_tir)
            sintheta_tir = np.sin(theta_1 + theta_tir)
            tir_ind_theta = tir_indices - 2
            tir_ind_theta[tir_ind_theta < 0] = 0
            costheta[tir_ind_theta, :] = costheta_tir
            sintheta[tir_ind_theta, :] = sintheta_tir

        # Rotate to global coords
        # Start with event 2 because the 0th event contains the initialized
        # values from before the field enters the sample. The 1st event contains
        # the values for the field after entering the sample, but before scattering
        for n in np.arange(2, self.nevents + 1):
            # Calculate the new x, y, z coordinates of the propagation direction 
            # using the following equations, which can be derived by using matrix
            # operations to perform a rotation about the y-axis by angle theta
            # followed by a rotation about the z-axis by angle phi
            Ex = ((En[0,n:,:]*costheta[n-2,:] + En[2,n:,:]*sintheta[n-2,:])*
                    cosphi[n-2,:]) - En[1,n:,:]*sinphi[n-2,:]
            Ey = ((En[0,n:,:]*costheta[n-2,:] + En[2,n:,:]*sintheta[n-2,:])*
                  sinphi[n-2,:]) + En[1,n:,:]*cosphi[n-2,:]
            Ez =  -En[0,n:,:]*sintheta[n-2,:] + En[2,n:,:]*costheta[n-2,:]  
            En[:,n:,:] = Ex, Ey, Ez
        
        # Calculate the structure factor field contribution.
        # Insert a row of zeros since first event does not change direction
        # Note that this will only work for normal incidence.
        theta2 = np.insert(theta,0,np.zeros(ntraj),axis=0) 
        qd = 4*np.array(np.abs(x)).max()*np.sin(theta2/2)  
        
        # calculate the step propagation factor
        step_cumul = np.abs(k)*np.cumsum(step, axis=0)#step #
        step_phase_factor = np.exp(1j*np.abs(k)*step_cumul)
        
        # multiply the fields by the phase propagation due to structure factor
        # of the initial trajectories
        # should multiply by 1 for trajectories do not have fine roughness
        ntraj_fine = int(np.round(ntraj * fine_roughness))
        En[0, 1:, :] = En[0, 1:, :] * step_phase_factor
        En[1, 1:, :] = En[1, 1:, :] * step_phase_factor
        En[2, 1:, :] = En[2, 1:, :] * step_phase_factor

        # Normalize
        En[0, :, :], En[1, :, :], En[2, :, :] = normalize(En[0, :, :], En[1, :, :], En[2, :, :], return_nan=False)
        
        self.fields = sc.Quantity(En,self.fields.units)

    def move(self, step):
        """
        Calculates positions of photon packets in all the trajectories.
        After each scattering event, the photon packet gets a new position
        based on the previous position, the step size, and the direction of
        propagation.
        
        Parameters
        ----------
        step : ndarray (structcol.Quantity [length])
            Step sizes between scattering events in each of the trajectories.
        
        """

        displacement = self.position
        if not isinstance(displacement, sc.Quantity):
            step = step.to('um').magnitude
            direction = self.direction.magnitude
        else:
            direction = self.direction
        displacement[:, 1:, :] = step * direction

        # The array of positions is a cumulative sum of all of the
        # displacements
        self.position[0] = np.cumsum(displacement[0,:,:], axis=0)
        self.position[1] = np.cumsum(displacement[1,:,:], axis=0)
        self.position[2] = np.cumsum(displacement[2,:,:], axis=0)

    def plot_coord(self, ntraj, three_dim=False):
        """
        Plots the cartesian coordinates of the trajectories as a function of
        the number of scattering events.
        
        Parameters
        ----------
        ntraj : int
            Number of trajectories.
        three_dim : bool
            If True, it plots the trajectories' coordinates in 3D.
        
        """

        colormap = plt.cm.gist_ncar
        colors = itertools.cycle([colormap(i) for i in
                                  np.linspace(0, 0.9, ntraj)])

        f, ax = plt.subplots(3, figsize=(8,17), sharex=True)

        ax[0].plot(np.arange(len(self.position[0,:,0].magnitude)),
                   self.position[0,:,:].magnitude, '-')
        ax[0].set_title('Positions during trajectories')
        ax[0].set_ylabel('x (' + str(self.position.units) + ')')

        ax[1].plot(np.arange(len(self.position[1,:,0].magnitude)),
                   self.position[1,:,:].magnitude, '-')
        ax[1].set_ylabel('y (' + str(self.position.units) + ')')

        ax[2].plot(np.arange(len(self.position[2,:,0].magnitude)),
                   self.position[2,:,:].magnitude, '-')
        ax[2].set_ylabel('z (' + str(self.position.units) + ')')
        ax[2].set_xlabel('scattering event')

        if three_dim == True:
            fig = plt.figure(figsize=(8,6))
            ax3D = fig.add_subplot(111, projection='3d')
            ax3D.set_xlabel('x (' + str(self.position.units) + ')')
            ax3D.set_ylabel('y (' + str(self.position.units) + ')')
            ax3D.set_zlabel('z (' + str(self.position.units) + ')')
            ax3D.set_title('Positions during trajectories')

            for n in np.arange(ntraj):
                ax3D.scatter(self.position[0,:,n].magnitude, 
                             self.position[1,:,n].magnitude,
                             self.position[2,:,n].magnitude,
                             color=next(colors))


def initialize(nevents, ntraj, n_medium, n_sample, boundary, seed=None,
               incidence_theta_min=sc.Quantity(0.,'rad'), 
               incidence_theta_max=sc.Quantity(0.,'rad'), 
               incidence_theta_data=None, 
               incidence_phi_min=sc.Quantity(0.,'rad'), 
               incidence_phi_max=sc.Quantity(2*np.pi,'rad'), 
               incidence_phi_data=None,
               plot_initial=False, 
               spot_size=sc.Quantity('1 um'), 
               sample_diameter=None,
               coarse_roughness=0.,
               coherent=False,
               fields=False):
    """
    Sets the trajectories' initial conditions (position, direction, weight,
    and polarization if set to True).
    The initial positions are determined randomly in the x-y plane.

    If boundary is a sphere, the initial z-positions are confined to the 
    surface of a sphere. If boundary is a film, the initial z-positions are set
    to zero.
    
    If incidence_theta_min and incidence_theta_max are both set to 0, the 
    initial propagation direction is set to be 1 at z, meaning that the photon 
    packets point straight down in z. The initial directions are corrected for 
    refraction, for either type of boundary and for any incidence angle.

    * Notes:
    - for sphere boundary, incidence angle currently must be 0
    
    Parameters
    ----------
    nevents: int
        Number of scattering events
    ntraj: int
        Number of trajectories
    n_medium: float (structcol.Quantity [dimensionless])
        Refractive index of the medium.
    n_sample: float (structcol.Quantity [dimensionless])
        Refractive index of the sample.
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options are
        'film' or 'sphere'
    seed: int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are actually random.
    incidence_theta_min: float (structcol.Quantity [angle])
        Minimum value for theta when it incides onto the sample.
        Should be >= 0 and < pi/2.
    incidence_theta_max: float (structcol.Quantity [angle])
        Maximum value for theta when it incides onto the sample.
        Should be >= 0 and < pi/2.
    incidence_theta_data: array (structcol.Quantity [angle]) (optional)
        Array of values for the incident theta for each trajectory. Length of 
        the array must therefore be the same as number of trajectories. If
        None, the code will randomly sample theta angles from a uniform 
        distribution between incidence_theta_min and incidence_theta_max. If
        user does not specify units, values must be in radians. 
    incidence_phi_min: float (structcol.Quantity [angle])
        Minimum value for phi when it incides onto the sample.
        Should be >= 0 and <= pi.
    incidence_phi_max: float (structcol.Quantity [angle])
        Maximum value for phi when it incides onto the sample.
        Should be >= 0 and <= pi.
    incidence_phi_data: array (structcol.Quantity [angle]) (optional)
        Array of values for the incident phi for each trajectory. Length of 
        the array must therefore be the same as number of trajectories. If
        None, the code will randomly sample phi angles from a uniform 
        distribution between incidence_phi_min and incidence_phi_max.  If
        user does not specify units, values must be in radians. 
    plot_inital: boolean
        If plot_initial is set to True, function will create a 3d plot showing
        initial positions and directions of trajectories before entering the 
        sphere and directly after refraction correction upon entering the 
        sphere.
    spot_size: float (structcol.Quantity [length])
        For film sample, side length of a square spot size. For sphere sample
        diameter of a circular spot size. 
    sample_diameter: None or float (None type or structcol.Quantity [length])
        Diameter of the sample. Default is None. Should be None if sample
        geometry is a film. Should be float equal to the sphere diameter if 
        sample is a sphere.
    coarse_roughness : float (can be structcol.Quantity [dimensionless])
        Coarse surface roughness should be included when the roughness is large
        on the scale of the wavelength of light. This means that light 
        encounters a locally smooth surface that has a slope relative to the 
        z=0 plane. Then the model corrects the Fresnel reflection and refraction 
        to account for the different angles of incidence due to the roughness. 
        The coarse_roughness parameter is the rms slope of the surface. If 
        included, it should be larger than 0. There is no upper bound, but when 
        the coarse roughness tends to infinity, the surface becomes too "spiky" 
        and light can no longer hit it, which reduces the reflectance down to 0. 
    fields: boolean
        If True, also returns the initial fields of trajectories
    coherent: boolean
        If True, assumes the intial relative phases between trajectories are zero.
        If coherent is set to True while fields is set to False, then the
        coherent value is ignored, since there can be no coherence without taking
        into account the fields. 
    
    Returns
    -------
    r0: 3D array-like (structcol.Quantity [length])
        Trajectory positions. Has shape of (3, number of events + 1, number 
        of trajectories). r0[0,0,:] contains random x-positions within a circle 
        on the x-y plane whose radius is the sphere radius. r0[1, 0, :] contains
        random y-positions within the same circle on the x-y plane. r0[2, 0, :]
        contains z-positions on the top hemisphere at the sphere boundary. The 
        rest of the elements are initialized to zero.
    k0: 3D array-like (structcol.Quantity [dimensionless])
        Initial direction of propagation. Has shape of (3, number of events,
        number of trajectories). k0[0,:,:] and k0[1,:,:] are initalized to zero,
        and k0[2,0,:] is initalized to 1.
    weight0: 3D array-like (structcol.Quantity [dimensionless])
        Initial weight. Has shape of (number of events, number of trajectories)
        - Note that the photon weight represents the fraction of 
        that particular photon that is propagated through the sample. It does 
        not represent the photon's weight relative to other photons. the weight0
        array is initialized to 1 because you start with the full weight of the 
        initial photons. If you wanted to make the relative weights of photons
        different, you would need to introduce a new variable (e.g relative 
        intensity) that me, NOT change the intialization of the weights array.
        - Also Note that the size of the weights array it nevents*ntraj, NOT
        nevents+1, ntraj. This may at first seem counterintuitive because
        physically, we can associate a weight to a photon at each position 
        (which would call for a dimension nevents+1), not at each event. 
        However, there is no need to keep track of the weight at the first 
        event; The weight, by definition, must initially be 1 for each photon. 
        Adding an additional row of ones to this array would be unecessary and
        would contribute to less readable code in the calculation of absorptance,
        reflectance, and transmittance. Therefore the weights array begins with 
        the weight of the photons after their first event.
    pol0: (optional) 3D array-like (structcol.Quantity[dimensionless])
        Initial polarization vector in global coordinates. Has shape of 
        (number of events, number of trajectories). Only returns initial 
        linearly, x-polarized light.
    kz0_rot : array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that  
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the 
        surface. The array size is (1, ntraj). Only returned if coarse_roughness
        is set to > 0. 
    kz0_refl : array_like (structcol.Quantity [dimensionless])
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global 
        coordinate system. The array size is (1, ntraj). Only returned if 
        coarse_roughness is set to > 0. 
    
    Reference
    ---------
    B. v. Ginneken, M. Stavridi, J. J. Koenderink, “Diffuse and specular 
    reflectance from rough surfaces”, Applied Optics, 37, 1 (1998) (has 
    definition of rsm slope of the surface).
    
    """
    
    if seed is not None:
        np.random.seed([seed]) # uncomment
        
    # get the spot size magnitude to multiply by initial x and y positions
    spot_size_magnitude = spot_size.to('um').magnitude
    
    # get the sample radius as a float 
    if isinstance(sample_diameter, sc.Quantity):
        sample_radius = sample_diameter.to('um').magnitude/2
    
    # Initial position. The position array has one more row than the direction
    # and weight arrays because it includes the starting positions on the x-y
    # plane
    r0 = np.zeros((3, nevents+1, ntraj))

    # Create an empty array of the initial direction cosines of the right size
    k0 = np.zeros((3, nevents, ntraj))
    
    # Initial weight
    weight0 = np.ones((nevents, ntraj))
    
    if boundary == 'film':
        
        # raise error if user inputs a value for sphere diameter
        if sample_diameter is not None:
            raise ValueError('for film geometry, sample_diameter must be set\
                             to None')
        # randomly choose x positions on interval [0,1]
        r0[0,0,:] = random((1,ntraj))*spot_size_magnitude 
        
        # randomly choose y positions on interval [0,1]
        r0[1,0,:] = random((1,ntraj))*spot_size_magnitude
        
        # initialize the incident angles theta and phi. The user can input 
        # data or sample randomly from a uniform distribution between a min and 
        # a max incident angles. 
        if incidence_theta_data is not None: 
            if len(incidence_theta_data) != ntraj:
                raise ValueError('length of incidence_theta_data must be equal\
                to number of trajectories')
            theta = incidence_theta_data
        else: 
            incidence_theta_min = incidence_theta_min.to('rad').magnitude
            incidence_theta_max = incidence_theta_max.to('rad').magnitude
            theta = np.random.uniform(incidence_theta_min, incidence_theta_max, ntraj)

        if incidence_phi_data is not None: 
            if len(incidence_phi_data) != ntraj:
                raise ValueError('length of incidence_phi_data must be equal\
                to number of trajectories')
            phi = incidence_phi_data
        else: 
            incidence_phi_min = incidence_phi_min.to('rad').magnitude
            incidence_phi_max = incidence_phi_max.to('rad').magnitude
            phi = np.random.uniform(incidence_phi_min, incidence_phi_max, ntraj)

        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        
    if boundary == 'sphere':
        
        # raise error if user forgets to input a value for the sphere diameter
        if sample_diameter is None:
            raise ValueError('for sphere geometry, sample_diameter must be \
                             a physical quantity, not None')
            
        # randomly choose r on interval [0,1] and multiply by spot size radius
        r = np.sqrt(random(ntraj))*spot_size_magnitude/2
        
        # randomly choose th on interval [0,2*pi]
        th = 2*np.pi*random(ntraj)
        
        # convert to x and y, so that the points are randomly distributed 
        # across the cross sectional area of the sphere
        # for details, see: https://mathworld.wolfram.com/DiskPointPicking.html
        r0[0,0,:] = r*np.cos(th) 
        r0[1,0,:] = r*np.sin(th)
        
        # calculate z-positions from x- and y-positions
        r0[2,0,:] = sample_radius-np.sqrt(sample_radius**2 - r0[0,0,:]**2 - r0[1,0,:]**2)
    
        # find the minus normal vectors of the sphere at the initial positions
        neg_normal = np.zeros((3, ntraj)) 
        r0_magnitude = np.sqrt(r0[0,0,:]**2 + r0[1,0,:]**2 + (r0[2,0,:]-sample_radius)**2)
        neg_normal[0,:] = -r0[0,0,:]/r0_magnitude
        neg_normal[1,:] = -r0[1,0,:]/r0_magnitude
        neg_normal[2,:] = -(r0[2,0,:]-sample_radius)/r0_magnitude
        
        # solve for theta and phi for these samples
        theta = np.arccos(neg_normal[2,:])
        cosphi = neg_normal[0,:]/np.sin(theta)
        sinphi = neg_normal[1,:]/np.sin(theta)

    # If there is no coarse roughness (e.g. surface is flat)
    if coarse_roughness == 0:
        # Refraction of incident light upon entering the sample
        # TODO: only real part of n_sample should be used                             
        # for the calculation of angles of integration? Or abs(n_sample)? 
        theta = refraction(theta, np.abs(n_medium), np.abs(n_sample))
        if isinstance(theta, sc.Quantity):
            theta = theta.magnitude
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
    
        # calculate new directions using refracted theta and initial phi
        k0[0,0,:] = sintheta * cosphi
        k0[1,0,:] = sintheta * sinphi
        k0[2,0,:] = costheta
    
        # plot the initial positions and directions of the trajectories
        if plot_initial == True and boundary == 'sphere':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_ylim([-sample_radius, sample_radius])
            ax.set_xlim([-sample_radius, sample_radius])
            ax.set_zlim([0, sample_radius])
            ax.set_title('Initial Positions')
            ax.view_init(-164,-155)
            X, Y, Z, U, V, W = [r0[0,0,:],r0[1,0,:],r0[2,0,:],k0[0,0,:], k0[1,0,:], k0[2,0,:]]
            ax.quiver(X, Y, Z, U, V, W, color = 'g')
            
            X, Y, Z, U, V, W = [r0[0,0,:],r0[1,0,:],r0[2,0,:],np.zeros(ntraj), np.zeros(ntraj), np.ones(ntraj)]
            ax.quiver(X, Y, Z, U, V, W)
            
            # draw wireframe hemisphere
            u, v = np.mgrid[0:2*np.pi:20j, np.pi/2:0:10j]
            x = sample_radius*np.cos(u)*np.sin(v)
            y = sample_radius*np.sin(u)*np.sin(v)
            z = sample_radius-sample_radius*np.cos(v)
            ax.plot_wireframe(x, y, z, color=[0.8,0.8,0.8])
            
            
        init_traj_props = [r0, k0, weight0]
    
    # if the surface has coarse roughness
    else:
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
    
        # calculate new directions using refracted theta and initial phi
        k0[0,0,:] = sintheta * cosphi
        k0[1,0,:] = sintheta * sinphi
        k0[2,0,:] = costheta
        
        k0, kz0_rot, kz0_refl = coarse_roughness_enter(k0, 
                                                 n_medium,
                                                 n_sample,
                                                 coarse_roughness,
                                                 boundary)
        init_traj_props = [r0, k0, weight0, kz0_rot, kz0_refl]
                                                 
    if fields:
        # The field is initialized with nevents+1 because we want to save
        # the value of the field from before the photon enters the sample
        fields0 = np.zeros((3, nevents+1, ntraj), dtype = 'complex')                  
        # initialize for unpolarized, incoherent light
        if coherent:
            phase_x = np.ones(ntraj)
            phase_y = np.ones(ntraj)
        else:
            phase_x = np.random.random(ntraj)*2*np.pi
            phase_y = np.random.random(ntraj)*2*np.pi
        fields0[0,0,:] = np.exp(phase_x*1j)
        fields0[1,0,:] = np.exp(phase_y*1j)
        fields0x, fields0y, _ = normalize(fields0[0,0,:], fields0[1,0,:], 0)
        fields0[0,0,:] = fields0x
        fields0[1,0,:] = fields0y
        
        # first step into the sample is same 
        fields0[0,1,:] = fields0x
        fields0[1,1,:] = fields0y
        init_traj_props.append(fields0)
    return init_traj_props
    


def calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen,
              radius2=None, 
              concentration=None, 
              pdi=None, 
              polydisperse=False,
              fields = False, 
              fine_roughness=0, 
              min_angle = 0.01, 
              num_angles = 200, 
              num_phis = 300,
              structure_type = 'glass', 
              form_type = 'sphere', 
              structure_s_data=None, 
              structure_qd_data=None, 
              n_matrix=None,
              effective_medium_struct=True, 
              effective_medium_form=True):
    """
    Calculates the phase function and scattering coefficient from either the
    single scattering model or Mie theory. Calculates the absorption coefficient
    from Mie theory.
    
    Parameters
    ----------
    radius : float (structcol.Quantity [length])
        Radius of scatterer. 
    n_particle : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the particle.
    n_sample : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample. 
    volume_fraction : float (structcol.Quantity [dimensionless])
        Volume fraction of the sample. 
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    radius2 : float (structcol.Quantity [length])
        Mean radius of secondary scatterer. Specify only if the system is 
        binary, meaning that there are two mean particle radii (for example,
        one small and one large).
    concentration : 2-element array (structcol.Quantity [dimensionless])
        Concentration of each scatterer if the system is binary. For 
        polydisperse monospecies systems, specify the concentration as 
        [0., 1.]. The concentrations must add up to 1. 
    pdi : 2-element array (structcol.Quantity [dimensionless])
        Polydispersity index of each scatterer if the system is polydisperse. 
        For polydisperse monospecies systems, specify the pdi as a 2-element
        array with repeating values (for example, [0.01, 0.01]).
    polydisperse : bool
        If True, it uses the polydisperse form and structure factors. If set to
        True, radius2, concentration, and pdi must be specified. 
    fields: bool
        If True, returns phase function as function of theta and phi, so
        it can be used in field calculations
    fine_roughness: float (structcol.Quantity [dimensionless])
        When the sample has surface roughness that is comparable to the 
        wavelength of light, then the first step is calculated with Mie theory
        because light "sees" the Mie scatterer first instead of the sample as a
        whole. After taking the first step, light is inside the sample and is
        scattered in in the usual way, with the phase function based on the 
        effective medium approximation. This parameter should be between 0 and 
        1 and corresponds to the fraction of the sample area that has fine 
        roughness. For ex, a value of 0.3 means that 30% of incident light will
        hit fine surface roughness (e.g. will "see" a Mie scatterer first). The 
        rest of the light will see a smooth surface, which could be flat or 
        have coarse roughness (long in the lengthscale of light).
    min_angle: float
        min_angle to prevent error because structure factor is zero at theta=0
    num_angles: int
        Sets the number of thetas at which phase function p will be calculated.
    num_phis: int
        Sets the number of phis at which phase function p will be calculated. 
        Only used if polarization is True. 
    structure_type: string or None
        structure factor desired for calculation. Can be 'glass', 'paracrystal', 
        'polydisperse', 'data', or None. 
    form_type: string or None
        form factor desired for calculation. Can be 'sphere', 'polydisperse', 
        or None.
    structure_s_data: None or 1d array
        if structure_type is 'data', the structure factor data must be provided
        here in the form of a one dimensional array 
    structure_qd_array: None of 1d array
        if structure_type is 'data', the qd data must be provided here in the 
        form of a one dimensional array 
    n_matrix : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the matrix. It must be specified when the fine
        roughness is > 0. When there is fine roughness, we assume that light  
        goes from the index of the matrix to the index of the scatterer. Thus 
        we assume that fine roughness particles are not embedded in an
        effective medium. 

    Returns
    -------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    mu_scat : float or 2-element array (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering 
        model. When fine_roughness is larger than 0, mu_scat is a 2-element
        array, where the first element is the scattering coefficient from either
        Mie theory or single scattering model, and the second element is the 
        scattering coefficient from Mie theory. 
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient of the sample as an effective medium.
    
    Notes
    -----
    The phase function is given by:
        p = diff. scatt. cross section / cscat
    The single scattering model calculates the differential cross section and
    the total cross section. In a non-absorbing system, we can choose to 
    calculate these from Mie theory:
        diff. scat. cross section = S11 / k^2
        p = S11 / (k^2 * cscat)
        (Bohren and Huffmann, chapter 13.3)
    """

    # calculate parameters for scattering calculations
    x_eff = None
    if effective_medium_form and effective_medium_struct:
        n_sample = n_sample
    if effective_medium_struct and not effective_medium_form:
        n_sample_eff = n_sample
        x_eff = size_parameter(wavelen, n_sample_eff, radius)
        n_sample = n_matrix
    if not effective_medium_form and not effective_medium_struct:
        n_sample = n_matrix

    k = 2 * np.pi * n_sample / wavelen    
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    # radius and radius2 should be in the same units (for polydisperse samples)    
    if radius2 is not None:
        radius2 = radius2.to(radius.units)
    if radius2 is None:
        radius2 = radius    
    
    # General number density formula for binary systems, converges to monospecies 
    # formula when the concentration of either particle goes to zero. When the
    # system is monospecies, define a concentration array to be able to use the
    # general formula.
    if concentration is None:
        concentration = sc.Quantity(np.array([1,0]), '')
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = 1/(radius.max()**3 + radius2.max()**3 * concentration[1]/concentration[0])
        term2 = 1/(radius2.max()**3 + radius.max()**3 * concentration[0]/concentration[1])
    np.seterr(divide='warn', invalid='warn')
    number_density = 3.0 * volume_fraction / (4.0 * np.pi) * (term1 + term2)
    # if the system is polydisperse, use the polydisperse form and structure 
    # factors
    if polydisperse == True:
        if radius2 is None or concentration is None or pdi is None:
            raise ValueError('must specify diameters, concentration, and pdi for polydisperperse systems')
        
        if len(np.atleast_1d(m)) > 1:
            raise ValueError('cannot handle polydispersity in core-shell particles')
        
        form_type = 'polydisperse'
        structure_type = 'polydisperse'
    
    # define the mean diameters in case the system is polydisperse    
    mean_diameters = sc.Quantity(np.array([2*radius.magnitude, 2*radius2.magnitude]),
                                 radius.units)
                         
    # calculate the absorption coefficient
    mu_abs = 4*np.pi*n_sample.imag/wavelen
    
    # Define angles at which phase function will be calculated, based on 
    # whether light is polarized or unpolarized
    # Scattering angles (typically from a small angle to pi). A non-zero small 
    # angle is needed because in the single scattering model, if the analytic 
    # formula is used, S(q=0) returns nan. To prevent any errors or warnings, 
    # set the minimum value of angles to be a small value, such as 0.01.        
    angles = sc.Quantity(np.linspace(min_angle, np.pi, num_angles), 'rad') 
    
    if fields:
        coordinate_system = 'cartesian'
        phis = sc.Quantity(np.linspace(min_angle, 2*np.pi, num_phis), 'rad') 
        phis, thetas = np.meshgrid(phis, angles) # theta dimension must come first     
    else:
        thetas = angles
        coordinate_system = 'scattering plane'
        phis=None

  
    # calculate the phase function
    p, cscat_total = phase_function(m, x, thetas, volume_fraction, 
                                    k, number_density,
                                    wavelen=wavelen, 
                                    diameters=mean_diameters, 
                                    concentration=concentration, 
                                    pdi=pdi, n_sample=n_sample,
                                    form_type=form_type,
                                    structure_type=structure_type,
                                    coordinate_system=coordinate_system,
                                    phis = phis,
                                    structure_s_data=structure_s_data,
                                    structure_qd_data=structure_qd_data,
                                    x_eff=x_eff)
    mu_scat = number_density * cscat_total
    
    # Here, the resulting units of mu_scat and mu_abs are nm^2/um^3. Thus, we 
    # simplify the units to 1/um 
    mu_scat = mu_scat.to('1/um')
    mu_abs = mu_abs.to('1/um')
  
    # if there is fine surface roughness, also calculate and return the scatt 
    # coeff from Mie theory. We assume that fine roughness particles are in the 
    # matrix and not in the effective sample medium. 
    if fine_roughness > 0.:
        if n_matrix is None:
            raise ValueError('need to specify n_matrix if fine_roughness > 0')
        m = index_ratio(n_particle, n_matrix)
        x = size_parameter(wavelen, n_matrix, radius)
        k = 2 * np.pi * n_matrix / wavelen  
        
        _, cscat_total_mie = phase_function(m, x, thetas, volume_fraction, 
                                            k, number_density, 
                                            wavelen=wavelen, 
                                            diameters=mean_diameters, 
                                            concentration=concentration, 
                                            pdi=pdi, n_sample=n_matrix,
                                            form_type=form_type,
                                            structure_type=None,
                                            coordinate_system=coordinate_system,
                                            phis=phis)
        mu_scat_mie = number_density * cscat_total_mie
        mu_scat_mie = mu_scat_mie.to('1/um')        
        mu_scat = sc.Quantity(np.array([mu_scat.magnitude, 
                                        mu_scat_mie.magnitude]), '1/um')
 
    return p, mu_scat, mu_abs
    

def phase_function(m, x, angles, volume_fraction, k, number_density,
                   wavelen=None, 
                   diameters=None, 
                   concentration=None, 
                   pdi=None, 
                   n_sample=None, 
                   form_type='sphere', 
                   structure_type='glass', 
                   coordinate_system = 'scattering plane', 
                   phis=None, 
                   structure_s_data=None, 
                   structure_qd_data=None,
                   x_eff=None):
    """
    Calculates the phase function (the phase function is the same for absorbing 
    and non-absorbing systems).
    
    Parameters:
    ----------
    m: float
        index ratio between the particle and sample
    x: float
        size parameter
    angles: array (sc.Quantity [rad])
        theta angles at which to calculate phase function
    volume_fraction: float (sc.Quantity [dimensionless])
    k: float (sc.Quantity [1/length])
        k vector. k = 2*pi*n_sample / wavelength
    number_density: float (sc.Quantity [1/length^3])
    wavelen: float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    diameters: float (structcol.Quantity [length])
        Mean diameters of secondary scatterer. Specify only if the system is 
        binary, meaning that there are two mean particle radii (for example,
        one small and one large).
    concentration: 2-element array (structcol.Quantity [dimensionless])
        Concentration of each scatterer if the system is binary. For 
        polydisperse monospecies systems, specify the concentration as 
        [0., 1.]. The concentrations must add up to 1. 
    pdi: 2-element array (structcol.Quantity [dimensionless])
        Polydispersity index of each scatterer if the system is polydisperse. 
        For polydisperse monospecies systems, specify the pdi as a 2-element
        array with repeating values (for example, [0.01, 0.01]).
    n_sample : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample. 
    form_type: str or None
        form factor desired for calculation. Can be 'sphere', 'polydisperse', 
        or None.
    structure_type: str or None
        structure factor desired for calculation. Can be 'glass', 'paracrystal', 
        'polydisperse', 'data', or None. 
    coordinate_system: string
        default value 'scattering plane' means scattering calculations will be 
        carried out in the basis defined by basis vectors parallel and 
        perpendicular to scattering plane. Variable also accepts value 
        'cartesian' which scattering calculations will be carried out in the 
        basis defined by basis vectors x and y in the lab frame, with z 
        as the direction of propagation.
    phis: array (sc.Quantity [rad])
        phi angles at which to calculate phase function
        structure_type: string or None
        structure factor desired for calculation. Can be 'glass', 'paracrystal', 
        'polydisperse', 'data', or None. 
    structure_s_data: None or 1d array
        if structure_type is 'data', the structure factor data must be provided
        here in the form of a one dimensional array 
    structure_qd_array: None of 1d array
        if structure_type is 'data', the qd data must be provided here in the 
        form of a one dimensional array 
        
    Returns:
    --------
    p: array
        phase function for unpolarized light    
    cscat_total: float
        total scattering cross section for unpolarized light
        
    """  
    ksquared = np.abs(k)**2  
    
    if form_type=='polydisperse':
        distance = diameters/2
    else:
        distance = diameters.max()/2 
    
    # Note that we ignore near fields throughout structcol since we assume 
    # that the scattering length is larger than the distance at which near
    # fields are significant (~order of the wavelength of light). In the 
    # future, we might want to include near field effects. In that case, we 
    # need to make sure to pass near_fields = True in 
    # mie.diff_scat_intensity_complex_medium(). The default is False. 
    # Also note that the diff_cscat_par and perp will actuallly be 
    # the values diff_cscat_x and y if coordinate_system is cartesian. 
    diff_cscat_par, diff_cscat_perp = \
         model.differential_cross_section(m, x, angles, volume_fraction,
                                             structure_type=structure_type,
                                             form_type=form_type,
                                             diameters=diameters,
                                             coordinate_system=coordinate_system,
                                             phis=phis,
                                             concentration=concentration,
                                             pdi=pdi, 
                                             wavelen=wavelen, 
                                             n_matrix=n_sample, 
                                             k=k, 
                                             distance=distance,
                                             structure_s_data=structure_s_data,
                                             structure_qd_data=structure_qd_data, 
                                             x_eff=x_eff)
    
    # If in cartesian coordinate system, integrate the differential cross
    # section using integration functions in mie.py that can handle cartesian
    # coordinates. Also includes absorption.
    # TODO make this work for polydisperse
    if coordinate_system=='cartesian':
        thetas_1d = angles[:,0] 
        phis_1d = phis[0,:]
        
        # note that the diff_cscat_par and perp calculated above
        # will actually be diff_cscat_x and y 
        cscat_total = mie.integrate_intensity_complex_medium(diff_cscat_par,
                                                             diff_cscat_perp,
                                                             distance,
                                                             thetas_1d, k,
                                                             coordinate_system='cartesian',
                                                             phis=phis_1d)[0]
    
    # If absorption and not cartesian coords, integrate the differential cross 
    # section using integration functions in mie.py that use absorption
    elif np.abs(k.imag.magnitude)> 0.:
        # TODO implement cartesian for polydisperse
        if form_type=='polydisperse' and len(concentration)>1:
            # When the system is binary and absorbing, we integrate the 
            # polydisperse differential cross section at the surface of each
            # component (meaning at a distance of each mean radius). Then we
            # do a number average the total cross sections. 
            cscat_total1, cscat_total_par1, cscat_total_perp1, _, _ = \
                mie.integrate_intensity_complex_medium(diff_cscat_par, 
                                                       diff_cscat_perp, 
                                                       distance[0],angles,k)  
            cscat_total2, cscat_total_par2, cscat_total_perp2, _, _ = \
                mie.integrate_intensity_complex_medium(diff_cscat_par, 
                                                       diff_cscat_perp, 
                                                       distance[1],angles,k)
            cscat_total = cscat_total1 * concentration[0] + cscat_total2 * concentration[1]
        
        else: 
            cscat_total = mie.integrate_intensity_complex_medium(diff_cscat_par, 
                                                                 diff_cscat_perp, 
                                                                 distance,
                                                                 angles,k,
                                                                 coordinate_system=coordinate_system)[0]     
    
    # if there is no absorption in the system, Integrate with function in model
    else:
              
        cscat_total_par = model._integrate_cross_section(diff_cscat_par,
                                                      1.0/ksquared, angles)
        cscat_total_perp = model._integrate_cross_section(diff_cscat_perp,
                                                      1.0/ksquared, angles)
        cscat_total = (cscat_total_par + cscat_total_perp)/2.0

    # calculate the phase function
    p = (diff_cscat_par + diff_cscat_perp)/(np.sum(diff_cscat_par + diff_cscat_perp))

    return(p, cscat_total)


def sample_angles(nevents, ntraj, p, min_angle=0.01):
    """
    Samples scattering angles (theta) and azimuthal angles (phi) 
    
    if phase function p is 1d, phi is sampled from uniform distribution, and 
    theta from phase function distribution.
    
    if phase function p is 2d, both theta and phi are sampled from p. Note that
    theta must come first in the shape of the phase function
    
    Parameters
    ----------
    nevents : int
        Number of scattering events.
    ntraj : int
        Number of trajectories.
    p : array_like (structcol.Quantity [dimensionless])
        Phase function values returned from 'phase_function'.
    min_angle: float
        min_angle to prevent error because structure factor is zero at theta=0
    
    Returns
    -------
    sintheta, costheta, sinphi, cosphi, theta, phi : ndarray
        Sampled azimuthal and scattering angles, and their sines and cosines.
    
    """   
   
    if isinstance(p,sc.Quantity):
        p = p.magnitude
    num_theta = len(p)
    
    # The direction for the first event is defined upon initialization
    # so we only need to sample nevents-1.
    # In previous versions of the code, we sampled nevents, which gave us an 
    # extra sampled angle that was never used. While this did not lead to incorrect
    # results, it led to inconsistencies in indexing which had the potential
    # to create future bugs. Sampling one less angle fixes this issue.
    nevents = nevents-1 
    
    # Scattering angles for the phase function calculation (typically from 0 to 
    # pi). A non-zero minimum angle is needed because in the single scattering 
    # model, if the analytic formula is used, S(q=0) returns nan.
    thetas = sc.Quantity(np.linspace(min_angle, np.pi, num_theta), 'rad') 
    thetas = thetas.magnitude
    
    if len(p.shape)==1:# if p depends on theta 
 
        # Random sampling of azimuthal angle phi from uniform distribution [0 -
        # 2pi]
        rand = np.random.random((nevents,ntraj))
        phi = 2*np.pi*rand
        
    
        # make sure probability is normalized
        prob = p * np.sin(thetas)*2*np.pi    # prob is integral of p in solid angle
        prob_norm = prob/sum(prob)           # normalize to make it add up to 1
        
        # Randomly sample scattering angle theta
        theta = np.array([np.random.choice(thetas, ntraj, p = prob_norm)
                          for i in range(nevents)])
        
    if len(p.shape)==2: # if p depends on theta and phi
        
        # get the number of phis from the shape of the phase function
        num_phi = p.shape[1]
            
        # sum for theta axis to get phi probabilities
        p_phi = np.sum(p, axis = 0)
            
        # define phi values from which to sample 
        phis = sc.Quantity(np.linspace(min_angle,2*np.pi, num_phi), 'rad') 
        phis = phis.magnitude
    
        # sample indices for phi values
        phi_ind = np.array([np.random.choice(num_phi, ntraj, p = p_phi/np.sum(p_phi))
                                for i in range(nevents)])
            
        # sample thetas based on sampled phi values
        theta_ind = np.zeros((nevents,ntraj))
        theta = np.zeros((nevents,ntraj))
        phi = np.zeros((nevents,ntraj))
        for i in range(nevents):
            for j in range(ntraj):
                p_theta = p[:,phi_ind[i,j]]*np.sin(thetas)
                theta_ind[i,j] = np.random.choice(num_theta, p = p_theta/np.sum(p_theta))
                theta[i,j] = thetas[int(theta_ind[i,j])]
                phi[i,j] = phis[int(phi_ind[i,j])]
            
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    return sintheta, costheta, sinphi, cosphi, theta, phi


def sample_step(nevents, ntraj, mu_scat, fine_roughness=0.):
    """
    Samples step sizes from exponential distribution.
    
    Parameters
    ----------
    nevents : int
        Number of scattering events.
    ntraj : int
        Number of trajectories.
    mu_scat : float or 2-element array (structcol.Quantity [1/length])
        Scattering coefficient. When fine_roughness is larger than 0, mu_scat 
        is a 2-element array, where the first element is the scattering 
        coefficient from either Mie theory or single scattering model, and the 
        second element is the scattering coefficient from Mie theory.
    fine_roughness : float (structcol.Quantity [dimensionless])
        Fraction of the sample area that has fine roughness. Should be between 
        0 and 1. For ex, a value of 0.3 means that 30% of incident light will 
        hit fine surface roughness (e.g. will "see" a Mie scatterer first). The 
        rest of the light will see a smooth surface, which could be flat or 
        have coarse roughness (long in the lengthscale of light). 

    Returns
    -------
    step : ndarray
        Sampled step sizes for all trajectories and scattering events.
    
    """
     
    if fine_roughness > 1. or fine_roughness < 0.:
        raise ValueError('fine roughness fraction must be between 0 and 1')
    
    # check whether mu_scat contains two values
    if len(np.array([mu_scat.magnitude]).flatten()) > 1:
        mu_scat, mu_scat_mie = mu_scat
    else:
        mu_scat_mie = None

    # Generate array of random numbers from 0 to 1
    rand = np.random.random((nevents,ntraj)) #uncomment

    # sample step sizes
    step = -np.log(1.0-rand) / mu_scat

    # If there is fine surface roughness, sample the first step from Mie theory
    # for the number of trajectories set by fine_roughness
    if mu_scat_mie is not None:
        ntraj_mie = int(round(ntraj * fine_roughness))
        rand_ntraj = np.random.random(ntraj_mie)
        step[0,0:ntraj_mie] = -np.log(1.0-rand_ntraj) / mu_scat_mie
    
    return step

def coarse_roughness_enter(k0, n_medium, n_sample,
                           coarse_roughness, boundary):
    '''
    Calculates new initial directions based on the coarse roughness of the sample.
    
    Parameters
    ----------
    k0: 3D array-like (structcol.Quantity [dimensionless])
        Initial direction of propagation. Has shape of (3, number of events,
        number of trajectories). k0[0,:,:] and k0[1,:,:] are initalized to zero,
        and k0[2,0,:] is initalized to 1.
    n_medium: float (structcol.Quantity [dimensionless])
        Refractive index of the medium.
    n_sample: float (structcol.Quantity [dimensionless])
        Refractive index of the sample.
    coarse_roughness : float (can be structcol.Quantity [dimensionless])
        Coarse surface roughness should be included when the roughness is large
        on the scale of the wavelength of light. This means that light 
        encounters a locally smooth surface that has a slope relative to the 
        z=0 plane. Then the model corrects the Fresnel reflection and refraction 
        to account for the different angles of incidence due to the roughness. 
        The coarse_roughness parameter is the rms slope of the surface. If 
        included, it should be larger than 0. There is no upper bound, but when 
        the coarse roughness tends to infinity, the surface becomes too "spiky" 
        and light can no longer hit it, which reduces the reflectance down to 0. 
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options are
        'film' or 'sphere.' Coarse roughness is currently only implemented for 
        a film.
    
    Returns
    -------
    k0_rough: 3D array-like (structcol.Quantity [dimensionless])
        Initial direction of propagation, corrected for coarse roughness.
    kz0_rot : array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that  
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the 
        surface. The array size is (1, ntraj). Only returned if coarse_roughness
        is set to > 0. 
    kz0_refl : array_like (structcol.Quantity [dimensionless])
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global 
        coordinate system. The array size is (1, ntraj). Only returned if 
        coarse_roughness is set to > 0. 
    
    '''
    if boundary == 'sphere':
        raise ValueError('course roughness not yet implemented for sphere\
                         boundary')
    nevents = k0.shape[1]
    ntraj = k0.shape[2]
    
    # get the first event only
    kx0 = k0[0,0,:]
    ky0 = k0[1,0,:]
    kz0 = k0[2,0,:]
    
    # sample the surface roughness angles theta_a
    theta_a_full = np.linspace(0., np.pi / 2, 500)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_a = P_theta_a(theta_a_full, coarse_roughness) / sum(P_theta_a(theta_a_full, coarse_roughness))
        if isinstance(prob_a, sc.Quantity):
            prob_a = prob_a.magnitude
    if np.isnan(prob_a).all(): 
        theta_a = np.zeros(ntraj)
    else: 
        theta_a = np.array([np.random.choice(theta_a_full, ntraj, p=prob_a) for i in range(1)]).flatten()
            
    # In case the surface is rough, then find new coordinates of initial 
    # directions after rotating the surface by an angle theta_a around y axis
    sintheta_a = np.sin(theta_a)
    costheta_a = np.cos(theta_a)
    
    kx0_rot = costheta_a * kx0 - sintheta_a * kz0
    ky0_rot = ky0
    kz0_rot = sintheta_a * kx0 + costheta_a * kz0

    # Find the new angles theta and phi between the incident trajectories and 
    # the normal to the new surface after the coordinate axis rotation
    theta_rot = np.arccos(kz0_rot / np.sqrt(kx0_rot**2 + ky0_rot**2 + kz0_rot**2))
    phi_rot = np.arccos(kx0_rot / np.sqrt(kx0_rot**2 + ky0_rot**2 + kz0_rot**2))

    # Refraction of incident light upon entering sample
    # TODO: only real part of n_sample should be used                             
    # for the calculation of angles of integration? Or abs(n_sample)? 
    theta_refr = refraction(theta_rot, n_medium, np.abs(n_sample))
    if isinstance(theta_refr, sc.Quantity):
        theta_refr = theta_refr.to('radians').magnitude
    if isinstance(phi_rot, sc.Quantity):
        phi_rot = phi_rot.to('radians').magnitude
    kx0_rot_refr = np.sin(theta_refr) * np.cos(phi_rot)
    ky0_rot_refr = np.sin(theta_refr) * np.sin(phi_rot)
    kz0_rot_refr = np.cos(theta_refr) 
    
    # Rotate the axes back so that the initial refracted directions are in 
    # old (global) coordinates by doing an axis rotation around y by 2pi-theta_a    
    kx0_refr = np.cos(2*np.pi-theta_a) * kx0_rot_refr - np.sin(2*np.pi-theta_a) * kz0_rot_refr
    ky0_refr = ky0_rot_refr
    kz0_refr = np.sin(2*np.pi-theta_a) * kx0_rot_refr + np.cos(2*np.pi-theta_a) * kz0_rot_refr    

    # Create an empty array of the initial direction cosines of the right size
    k0_rough = np.zeros((3, nevents, ntraj))
    
    # Fill up the first row (corresponding to the first scattering event) of the
    # direction cosines array with the randomly generated angles:
    k0_rough[0,0,:] = kx0_refr
    k0_rough[1,0,:] = ky0_refr
    k0_rough[2,0,:] = kz0_refr

    # Calculate Fresnel reflected directions, which are the same as the initial
    # directions in the local coordinate sytem but with a rotation of pi in phi_rot
#    kx0_rot_refl = -kx0_rot
#    ky0_rot_refl = -ky0_rot
#    kz0_rot_refl = kz0_rot

    # Calculate Fresnel reflected directions, which are the same as the initial
    # directions in the local coordinate sytem but flipping the z sign
    kx0_rot_refl = kx0_rot
    ky0_rot_refl = ky0_rot
    kz0_rot_refl = -kz0_rot

    # Rotate the axes back so that the reflected directions are in 
    # old (global) coordinates by doing an axis rotation around y by 2pi-theta_a    
    kx0_refl = np.cos(2*np.pi-theta_a) * kx0_rot_refl - np.sin(2*np.pi-theta_a) * kz0_rot_refl
    ky0_refl = ky0_rot_refl
    kz0_refl = np.sin(2*np.pi-theta_a) * kx0_rot_refl + np.cos(2*np.pi-theta_a) * kz0_rot_refl    
    
    return k0_rough, kz0_rot, kz0_refl


def P_theta_a(theta_a, r):
    """
    Calculates the probability of surface slope angles as a function of 
    surface roughness parameter r.
    
    Parameters
    ----------
    theta_a : array 
        Surface roughness angle between the slope of the surface and the 
        z=0 plane. 
    r : float (can be structcol.Quantity [dimensionless])
        Surface roughness parameter or rms slope of the surface
    
    Returns
    -------
    Probability of that the surface will have certain slope angles.
    
    Reference
    ---------
    B. v. Ginneken, M. Stavridi, J. J. Koenderink, “Diffuse and specular 
    reflectance from rough surfaces”, Applied Optics, 37, 1 (1998) (has 
    definition of rsm slope of the surface).
    
    """
    term1 = np.sin(theta_a) / r**2 / (np.cos(theta_a))**3
    term2 = np.exp(-(np.tan(theta_a))**2 / (2*r**2))

    return term1 * term2
