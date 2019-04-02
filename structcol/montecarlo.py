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
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import pymie as pm
from pymie import mie, size_parameter, index_ratio
from pymie import multilayer_sphere_lib as msl
from . import model
from . import refraction
import numpy as np
from numpy.random import random as random
import structcol as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

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
    position : ndarray (structcol.Quantity [length])
        array of position vectors in cartesian coordinates of n trajectories
    direction : ndarray (structcol.Quantity [dimensionless])
        array of direction of propagation vectors in cartesian coordinates
        of n trajectories after every scattering event
    weight : ndarray (structcol.Quantity [dimensionless])
        array of photon packet weights for absorption modeling of n
        trajectories
    nevents : int
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
    plot_coord(ntraj, three_dim=False)
        plot positions of trajectories as a function of number scattering
        events.
    
    """

    def __init__(self, position, direction, weight):
        """
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
        Calculates the directions of propagation (or direction cosines) after
        scattering.
        At a scattering event, a photon packet adopts a new direction of
        propagation, which is randomly sampled from the phase function.
        
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

        # Calculate the new x, y, z coordinates of the propagation direction
        # using the following equations, which can be derived by using matrix
        # operations to perform a rotation about the y-axis by angle theta
        # followed by a rotation about the z-axis by angle phi
        for n in np.arange(1,self.nevents):
            kx = ((kn[0,n-1,:]*costheta[n-1,:] + kn[2,n-1,:]*sintheta[n-1,:])*
                  cosphi[n-1,:]) - kn[1,n-1,:]*sinphi[n-1,:]

            ky = ((kn[0,n-1,:]*costheta[n-1,:] + kn[2,n-1,:]*sintheta[n-1,:])*
                  sinphi[n-1,:]) + kn[1,n-1,:]*cosphi[n-1,:]

            kz = -kn[0,n-1,:]*sintheta[n-1,:] + kn[2,n-1,:]*costheta[n-1,:]

            kn[:,n,:] = kx, ky, kz

        # Update all the directions of the trajectories
        self.direction = sc.Quantity(kn, self.direction.units)


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
        displacement[:, 1:, :] = step * self.direction

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

        ax[0].plot(np.arange(len(self.position[0,:,0])),
                   self.position[0,:,:], '-')
        ax[0].set_title('Positions during trajectories')
        ax[0].set_ylabel('x (' + str(self.position.units) + ')')

        ax[1].plot(np.arange(len(self.position[1,:,0])),
                   self.position[1,:,:], '-')
        ax[1].set_ylabel('y (' + str(self.position.units) + ')')

        ax[2].plot(np.arange(len(self.position[2,:,0])),
                   self.position[2,:,:], '-')
        ax[2].set_ylabel('z (' + str(self.position.units) + ')')
        ax[2].set_xlabel('scattering event')

        if three_dim == True:
            fig = plt.figure(figsize = (8,6))
            ax3D = fig.add_subplot(111, projection='3d')
            ax3D.set_xlabel('x (' + str(self.position.units) + ')')
            ax3D.set_ylabel('y (' + str(self.position.units) + ')')
            ax3D.set_zlabel('z (' + str(self.position.units) + ')')
            ax3D.set_title('Positions during trajectories')

            for n in np.arange(ntraj):
                ax3D.scatter(self.position[0,:,n], self.position[1,:,n],
                             self.position[2,:,n], color=next(colors))

def initialize(nevents, ntraj, n_medium, n_sample, boundary, seed=None,
               incidence_angle=0., plot_initial=False):

    """
    Sets the trajectories' initial conditions (position, direction, and weight).
    The initial positions are determined randomly in the x-y plane.
    
    If boundary is a sphere, the initial z-positions are confined to the 
    surface of a sphere. If boundary is a film, the initial z-positions are set
    to zero.
    
    If incidence angle is set to 0, the initial propagation direction
    is set to be 1 at z, meaning that the photon packets point straight down in z.
    The initial directions are corrected for refraction, for either type of
    boundary and for any incidence angle.
    
    **notes:
        - for sphere boundary, incidence angle currently must be 0
        - spot size is not implemented--you must multiply by number outside 
          of this function to implement the spot size
        - you must multiply returned r0 by sphere diameter to get correct
          initial positions for sphere boundary
    
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
    incidence_angle: float
        Maximum value for theta when it incides onto the sample.
        Should be between 0 and pi/2.
    plot_inital: boolean
        If plot_initial is set to True, function will create a 3d plot showing
        initial positions and directions of trajectories before entering the 
        sphere and directly after refraction correction upon entering the 
        sphere
    
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
    
    """
    if seed is not None:
        np.random.seed([seed])
    
    # Initial position. The position array has one more row than the direction
    # and weight arrays because it includes the starting positions on the x-y
    # plane
    r0 = np.zeros((3, nevents+1, ntraj))
    
    # Create an empty array of the initial direction cosines of the right size
    k0 = np.zeros((3, nevents, ntraj))
    
    # Initial weight
    weight0 = np.ones((nevents, ntraj))
    
    if boundary == 'film':
        # randomly choose x positions on interval [0,1]
        r0[0,0,:] = random((1,ntraj))
        
        # randomly choose y positions on interval [0,1]
        r0[1,0,:] = random((1,ntraj))

        # Random sampling of azimuthal angle phi from uniform distribution [0 -
        # 2pi] for the first scattering event
        rand_phi = random((1,ntraj))
        phi = 2*np.pi*rand_phi
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        # Random sampling of scattering angle theta from uniform distribution [0 -
        # pi] for the first scattering event
        rand_theta = random((1,ntraj))
        theta = rand_theta * incidence_angle

    if boundary == 'sphere':
        # randomly choose r on interval [0,1]
        r = np.sqrt(random(ntraj))
        
        # randomly choose th on interval [0,2*pi]
        th = 2*np.pi*random(ntraj)
        
        # randomly choose x and y-positions within sphere radius
        r0[0,0,:] = r*np.cos(th) 
        r0[1,0,:] = r*np.sin(th)
        
        # calculate z-positions from x- and y-positions
        r0[2,0,:] = 1-np.sqrt(1 - r0[0,0,:]**2 - r0[1,0,:]**2)
    
        # find the minus normal vectors of the sphere at the initial positions
        neg_normal = np.zeros((3, ntraj)) 
        r0_magnitude = np.sqrt(r0[0,0,:]**2 + r0[1,0,:]**2 + (r0[2,0,:]-1)**2)
        neg_normal[0,:] = -r0[0,0,:]/r0_magnitude
        neg_normal[1,:] = -r0[1,0,:]/r0_magnitude
        neg_normal[2,:] = -(r0[2,0,:]-1)/r0_magnitude
        
        # solve for theta and phi for these samples
        theta = np.arccos(neg_normal[2,:])
        cosphi = neg_normal[0,:]/np.sin(theta)
        sinphi = neg_normal[1,:]/np.sin(theta)

    # Refraction of incident light upon entering the sample
    # TODO: only real part of n_sample should be used                             
    # for the calculation of angles of integration? Or abs(n_sample)? 
    theta = refraction(theta, n_medium, np.abs(n_sample))
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
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_zlim([0, 1])
        ax.set_title('Initial Positions')
        ax.view_init(-164,-155)
        X, Y, Z, U, V, W = [r0[0,0,:],r0[1,0,:],r0[2,0,:],k0[0,0,:], k0[1,0,:], k0[2,0,:]]
        ax.quiver(X, Y, Z, U, V, W, color = 'g')
        
        X, Y, Z, U, V, W = [r0[0,0,:],r0[1,0,:],r0[2,0,:],np.zeros(ntraj), np.zeros(ntraj), np.ones(ntraj)]
        ax.quiver(X, Y, Z, U, V, W)
        
        # draw wireframe hemisphere
        u, v = np.mgrid[0:2*np.pi:20j, np.pi/2:0:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = (1-np.cos(v))
        ax.plot_wireframe(x, y, z, color=[0.8,0.8,0.8])

    return r0, k0, weight0

def calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen,
              radius2=None, concentration=None, pdi=None, polydisperse=False,
              mie_theory = False):
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
    mie_theory : bool
        If True, the phase function and scattering coefficient is calculated 
        from Mie theory. If False (default), they are calculated from the 
        single scattering model, which includes a correction for the structure
        factor
    
    Returns
    -------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    mu_scat : float (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering model.
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
    
    # Scattering angles (typically from a small angle to pi). A non-zero small 
    # angle is needed because in the single scattering model, if the analytic 
    # formula is used, S(q=0) returns nan. To prevent any errors or warnings, 
    # set the minimum value of angles to be a small value, such as 0.01.
    min_angle = 0.01            
    angles = sc.Quantity(np.linspace(min_angle, np.pi, 200), 'rad') 

    k = 2 * np.pi * n_sample / wavelen     
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    # radius and radius2 should be in the same units (for polydisperse samples)    
    if radius2 is not None:
        radius2 = radius2.to(radius.units)
    if radius2 is None:
        radius2 = radius    
    
    # For now, set the number_density to be the average number_density if the 
    # system is polydisperse
    # TODO: should the number_density account for polydispersity as well?
    number_density1 = 3.0 * volume_fraction / (4.0 * np.pi * radius.max()**3)    
    number_density2 = 3.0 * volume_fraction / (4.0 * np.pi * radius2.max()**3)
    number_density = (number_density1 + number_density2)/2
    
    # if the system is polydisperse, use the polydisperse form and structure 
    # factors
    if polydisperse == True:
        if radius2 is None or concentration is None or pdi is None:
            raise ValueError('must specify diameters, concentration, and pdi for polydisperperse systems')
        
        if len(np.atleast_1d(m)) > 1:
            raise ValueError('cannot handle polydispersity in core-shell particles')
        
        form_type = 'polydisperse'
        structure_type = 'polydisperse'
    else:
        form_type = 'sphere'
        structure_type = 'glass'
        
    # define the mean diameters in case the system is polydisperse    
    mean_diameters = sc.Quantity(np.array([2*radius.magnitude, 2*radius2.magnitude]),
                                 radius.units)
                         
    # calculate the absorption coefficient
    if np.abs(n_sample.imag.magnitude) > 0.0:
            
       # The absorption coefficient can be calculated from the imaginary 
        # component of the samples's refractive index
        mu_abs = 4*np.pi*n_sample.imag/wavelen
        
#        # Calculate absorption coefficient for 1 particle (because there isn't
#        # a structure factor for absorption)
#        nstop = mie._nstop(np.array(x).max())
#        # if the index ratio m is an array with more than 1 element, it's a 
#        # multilayer particle
#        if len(np.atleast_1d(m)) > 1:
#            coeffs = msl.scatcoeffs_multi(m, x)
#            cabs_part = mie._cross_sections_complex_medium_sudiarta(coeffs[0], 
#                                                                    coeffs[1], 
#                                                                    x,radius)[1]
#            if cabs_part.magnitude < 0.0:
#                cabs_part = 0.0 * cabs_part.units
#        else:
#            al, bl = mie._scatcoeffs(m, x, nstop)   
#            cl, dl = mie._internal_coeffs(m, x, nstop)
#            x_scat = size_parameter(wavelen, n_particle, radius)
#            cabs_part = mie._cross_sections_complex_medium_fu(al, bl, cl, dl, 
#                                                              radius,n_particle, 
#                                                              n_sample, x_scat, 
#                                                              x, wavelen)[1]                                                      
#        mu_abs = cabs_part * number_density

    else:
        cross_sections = mie.calc_cross_sections(m, x, wavelen/n_sample)  
        cabs_part = cross_sections[2]                                               
        mu_abs = cabs_part * number_density
      
    # calculate the phase function
    p, p_par, p_perp, cscat_total = phase_function(m, x, angles, volume_fraction, 
                                                   k, number_density,
                                                   wavelen=wavelen, 
                                                   diameters=mean_diameters, 
                                                   concentration=concentration, 
                                                   pdi=pdi, n_sample=n_sample,
                                                   form_type=form_type,
                                                   structure_type=structure_type,
                                                   mie_theory=mie_theory)

    mu_scat = number_density * cscat_total

    # Here, the resulting units of mu_scat and mu_abs are nm^2/um^3. Thus, we 
    # simplify the units to 1/um 
    mu_scat = mu_scat.to('1/um')
    mu_abs = mu_abs.to('1/um')
    
    return p, mu_scat, mu_abs
    

def phase_function(m, x, angles, volume_fraction, k, number_density,
                   wavelen=None, diameters=None, concentration=None, pdi=None, 
                   n_sample=None, form_type='sphere', structure_type='glass', 
                   mie_theory=False):
    """
    Calculates the phase function (the phase function is the same for absorbing 
    and non-absorbing systems)
    
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
        'polydisperse', or None. 
    mie_theory: bool
        If TRUE, phase function is calculated according to Mie theory 
        (assuming no contribution from structure factor). If FALSE, phase
        function is calculated according to single scattering theory 
        (which uses Mie and structure factor contributions)
        
    Returns:
    --------
    p: array
        phase function for unpolarized light 
    p_par: array
        phase function for parallel polarized light   
    p_perp: array
        phase function for perpendicularly polarized light   
    cscat_total: float
        total scattering cross section for unpolarized light
        
    """  
    ksquared = np.abs(k)**2  
    
    if form_type=='polydisperse':
        distance = diameters/2
    else:
        distance = diameters.max()/2 # TODO: need to figure out which integration distance to use 

    # If mie_theory = True, calculate the phase function for 1 particle 
    # using Mie theory (excluding the structure factor)
    if mie_theory == True:
        structure_type = None
  
    diff_cscat_par, diff_cscat_perp = \
         model.differential_cross_section(m, x, angles, volume_fraction,
                                             structure_type=structure_type,
                                             form_type=form_type,
                                             diameters=diameters,
                                             concentration=concentration,
                                             pdi=pdi, wavelen=wavelen, 
                                             n_matrix=n_sample, k=k, 
                                             distance=distance)
                                       
    # Integrate the differential cross section to get the total cross section
    if np.abs(k.imag.magnitude) > 0.:      
        if form_type=='polydisperse' and len(concentration)>1:
            cscat_total1, cscat_total_par1, cscat_total_perp1, _, _ = \
                mie.integrate_intensity_complex_medium(diff_cscat_par, 
                                                       diff_cscat_perp, 
                                                       distance[0],angles,k)  
            cscat_total2, cscat_total_par2, cscat_total_perp2, _, _ = \
                mie.integrate_intensity_complex_medium(diff_cscat_par, 
                                                       diff_cscat_perp, 
                                                       distance[1],angles,k)
            cscat_total = cscat_total1 * concentration[0] + cscat_total2 * concentration[1]
            cscat_total_par = cscat_total_par1 * concentration[0] + cscat_total_par2 * concentration[1]
            cscat_total_perp = cscat_total_perp1 * concentration[0] + cscat_total_perp2 * concentration[1]
        
        else: 
            cscat_total, cscat_total_par, cscat_total_perp, diff_cscat_par2, diff_cscat_perp2 = \
                mie.integrate_intensity_complex_medium(diff_cscat_par, 
                                                       diff_cscat_perp, 
                                                       distance,angles,k)  
            
        # to calculate the phase function when there is absorption, we  
        # use the far-field Mie solutions because the near field diff cross 
        # section behaves weirdly. To make sure we use the far-field 
        # solutions, set k = None.                                               
        diff_cscat_par_ff, diff_cscat_perp_ff = \
            model.differential_cross_section(m, x, angles, volume_fraction,
                                             structure_type=structure_type,
                                             form_type=form_type,
                                             diameters=diameters,
                                             concentration=concentration,
                                             pdi=pdi, wavelen=wavelen, 
                                             n_matrix=n_sample, k=None, distance=distance)
        cscat_total_par_ff = model._integrate_cross_section(diff_cscat_par_ff,
                                                      1.0/ksquared, angles)
        cscat_total_perp_ff = model._integrate_cross_section(diff_cscat_perp_ff,
                                                      1.0/ksquared, angles)
        cscat_total_ff = (cscat_total_par_ff + cscat_total_perp_ff)/2.0                                     
        
        p = (diff_cscat_par_ff + diff_cscat_perp_ff)/(ksquared * 2 * cscat_total_ff)
        p_par = diff_cscat_par_ff/(ksquared * 2 * cscat_total_par_ff)
        p_perp = diff_cscat_perp_ff/(ksquared * 2 * cscat_total_perp_ff)
    
    # if there is no absorption in the system
    else:
        cscat_total_par = model._integrate_cross_section(diff_cscat_par,
                                                      1.0/ksquared, angles)
        cscat_total_perp = model._integrate_cross_section(diff_cscat_perp,
                                                      1.0/ksquared, angles)
        cscat_total = (cscat_total_par + cscat_total_perp)/2.0
    
        p = (diff_cscat_par + diff_cscat_perp)/(ksquared * 2 * cscat_total)
        p_par = diff_cscat_par/(ksquared * 2 * cscat_total_par)
        p_perp = diff_cscat_perp/(ksquared * 2 * cscat_total_perp)
        
        diff_cscat_par2 = diff_cscat_par / (np.abs(k)**2)
        diff_cscat_perp2 = diff_cscat_perp / (np.abs(k)**2)
        
    return(p, p_par, p_perp, cscat_total)

    
def sample_angles(nevents, ntraj, p):
    """
    Samples azimuthal angles (phi) from uniform distribution, and scattering
    angles (theta) from phase function distribution.
    
    Parameters
    ----------
    nevents : int
        Number of scattering events.
    ntraj : int
        Number of trajectories.
    p : array_like (structcol.Quantity [dimensionless])
        Phase function values returned from 'phase_function'.
    
    Returns
    -------
    sintheta, costheta, sinphi, cosphi, theta, phi : ndarray
        Sampled azimuthal and scattering angles, and their sines and cosines.
    
    """
    
    # Scattering angles for the phase function calculation (typically from 0 to 
    # pi). A non-zero minimum angle is needed because in the single scattering 
    # model, if the analytic formula is used, S(q=0) returns nan.
    min_angle = 0.01            
    angles = sc.Quantity(np.linspace(min_angle,np.pi, 200), 'rad')  

    # Random sampling of azimuthal angle phi from uniform distribution [0 -
    # 2pi]
    rand = np.random.random((nevents,ntraj))
    phi = 2*np.pi*rand
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    # Random sampling of scattering angle theta
    prob = p * np.sin(angles)*2*np.pi    # prob is integral of p in solid angle
    prob_norm = prob/sum(prob)           # normalize to make it add up to 1

    theta = np.array([np.random.choice(angles, ntraj, p = prob_norm)
                      for i in range(nevents)])
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    return sintheta, costheta, sinphi, cosphi, theta, phi


def sample_step(nevents, ntraj, mu_abs, mu_scat):
    """
    Samples step sizes from exponential distribution.
    
    Parameters
    ----------
    nevents : int
        Number of scattering events.
    ntraj : int
        Number of trajectories.
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient.
    mu_scat : float (structcol.Quantity [1/length])
        Scattering coefficient.
    
    Returns
    -------
    step : ndarray
        Sampled step sizes for all trajectories and scattering events.
    
    """
    # Calculate total extinction coefficient
    mu_total = mu_scat + mu_abs 

    # Generate array of random numbers from 0 to 1
    rand = np.random.random((nevents,ntraj))

    step = -np.log(1.0-rand) / mu_total

    return step