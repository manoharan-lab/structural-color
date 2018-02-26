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

from . import mie, model, index_ratio, size_parameter
import numpy as np
from numpy.random import random as random
import structcol as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import warnings
from scipy.optimize import fsolve

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
        Calculates absorption of photon packet after each scattering event.
        Absorption is modeled as a reduction of a photon packet's weight
        every time it gets scattered using Beer-Lambert's law.
        
        Parameters
        ----------
        mu_abs : ndarray (structcol.Quantity [1/length])
            Absorption coefficient of packet 
        step_size: ndarray (structcol.Quantity [length])
            Step size of packet (sampled from scattering lengths).
        
        """
        
        # Beer-Lambert
        #weight = np.exp(-mu_abs*np.cumsum(step_size[:,:], axis=0))
        weight = self.weight*np.exp(-mu_abs*np.cumsum(step_size[:,:], axis=0))
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


def select_events(inarray, events):
    '''
    Selects the items of inarray according to event coordinates
    
    Parameters
    ----------
    inarray: 2D array
        Should have axes corresponding to events, trajectories
    events: 1D array
        Should have length corresponding to ntrajectories.
        Non-zero entries correspond to the event of interest
    
    Returns
    -------
    1D array: contains only the elements of inarray corresponding to non-zero events values.
    
    '''
    # there is no 0th event, so disregard a 0 (or less) in the events array
    valid_events = (events > 0)
    
    # The 0th element in arrays such as direction refer to the 1st event
    # so subtract 1 from all the valid events to correct for array indexing
    ev = events[valid_events].astype(int) - 1
    
    # find the trajectories where there are valid events
    tr = np.where(valid_events)[0]

    # want output of the same form as events
    outarray = np.zeros(len(events))
    
    # get an output array with elements corresponding to the input events
    outarray[valid_events] = inarray[ev, tr]
    if isinstance(inarray, sc.Quantity):
        outarray = sc.Quantity(outarray, inarray.units)
    return outarray
    
def find_exit_intersect(x0,y0,z0, x1, y1, z1, radius):
    """
    finds the point at which an exiting trajectory intersect with the boundary 
    of the sphere
    
    Parameters
    ----------
    x0: float
        initial x-position of trajectory
    y0: float
        initial y-position of trajectory
    z0: float
        initial z-position of trajectory
    x1: float
        x-position of trajectory after exit
    y1: float
        y-position of trajectory after exit
    z1: float
        z-position of trajectory after exit
    radius : float
        radius of spherical boundary 

    Returns
    ----------
        tuple (x, y, z) point of intersection     
    
    """
    def equations(params):
        x,y,z = params
        return((x-x0)/(x1-x0)-(y-y0)/(y1-y0), (z-z0)/(z1-z0)-(y-y0)/(y1-y0), x**2 + y**2 + z**2-radius**2 )

    intersect_pt, infodict, ler, mesg = fsolve(equations,(x1,y1,z1), full_output = True) # initial guess is x0,y0,z0

    return intersect_pt[0], intersect_pt[1], intersect_pt[2]
    
# vectorize above function    
find_exit_intersect_vec = np.vectorize(find_exit_intersect)

def exit_kz(x, y, z, indices, radius, n_inside, n_outside):
    '''
    returns kz of exit trajectory, corrected for refraction at the spherical
    boundary
    
    Parameters
    ----------
    x: 1D array
        x values for each trajectory and event
    y: 1D array
        y values for each trajectory and event
    z: 1D array
        z values for each trajectory and event
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    radius: float
        radius of sphere boundary
    n_inside: float
        refractive index inside sphere boundary
    n_outside: float
        refractive index outside sphere boundary
    
    Returns
    -------
    k2z: 1D array (length ntraj)
        z components of refracted kz upon trajectory exit
    
    '''
    # find unit vectors k1, normal vector at exit, and angle between normal and k1
    k1, norm, theta_1 = get_angles_sphere(x, y, z, radius, indices)

    # take cross product of k1 and sphere normal vector to find vector to rotate
    # around    
    kr = np.transpose(np.cross(np.transpose(k1),np.transpose(norm)))
    
    # TODO make sure signs work out
    # use Snell's law to calculate angle between k2 and normal vector
    theta_2 = refraction(theta_1, n_inside, n_outside)    
    
    # angle to rotate around is theta_2-theta_1
    theta = theta_2-theta_1
    
    # perform the rotation
    k2z = rotate_refract(norm, kr, theta, k1)
    
    return k2z

def rotate_refract(abc, uvw, theta, xyz):
    '''
    rotates unit vector <xyz> by angle theta around unit vector <uvw>,
    where abs is a point on the vector we are rotating around
    
    Parameters
    ----------
    abc: 3D array
       point (a,b,c) on vector to rotate around. Length is number of exit
       trajectories we are considering
    uvw: 3D array
        unit vector to rotate around. Length is number of exit trajectories
        we are considering
    theta: 1D array
        angle we are rotating by. Length is number if exit trajectories we are
        considering
    xyz: 3D array
        vector to rotate. Length is number of exit trajectories we are 
        considering

    Returns
    -------
    k2z: 1D array (length ntraj)
        z components of refracted kz upon trajectory exit
        
    Note: see more on rotations at
    https://sites.google.com/site/glennmurray/Home/rotation-matrices
    -and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
    
    '''
    a = abc[0,:]
    b = abc[1,:]
    c = abc[2,:]
    u = uvw[0,:]
    v = uvw[1,:]
    w = uvw[2,:]
    x = xyz[0,:]
    y = xyz[1,:]
    z = xyz[2,:]
    
    # rotation matrix 
    k2z = (c*(u**2 + v**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta)) + z*np.cos(theta) + (-b*u + a*v - v*x + u*y)*np.sin(theta) 
    return k2z

def get_angles(kz, indices):
    '''
    Returns specified angles (relative to global z) from kz components
    
    Parameters
    ----------
    kz: 2D array
        kz values, with axes corresponding to events, trajectories
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    
    Returns
    -------
    1D array of pint quantities (length Ntraj)
    
    '''
    # select scattering events resulted in exit
    cosz = select_events(kz, indices)
    
    # calculate angle to normal from cos_z component (only want magnitude)
    return sc.Quantity(np.arccos(np.abs(cosz)),'')

def get_angles_sphere(x, y, z, radius, indices, incident = False, plot_exits = False):
    '''
    Returns angles relative to vector normal to sphere at point on 
    boundary. Currently works only for incident light in 
    the +z direction
    
    Parameters
    ----------
    x: 2D array
        x position values, with axes corresponding to (1 + events, trajectories)
        there is one more x position than events because it takes two positions
        to define an event
    y: 2D array
        y position values, with axes corresponding to (1 + events, trajectories)
        there is one more y position than events because it takes two positions
        to define an event
    z: 2D array
        z position values, with axes corresponding to (1 + events, trajectories)
        there is one more z position than events because it takes two positions
        to define an event
    radius: float
        radius of the sphere boundary 
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
        index = 1 corresponds to first event, or 0th element in events array
    incident: boolean
        If set to True, function finds the angles between incident light
        travelling in the +z direction and the sphere boundary where the 
        trajectory enters. If set to False, function finds the angles between
        the trajectories inside the sphere and the normal at the sphere 
        boundary where the trajectory exits.
    plot_exits : boolean
        If set to True, function will plot the last point of trajectory inside 
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits, 
        making one plot for reflection and one plot for transmission
    
    Returns
    -------
    k1: 2D array of shape (3, ntraj)
        direction vector of trajectory leaving sphere
    norm: 1D array of shape (3, ntraj)
        vector normal to sphere at the exit point of the trajectory
    angles_norm: 1D array of pint quantities (length Ntraj)
        angle between k1 and the normal vector at the exit point of the
        trajectory
    '''
    # Subtract radius from z to center the sphere at 0,0,0. This makes the 
    # following calculations much easier
    z = z - radius
    
    if incident:
        select_x1 = select_events(x, indices)
        select_y1 = select_events(y, indices)
        select_z1 = select_events(z, indices)
        
        select_x0 = select_x1
        select_y0 = select_y1
        select_z0 = select_z1 + 1
        
        x_inter = select_x1
        y_inter = select_y1
        z_inter = select_z1
    else:
    
        # get positions outside of sphere boundary from after exit (or entrance if 
        # this is for first event)
        select_x1 = select_events(x[1:,:], indices)
        select_y1 = select_events(y[1:,:], indices)
        select_z1 = select_events(z[1:,:], indices)
        
        # get positions inside sphere boundary from before exit
        select_x0 = select_events(x[:len(x)-1,:],indices)
        select_y0 = select_events(y[:len(y)-1,:],indices)
        select_z0 = select_events(z[:len(z)-1,:],indices)
        
        # get positions at sphere boundary from exit
        x_inter, y_inter, z_inter = find_exit_intersect_vec(select_x0,
                                                            select_y0,
                                                            select_z0,
                                                            select_x1,
                                                            select_y1,
                                                            select_z1, radius)
                                                        
    # calculate the magnitude of exit vector to divide to make a unit vector
    mag = np.sqrt((select_x1-select_x0)**2 + (select_y1-select_y0)**2 
                                           + (select_z1-select_z0)**2)
                                           
    # calculate the vector normal to the sphere boundary at the exit
    norm = np.zeros((3,len(x_inter)))
    norm[0,:] = x_inter
    norm[1,:] = y_inter
    norm[2,:] = z_inter
    norm = norm/radius
    
    # calculate the normalized k1 vector 
    k1 = np.zeros((3,len(x_inter)))
    k1[0,:] = select_x1 - select_x0
    k1[1,:] = select_y1 - select_y0
    k1[2,:] = select_z1 - select_z0
    k1 = k1/mag
    
    # calculate the dot product between the vector normal to the sphere and the 
    # exit vector, and divide by their magnitudes. Then use arccos to find 
    # the angles 
    dot_norm = np.nan_to_num(norm[0,:]*k1[0,:] + 
                             norm[1,:]*k1[1,:] +
                             norm[2,:]*k1[2,:])

    # if the dot product is <0, force it to zero.
    # a negative dot product cannot physically occur because it implies
    # that the angle between k1 and the normal is > 90 degrees. Testing of the
    # code that in some cases, very small (< magnitude 0.002) negative numbers
    # are found from the dot product. This suggests that the solution for the 
    # interset between the sphere and the k1 vector is slightly off. Since we 
    # know that we cannot have a negative dot product, we instead force it to 
    # zero, meaning that we assume an angle of 90 degrees between the sphere 
    # normal and the k1 vector 
    dot_norm[dot_norm < 0] = 0
    angles_norm = np.nan_to_num(np.arccos(dot_norm))
    angles_norm = sc.Quantity(angles_norm, '')
    
    dot_z = np.nan_to_num(abs(select_z1-select_z0)/mag)
    angles_z = np.nan_to_num(np.arccos(dot_z))    
    
    # plot the points before exit, after exit, and on exit boundary
    if plot_exits == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(select_x0,select_y0,select_z0, c = 'b')
        ax.scatter(select_x1,select_y1,select_z1, c = 'g')
        ax.scatter(x_inter,y_inter,z_inter, c='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')    
        
        u, v = np.mgrid[0:2*np.pi:20j, np.pi:0:10j]
        x = radius*np.cos(u)*np.sin(v)
        y = radius*np.sin(u)*np.sin(v)
        z = radius*(-np.cos(v))
        ax.plot_wireframe(x, y, z, color=[0.8,0.8,0.8])    
    
    return k1, norm, angles_norm

def fresnel_pass_frac(kz, indices, n_before, n_inside, n_after):
    '''
    Returns weights of interest reduced by fresnel reflection across two interfaces,
    For example passing through a coverslip.

    Parameters
    ----------
    kz: 2D array
        kz values, with axes corresponding to events, trajectories
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    n_before: float
        Refractive index of the medium light is coming from
    n_inside: float
        Refractive index of the boundary material (e.g. glass coverslip)
    n_after: float
        Refractive index of the medium light is going to
    
    Returns
    -------
    1D array of length Ntraj
    
    '''
    # Allow single interface by passing in None as n_inside
    if n_inside is None:
        n_inside = n_before

    # find angles before
    theta_before = get_angles(kz, indices)
    # find angles inside
    theta_inside = refraction(theta_before, n_before, n_inside)
    # if theta_inside is nan (because the trajectory doesn't exit due to TIR), 
    # then replace it with pi/2 (the trajectory goes sideways infinitely) to 
    # avoid errors during the calculation of stuck trajectories
    theta_inside[np.isnan(theta_inside)] = np.pi/2.0

    # find fraction passing through both interfaces
    trans_s1, trans_p1 = model.fresnel_transmission(n_before, n_inside, theta_before) # before -> inside
    trans_s2, trans_p2 = model.fresnel_transmission(n_inside, n_after, theta_inside)  # inside -> after
    fresnel_trans = (trans_s1 + trans_p1)*(trans_s2 + trans_p2)/4.

    # find fraction reflected off both interfaces before transmission
    refl_s1, refl_p1 = model.fresnel_reflection(n_inside, n_after, theta_inside)  # inside -> after
    refl_s2, refl_p2 = model.fresnel_reflection(n_inside, n_before, theta_inside) # inside -> before
    fresnel_refl = (refl_s1 + refl_p1)*(refl_s2 + refl_p2)/4.

    # Any number of higher order reflections off the two interfaces
    # Use converging geometric series 1+a+a**2+a**3...=1/(1-a)
    return fresnel_trans/(1-fresnel_refl+eps)
    
def fresnel_pass_frac_sphere(radius, indices, n_before, n_inside, n_after, 
                             x, y, z, incident=False, plot_exits=False):
    '''
    Returns weights of interest reduced by fresnel reflection across two 
    interfaces, For example passing through a coverslip.
    
    Note: if n_inside = None, returns weights of interest reduced accross one
    interface. This code has not been tested for case of some sort of coverslip
    covering a sphere. It is currently only used for case of passing from
    sphere directly to air.

    Parameters
    ----------
    radius: float
        radius of the sphere boundary 
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    n_before: float
        Refractive index of the medium light is coming from
    n_inside: float
        Refractive index of the boundary material (e.g. glass coverslip)
    n_after: float
        Refractive index of the medium light is going to
    x: 2D array
        x position values, with axes corresponding to (1 + events, trajectories)
        there is one more x position than events because it takes two positions
        to define an event
    y: 2D array
        y position values, with axes corresponding to (1 + events, trajectories)
        there is one more y position than events because it takes two positions
        to define an event
    z: 2D array
        z position values, with axes corresponding to (1 + events, trajectories)
        there is one more z position than events because it takes two positions
        to define an event
    plot_exits : boolean
        if set to True, function will plot the last point of trajectory inside 
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits, 
        making one plot for reflection and one plot for transmission
   
    Returns
    -------
    1D array of length Ntraj
    
    '''
    #Allow single interface by passing in None as n_inside
    if n_inside is None:
        n_inside = n_before

    #find angles before
    k1, norm, theta_before = get_angles_sphere(x,y,z,radius, indices, incident = incident, plot_exits = plot_exits)
    
    #find angles inside
    theta_inside = refraction(theta_before, n_before, n_inside)
    
    # if theta_inside is nan (because the trajectory doesn't exit due to TIR), 
    # then replace it with pi/2 (the trajectory goes sideways infinitely) to 
    # avoid errors during the calculation of stuck trajectories
    theta_inside[np.isnan(theta_inside)] = np.pi/2.0

    #find fraction passing through both interfaces
    trans_s1, trans_p1 = model.fresnel_transmission(n_before, n_inside, theta_before) # before -> inside
    trans_s2, trans_p2 = model.fresnel_transmission(n_inside, n_after, theta_inside)  # inside -> after
    fresnel_trans = (trans_s1 + trans_p1)*(trans_s2 + trans_p2)/4.

    #find fraction reflected off both interfaces before transmission
    refl_s1, refl_p1 = model.fresnel_reflection(n_inside, n_after, theta_inside)  # inside -> after
    refl_s2, refl_p2 = model.fresnel_reflection(n_inside, n_before, theta_inside) # inside -> before
    fresnel_refl = (refl_s1 + refl_p1)*(refl_s2 + refl_p2)/4.

    #Any number of higher order reflections off the two interfaces
    #Use converging geometric series 1+a+a**2+a**3...=1/(1-a)
    return k1, norm, fresnel_trans/(1-fresnel_refl+eps)

def detect_correct(kz, weights, indices, n_before, n_after, thresh_angle):
    '''
    Returns weights of interest within detection angle
    
    Parameters
    ----------
    kz: 2D array
        kz values, with axes corresponding to events, trajectories
    weights: 2D array
        weights values, with axes corresponding to events, trajectories
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    n_before: float
        Refractive index of the medium light is coming from
    n_after: float
        Refractive index of the medium light is going to
    thresh_angle: float
        Detection angle to compare with output angles
    
    Returns
    -------
    1D array of length Ntraj
    
    '''

    # find angles when crossing interface
    theta = refraction(get_angles(kz, indices), n_before, n_after)
    theta[np.isnan(theta)] = np.inf # this avoids a warning

    # choose only the ones inside detection angle
    filter_weights = weights.copy()
    filter_weights[theta > thresh_angle] = 0
    return filter_weights

def refraction(angles, n_before, n_after):
    '''
    Returns angles after refracting through an interface
    
    Parameters
    ----------
    angles: float or array of floats
        angles relative to normal before the interface
    n_before: float
        Refractive index of the medium light is coming from
    n_after: float
        Refractive index of the medium light is going to
    
    '''
    snell = n_before / n_after * np.sin(angles)
    snell[abs(snell) > 1] = np.nan # this avoids a warning
    return np.arcsin(snell)

def calc_refl_trans(trajectories, z_low, cutoff, n_medium, n_sample,
                    n_front=None, n_back=None, detection_angle=np.pi/2, return_extra = False):
    """
    Counts the fraction of reflected and transmitted trajectories after a cutoff.
    Identifies which trajectories are reflected or transmitted, and at which
    scattering event. Includes Fresnel reflection correction. Then
    counts the fraction of reflected trajectories that are detected.
    
    Parameters
    ----------
    trajectories : Trajectory object
        Trajectory object of which the reflection is to be calculated.
    z_low : float (structcol.Quantity [length])
        Initial z-position that defines the beginning of the simulated sample.
        Should be set to 0.
    cutoff : float (structcol.Quantity [length])
        Final z-cutoff that determines the effective thickness of the simulated
        sample.
    n_medium : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the medium.
    n_sample : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample.
    n_front : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the front cover of the sample (default None)
    n_back : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the back cover of the sample (default None)
    detection_angle : float
        Range of angles of detection. Only the packets that come out of the
        sample within this range will be detected and counted. Should be
        0 < detection_angle <= pi/2, where 0 means that no angles are detected,
        and pi/2 means that all the backscattering angles are detected.
    
    Returns
    -------
    reflectance: float
        Fraction of reflected trajectories, including the Fresnel correction
        but not considering the range of the detector.
    transmittance: float
        Fraction of transmitted trajectories, including the Fresnel correction
        but not considering the range of the detector.
    Note: absorptance of the sample can be found by 1 - reflectance - transmittance
    
    """
    # set up the values we need as numpy arrays
    z = trajectories.position[2]
    if isinstance(z, sc.Quantity):
        z = z.to('um').magnitude
    kx, ky, kz = trajectories.direction
    if isinstance(kx, sc.Quantity):
        kx = kx.magnitude
        ky = ky.magnitude
        kz = kz.magnitude
    weights = trajectories.weight
    if isinstance(weights, sc.Quantity):
        weights = weights.magnitude
    if isinstance(z_low, sc.Quantity):
        z_low = z_low.to('um').magnitude
    if isinstance(cutoff, sc.Quantity):
        cutoff = cutoff.to('um').magnitude

    ntraj = z.shape[1]
    
    # rescale z in terms of integer numbers of sample thickness
    z_floors = np.floor((z - z_low)/(cutoff - z_low))

    # potential exits whenever trajectories cross any boundary
    potential_exits = ~(np.diff(z_floors, axis = 0)==0)

    # find all kz with magnitude large enough to exit
    no_tir = abs(kz) > np.cos(np.arcsin(n_medium / n_sample))

    # exit in positive direction (transmission) iff crossing odd boundary
    pos_dir = np.mod(z_floors[:-1]+1*(z_floors[1:]>z_floors[:-1]), 2).astype(bool)

    # construct boolean arrays of all valid exits in pos & neg directions
    high_bool = potential_exits & no_tir & pos_dir
    low_bool = potential_exits & no_tir & ~pos_dir

    # find first valid exit of each trajectory in each direction
    # note we convert to 2 1D arrays with len = Ntraj
    # need vstack to reproduce earlier behaviour:
    # an initial row of zeros is used to distinguish no events case
    low_event = np.argmax(np.vstack([np.zeros(ntraj),low_bool]), axis=0)
    high_event = np.argmax(np.vstack([np.zeros(ntraj),high_bool]), axis=0)

    # find all trajectories that did not exit in each direction
    no_low_exit = (low_event == 0)
    no_high_exit = (high_event == 0)

    # find positions where low_event is less than high_event
    # note that either < or <= would work here. They are only equal if both 0.
    low_smaller = (low_event < high_event)

    # find all trajectory outcomes
    # note ambiguity for trajectories that did not exit in a given direction
    low_first = no_high_exit | low_smaller
    high_first = no_low_exit | (~low_smaller)
    never_exit = no_low_exit & no_high_exit

    # find where each trajectory first exits
    refl_indices = low_event * low_first
    trans_indices = high_event * high_first
    stuck_indices = never_exit * (z.shape[0]-1)

    # calculate initial weights that actually enter the sample after fresnel
    init_dir = np.cos(refraction(get_angles(kz, np.ones(ntraj)), n_sample, n_medium))
    # init_dir is reverse-corrected for refraction. = kz before medium/sample interface
    inc_fraction = fresnel_pass_frac(np.array([init_dir]), np.ones(ntraj), n_medium, n_front, n_sample)

    # calculate outcome weights from all trajectories
    refl_weights = inc_fraction * select_events(weights, refl_indices)
    trans_weights = inc_fraction * select_events(weights, trans_indices)
    stuck_weights = inc_fraction * select_events(weights, stuck_indices)
    absorb_weights = inc_fraction - refl_weights - trans_weights - stuck_weights

    # warn user if too many trajectories got stuck
    stuck_frac = np.sum(stuck_weights) / np.sum(inc_fraction) * 100
    stuck_traj_warn = " \n{0}% of trajectories did not exit the sample. Increase Nevents to improve accuracy.".format(str(int(stuck_frac)))
    if stuck_frac >= 20: warnings.warn(stuck_traj_warn)

    # correct for non-TIR fresnel reflection upon exiting
    reflected = refl_weights * fresnel_pass_frac(kz, refl_indices, n_sample, n_front, n_medium)
    transmitted = trans_weights * fresnel_pass_frac(kz, trans_indices, n_sample, n_back, n_medium)
    refl_fresnel = refl_weights - reflected
    trans_fresnel = trans_weights - transmitted

    # find fraction of known outcomes that are successfully transmitted or reflected
    known_outcomes = np.sum(absorb_weights + reflected + transmitted)
    refl_frac = np.sum(reflected) / known_outcomes
    trans_frac = np.sum(transmitted) / known_outcomes
    
    # need to distribute ambiguous trajectory weights.
    # stuck are 50/50 reflected/transmitted since they are randomized.
    # non-TIR fresnel are treated as new trajectories at the appropriate interface.
    # This means reversed R/T ratios for fresnel reflection at transmission interface.
    extra_refl = refl_fresnel * refl_frac + trans_fresnel * trans_frac + stuck_weights * 0.5
    extra_trans = trans_fresnel * refl_frac + refl_fresnel * trans_frac + stuck_weights * 0.5

    # correct for effect of detection angle upon leaving sample
    inc_refl = (1 - inc_fraction) # fresnel reflection incident on sample
    inc_refl = detect_correct(np.array([init_dir]), inc_refl, np.ones(ntraj), n_medium, n_medium, detection_angle)
    trans_detected = detect_correct(kz, transmitted, trans_indices, n_sample, n_medium, detection_angle)
    refl_detected = detect_correct(kz, reflected, refl_indices, n_sample, n_medium, detection_angle)
    trans_det_frac = np.max([np.sum(trans_detected),eps]) / np.max([np.sum(transmitted), eps])
    refl_det_frac = np.max([np.sum(refl_detected),eps]) / np.max([np.sum(reflected), eps]) 

    # calculate transmittance and reflectance for each trajectory (in terms of trajectory weights)
    transmittance = trans_detected + extra_trans * trans_det_frac
    reflectance = refl_detected + extra_refl * refl_det_frac + inc_refl

    #calculate mean reflectance and transmittance for all trajectories
    if return_extra:
        # divide by ntraj to get reflectance per trajectory
        return refl_indices, trans_indices, reflected/ntraj, trans_frac, refl_frac, refl_fresnel/ntraj, trans_fresnel/ntraj, inc_refl/ntraj, np.sum(reflectance)/ntraj
    else:
        return (np.sum(reflectance)/ntraj, np.sum(transmittance/ntraj))


def calc_refl_trans_sphere(trajectories, n_medium, n_sample, radius, p, mu_abs, mu_scat,
                           detection_angle = np.pi/2, plot_exits = False, tir = False,
                           run_tir = True, return_extra = False, call_depth = 0, max_call_depth = 20):
    """
    Counts the fraction of reflected and transmitted trajectories for an 
    assembly with a spherical boundary. Identifies which trajectories are 
    reflected or transmitted, and at which scattering event. Then calculates 
    the fraction of reflected and transmitted trajectories.
    
    Parameters
    ----------
    trajectories : Trajectory object
        Trajectory object of which the reflection is to be calculated.
 
    n_medium: float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the medium.
    n_sample: float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample.
    radius : float (structcol.Quantity [length])
        radius of spherical boundary
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    mu_scat : float (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering model.
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient from Mie theory.
    detection_angle: float
        Range of angles of detection. Only the packets that come out of the
        sample within this range will be detected and counted. Should be
        0 < detection_angle <= pi/2, where 0 means that no angles are detected,
        and pi/2 means that all the backscattering angles are detected.
    plot_exits: boolean
        If set to True, function will plot the last point of trajectory inside 
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits, 
        making one plot for reflection and one plot for transmission
    tir: boolean
        This boolean is not intended to be set by the user. It's purpose is to 
        keep track of whether calc_refl_trans_sphere() is running for the trajectories
        initially being sent into the sphere or for the fresnel reflected (tir)
        trajectories that are trapped in the sphere. It's default value is
        False, and it is changed to True when calc_refl_trans_sphere() is 
        recursively called for calculating the reflectance from fresnel 
        reflected trajectories
    run_tir: boolean
        If set to True, function will calculate new trajectories for weights 
        that are fresnel reflected back into the sphere upon exit (There is
        almost always at least some small weight that is reflected back into
        sphere). If set to False, fresnel reflected trajectories are evenly 
        distributed to reflectance and transmittance.       
    call_depth: int
        This argument is not intended to be set by the user. Call_depth keeps 
        track of the recursion call_depth. It's default value is 0, and upon
        each recursive call to calc_refl_trans_sphere(), it is increased by 1. 
    max_call_depth: int
        This argument determines the maximum number of recursive calls that can
        be made to calc_refl_trans_sphere(). The default value is 20, but it 
        can be changed by the user if desired. The user should note that there
        are diminishing returns for higher max_call_depth, as the remaining 
        fresnel reflected trajectories after 20 calls are primarily stuck in 
        shallow angle paths around the perimeter of the sphere that will never 
        exit.

    Returns
    ----------
    reflectance: float
        Fraction of reflected trajectories, including the Fresnel correction
        but not considering the range of the detector.
    transmittance: float
        Fraction of transmitted trajectories, including the Fresnel correction
        but not considering the range of the detector.
    Note: absorptance of the sample can be found by 1 - reflectance - transmittance
    
    """   
    # set up the values we need as numpy arrays
    x, y, z = trajectories.position
    if isinstance(z, sc.Quantity):
        x = x.to('um').magnitude
        y = y.to('um').magnitude
        z = z.to('um').magnitude
    kx, ky, kz = trajectories.direction
    if isinstance(kx, sc.Quantity):
        kx = kx.magnitude
        ky = ky.magnitude
        kz = kz.magnitude
    weights = trajectories.weight
    if isinstance(weights, sc.Quantity):
        weights = weights.magnitude
    if isinstance(radius, sc.Quantity):
        radius = radius.to('um').magnitude
    
    # get the number of trajectories
    ntraj = z.shape[1]
    nevents = kz.shape[0]

    # potential exits whenever trajectories are outside sphere boundary
    potential_exits = (x[1:,:]**2 + y[1:,:]**2 + (z[1:,:]-radius)**2) > radius**2
    potential_exit_indices = np.argmax(np.vstack([np.zeros(ntraj), potential_exits]), axis=0)
    
    # exit in positive direction (transmission)
    kz_correct = exit_kz(x, y, z, potential_exit_indices, radius, n_sample, n_medium)
    pos_dir = kz_correct > 0
    
    # construct boolean arrays of all valid exits in pos & neg directions
    high_bool = potential_exits & pos_dir
    low_bool = potential_exits & ~pos_dir    
    # find first valid exit of each trajectory in each direction
    # note we convert to 2 1D arrays with len = Ntraj
    # need vstack to reproduce earlier behaviour:
    # an initial row of zeros is used to distinguish no events case
    low_event = np.argmax(np.vstack([np.zeros(ntraj),low_bool]), axis=0)
    high_event = np.argmax(np.vstack([np.zeros(ntraj),high_bool]), axis=0)

    # find all trajectories that did not exit in each direction
    no_low_exit = (low_event == 0)
    no_high_exit = (high_event == 0)

    # find positions where low_event is less than high_event
    # note that either < or <= would work here. They are only equal if both 0.
    low_smaller = (low_event < high_event)

    # find all trajectory outcomes
    # note ambiguity for trajectories that did not exit in a given direction
    low_first = no_high_exit | low_smaller
    high_first = no_low_exit | (~low_smaller)
    never_exit = no_low_exit & no_high_exit

    # find where each trajectory first exits
    refl_indices = low_event * low_first
    trans_indices = high_event * high_first
    stuck_indices = never_exit * (z.shape[0]-1)

    # for now, we assume initial direction is in +z
    init_dir = np.ones(ntraj)

    # init_dir is reverse-corrected for refraction. = kz before medium/sample interface
    # calculate initial weights that actually enter the sample after fresnel
    if tir == False:
        _, _, inc_fraction = fresnel_pass_frac_sphere(radius, np.ones(ntraj), n_medium,
                                                None, n_sample, x, y, z, incident = True)    
    else:
        inc_fraction = np.ones(ntraj)

    # calculate outcome weights from all trajectories
    refl_weights = inc_fraction * select_events(weights, refl_indices)
    trans_weights = inc_fraction * select_events(weights, trans_indices)
    stuck_weights = inc_fraction * select_events(weights, stuck_indices)
    absorb_weights = inc_fraction - refl_weights - trans_weights - stuck_weights

    # warn user if too many trajectories got stuck
    stuck_frac = np.sum(stuck_weights) / np.sum(inc_fraction) * 100
    stuck_traj_warn = " \n{0}% of trajectories did not exit the sample. Increase Nevents to improve accuracy.".format(str(int(stuck_frac)))
    if stuck_frac >= 20: warnings.warn(stuck_traj_warn)

    # correct for non-TIR fresnel reflection upon exiting
    k1_refl, norm_refl, fresnel_pass_frac_refl = fresnel_pass_frac_sphere(radius,refl_indices, n_sample, None, n_medium, x, y, z, 
                                                        plot_exits = plot_exits)
    reflected = refl_weights * fresnel_pass_frac_refl
    if plot_exits == True:
        plt.gca().set_title('Reflected exits')
        plt.gca().view_init(-164,-155)
    k1_trans, norm_trans, fresnel_pass_frac_trans = fresnel_pass_frac_sphere(radius, trans_indices, n_sample, None, n_medium, x, y, z, 
                                                        plot_exits = plot_exits)
    transmitted = trans_weights * fresnel_pass_frac_trans
    if plot_exits == True:
        plt.gca().set_title('Transmitted exits')
        plt.gca().view_init(-164,-155)
    refl_fresnel = refl_weights - reflected
    trans_fresnel = trans_weights - transmitted

    # find fraction that are successfully transmitted or reflected
    refl_frac = np.sum(reflected) / ntraj
    trans_frac = np.sum(transmitted) / ntraj

    # correct for effect of detection angle upon leaving sample
    # TODO: get working for other detector angles
    inc_refl = 1 - inc_fraction # fresnel reflection incident on sample
    inc_refl = detect_correct(np.array([init_dir]), inc_refl, np.ones(ntraj), n_medium, n_medium, detection_angle)
    
    trans_detected = transmitted
    #trans_detected = detect_correct(kz, transmitted, trans_indices, n_sample, n_medium, detection_angle)
    trans_det_frac = np.max([np.sum(trans_detected),eps]) / np.max([np.sum(transmitted), eps])

    refl_detected = reflected
    #refl_detected = detect_correct(kz, reflected, refl_indices, n_sample, n_medium, detection_angle)
    refl_det_frac = np.max([np.sum(refl_detected),eps]) / np.max([np.sum(reflected), eps]) 

    # calculate mean transmittance and reflectance for all trajectories (in terms of trajectory weights)
    reflectance_mean = refl_frac + np.sum(inc_refl)/ntraj
    transmittance_mean = trans_frac

    # calculate new trajectories and reflectance if a significant amount of 
    # light stays inside the sphere due to fresnel reflection
    if run_tir and call_depth < max_call_depth and np.sum(refl_fresnel + trans_fresnel + stuck_weights)/ntraj > .01:
        
        # new weights are the weights that are fresnel reflected back into the 
        # sphere
        nevents = trajectories.nevents
        weights_tir = np.zeros((nevents,ntraj))
        weights_tir[:,:] = refl_fresnel + trans_fresnel + stuck_weights
        weights_tir = sc.Quantity(weights_tir, '')
        
        # new positions are the positions at the exit boundary
        positions = np.zeros((3,nevents+1,ntraj))
        indices = refl_indices + trans_indices
        # get positions outside of sphere boundary from after exit
        select_x1 = select_events(x[1:,:], indices)
        select_y1 = select_events(y[1:,:], indices)
        select_z1 = select_events(z[1:,:], indices)   
        
        # get positions inside sphere boundary from before exit
        select_x0 = select_events(x[:len(x)-1,:],indices)
        select_y0 = select_events(y[:len(y)-1,:],indices)
        select_z0 = select_events(z[:len(z)-1,:],indices)
        
        # get positions at sphere boundary from exit
        x_inter, y_inter, z_inter = find_exit_intersect_vec(select_x0,
                                                            select_y0,
                                                            select_z0,
                                                            select_x1,
                                                            select_y1,
                                                            select_z1, radius)
                                                            
        # new directions are 
        directions = np.zeros((3,nevents,ntraj))
        directions = sc.Quantity(directions, '')
        
        # dot the normal vector with the direction at exit 
        select_kx = select_events(kx, indices)
        select_ky = select_events(ky, indices)
        select_kz = select_events(kz, indices)
        
        #
        dot_kin_normal = np.nan_to_num(np.array([select_kx*x_inter/radius, select_ky*y_inter/radius, select_kz*z_inter/radius])) 
        thetas = np.nan_to_num(np.arccos(dot_kin_normal))
        k_refl = np.array([select_kx,select_ky,select_kz]*(np.cos(thetas)+np.sin(thetas)))

        directions[:,0,:] = k_refl
        directions[0,0,:] = directions[0,0,:] + select_events(kx, stuck_indices)
        directions[1,0,:] = directions[1,0,:] + select_events(ky, stuck_indices)
        directions[2,0,:] = directions[2,0,:] + select_events(kz, stuck_indices)
        
        # set the initial positions at the sphere boundary
        positions[0,0,:] = x_inter + select_events(x[1:,:], stuck_indices)
        positions[1,0,:] = y_inter + select_events(y[1:,:], stuck_indices)
        positions[2,0,:] = z_inter + select_events(z[1:,:], stuck_indices)

        # TODO: get rid of trajectories whose initial weights are 0
        # find indices where initial weights are 0
#        indices = np.where(weights_tir[0,:] == 0)
#        if indices[0].size > 0:
#            weights_tir = np.delete(weights_tir,indices)
#            positions = np.delete(positions, indices, axis = 0)
#            directions = np.delete(directions, indices,axis = 0)
        
        # create new trajectories object
        trajectories_tir = Trajectory(positions, directions, weights_tir)
        # Generate a matrix of all the randomly sampled angles first 
        sintheta, costheta, sinphi, cosphi, _, _ = sample_angles(nevents, ntraj, p)

        # Create step size distribution
        step = sample_step(nevents, ntraj, mu_abs, mu_scat)
    
        # Run photons
        trajectories_tir.absorb(mu_abs, step)
        trajectories_tir.scatter(sintheta, costheta, sinphi, cosphi)         
        trajectories_tir.move(step)

        # Calculate reflection and transmition 
        reflectance_tir, transmittance_tir = calc_refl_trans_sphere(trajectories_tir, 
                                                                    n_medium, n_sample, 
                                                                    radius, p, mu_abs, mu_scat, 
                                                                    plot_exits = plot_exits,
                                                                    tir = True, call_depth = call_depth+1)
        return (reflectance_tir + reflectance_mean, transmittance_tir + transmittance_mean)
        
    else:    
        # need to distribute ambiguous trajectory weights.
        # stuck are 50/50 reflected/transmitted since they are randomized.
        # non-TIR fresnel are treated as new trajectories at the appropriate interface.
        # This means reversed R/T ratios for fresnel reflection at transmission interface.
        extra_refl = 0.5*(refl_fresnel + trans_fresnel + stuck_weights)
        extra_trans = 0.5*(trans_fresnel + refl_fresnel + stuck_weights)
        #calculate mean reflectance and transmittance for all trajectories
        
        # calculate transmittance and reflectance for each trajectory (in terms of trajectory weights)
        transmittance = trans_detected + extra_trans * trans_det_frac
        reflectance = refl_detected + extra_refl * refl_det_frac + inc_refl
        
        # calculate mean reflectance and transmittance for all trajectories
        reflectance_mean = np.sum(reflectance)/ntraj
        transmittance_mean = np.sum(transmittance)/ntraj
        
        if return_extra == True:
            return (k1_refl, k1_trans, norm_refl, norm_trans, reflectance_mean, transmittance_mean)
        
        else:               
            return (reflectance_mean, transmittance_mean) 

def initialize(nevents, ntraj, n_medium, n_sample, seed=None, incidence_angle=0.):

    """
    Sets the trajectories' initial conditions (position, direction, and weight).
    The initial positions are determined randomly in the x-y plane (the initial
    z-position is at z = 0). The default initial propagation direction is set to
    be kz = 1, meaning that the photon packets point straight down in z. The 
    initial weight is currently determined to be a value of choice.
    
    Parameters
    ----------
    nevents : int
        Number of scattering events
    ntraj : int
        Number of trajectories
    n_medium : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the medium.
    n_sample : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample.
    seed : int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are actually random.
    incidence_angle : float
        Maximum value for theta when it incides onto the sample.
        Should be between 0 and pi/2.
    
    Returns
    -------
    r0 : array_like (structcol.Quantity [length])
        Initial position.
    k0 : array_like (structcol.Quantity [dimensionless])
        Initial direction of propagation.
    weight0 : array_like (structcol.Quantity [dimensionless])
        Initial weight. 
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
    r0[0,0,:] = random((1,ntraj))
    r0[1,0,:] = random((1,ntraj))

    # Create an empty array of the initial direction cosines of the right size
    k0 = np.zeros((3, nevents, ntraj))

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

    # Refraction of incident light upon entering sample
    theta = refraction(theta, n_medium, n_sample)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Fill up the first row (corresponding to the first scattering event) of the
    # direction cosines array with the randomly generated angles:
    # kx = sintheta * cosphi
    # ky = sintheta * sinphi
    # kz = costheta
    k0[0,0,:] = sintheta * cosphi
    k0[1,0,:] = sintheta * sinphi
    k0[2,0,:] = costheta

    # Initial weight
    weight0 = np.ones((nevents, ntraj))

    return r0, k0, weight0


def initialize_sphere(nevents, ntraj, n_medium, n_sample, radius, seed=None, 
                      incidence_angle=0., plot_initial=False):
    """
    Sets the trajectories' initial conditions (position, direction, and weight).
    The initial positions are determined randomly in the x-y plane. The initial
    z-positions are confined to the surface of a sphere. The initial propagation
    direction is set to be 1 at z, meaning that the photon packets point 
    straight down in z.
    
    Parameters
    ----------
    nevents : int
        Number of scattering events
    ntraj : int
        Number of trajectories
    n_medium : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the medium.
    n_sample : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample.
    radius : float (structcol.Quantity [length])
        radius of spherical boundary
    seed : int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are actually random.
    incidence_angle : float
        Maximum value for theta when it incides onto the sample.
        Should be between 0 and pi/2.
    plot_inital : boolean
        If plot_initial is set to True, function will create a 3d plot showing
        initial positions and directions of trajectories before entering the 
        sphere and directly after refraction correction upon entering the 
        sphere
    
    Returns
    ----------
    r0 : 2D array_like (structcol.Quantity [length])
        Trajectory positions. Has shape of (3, number of events + 1, number 
        of trajectories). r0[0,0,:] contains random x-positions within a circle 
        on the x-y plane whose radius is the sphere radius. r0[1, 0, :] contains
        random y-positions within the same circle on the x-y plane. r0[2, 0, :]
        contains z-positions on the top hemisphere at the sphere boundary. The 
        rest of the elements are initialized to zero.
    k0 : array_like (structcol.Quantity [dimensionless])
        Initial direction of propagation. Has shape of (3, number of events,
        number of trajectories). k0[0,:,:] and k0[1,:,:] are initalized to zero,
        and k0[2,0,:] is initalized to 1.
    weight0 : array_like (structcol.Quantity [dimensionless])
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
        
    **note: currently only works for normal incidence
    """

    if seed is not None:
        np.random.seed([seed])

    # Initial position. The position array has one more row than the direction
    # and weight arrays because in includes the starting positions on the x-y
    # plane
    r0 = np.zeros((3, nevents+1, ntraj))
    if isinstance(radius, sc.Quantity):
        radius = radius.to('um').magnitude

    # randomly choose x-positions within sphere radius
    r0[0,0,:] = 2*radius * (random((1,ntraj))-.5)
    
    # randomly choose y-positions within sphere radius contrained by x-positions
    for i in range(ntraj):    
        r0[1,0,i] = 2*np.sqrt(radius**2-r0[0,0,i]**2) * (random((1))-.5)
        
    # calculate z-positions from x- and y-positions
    r0[2,0,:] = radius-np.sqrt(radius**2 - r0[0,0,:]**2 - r0[1,0,:]**2)

    # Create an empty array of the initial direction cosines of the right size
    k0 = np.zeros((3, nevents, ntraj))
    
    # find the minus normal vectors of the sphere at the initial positions
    neg_normal = np.zeros((3, ntraj)) # 3 components for each trajectory
    r0_magnitude = np.sqrt(r0[0,0,:]**2 + r0[1,0,:]**2 + (r0[2,0,:]-radius)**2)
    neg_normal[0,:] = -r0[0,0,:]/r0_magnitude
    neg_normal[1,:] = -r0[1,0,:]/r0_magnitude
    neg_normal[2,:] = -(r0[2,0,:]-radius)/r0_magnitude
    
    # solve for theta and phi for these samples
    theta = np.arccos(neg_normal[2,:])
    cosphi = neg_normal[0,:]/np.sin(theta)
    sinphi = neg_normal[1,:]/np.sin(theta)
    
    # refraction of incident light upon entering the sample
    theta = refraction(theta, n_medium, n_sample) 
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    
    # calculate new directions using refracted theta and initial phi
    k0[0,0,:] = sintheta * cosphi
    k0[1,0,:] = sintheta * sinphi
    k0[2,0,:] = costheta

    # Initial weight
    weight0 = np.ones((nevents, ntraj))
    
    if plot_initial == True:
        # plot the initial positions and directions of the trajectories
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim([-radius, radius])
        ax.set_xlim([-radius, radius])
        ax.set_zlim([0, radius])
        ax.set_title('Initial Positions')
        ax.view_init(-164,-155)
        X, Y, Z, U, V, W = [r0[0,0,:],r0[1,0,:],r0[2,0,:],k0[0,0,:], k0[1,0,:], k0[2,0,:]]
        ax.quiver(X, Y, Z, U, V, W, color = 'g')
        
        X, Y, Z, U, V, W = [r0[0,0,:],r0[1,0,:],r0[2,0,:],np.zeros(ntraj), np.zeros(ntraj), np.ones(ntraj)]
        ax.quiver(X, Y, Z, U, V, W)
        
        # draw wireframe hemisphere
        u, v = np.mgrid[0:2*np.pi:20j, np.pi/2:0:10j]
        x = radius*np.cos(u)*np.sin(v)
        y = radius*np.sin(u)*np.sin(v)
        z = radius*(1-np.cos(v))
        ax.plot_wireframe(x, y, z, color=[0.8,0.8,0.8])

    return r0, k0, weight0

def calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen,
              shell_radius=None, phase_mie=False, mu_scat_mie=False):
    """
    Calculates the phase function and scattering coefficient from either the
    single scattering model or Mie theory. Calculates the absorption coefficient
    from Mie theory.
    
    Parameters
    ----------
    radius : float (structcol.Quantity [length])
        Radius of scatterer. If particles are core-shell, should be radius of 
        only the core. 
    n_particle : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the particle.
    n_sample : float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample. If particles are core-shell, should be 
        calculated with the volume fraction of only the cores. 
    volume_fraction : float (structcol.Quantity [dimensionless])
        Volume fraction of the sample. If particles are core-shell, should be 
        volume fraction of only the cores. 
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    shell_radius : float (structcol.Quantity [length]) or None
        Radius of core + shell particle if particles are core-shell.
    phase_mie : bool
        If True, the phase function is calculated from Mie theory. If False
        (default), it is calculated from the single scattering model, which
        includes a correction for the structure factor
    mu_scat_mie : bool
        If True, the scattering coefficient is calculated from Mie theory. If 
        False, it is calculated from the single scattering model 
    
    Returns
    -------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    mu_scat : float (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering model.
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient from Mie theory.
    
    Notes
    -----
    The phase function is given by:
        p = diff. scatt. cross section / cscat
    The single scattering model calculates the differential cross section and
    the total cross section. If we choose to calculate these from Mie theory:
        diff. scat. cross section = S11 / k^2
        p = S11 / (k^2 * cscat)
        (Bohren and Huffmann, chapter 13.3)
    
    """
    
    # Scattering angles (typically from a small angle to pi). A non-zero small 
    # angle is needed because in the single scattering model, if the analytic 
    # formula is used, S(q=0) returns nan. To prevent any errors or warnings, 
    # set the minimum value of angles to be a small value, such as 0.01.
    min_angle = 0.01            
    angles = sc.Quantity(np.linspace(min_angle,np.pi, 200), 'rad') 

    # if shell_radius is None, then the particles are not core-shell
    if shell_radius == None: 
        shell_radius = radius
        
    # if particles are core-shells, calculate volume fractions of only cores 
    # and of core-shells. If particles are not core-shells, 
    # volume_fraction_shell = volume_fraction_core = volume fraction of particles
    volume_fraction_core = volume_fraction    
    volume_fraction_shell = volume_fraction * (shell_radius / radius)**3
    
    number_density = 3.0 * volume_fraction_shell / (4.0 * np.pi * shell_radius**3)
    ksquared = (2 * np.pi *n_sample / wavelen)**2
    m = index_ratio(n_particle, n_sample)
    x_core = size_parameter(wavelen, n_sample, radius)
    x_shell = size_parameter(wavelen, n_sample, shell_radius)
    
    # Calculate the absorption coefficient from Mie theory
    ## Use wavelen/n_sample: wavelength of incident light *in media*
    ## (usually this would be the wavelength in the effective index of the
    ## particle-matrix composite)
    cross_sections = mie.calc_cross_sections(m, x_core, wavelen/n_sample)  
    cabs = cross_sections[2]
    
    mu_abs = (cabs * number_density)

    # If phase_mie is set to True, calculate the phase function from Mie theory
    if phase_mie == True:
        S2squared, S1squared = mie.calc_ang_dist(m, x_core, angles)
        S11 = (S1squared + S2squared)/2
        cscat = cross_sections[0]
        p = S11 / (ksquared * cscat)

    # Calculate the differential and total cross sections from the single
    # scattering model
    diff_sigma_par, diff_sigma_per = \
        model.differential_cross_section(m, x_core, angles, volume_fraction_core, 
                                         x_shell, volume_fraction_shell)
    sigma_total_par = model._integrate_cross_section(diff_sigma_par,
                                                     1.0/ksquared, angles)
    sigma_total_perp = model._integrate_cross_section(diff_sigma_per,
                                                      1.0/ksquared, angles)
    sigma_total = (sigma_total_par + sigma_total_perp)/2.0

    # If phase_mie is set to False, use the phase function from the model
    if phase_mie == False:
        p = (diff_sigma_par + diff_sigma_per)/(ksquared * 2 * sigma_total)

    # If mu_scat_mie is set to True, use the scattering coeff from Mie theory
    if mu_scat_mie == True:
        cscat = cross_sections[0]
        mu_scat = cscat * number_density

    # If mu_scat_mie is set to False, use the scattering coeff from the model
    if mu_scat_mie == False:
        mu_scat = number_density * sigma_total

    # Here, the resulting units of mu_scat and mu_abs are nm^2/um^3. Thus, we 
    # simplify the units to 1/um 
    mu_scat = mu_scat.to('1/um')
    mu_abs = mu_abs.to('1/um')
    
    return p, mu_scat, mu_abs


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
    mu_total = mu_abs + mu_scat

    # Generate array of random numbers from 0 to 1
    rand = np.random.random((nevents,ntraj))

    step = -np.log(1.0-rand) / mu_total

    return step