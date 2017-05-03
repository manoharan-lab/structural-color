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
        
        # beer lambert
        weight = np.exp(-mu_abs*np.cumsum(step_size[:,:], axis=0))
        
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
    valid_events = (events > 0)
    ev = events[valid_events].astype(int) - 1
    tr = np.where(valid_events)[0]

    #want output of the same form as events
    outarray = np.zeros(len(events))
    outarray[valid_events] = inarray[ev, tr]
    if isinstance(inarray, sc.Quantity):
        outarray = sc.Quantity(outarray, inarray.units)
    return outarray

def get_angles(kz, indices):
    '''
    Returns specified angles (relative to normal) from kz components

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

def fresnel_pass_frac(kz, indices, n_before, n_after):
    '''
    Returns weights of interest reduced by fresnel reflection

    Parameters
    ----------
    kz: 2D array
        kz values, with axes corresponding to events, trajectories
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    n_before: float
        Refractive index of the medium light is coming from
    n_after: float
        Refractive index of the medium light is going to

    Returns
    -------
    1D array of length Ntraj
    '''

    #find angles when crossing interface
    theta = get_angles(kz, indices)

    #find fresnel 
    trans_s, trans_p = model.fresnel_transmission(n_before, n_after, theta)
    return (trans_s + trans_p)/2

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
                    detection_angle=np.pi/2):
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
    Note: absorbance by the sample can be found by 1 - reflectance - transmittance

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
    pos_dir = np.mod(np.floor(np.maximum(z_floors[1:], z_floors[:-1])), 2).astype(bool)

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
    init_weight = weights[0]
    init_dir = np.cos(refraction(get_angles(kz, np.ones(ntraj)), n_sample, n_medium))
    # init_dir is reverse-corrected for refraction. = kz before medium/sample interface
    inc_fraction = fresnel_pass_frac(np.array([init_dir]), np.ones(ntraj), n_medium, n_sample)

    # calculate outcome weights from all trajectories
    refl_weights = inc_fraction * select_events(weights, refl_indices)
    trans_weights = inc_fraction * select_events(weights, trans_indices)
    stuck_weights = inc_fraction * select_events(weights, stuck_indices)
    absorb_weights = inc_fraction * init_weight - refl_weights - trans_weights - stuck_weights
    
    # correct for non-TIR fresnel reflection upon exiting
    reflected = refl_weights * fresnel_pass_frac(kz, refl_indices, n_sample, n_medium)
    transmitted = trans_weights * fresnel_pass_frac(kz, trans_indices, n_sample, n_medium)
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
    inc_refl = init_weight * (1 - inc_fraction) # fresnel reflection incident on sample
    inc_refl = detect_correct(np.array([init_dir]), inc_refl, np.ones(ntraj), n_medium, n_medium, detection_angle)
    trans_detected = detect_correct(kz, transmitted, trans_indices, n_sample, n_medium, detection_angle)
    refl_detected = detect_correct(kz, reflected, refl_indices, n_sample, n_medium, detection_angle)

    # calculate transmittance and reflectance for each trajectory (in terms of trajectory weights)
    transmittance = trans_detected + extra_trans * np.sum(trans_detected) / np.sum(transmitted)
    reflectance = refl_detected + extra_refl * np.sum(refl_detected) / np.sum(reflected) + inc_refl

    #calculate mean reflectance and transmittance for all trajectories
    return (np.sum(reflectance)/np.sum(init_weight), np.sum(transmittance/np.sum(init_weight)))    

def calc_reflection_sphere(x, y, z, ntraj, n_matrix, n_sample, kx, ky, kz,
                           radius):
    """
    Counts the fraction of reflected trajectories for a photonic glass with a
    spherical boundary.

    Identifies which trajectories are reflected or transmitted, and at which
    scattering event. Then counts the fraction of reflected trajectories.

    Parameters
    ----------
    x, y, z : array_like (structcol.Quantity [length])
        x, y, z-coordinates of position array.
    ntraj : int
        Number of trajectories.
    n_matrix : float
        Refractive index of the matrix.
    n_sample : float
        Refractive index of the sample.
    kx, ky, kz : array_like (structcol.Quantity [dimensionless])
        x, y, and z components of the direction cosines.
    radius : float
        radius of spherical boundary

    Returns
    ----------
    R_fraction : float
        Fraction of reflected trajectories.

    """
    #TODO this code has not been vectorized like the non-spherical case above
    refl_row_indices = []
    refl_col_indices = []
    trans_row_indices = []
    trans_col_indices = []

    def cutoff(x,y):
        if (x**2 + y**2) < radius**2:
            return radius + np.sqrt(radius**2 - x**2 - y**2)
        else:
            return radius

    def z_low(x,y):
        if (x**2 + y**2) < radius**2:
            return radius - np.sqrt(radius**2 - x**2 - y**2)
        else:
            return radius

    # For each trajectory, find the first scattering event after which the
    # packet exits the system by either getting reflected (z-coord < z_low) or
    # transmitted (z-coord > cutoff):

    for tr in np.arange(ntraj):
        x_tr = x[:,tr]
        y_tr = y[:,tr]
        z_tr = z[:,tr]
        kz_tr = kz[:,tr]

        # If there are any z-positions in the trajectory that are larger
        # than the cutoff (which means the packet has been transmitted), then
        # find the index of the first scattering event at which this happens.
        # If no packet gets transmitted, then leave as NaN.

        #if any(x_tr**2 + y_tr**2 + z_tr**2 > radius):


        for i in range(0,len(z_tr)):
            if z_tr[i] > cutoff(x_tr[i], y_tr[i]):
                trans_row = i
                break
            else:
                trans_row = np.NaN

        # If there are any z-positions in the trajectory that are smaller
        # than z_low (which means the packet has been reflected), then find
        # the index of the first scattering event at which this happens.
        # If no packet gets reflected, then leave as NaN.
        for i in range(0,len(z_tr)):
            if i > 0: # there will not be reflection before first event
                if z_tr[i] < z_low(x_tr[i], y_tr[i]) and kz_tr[i-1]<0:
                    refl_row = i
                    break
                else:
                    refl_row = np.NaN
        # If a packet got transmitted but not reflected in the trajectory,
        # then append the index at which it gets transmitted
        if (type(trans_row) == int and type(refl_row) != int):
            trans_row_indices.append(trans_row)
            trans_col_indices.append(tr)

        # If a packet got reflected but not transmitted in the trajectory,
        # then append the index at which it gets reflected
        if (type(refl_row) == int and type(trans_row) != int):
            refl_row_indices.append(refl_row)
            refl_col_indices.append(tr)

        # If a packet gets both reflected and transmitted, choose whichever
        # happens first
        if (type(trans_row) == int and type(refl_row) == int):
            if trans_row < refl_row:
                trans_row_indices.append(trans_row)
                trans_col_indices.append(tr)
            if trans_row < trans_row:
                refl_row_indices.append(refl_row)
                refl_col_indices.append(tr)

    # create arrays from reflection row and column lists
    #refl_event = np.array(refl_row_indices)-1
    #refl_traj = np.array(refl_col_indices)

    # TODO: add fresnel correction for sphere instead of just for plane
    # calculate fresnel reflectances

    #refl_indices = np.zeros(ntraj)
    #refl_indices[refl_traj] = refl_event

#    refl_fresnel_1, refl_fresnel_2 = fresnel_refl(n_sample, n_matrix, kz, refl_indices)

    # calculate reflected fraction
    refl_fraction = np.array(len(refl_row_indices)) / ntraj
    #refl_fraction = refl_fresnel_1 + (refl_fraction - refl_fresnel_2)*(1- refl_fresnel_1)

    return refl_fraction


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
    weight0 = np.zeros((nevents, ntraj))
    weight0[:,:] = 1.

    return r0, k0, weight0


def initialize_sphere(nevents, ntraj, radius, seed=None, initial_weight = 1):
    """
    Sets the trajectories' initial conditions (position, direction, and weight).

    The initial positions are determined randomly in the x-y plane (the initial
    z-position is at z = 0). The initial propagation direction is set to be 1
    at z, meaning that the photon packets point straight down in z. The initial
    weight is currently determined to be a value of choice.

    Parameters
    ----------
    nevents : int
        Number of scattering events
    ntraj : int
        Number of trajectories
    radius: float
        radius of the bounding sphere
    seed : int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are actually random.

    Returns
    ----------
    r0 : array_like (structcol.Quantity [length])
        Initial position.
    k0 : array_like (structcol.Quantity [dimensionless])
        Initial direction of propagation.
    weight0 : array_like (structcol.Quantity [dimensionless])
        Initial weight.

    """

    if seed is not None:
        np.random.seed([seed])

    # Initial position. The position array has one more row than the direction
    # and weight arrays because in includes the starting positions on the x-y
    # plane
    r0 = np.zeros((3, nevents+1, ntraj))
    r = radius.magnitude*random((1,ntraj))
    t = 2*np.pi*random((1,ntraj))
    r0[0,0,:] = r*np.cos(t)
    r0[1,0,:] = r*np.sin(t)
    r0[2,0,:] = radius.magnitude-np.sqrt(radius.magnitude**2 -
                                         r0[0,0,:]**2 - r0[1,0,:]**2)

    # Initial direction
    eps = 1.e-9
    k0 = np.zeros((3, nevents, ntraj))
    k0[2,0,:] = 1. - eps

    # Initial weight
    weight0 = np.zeros((nevents, ntraj))
    weight0[0,:] = initial_weight

    return r0, k0, weight0


def calc_scat(radius, n_particle, n_sample, volume_fraction, wavelen,
              phase_mie=False, mu_scat_mie=False):
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

    number_density = 3.0 * volume_fraction / (4.0 * np.pi * radius**3)
    ksquared = (2 * np.pi *n_sample / wavelen)**2
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    # Calculate the absorption coefficient from Mie theory
    ## Use wavelen/n_sample: wavelength of incident light *in media*
    ## (usually this would be the wavelength in the effective index of the
    ## particle-matrix composite)
    cross_sections = mie.calc_cross_sections(m, x, wavelen/n_sample)
    cabs = cross_sections[2]
    
    mu_abs = (cabs * number_density)

    # If phase_mie is set to True, calculate the phase function from Mie theory
    if phase_mie == True:
        S2squared, S1squared = mie.calc_ang_dist(m, x, angles)
        S11 = (S1squared + S2squared)/2
        cscat = cross_sections[0]
        p = S11 / (ksquared * cscat)

    # Calculate the differential and total cross sections from the single
    # scattering model
    diff_sigma_par, diff_sigma_per = \
        model.differential_cross_section(m, x, angles, volume_fraction)
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
