# -*- coding: utf-8 -*-
# Copyright 2016 Vinothan N. Manoharan, Victoria Hwang, Annie Stephenson
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
    absorb(mu_abs, mu_scat)
        calculate absorption at each scattering event with given absorption
        and scattering coefficients.
    scatter(sintheta, costheta, sinphi, cosphi)
        calculate directions of propagation after each scattering event with
        given randomly sampled scattering and azimuthal angles.
    move(lscat)
        calculate new positions of the trajectory with given scattering length,
        obtained from either Mie theory or the single scattering model.
    plot_coord(ntraj, three_dim=False)
        plot positions of trajectories as a function of number scattering
        events.

    """

    def __init__(self, position, direction, weight, nevents):
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
        nevents : see Class attributes

        """

        self.position = position
        self.direction = direction
        self.weight = weight
        self.nevents = nevents


    def absorb(self, mu_abs, step_size):
        """
        Calculates absorption of photon packet after each scattering event.

        Absorption is modeled as a reduction of a photon packet's weight
        every time it gets scattered using Beer-Lambert's law.

        Parameters
        ----------
        mu_abs : ndarray (structcol.Quantity [1/length])
            Absorption coefficient of packet.
        step_size: ndarray (structcol.Quantity [length])
            Scattering coefficient of packet.

        """

        # beer lambert
        self.weight = np.exp(-mu_abs*np.cumsum(step_size[:,:], axis=0))



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

        kn = self.direction

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
        self.direction = kn

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

def trajectory_status(z, low_lim, high_lim):
    """
    Determine the outcome of each trajectory for a given sample thickness.

    Parameters
    ----------
    z : array_like (structcol.Quantity [length])
        z-coordinates of position array.
    low_lim : float (structcol.Quantity [length])
        Lower limit that defines the beginning of the simulated sample.
        Usually set to the starting trajectory position and 0.
    high_lim : float (structcol.Quantity [length])
        Upper limit that defines the end of the simulated sample.
        Usually set to the sample's effective thickness.

    Returns
    -------
    exit_low_traj : array
        Indices of trajectories that are reflected.
        Reflection here means exiting the sample in the lower direction.
    exit_low_event : array
        Event indices corresponding to exit in the lower direction.
        Elementwise correspondence to the trajectories in exit_low_traj.
    exit_high_traj : array
        Indices of trajectories that are transmitted.
        Transmission here means exiting the sample in the higher direction.
    exit_high_event : array
        Event indices corresponding to exit in the higher direction.
        Elementwise correspondence to the trajectories in exit_high_traj.
    never_exit_traj : array
        Indices of trajectories that do not exit the sample.
        These trajectories would eventually be reflected or transmitted,
        if allowed to undergo more scattering events.
    """

    # find all events of all trajectories outside limits (low_lim, high_lim)
    low_bool = (z < low_lim)
    high_bool = (z > high_lim)

    # find first exit event of each trajectory in each direction
    # note we convert to 1D array with len = Ntraj
    low_event = np.argmax(low_bool, axis=0)
    high_event = np.argmax(high_bool, axis=0)

    # find all trajectories that did not exit in each direction
    no_low = (low_event == 0)
    no_high = (high_event == 0)

    # find positions where low_event is less than high_event
    # note that either < or <= would work here. They are only equal if both 0.
    low_smaller = (low_event < high_event)

    # find all trajectory outcomes
    # note ambiguity for trajectories that did not exit in a given direction
    low_exit = no_high | low_smaller
    high_exit = no_low | (~low_smaller)
    never_exit = no_low & no_high

    # find where each trajectory first exits
    first_low = low_event * low_exit
    first_high = high_event * high_exit

    #organize values and return
    exit_low_traj = np.where(first_low > 0)[0]
    exit_low_event = first_low[first_low > 0]
    exit_high_traj = np.where(first_high > 0)[0]
    exit_high_event= first_high[first_high > 0]
    never_exit_traj = np.where(never_exit > 0)[0]

    return (exit_low_traj, exit_low_event, exit_high_traj, exit_high_event, never_exit_traj)


def calc_reflection(z, z_low, cutoff, ntraj, n_matrix, n_sample, kx, ky, kz,
                    weights=None, detection_angle=np.pi/2):
    """
    Counts the fraction of reflected trajectories after a cutoff.

    Identifies which trajectories are reflected or transmitted, and at which
    scattering event. Includes Fresnel reflection correction. Then
    counts the fraction of reflected trajectories that are detected.

    Parameters
    ----------
    z : array_like (structcol.Quantity [length])
        z-coordinates of position array.
    z_low : float (structcol.Quantity [length])
        Initial z-position that defines the beginning of the simulated sample.
        Should be set to 0.
    cutoff : float (structcol.Quantity [length])
        Final z-cutoff that determines the effective thickness of the simulated
        sample.
    ntraj : int
        Number of trajectories.
    n_matrix : float
        Refractive index of the matrix.
    n_sample : float
        Refractive index of the sample.
    kx, ky, kz : array_like (structcol.Quantity [dimensionless])
        x, y, and z components of the direction cosines.
    detection_angle : float
        Range of angles of detection. Only the packets that come out of the
        sample within this range will be detected and counted. Should be
        0 < detection_angle <= pi/2, where 0 means that no angles are detected,
        and pi/2 means that all the backscattering angles are detected.

    Returns
    -------
    refl_fraction_corrected : float
        Fraction of reflected trajectories, including the Fresnel correction
        (which includes total internal reflection), and are within the range
        of the detector.

    """
    if weights is None:
        weights = np.ones((kx.shape[0],ntraj))

    refl_col_indices, refl_row_indices, trans_col_indices, trans_row_indices, _ = trajectory_status(z, z_low, cutoff)

    ## Include total internal reflection correction if there is any reflection:

    # If there aren't any reflected packets, then no need to calculate TIR
    if not refl_row_indices.tolist():
        refl_fraction_corrected = 0.0
        theta_r = np.NaN
        phi_r = np.NaN
        print("No photons are reflected because cutoff is too small.")
    else:
        # Now we want to find the scattering and azimuthal angles of the
        # packets as they exit the sample, to see if they would get reflected
        # back into the sample due to TIR.

        # refl_row_indices is the list of indices corresponding to the
        # scattering events immediately after a photon packet gets reflected.
        # Thus, to get the scattering event immediately before the packet exits
        # the sample, we subtract 1. refl_col_indices is the list of indices
        # corresponding to the trajectories in which a photon packet gets
        # reflected.
        ev = np.array(refl_row_indices)-1
        tr = np.array(refl_col_indices)

        # kx, ky, and kz are the direction cosines
        cos_x = kx[ev,tr]
        cos_y = ky[ev,tr]
        cos_z = kz[ev,tr]

        # Find the propagation angles of the photon packets when they are
        # exiting the sample. Count how many of the angles are within the total
        # internal reflection range, and calculate a corrected reflection
        # fraction

        #convert cartesian coordinates into spherical coordinate angles
        theta_r = sc.Quantity(np.arccos(cos_z), 'rad')
        phi_r = np.arctan2(cos_y, cos_x) #angle from [-pi, pi]
        phi_r = sc.Quantity(phi_r + 2*np.pi*(phi_r<0), 'rad') #angle from [0, 2pi]

        # Calculate the Fresnel reflection of all the reflected trajectories
        refl_fresnel_inc, refl_fresnel_out, theta_r, weights_refl = \
            fresnel_refl(n_sample, n_matrix, kz, ev, tr, weights)
            
        # For the trajectories that make it out of the sample after the TIR
        # correction, calculate the thetas after refraction at the interface.
        # The refracted theta is the theta in the global coordinate system.
        refracted_theta = np.pi - np.arcsin(n_sample / n_matrix *
                                            np.sin(np.pi-theta_r))

        # Out of the trajectories that make it out of the sample, find the ones
        # that are within the detector range after being refracted at the
        # interface
        detected_refl_fresnel_out = \
           refl_fresnel_out[np.where(refracted_theta >
                                     (np.pi-detection_angle))]
        weights_refl = weights_refl[np.where(refracted_theta >
                                             (np.pi-detection_angle))]
        refl_fraction = np.array(len(detected_refl_fresnel_out)) / ntraj

        # Only keep the refracted theta that are within angle of detection
        refl_fresnel_out_avg = np.sum(detected_refl_fresnel_out) / ntraj
        refl_fresnel_inc_avg = np.sum(refl_fresnel_inc) / ntraj
        weights_refl_avg = np.sum(weights_refl) / len(weights_refl)
        refl_fraction_corrected = (refl_fresnel_inc_avg +
                                   (refl_fraction - refl_fresnel_out_avg) *
                                   (1- refl_fresnel_inc_avg))*weights_refl_avg

    return refl_fraction_corrected


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
#    refl_fresnel_1, refl_fresnel_2 = fresnel_refl(n_sample, n_matrix, kz,
#                                                  refl_event, refl_traj)

    # calculate reflected fraction
    refl_fraction = np.array(len(refl_row_indices)) / ntraj
    #refl_fraction = refl_fresnel_1 + (refl_fraction - refl_fresnel_2)*(1- refl_fresnel_1)

    return refl_fraction


def initialize(nevents, ntraj, seed=None, incidence_angle=0.):

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

def calc_scat(radius, n_particle, n_sample, volume_fraction, angles, wavelen,
              phase_mie=False, lscat_mie=False):
    """
    Calculates the phase function and scattering length from either the single
    scattering model or Mie theory. Calculates the absorption length from Mie
    theory.

    Parameters
    ----------
    radius : float (structcol.Quantity [length])
        Radius of scatterer.
    n_particle : float
        Refractive index of the particle.
    n_sample : float
        Refractive index of the sample.
    volume_fraction : float
        Volume fraction of the sample.
    angles : array_like (structcol.Quantity [rad])
        Scattering angles (typically from a small angle to pi). A non-zero
        small angle is needed because in the single scattering model, if the
        analytic formula is used, S(q=0) returns nan. To prevent any errors or
        warnings, set the minimum value of angles to be a small value, such
        as 0.01.
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    phase_mie : bool
        If True, the phase function is calculated from Mie theory. If False
        (default), it is calculated from the single scattering model, which
        includes a correction for the structure factor
    lscat_mie : bool
        If True, the scattering length is calculated from Mie theory. If False,
        it is calculated from the single scattering model

    Returns
    -------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    lscat : float (structcol.Quantity [length])
        Scattering length from either Mie theory or single scattering model.
    labs : float (structcol.Quantity [length])
        Absorption length from Mie theory.

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

    number_density = 3.0 * volume_fraction / (4.0 * np.pi * radius**3)
    ksquared = (2 * np.pi *n_sample / wavelen)**2
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    # Calculate the absorption length from Mie theory
    ## Use wavelen/n_sample: wavelength of incident light *in media*
    ## (usually this would be the wavelength in the effective index of the
    ## particle-matrix composite)
    cross_sections = mie.calc_cross_sections(m, x, wavelen/n_sample)
    cabs = cross_sections[2]
    labs = 1 / (cabs * number_density)

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

    # If lscat_mie is set to True, use the scattering length from Mie theory
    if lscat_mie == True:
        cscat = cross_sections[0]
        lscat = 1 / (cscat * number_density)

    # If lscat_mie is set to False, use the scattering length from the model
    if lscat_mie == False:
        lscat = 1 / number_density / sigma_total

    # Here, the resulting units of lscat and labs are um^3/nm^2. Thus, we
    # simplify the units to um
    lscat = lscat.to('um')
    labs = labs.to('um')

    return p, lscat, labs


def sample_angles(nevents, ntraj, p, angles):
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
    angles : array_like (structcol.Quantity [rad])
        Scattering angles (typically from 0 to pi).

    Returns
    -------
    sintheta, costheta, sinphi, cosphi, theta, phi : ndarray
        Sampled azimuthal and scattering angles, and their sines and cosines.

    """

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


def fresnel_refl(n_sample, n_matrix, kz, refl_event, refl_traj, weights):
    """
    Calculates the reflectance at the interface of two refractive indices using
    the fresnel equations. This calculation will include total internal
    reflection

    Parameters
    ----------
    n_matrix : float
        Refractive index of the matrix.
    n_sample : float
        Refractive index of the sample.
    kz : array_like (structcol.Quantity [dimensionless])
        x components of the direction cosines.
    refl_event : array
        Indices of reflection events.
    refl_traj : array_like (structcol.Quantity [dimensionless])
        Indices of reflected trajectories.

    Returns
    -------
    refl_fresnel_inc : array
        Array of Fresnel reflectance fractions of light reflected for each
        photon due to the interface when the trajectory first enters the
        sample.
    refl_fresnel_out : array
        Array of Fresnel reflectance fractions of light reflected for each
        photon due to the interface when the trajectory leaves the sample.
    theta_out : array
        Array of the scattering angles that make it out of the sample after
        eliminating the trajectories that get totally internally reflected.
    weights_refl: array
        Array of the weights of the trajectories make it out of the sample
        after eliminating the trajectories that get totally internally
        reflected.

    """
    # TODO: add option to modify theta calculation to incorperate curvature of
    # sphere

    # Calculate fresnel for incident light going from medium to sample
    theta_inc = np.arccos(kz[0,:])
    refl_s_inc, refl_p_inc = \
        model.fresnel_reflection(n_matrix, n_sample, sc.Quantity(theta_inc, ''))
    refl_fresnel_inc = .5*(refl_s_inc + refl_p_inc)

    # Calculate fresnel for reflected light going from sample to medium
    theta_out = np.arccos(-kz[refl_event,refl_traj])
    refl_s_out, refl_p_out = \
        model.fresnel_reflection(n_sample, n_matrix, sc.Quantity(theta_out, ''))
    refl_fresnel_out = .5*(refl_s_out + refl_p_out)

    # Find the thetas that do not get TIR'd
    theta_out = np.pi-theta_out[np.where(refl_fresnel_out < 1)]
    
    weights_refl = weights[refl_event, refl_traj]
    weights_refl = weights_refl[np.where(refl_fresnel_out<1)]
    refl_fresnel_out = refl_fresnel_out[refl_fresnel_out < 1]

    return refl_fresnel_inc, refl_fresnel_out, theta_out, weights_refl

