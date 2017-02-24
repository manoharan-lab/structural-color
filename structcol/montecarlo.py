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

from . import mie, index_ratio, size_parameter
import numpy as np
from numpy.random import random as random
import structcol as sc
from structcol import model
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
        dimensions of 
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
        obtained from Mie theory.
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


    def absorb(self, mu_abs, mu_scat):
        """
        Calculates absorption of photon packet after each scattering event.

        Absorption is modeled as a reduction of a photon packet's weight 
        every time it gets scattered. Currently, absorption is not modeled
        independently of scattering.
        
        Parameters
        ----------
        mu_abs : float (structcol.Quantity [1/length])
            Absorption coefficient of packet.
        mu_scat: float (structcol.Quantity [1/length])
            Scattering coefficient of packet.

        Returns 
        -------
        array_like
            New weight of packet 
        
        """

        # Extinction coefficient is sum of absorption and scattering coeff.
        mu_total = mu_abs + mu_scat
        
        # At each scattering event, the photon packet loses part of its weight        
        delta_weight = self.weight * mu_abs / mu_total

        self.weight = self.weight - delta_weight
        
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
        # followed by a rotation about the z-axis by anlge phi
        for n in np.arange(1,self.nevents):
            kx = (kn[0,n-1,:]*costheta[n-1,:] + kn[2,n-1,:]*sintheta[n-1,:])*cosphi[n-1,:] - kn[1,n-1,:]*sinphi[n-1,:]
                    
            ky = (kn[0,n-1,:]*costheta[n-1,:] + kn[2,n-1,:]*sintheta[n-1,:])*sinphi[n-1,:] + kn[1,n-1,:]*cosphi[n-1,:]

            kz = -kn[0, n-1, :]*sintheta[n-1,:] + kn[2, n-1, :]*costheta[n-1,:]

            kn[:,n,:] = kx, ky, kz
            
        # Update all the directions of the trajectories
        self.direction = kn


    def move(self, lscat):
        """
        Calculates positions of photon packets in all the trajectories.

        After each scattering event, the photon packet gets a new position 
        based on the previous position, the step size, and the direction of 
        propagation.

        Parameters
        ----------
        lscat : float (structcol.Quantity [length])
            Scattering length from Mie theory, which is used as the step size 
            between scattering events in the trajectories.
      
        """

        displacement = self.position
        displacement[:, 1:, :] = lscat * self.direction

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
        colors = itertools.cycle([colormap(i) for i in np.linspace(0, 0.9, ntraj)])

        f, ax = plt.subplots(3, figsize=(8,17), sharex=True)

        ax[0].plot(np.arange(len(self.position[0,:,0])), self.position[0,:,:], '-')
        ax[0].set_title('Positions during trajectories')
        ax[0].set_ylabel('x (' + str(self.position.units) + ')')

        ax[1].plot(np.arange(len(self.position[1,:,0])), self.position[1,:,:], '-')
        ax[1].set_ylabel('y (' + str(self.position.units) + ')')

        ax[2].plot(np.arange(len(self.position[2,:,0])), self.position[2,:,:], '-')
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

    
def fresnel_refl(n_sample, n_matrix, kz, refl_event, refl_traj):
    """
    calculates the reflectance at the interface of two refractive indeces using
    the fresnel equations. This calculation will include total internal reflection

    Parameters
    ----------
    n_matrix : float
        Refractive index of the matrix.
    n_sample : float
        Refractive index of the sample.
    kz : array_like (structcol.Quantity [dimensionless])
        x components of the direction cosines. 
    refl_event : array
        indices of reflection events
    refl_traj : array_like (structcol.Quantity [dimensionless])
        indices of reflected trajectories
    
    Returns
    -------
   refl_fresnel_incident_avg : float
      fraction of light reflected due to the interface when the trajectory first 
      enters the sample
   
   refl_fresnel_return_avg : float
      fraction of light reflected due to the interface leaving the sample 
      traveling into the surounding medium
    """
    # TODO: add option to modify theta calculation to incorperate curvature of sphere    
    
    # calculate fresnel for incident light going from medium to sample
    theta = np.arccos(kz[0,:])
    refl_s, refl_p = model.fresnel_reflection(n_matrix, n_sample, sc.Quantity(theta, ''))
    refl_fresnel = .5*(refl_s + refl_p)
    refl_fresnel_incident_avg = np.mean(refl_fresnel)
    
    # calculate fresnel for reflected light going from sample to medium
    theta = np.arccos(-kz[refl_event,refl_traj])
    refl_s, refl_p = model.fresnel_reflection(n_sample, n_matrix, sc.Quantity(theta, ''))
    refl_fresnel = .5*(refl_s + refl_p)
    refl_fresnel_return_avg = np.sum(refl_fresnel)/kz.shape[1]
    
    return refl_fresnel_incident_avg, refl_fresnel_return_avg
    


def refl_trans_counter(z, z_low, cutoff, ntraj, n_matrix, n_sample, kx, ky, kz):
    """
    Counts the fraction of reflected trajectories after a cutoff.
    
    Identifies which trajectories are reflected or transmitted, and at which 
    scattering event. Includes total internal reflection correction. Then 
    counts the fraction of reflected trajectories. 

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
    
    Returns
    -------
    refl_fraction_corrected : float
        Fraction of reflected trajectories, including the total internal 
        reflection correction.
    theta_r : array_like (structcol.Quantity [rad])
        Scattering angles when the photon packets exit the sample (defined with
        respect to global coordinate system of the sample).
    phi_r : array_like (structcol.Quantity [rad]) 
        Azimuthal angles when the photon packets exit the sample (defined with
        respect to global coordinate system of the sample).
        
    """

    refl_row_indices = []
    refl_col_indices = []
    trans_row_indices = []
    trans_col_indices = []
    
    # For each trajectory, find the first scattering event after which the  
    # packet exits the system by either getting reflected (z-coord < z_low) or
    # transmitted (z-coord > cutoff):
    for tr in np.arange(ntraj):
        z_tr = z[:,tr]

        # If there are any z-positions in the trajectory that are larger
        # than the cutoff (which means the packet has been transmitted), then   
        # find the index of the first scattering event at which this happens. 
        # If no packet gets transmitted, then leave as NaN.
        if any(z_tr > cutoff):
            z_trans = next(zi for zi in z_tr if zi > cutoff)
            trans_row = z_tr.tolist().index(z_trans)
        else:
            trans_row = np.NaN

        # If there are any z-positions in the trajectory that are smaller
        # than z_low (which means the packet has been reflected), then find  
        # the index of the first scattering event at which this happens. 
        # If no packet gets reflected, then leave as NaN.
        if any(z_tr < z_low):
            z_refl = next(zi for zi in z_tr if zi < z_low)
            refl_row = z_tr.tolist().index(z_refl)
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
            if refl_row < trans_row:
                refl_row_indices.append(refl_row)
                refl_col_indices.append(tr)


    ## Include total internal reflection correction if there is any reflection:
    
    # If there aren't any reflected packets, then no need to calculate TIR
    if not refl_row_indices:
        refl_fraction_corrected = 0.0
        theta_r = np.NaN
        phi_r = np.NaN
        print("No photons are reflected because cutoff is too small.")
    else:
        # Calculate total internal reflection angle
#       sin_alpha_sample = np.sin(np.pi - np.pi/2) * n_matrix/n_sample
#
#        if sin_alpha_sample >= 1:
#            theta_min_refracted = np.pi/2.0
#        else:
#            theta_min_refracted = np.pi - np.arcsin(sin_alpha_sample)

        # Now we want to find the scattering and azimuthal angles of the packets
        # as they exit the sample, to see if they would get reflected back into 
        # the sample due to TIR.
        theta_r = []
        phi_r = []
#        count = 0

        # R_row_indices is the list of indices corresponding to the scattering
        # events immediately after a photon packet gets reflected. Thus, to get the 
        # scattering event immediately before the packet exits the sample, we 
        # subtract 1.  R_col_indices is the list of indices corresponding to the 
        # trajectories in which a photon packet gets reflected. 
        ev = np.array(refl_row_indices)-1
        tr = np.array(refl_col_indices)

        # kx, ky, and kz are the direction cosines
        cos_x = kx[ev,tr]
        cos_y = ky[ev,tr]
        cos_z = kz[ev,tr]

        # Find the propagation angles of the photon packets when they are exiting
        # the sample. Count how many of the angles are within the total internal 
        # reflection range, and calculate a corrected reflection fraction
        for i in range(len(cos_x)):

            # Solve for correct theta and phi from the direction cosines,
            # accounting for parity of sin and cos functions
            # cos_x = sinθ * cosφ 
            # cos_y = sinθ * sinφ  
            # cos_z = cosθ 

            # The arccos function in numpy takes values from 0 to pi. When we solve
            # for theta, this is fine because theta goes from 0 to pi.
            theta = np.arccos(cos_z[i])      
            theta_r.append(theta)            
        
            # However, phi goes from 0 to 2 pi, which means we need to account for 
            # two possible solutions of arccos so that they span the 0 - 2pi range. 
            phi1 = np.arccos(cos_x[i] / np.sin(theta))
        
            # I define pi as a quantity in radians, so that phi is in radians.        
            pi = sc.Quantity(np.pi,'rad')
            phi2 = - np.arccos(cos_x[i] / np.sin(theta)) + 2*pi
    
            # Same for arcsin. 
            phi3 = np.arcsin(cos_y[i] / np.sin(theta))
            if phi3 < 0:
                phi3 = phi3 + 2*pi 
            phi4 = - np.arcsin(cos_y[i] / np.sin(theta)) + pi 

            # Now we need to figure out which phi in each pair is the correct one,
            # since only either phi1 or phi2 will match either phi3 or phi4
            A = np.array([abs(phi1-phi3),abs(phi1-phi4),abs(phi2-phi3),abs(phi2-phi4)])
        
            # Find which element in A is the minimum        
            B = A.argmin(0)

            # If the first element in A is the minimum, then the correct solution 
            # is phi1 = phi3, so append their average:
            if B == 0:
                phi_r.append((phi1+phi3)/2)
            elif B == 1:
                phi_r.append((phi1+phi4)/2)
            elif B == 2:
                phi_r.append((phi2+phi3)/2)
            elif B == 3:
                phi_r.append((phi2+phi4)/2)

            # Count how many of the thetas correspond to the range of total 
            # internal reflection
#            if theta < theta_min_refracted:
#                count = count + 1

        # Calculate corrected reflection fraction
#        refl_fraction_corrected = np.array(len(refl_row_indices) - count) / ntraj
    refl_fraction_corrected = np.array(len(refl_row_indices)) / ntraj
        
    # added by Annie
    refl_event = np.array(refl_row_indices)-1
    refl_traj = np.array(refl_col_indices)
    refl_fresnel_1, refl_fresnel_2 = fresnel_refl(n_sample, n_matrix, kz, refl_event, refl_traj)
    refl_fraction_corrected = refl_fresnel_1 + (refl_fraction_corrected - refl_fresnel_2)*(1- refl_fresnel_1) 

    return refl_fraction_corrected#, theta_r, phi_r


def refl_trans_counter_sphere(x, y, z, ntraj, n_matrix, n_sample, kx, ky, kz, radius):
    """
    Counts the fraction of reflected trajectories for a photonic ball.
    
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
        radius of photonic ball
    
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
    #refl_fresnel_1, refl_fresnel_2 = fresnel_refl(n_sample, n_matrix, kz, refl_event, refl_traj)
    
    # calculate reflected fraction
    refl_fraction = np.array(len(refl_row_indices)) / ntraj
    #refl_fraction = refl_fresnel_1 + (refl_fraction - refl_fresnel_2)*(1- refl_fresnel_1) 

    return refl_fraction
    

def initialize_sphere(nevents, ntraj, radius, seed=None):
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
        radius of the photonic ball
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
    r0[2,0,:] = radius.magnitude-np.sqrt(radius.magnitude**2-r0[0,0,:]**2-r0[1,0,:]**2)

    # Initial direction
    eps = 1.e-9
    k0 = np.zeros((3, nevents, ntraj))
    k0[2,0,:] = 1. - eps

    # Initial weight
    weight0 = np.zeros((nevents, ntraj))
    weight0[0,:] = 0.001                  # (figure out how to determine this)


    return r0, k0, weight0


def initialize(nevents, ntraj, seed=None, initial_weight=0.001, eps = 1.e-9):
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
    initial_weight : float
        Initial weight of the photon packet. (Note: we still need to decide how
        to determine this value). 
    eps : float
        Difference between the initial z-direction cosine value and 1. The
        initial z-direction value should not be exactly 1.0 because this leads to
        a divide-by-zero error in the 'scatter' function, when it calculates the 
        denominator of the equations for the new directions of propagation.
        eps should be smaller than the tolerance parameter 'atol' in the 'scatter'
        function.

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

    # Initial direction
    k0 = np.zeros((3, nevents, ntraj))
    k0[2,0,:] = 1. - eps

    # Initial weight
    weight0 = np.zeros((nevents, ntraj))
    weight0[0,:] = initial_weight                

    return r0, k0, weight0


def phase_function(radius, n_particle, n_sample, angles, wavelen):
    """
    Calculates the phase function from Mie theory.

    Parameters
    ----------  
    radius : float (structcol.Quantity [length])
        Radius of scatterer.
    n_particle : float
        Refractive index of the particle.
    n_sample : float
        Refractive index of the sample.
    angles : array_like (structcol.Quantity [rad])
        Scattering angles (typically from 0 to pi).
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    
    Returns 
    -------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function 
    
    Notes 
    -----
    p = diff. scatt. cross section / cscat    
    diff. scat. cross section = S11 / k^2
    p = S11 / (k^2 * cscat)
    (Bohren and Huffmann, chapter 13.3)
    
    """

    angles = angles.to('rad')
    ksquared = (2 * np.pi *n_sample / wavelen)**2

    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    S2squared, S1squared = mie.calc_ang_dist(m, x, angles)
    S11 = (S1squared + S2squared)/2
    cscat = mie.calc_cross_sections(m, x, wavelen/n_sample)[0]
    
    p = S11 / (ksquared * cscat)

    return p


def scat_abs_length(radius, n_particle, n_sample, volume_fraction, wavelen):
    """
    Calculates the scattering and absorption lengths from Mie theory.

    Parameters
    ---------- 
    radius : float (structcol.Quantity [length])
        Radius of scatterer.
    n_particle : float 
        Refractive index of the particle.
    n_sample : float 
        Refractive index of the sample.
    volume_fraction : float
        Volume fraction of scatterers in the sample.
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.

    Returns
    ------- 
    lscat : float (structcol.Quantity [length])
        Scattering length.
    labs : float (structcol.Quantity [length])
        Absorption length. 
    
    """

    number_density = 3.0 * volume_fraction / (4.0 * np.pi * radius**3)
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    # Use wavelen/n_sample: wavelength of incident light *in media* (usually 
    # this would be the wavelength in the effective index of the 
    # particle-matrix composite)
    cross_sections = mie.calc_cross_sections(m, x, wavelen/n_sample)
    cscat = cross_sections[0]
    cabs = cross_sections[2]

    lscat = 1 / (cscat * number_density)
    labs = 1 / (cabs * number_density) 

    # Here, the resulting units of lscat and labs are um^3/nm^2. Thus, we 
    # simplify the units to um
    lscat = lscat.to('um')
    labs = labs.to('um')

    return lscat, labs


def sampling(nevents, ntraj, p, angles):
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

    # Random sampling of azimuthal angle phi from uniform distribution [0 - 2pi]
    rand = np.random.random((nevents,ntraj))
    phi = 2*np.pi*rand
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    # Random sampling of scattering angle theta
    prob = p * np.sin(angles)*2*np.pi    # prob is integral of p in solid angle
    prob_norm = prob/sum(prob)           # normalize to make it add up to 1

    theta = np.array([np.random.choice(angles, ntraj, p = prob_norm) for i in range(nevents)])
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    return sintheta, costheta, sinphi, cosphi, theta, phi
    
