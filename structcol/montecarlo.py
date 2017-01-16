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
#np.random.seed([10])
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
        obtained from Mie theory.
    plot_coord(ntraj, three_dim=False)
        plot positions of trajectories as a function of number scattering
        events.

    """

    def __init__(self, position, direction, weight, nevents):
        """
        position : dimensions of (3, nevents+1, ntrajectories)
        direction : dimensions of (3, nevents, ntrajectories)
        weight : dimensions of (nevents, ntrajectories)
        
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
        ----------
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

        References
        ----------
        ..  [1] L. Wang, S. L. Jacquesa, L. Zhengb, "MCML - Monte Carlo  
            modeling of light transport in multi-la.yered. tissues," Computer  
            Methods and Programs in Biomedicine, vol. 47, 131-146, 1995.

        """

        kn = self.direction

        # TODO: the hard-coded epsilon factor is worrying; it's never a good
        # idea to hard-code constants like this. What determines epsilon? What
        # happens if you change the value? See if you can rewrite the code so
        # that the tolerance factor either isn't needed (consider using np.isclose()).

        # RESPONSE: I used the epsilon factor because [1] recommends using
        # simplified formulas for the cases where the z-direction cosine is 
        # close to 1, since trigonometric functions are computationally intensive.
        # Thus, I chose a random value epsilon that is the difference between 
        # the z-direction and 1 to determine when these are the same. Nonetheless,
        # I replaced the hard coded epsilon with np.isclose(), althought 
        # np.isclose() also seems to have hard coded tolerance parameters. I 
        # seem to get the same results with either method.

        # Calculate the new x, y, z coordinates of the propagation direction 
        # by multiplying (cross product?) the previous propagation direction 
        # by the scattering and azimuthal angles of the corresponding event. 
        # This is to go from the local spherical coordinate system to the 
        # global cartesian coordinate system.
        for n in np.arange(1,self.nevents):
            
            # kz is z-component of the propagation direction. If kz is not 
            # close to 1, the set of equations to calculate the new propagation 
            # direction are [1]:
            denom = (1-kn[2,n-1,:]**2)**0.5

            kx = sintheta[n-1,:] * (kn[0,n-1,:] * kn[2,n-1,:] * cosphi[n-1,:] -
                    kn[1,n-1,:] * sinphi[n-1,:]) / denom + kn[0,n-1,:] * costheta[n-1,:]

            ky = sintheta[n-1,:] * (kn[1,n-1,:] * kn[2,n-1,:] * cosphi[n-1,:] +
                    kn[0,n-1,:] * sinphi[n-1,:]) / denom + kn[1,n-1,:] * costheta[n-1,:]

            kz = -sintheta[n-1,:] * cosphi[n-1,:] * denom + kn[2,n-1,:] * costheta[n-1,:]

            # If kz is close to 1 (propagation direction is straight down in 
            # z-axis) and >= 0, the set of equatons are [1]:
            kx2 = sintheta[n-1,:] * cosphi[n-1,:]
            ky2 = sintheta[n-1,:] * sinphi[n-1,:]
            kz2 = costheta[n-1,:]

            # If kz is close to 1 and < 0, the set of equatons are [1]:
            ky3 = -sintheta[n-1,:] * sinphi[n-1,:]
            kz3 = -costheta[n-1,:]

            # Choose which set of equations to use based on the conditions
            # described above
        
            kn[:,n,:] = np.where(np.logical_not(np.isclose(1-denom**2, 1, rtol=1e-05, atol=1e-08, equal_nan=False)), (kx, ky, kz),
                                 np.where(kn[2,n-1,:]>= 0., (kx2, ky2, kz2), (kx2, ky3, kz3)))
            
            # (KEEP UNTIL FURTHER REVISIONS)
            #eps = 1e-4
            #kn[:,n,:] = np.where(denom > eps, (kx, ky, kz),
            #                     np.where(kn[2,n-1,:]>= 0., (kx2, ky2, kz2), (kx2, ky3, kz3)))

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


def RTcounter(z, z_low, cutoff, ntraj, n_matrix, n_sample, kx, ky, kz):
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
    ----------
    R_fraction_corrected : float
        Fraction of reflected trajectories, including the total internal 
        reflection correction.
    theta_r : array_like (structcol.Quantity [rad])
        Scattering angles when the photon packets exit the sample (defined with
        respect to global coordinate system of the sample).
    phi_r : array_like (structcol.Quantity [rad]) 
        Azimuthal angles when the photon packets exit the sample (defined with
        respect to global coordinate system of the sample).
        
    """

    R_row_indices = []
    R_col_indices = []
    T_row_indices = []
    T_col_indices = []
    
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
            z_T = next(zi for zi in z_tr if zi > cutoff)
            T_row = z_tr.tolist().index(z_T)
        else:
            T_row = np.NaN

        # If there are any z-positions in the trajectory that are smaller
        # than z_low (which means the packet has been reflected), then find  
        # the index of the first scattering event at which this happens. 
        # If no packet gets reflected, then leave as NaN.
        if any(z_tr < z_low):
            z_R = next(zi for zi in z_tr if zi < z_low)
            R_row = z_tr.tolist().index(z_R)
        else:
            R_row = np.NaN

        # If a packet got transmitted but not reflected in the trajectory, 
        # then append the index at which it gets transmitted
        if (type(T_row) == int and type(R_row) != int):
            T_row_indices.append(T_row)
            T_col_indices.append(tr)

        # If a packet got reflected but not transmitted in the trajectory, 
        # then append the index at which it gets reflected
        if (type(R_row) == int and type(T_row) != int):
            R_row_indices.append(R_row)
            R_col_indices.append(tr)

        # If a packet gets both reflected and transmitted, choose whichever 
        # happens first
        if (type(T_row) == int and type(R_row) == int):
            if T_row < R_row:
                T_row_indices.append(T_row)
                T_col_indices.append(tr)
            if R_row < T_row:
                R_row_indices.append(R_row)
                R_col_indices.append(tr)


    ## Include total internal reflection correction:

    # Calculate total internal reflection angle
    sin_alpha_sample = np.sin(np.pi - np.pi/2) * n_matrix/n_sample

    if sin_alpha_sample >= 1:
        theta_min_refracted = np.pi/2.0
    else:
        theta_min_refracted = np.pi - np.arcsin(sin_alpha_sample)

    # Now we want to find the scattering and azimuthal angles of the packets
    # as they exit the sample, to see if they would get reflected back into 
    # the sample due to TIR.
    theta_r = []
    phi_r = []
    count = 0

    # R_row_indices is the list of indices corresponding to the scattering
    # events immediately after a photon packet gets reflected. Thus, to get the 
    # scattering event immediately before the packet exits the sample, we 
    # subtract 1.  R_col_indices is the list of indices corresponding to the 
    # trajectories in which a photon packet gets reflected. 
    ev = np.array(R_row_indices)-1
    tr = np.array(R_col_indices)
    
    # kx, ky, and kz are the direction cosines
    cosA = kx[ev,tr]
    cosB = ky[ev,tr]
    cosC = kz[ev,tr]

    # Find the propagation angles of the photon packets when they are exiting
    # the sample. Count how many of the angles are within the total internal 
    # reflection range, and calculate a corrected reflection fraction
    for i in range(len(cosA)):

        # Solve for correct theta and phi from the direction cosines,
        # accounting for parity of sin and cos functions
        # cosA = sinθ * cosφ 
        # cosB = sinθ * sinφ  
        # cosC = cosθ 

        # The arccos function in numpy takes values from 0 to pi. When we solve
        # for theta, this is fine because theta goes from 0 to pi.
        theta = np.arccos(cosC[i])      
        theta_r.append(theta)            
        
        # However, phi goes from 0 to 2 pi, which means we need to account for 
        # two possible solutions of arccos so that they span the 0 - 2pi range. 
        phi1 = np.arccos(cosA[i] / np.sin(theta))
        
        # I define pi as a quantity in radians, so that phi is in radians.        
        pi = sc.Quantity(np.pi,'rad')
        phi2 = - np.arccos(cosA[i] / np.sin(theta)) + 2*pi

        # Same for arcsin. 
        phi3 = np.arcsin(cosB[i] / np.sin(theta))
        if phi3 < 0:
            phi3 = phi3 + 2*pi 
        phi4 = - np.arcsin(cosB[i] / np.sin(theta)) + pi 

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
        if theta < theta_min_refracted:
            count = count + 1

        # (Keep until further revisions):
        #phi2 = -np.arccos(cosA[i] / np.sin(theta))
        #phi3 = np.arcsin(cosB[i] / np.sin(theta))
        #phi4 = np.pi - np.arcsin(cosB[i] / np.sin(theta))
        #A = np.array([abs(phi1-phi3),abs(phi1-phi4),abs(phi2-phi3),abs(2*np.pi+phi2-phi4)])
        #elif B == 3:
        #    phi_r.append((2*np.pi+phi2+phi4)/2)

    # Calculate corrected reflection fraction
    R_fraction_corrected = np.array(len(R_row_indices) - count) / ntraj

    return R_fraction_corrected, theta_r, phi_r


def initialize(nevents, ntraj, seed=None):
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
    r0[0,0,:] = random((1,ntraj))
    r0[1,0,:] = random((1,ntraj))

    # Initial direction
    eps = 1.e-9
    k0 = np.zeros((3, nevents, ntraj))
    k0[2,0,:] = 1. - eps

    # Initial weight
    weight0 = np.zeros((nevents, ntraj))
    weight0[0,:] = 0.001                  # (figure out how to determine this)

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
    ----------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function 
    
    Notes 
    ----------
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
    ---------- 
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
    # TODO: why is it necessary to convert to micrometers? If the code depends
    # on the length scale being in micrometers, this should be explicitly
    # documented (or should use a function decorator to ensure that the
    # arguments have the correct dimensions)
    
    # RESPONSE: I converted to um because I don't know how to make pint simplify
    # the units when they are something like "nm^3/um^2". So I'm forcing the 
    # simplification here
    lscat = lscat.to('um')

    labs = 1 / (cabs * number_density)
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
    ---------- 
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
