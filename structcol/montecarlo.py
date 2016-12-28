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
        array of photon packet weights during for n trajectories
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
        # TODO: give the expected dimensions of each of these arrays (e.g. (3, nevents, ntrajectories))
        self.position = position
        self.direction = direction
        self.weight = weight
        self.nevents = nevents

    def absorb(self, mu_abs, mu_scat):
        """
        Calculates absorption of packet.

        mu_abs: absorption coefficient
        mu_scat: scattering coefficient

        TODO: add docstring in numpy docstring format
        (see https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
        TODO: please explain what is happening here.
        """

        mu_total = mu_abs + mu_scat
        delta_weight = self.weight * mu_abs / mu_total

        self.weight = self.weight - delta_weight

        return self.weight

    def scatter(self, sintheta, costheta, sinphi, cosphi):
        """
        Calculates the directions of propagation after scattering.

        sintheta, costheta, sinphi, cosphi: scattering and azimuthal angles
        sampled from the phase function.

        TODO: add docstring (see https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
        """

        kn = self.direction

        # TODO: the hard-coded epsilon factor is worrying; it's never a good
        # idea to hard-code constants like this. What determines epsilon? What
        # happens if you change the value? See if you can rewrite the code so
        # that the tolerance factor either isn't needed (consider using np.isclose()).
        # also explain what you are doing here.
        eps = 1e-4

        #TODO: please explain and comment what is going on here.  The code is very opaque.
        for n in np.arange(1,self.nevents):

            denom = (1-kn[2,n-1,:]**2)**0.5

            # if kz is not close to 1
            kx = sintheta[n-1,:] * (kn[0,n-1,:] * kn[2,n-1,:] * cosphi[n-1,:] -
                    kn[1,n-1,:] * sinphi[n-1,:]) / denom + kn[0,n-1,:] * costheta[n-1,:]

            ky = sintheta[n-1,:] * (kn[1,n-1,:] * kn[2,n-1,:] * cosphi[n-1,:] +
                    kn[0,n-1,:] * sinphi[n-1,:]) / denom + kn[1,n-1,:] * costheta[n-1,:]

            kz = -sintheta[n-1,:] * cosphi[n-1,:] * denom + kn[2,n-1,:] * costheta[n-1,:]

            # if kz is close to 1 and >= 0:
            kx2 = sintheta[n-1,:] * cosphi[n-1,:]
            ky2 = sintheta[n-1,:] * sinphi[n-1,:]
            kz2 = costheta[n-1,:]

            # if kz is close to 1 and < 0:
            ky3 = -sintheta[n-1,:] * sinphi[n-1,:]
            kz3 = -costheta[n-1,:]

            kn[:,n,:] = np.where(denom > eps, (kx, ky, kz),
                                 np.where(kn[2,n-1,:]>= 0., (kx2, ky2, kz2), (kx2, ky3, kz3)))

        # update all the directions of the trajectories
        self.direction = kn

        # TODO: why is a return value needed here? If the method modifies the
        # attributes of the object, no return value should be necessary (the
        # user would then just look at trajectoryobject.direction).
        # Alternatively, write the method so that it returns a different array
        # rather than modifying the array held in the existing object.

        return self.direction


    def move(self, lscat):
        """
        Calculates new positions in trajectories.

        lscat: scattering length from Mie theory (step size in trajectories).

        TODO: add docstring in numpy docstring format
        """

        displacement = self.position
        displacement[:, 1:, :] = lscat * self.direction

        self.position[0] = np.cumsum(displacement[0,:,:], axis=0)
        self.position[1] = np.cumsum(displacement[1,:,:], axis=0)
        self.position[2] = np.cumsum(displacement[2,:,:], axis=0)

        # TODO: why is a return value needed here? see above
        return self.position


    def plot_coord(self, ntraj, three_dim=False):
        """
        Plots the trajectories' cartesian coordinates as a function of
        the number of scattering events (or 'time').
        three_dim = True: plots the coordinates in 3D.

        TODO: add docstring in numpy docstring format
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
    Counts the fraction of reflected and transmitted trajectories
    after a cutoff.
    Identifies which trajectories are reflected or transmitted,
    and at which scattering event.
    Includes total internal reflection correction.
    kx, ky, kz: direction cosines calculated during the simulation.

    TODO: add docstring in numpy docstring format
    """

    R_row_indices = []
    R_col_indices = []
    T_row_indices = []
    T_col_indices = []

    # TODO: add comments to explain what you are doing
    for tr in np.arange(ntraj):
        z_tr = z[:,tr]

        if any(z_tr > cutoff):
            z_T = next(zi for zi in z_tr if zi > cutoff)
            T_row = z_tr.tolist().index(z_T)
        else:
            T_row = np.NaN

        if any(z_tr < z_low):
            z_R = next(zi for zi in z_tr if zi < z_low)
            R_row = z_tr.tolist().index(z_R)
        else:
            R_row = np.NaN


        if (type(T_row) == int and type(R_row) != int):
            T_row_indices.append(T_row)
            T_col_indices.append(tr)

        if (type(R_row) == int and type(T_row) != int):
            R_row_indices.append(R_row)
            R_col_indices.append(tr)

        # if a trajectory both reflects and transmits, choose whichever happens first
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

    # TODO: add comments to explain what you are doing
    theta_r = []
    phi_r = []
    count = 0

    ev = np.array(R_row_indices)-1
    tr = np.array(R_col_indices)
    cosA = kx[ev,tr]
    cosB = ky[ev,tr]
    cosC = kz[ev,tr]

    for i in range(len(cosA)):

        # Solve for correct theta and phi from the direction cosines,
        # accounting for parity of sin and cos functions

        theta = np.arccos(cosC[i])
        phi1 = np.arccos(cosA[i] / np.sin(theta))
        phi2 = -np.arccos(cosA[i] / np.sin(theta))

        phi3 = np.arcsin(cosB[i] / np.sin(theta))
        phi4 = np.pi - np.arcsin(cosB[i] / np.sin(theta))

        A = np.array([abs(phi1-phi3),abs(phi1-phi4),abs(phi2-phi3),abs(2*np.pi+phi2-phi4)])
        B = A.argmin(0)

        if B == 0:
            phi_r.append((phi1+phi3)/2)
        elif B == 1:
            phi_r.append((phi1+phi4)/2)
        elif B == 2:
            phi_r.append((phi2+phi3)/2)
        elif B == 3:
            phi_r.append((2*np.pi+phi2+phi4)/2)

        theta_r.append(theta)

        # Count how many of the thetas correspond to the range of total internal reflection
        if theta < theta_min_refracted:
            count = count + 1


    # Calculate corrected reflection fraction

    R_fraction_corrected = np.array(len(R_row_indices) - count) / ntraj


    return R_fraction_corrected, theta_r, phi_r


def initialize(nevents, ntraj, seed=None):
    """
    Sets the trajectory's initial conditions (position, direction, and weight).

    nevents: number of scattering events
    ntraj: number of trajectories

    TODO: add docstring in numpy docstring format
    """

    if seed is not None:
        np.random.seed([seed])

    # initial position
    r0 = np.zeros((3, nevents+1, ntraj))
    r0[0,0,:] = random((1,ntraj))
    r0[1,0,:] = random((1,ntraj))

    # initial direction
    eps = 1.e-9
    k0 = np.zeros((3, nevents, ntraj))
    k0[2,0,:] = 1. - eps

    # initial weight
    weight0 = np.zeros((nevents, ntraj))
    weight0[0,:] = 0.001                        # (figure out how to determine this)

    return r0, k0, weight0


def phase_function(radius, n_particle, n_sample, angles, wavelen):
    """
    Calculates the phase function from Mie theory.

    wavelen, radius, angles: must be entered as Quantity
    angles: scattering angles (typically from 0 to pi)
    p: phase_function

    p = diff. scatt. cross section / cscat    (Bohren and Huffmann 13.3)
    diff. scat. cross section = S11 / k^2
    p = S11 / (k^2 * cscat)

    TODO: add docstring in numpy docstring format
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

    radius, n_particle, n_sample, wavelen: must be entered as Quantity to allow
    specifying units

    wavelen: structcol.Quantity [length]. Wavelength in vacuum.
    wavelen/n_sample: wavelength of incident light *in media* (usually this would be the
    wavelength in the effective index of the particle-matrix composite)

    TODO: add docstring in numpy docstring format
    """

    number_density = 3.0 * volume_fraction / (4.0 * np.pi * radius**3)
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)

    cross_sections = mie.calc_cross_sections(m, x, wavelen/n_sample)
    cscat = cross_sections[0]
    cabs = cross_sections[2]

    lscat = 1 / (cscat * number_density)
    # TODO: why is it necessary to convert to micrometers? If the code depends
    # on the length scale being in micrometers, this should be explicitly
    # documented (or should use a function decorator to ensure that the
    # arguments have the correct dimensions)
    lscat = lscat.to('um')

    labs = 1 / (cabs * number_density)
    labs = labs.to('um')

    return lscat, labs


def sampling(nevents, ntraj, p, angles):
    """
    Samples azimuthal angles from uniform distribution,
    and scattering angles from phase function.

    TODO: add docstring in numpy docstring format
    What is p?  A function or array?
    """

    # random sampling of azimuthal angle phi from uniform distribution [0 - 2pi]
    rand = np.random.random((nevents,ntraj))
    phi = 2*np.pi*rand
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    # random sampling of scattering angle theta
    prob = p * np.sin(angles)*2*np.pi    # prob is integral of p in solid angle
    prob_norm = prob/sum(prob)           # normalize to make it add up to 1

    theta = np.array([np.random.choice(angles, ntraj, p = prob_norm) for i in range(nevents)])
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    return sintheta, costheta, sinphi, cosphi, theta, phi
