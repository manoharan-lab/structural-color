# -*- coding: utf-8 -*-
# Copyright 2016 Vinothan N. Manoharan, Victoria Hwang, Anna B. Stephenson,
# Solomon Barkley
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

import structcol as sc
import pymie
from pymie import mie
from . import refraction
from . import normalize
from . import select_events
import numpy as np
from numpy.random import random as random
import xarray as xr
import matplotlib.pyplot as plt
import itertools
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
        # TODO: remove phase and polarization as they have been replaced by
        # fields
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

    @property
    def ntrajectories(self):
        return self.weight.shape[1]

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

        # the 0th event is the initial direction which does not change when the
        # photon first enters the sample. It only changes after the first
        # scattering event. The trajectory steps first, then scatters. The
        # reason that we don't extend the arange to nevents + 1 is that we
        # still count this original initialized direction as an event, since
        # the step size must be sampled once it enters the material.

        # Calculate the new x, y, z coordinates of the propagation
        # direction using the following equations, which can be derived by
        # using matrix operations to perform a rotation about the y-axis by
        # angle theta followed by a rotation about the z-axis by angle phi
        # see pg 105 in A.B. Stephenson lab notebook 1 for derivation and
        # notes

        # this is the product of the rotation matrices R_z(phi).R_y(theta)
        # calculated for each event in each trajectory
        # shape of kn is [3,nevents,ntraj]
        # shape of R is [3,3,nevents,ntraj]
        R = np.array([[costheta*cosphi, -sinphi, sintheta*cosphi],
                      [costheta*sinphi, cosphi, sintheta*sinphi],
                      [-sintheta, np.zeros(sinphi.shape), costheta]])

        # could vectorize this loop if numpy had a cumulative dot product
        # ufunc.  But np.cumprod only does element by element.
        for n in np.arange(1, self.nevents):
            # now use Einstein summation to take the dot product of each
            # rotation matrix at each event in each trajectory with the
            # wavevector
            kn[:, n, :] = np.einsum('ijl,jl->il',
                                    R[:, :, n-1, :], kn[:, n-1, :])
            # Annie's equivalent code:
            # kn[0, n, :] = (((kn[0, n - 1, :] * costheta[n - 1, :]
            #                 + kn[2, n - 1, :] * sintheta[n - 1, :])
            #                 * cosphi[n - 1, :])
            #                - kn[1, n - 1, :] * sinphi[n - 1, :])
            # kn[1, n, :] = (((kn[0, n - 1, :] * costheta[n - 1, :]
            #                 + kn[2, n - 1, :] * sintheta[n - 1, :])
            #                 * sinphi[n - 1, :])
            #                + kn[1, n - 1, :] * cosphi[n - 1, :])
            # kn[2, n, :] = (-kn[0, n - 1, :] * sintheta[n - 1, :]
            #                + kn[2, n - 1, :] * costheta[n - 1, :])

        # Update all the directions of the trajectories
        self.direction = sc.Quantity(kn, self.direction.units)

    def calc_fields(self, theta, phi, sintheta, costheta, sinphi, cosphi,
                    n_particle, n_sample, radius, wavelen, step,
                    volume_fraction, fine_roughness=0, tir_refl_bool=None):
        """
        Calculates local x and y polarization rotated in reference frame where
        initial polarization is x-polarized. Assumes the incident light is in
        +z direction

        Within one trajectory, fields is accounted for by calculating
        the form factor using Mie theory, which gives the scattered fields
        and phase.

        To calculate the effects of interference between different
        trajectories, we include the phase shift calculated from Mie theory, as
        well as the phase shift due to the distances travelled. The structure
        factor contribution comes in through the phase shift due to the
        distances travelled.

        Here is an outline of how it is implemented:

        We start by calculating the amplitude scattering matrix in the
        parallel/perpendicular basis. We then multiply the matrix by the
        initial fields. This gives the scattered fields purely due to
        the form factor.

        Then we add these phase shifts to the phase shift incurred due to
        distance travelled, calculated as k*distance.

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
            Fraction of the sample area that has fine roughness. Should be
            between 0 and 1. For ex, a value of 0.3 means that 30% of incident
            light will hit fine surface roughness (e.g. will "see" a Mie
            scatterer first). The rest of the light will see a smooth surface,
            which could be flat or have coarse roughness (long in the
            lengthscale of light).
        tir_refl_bool: 2d array of booleans (shape: nevents, ntraj)
            Describes whether a trajectory gets totally internally reflected at
            any event and also exits in the negative direction to contribute to
            reflectance

        Calculates:
        ----------
        En: ndarray, shape: (3, nevents, ntrajectories)
            Electric field vector for each trajectory and event
            in global coordinates
        """

        # until refactoring, convert DataArrays to numpy
        if isinstance(n_particle, xr.DataArray):
            n_particle = n_particle.to_numpy().squeeze()
        if isinstance(n_sample, xr.DataArray):
            # drop VOLFRAC dimension, which will be included in all effective
            # index calculations.
            if sc.Coord.VOLFRAC in n_sample.coords:
                n_sample = n_sample.isel({sc.Coord.VOLFRAC: 0}, drop=True)
            n_sample = n_sample.to_numpy()

        m = np.atleast_2d(n_particle/n_sample)
        x = pymie.size_parameter(wavelen, n_sample, radius)
        k = 2 * np.pi * n_sample / wavelen.magnitude
        step = step.magnitude
        # TODO: fix the bug in the above code.  If the step size and wavelength
        # are specified in different units, the results will be off by a lot.
        # test_fields uses different units.  The commented code below is how
        # this should look:
        # m = sc.index.ratio(n_particle, n_sample)
        # x = sc.size_parameter(wavelen, n_sample, radius)
        # k = sc.wavevector(n_sample).magnitude
        # step = step.to_preferred().magnitude
        ntraj = theta.shape[1]

        # calculate the mie amplitude scattering matrix
        # we need to calculate the full matrix, rather than just the vector
        # scattering amplitude, because each matrix element contributes to
        # the changes in E field
        #
        # amplitude_scattering_matrix() is set up to broadcast over phi (for
        # each theta we calculate all the matrix for all phi values). Here we
        # want to calculate the matrix for every event (for each trajectory i
        # and event j we calculate the matrix for the combination of (theta_ij,
        # phi_ij) in that event).  We therefore calculate the matrix for each
        # theta (passing a flat array), then reshape and modify for each phi
        S = mie.amplitude_scattering_matrix(m, x, theta.ravel())

        # for clarity of indexing (0->1) we add a zero element to the list
        S = [0] + list(S)
        # Reshape to (..., nevents, ntraj). Also because calc_fields() is not
        # yet vectorized, we remove the wavelength axis from each element
        for i in (1,2,3,4):
            S[i] = S[i][0].reshape(theta.shape)
        # now account for phi
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        S1 = S[2]*(sinphi)**2 + S[1]*(cosphi)**2
        S2 = S[2]*(cosphi)**2 + S[1]*(sinphi)**2
        S3 = S[2]*sinphi*cosphi - S[1]*sinphi*cosphi
        S4 = S[2]*cosphi*sinphi - S[1]*cosphi*sinphi

        # mutliply the scat amp mats
        En = self.fields
        if isinstance(En, sc.Quantity):
            En = En.magnitude

        # En has shape (3, nevents+1, ntraj)
        Ex = En[0, 0, :]
        Ey = En[1, 0, :]

        # Ex and Ey are the initialized as the incident field vectors. To get
        # the Ex and Ey at each event, we have to multiply by the scattering
        # amplitude matrix, cumulatively for each event.
        # this gives us the local Ex and Ey vectors
        # Reminder: there is one less sampled angle than event number, because
        # the first event propogates straight into the sample.
        # Note: this basis assumes that
        # the direction of propagation is the +z direction.
        for n in np.arange(2, self.nevents + 1):
            Ex = S2[n-2, :] * Ex + S3[n-2, :] * Ey
            Ey = S4[n-2, :] * Ex + S1[n-2, :] * Ey
            # 0th event is before sample, the 1st event has no rotation
            En[0, n, :] = Ex
            En[1, n, :] = Ey

        # Deal with tir
        if tir_refl_bool is not None:
            # get indices for the first TIR event for each trajectory
            tir_indices = np.argmax(np.vstack([np.zeros(ntraj),
                                               tir_refl_bool]), axis=0)

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
        # values from before the field enters the sample. The 1st event
        # contains the values for the field after entering the sample, but
        # before scattering

        # this is the product of the rotation matrices R_z(phi).R_y(theta)
        # calculated for each event in each trajectory
        # shape of kn is [3,nevents,ntraj]
        # shape of R is [3,3,nevents,ntraj]
        R = np.array([[costheta*cosphi, -sinphi, sintheta*cosphi],
                      [costheta*sinphi, cosphi, sintheta*sinphi],
                      [-sintheta, np.zeros(sinphi.shape), costheta]])

        for n in np.arange(2, self.nevents + 1):
            # Calculate the new x, y, z coordinates of the propagation
            # direction using the following equations, which can be derived by
            # using matrix operations to perform a rotation about the y-axis by
            # angle theta followed by a rotation about the z-axis by angle phi
            #
            # Einstein summation to take the dot product of each rotation
            # matrix at each event in each trajectory with the wavevector
            # (the n: ensures that all subsequent fields are also rotated)
            En[:, n:, :] = np.einsum('ijl,jkl->ikl',
                                     R[:, :, n-2, :], En[:, n:, :])
            # Annie's equivalent code:
            # Ex = ((En[0,n:,:]*costheta[n-2,:] + En[2,n:,:]*sintheta[n-2,:])*
            #         cosphi[n-2,:]) - En[1,n:,:]*sinphi[n-2,:]
            # Ey = ((En[0,n:,:]*costheta[n-2,:] + En[2,n:,:]*sintheta[n-2,:])*
            #       sinphi[n-2,:]) + En[1,n:,:]*cosphi[n-2,:]
            # Ez =  -En[0,n:,:]*sintheta[n-2,:] + En[2,n:,:]*costheta[n-2,:]
            # En[:,n:,:] = Ex, Ey, Ez

        # Calculate the structure factor field contribution.
        # Insert a row of zeros since first event does not change direction
        # Note that this will only work for normal incidence.
        theta2 = np.insert(theta,0,np.zeros(ntraj),axis=0)

        # calculate the step propagation factor
        step_cumul = np.abs(k)*np.cumsum(step, axis=0)#step #
        step_phase_factor = np.exp(1j*np.abs(k)*step_cumul)

        # multiply the fields by the phase propagation due to structure factor
        # of the initial trajectories
        # should multiply by 1 for trajectories do not have fine roughness
        ntraj_fine = int(round(ntraj * fine_roughness))
        En[0, 1:, :] = En[0, 1:, :] * step_phase_factor
        En[1, 1:, :] = En[1, 1:, :] * step_phase_factor
        En[2, 1:, :] = En[2, 1:, :] * step_phase_factor

        # Normalize
        En[0, :, :], En[1, :, :], En[2, :, :] = normalize(En[0, :, :],
                                                          En[1, :, :],
                                                          En[2, :, :],
                                                          return_nan=False)

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

        # define coords/dims for position array (shape: [3, nevents+1, ntraj])
        pos_coords = [("vector", ["x", "y", "z"]),
                      ("event", np.arange(0, self.nevents+1)),
                      ("trajectory", np.arange(self.ntrajectories))]

        # strip and save units
        if isinstance(self.position, sc.Quantity):
            units = self.position.to('um').units
            displacement = xr.DataArray(self.position.magnitude,
                                        coords=pos_coords)
        else:
            units = 1
            displacement = xr.DataArray(self.position, coords=pos_coords)

        # calculate vector displacement after each event
        displacement[dict(event=slice(1, None))] = (step.to('um').magnitude
                                                    * self.direction.magnitude)

        # The array of positions is a cumulative sum of all of the
        # displacements. Note: we use da.cumsum() because
        # da.cumulative("event").sum() will be much slower if numbagg is not
        # installed. See https://github.com/pydata/xarray/issues/6528
        self.position = displacement.cumsum("event").to_numpy() * units

    def plot_coord(self, ntraj, three_dim=False): # pragma: no cover
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

        if three_dim:
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


def initialize(nevents, ntraj, n_medium, n_sample, boundary, rng=None,
               incidence_theta_min=sc.Quantity(0.,'rad'),
               incidence_theta_max=sc.Quantity(0.,'rad'),
               incidence_theta_data=None,
               incidence_phi_min=sc.Quantity(0.,'rad'),
               incidence_phi_max=sc.Quantity(2*np.pi,'rad'),
               incidence_phi_data=None,
               plot_initial=False,
               spot_size=sc.Quantity('1.0 um'),
               sample_diameter=None,
               coarse_roughness=0.,
               coherent=False,
               polarized=True,
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
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package
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
    plot_initial: boolean
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
        z=0 plane. Then the model corrects the Fresnel reflection and
        refraction to account for the different angles of incidence due to the
        roughness. The coarse_roughness parameter is the rms slope of the
        surface. If included, it should be larger than 0. There is no upper
        bound, but when the coarse roughness tends to infinity, the surface
        becomes too "spiky" and light can no longer hit it, which reduces the
        reflectance down to 0.
    fields: boolean
        If True, also returns the initial fields of trajectories
    coherent: boolean
        If True, assumes the intial relative phases between trajectories are
        zero. If coherent is set to True while fields is set to False, then the
        coherent value is ignored, since there can be no coherence without
        taking into account the fields.

    Returns
    -------
    r0: 3D array-like (structcol.Quantity [length])
        Trajectory positions. Has shape of (3, number of events + 1, number of
        trajectories). r0[0,0,:] contains random x-positions within a circle on
        the x-y plane whose radius is the sphere radius. r0[1, 0, :] contains
        random y-positions within the same circle on the x-y plane. r0[2, 0, :]
        contains z-positions on the top hemisphere at the sphere boundary. The
        rest of the elements are initialized to zero.
    k0: 3D array-like (structcol.Quantity [dimensionless])
        Initial direction of propagation. Has shape of (3, number of events,
        number of trajectories). k0[0,:,:] and k0[1,:,:] are initalized to
        zero, and k0[2,0,:] is initalized to 1.
    weight0: 3D array-like (structcol.Quantity [dimensionless])
        Initial weight. Has shape of (number of events, number of trajectories)
        - Note that the photon weight represents the fraction of that
        particular photon that is propagated through the sample. It does not
        represent the photon's weight relative to other photons. the weight0
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
        would contribute to less readable code in the calculation of
        absorptance, reflectance, and transmittance. Therefore the weights
        array begins with the weight of the photons after their first event.
    pol0: (optional) 3D array-like (structcol.Quantity[dimensionless])
        Initial polarization vector in global coordinates. Has shape of
        (number of events, number of trajectories). Only returns initial
        linearly, x-polarized light.
    kz0_rot : array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the
        surface. The array size is (1, ntraj). Only returned if
        coarse_roughness is set to > 0.
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
    # until refactoring, convert index DataArrays to numpy
    if isinstance(n_medium, xr.DataArray):
        n_medium = n_medium.to_numpy()
    if isinstance(n_sample, xr.DataArray):
        # drop VOLFRAC dimension, which will be included in all effective index
        # calculations.
        if sc.Coord.VOLFRAC in n_sample.coords:
            n_sample = n_sample.isel({sc.Coord.VOLFRAC: 0}, drop=True)
        n_sample = n_sample.to_numpy()

    if rng is None:
        rng = sc.rng

    # get the spot size magnitude to multiply by initial x and y positions
    spot_size_magnitude = spot_size.to('um').magnitude

    # get the sample radius as a float
    if isinstance(sample_diameter, sc.Quantity):
        sample_radius = sample_diameter.to('um').magnitude/2

    # strip units from dimensionless quantities
    if isinstance(n_medium, sc.Quantity):
        n_medium = n_medium
    if isinstance(n_sample, sc.Quantity):
        n_sample = n_sample

    # Initial position. The position array has one more row than the direction
    # and weight arrays because it includes the starting positions on the x-y
    # plane.  Shape should be [3, nevents+1, ntraj]
    r0 = xr.DataArray(dims=["component", "event", "trajectory"],
                      coords = {"component": ["x", "y", "z"],
                                "event": range(nevents+1),
                                "trajectory": range(ntraj)})

    # Create an empty array of the initial direction cosines of the right size
    # Shape should be [3, nevents, ntraj], so 1 event smaller than r0.
    k0 = xr.zeros_like(r0.isel(event=slice(0, -1)))

    # Initial weight
    weight0 = xr.ones_like(k0.sel(component='x')).drop_vars("component")

    if boundary == 'film':

        # raise error if user inputs a value for sphere diameter
        if sample_diameter is not None:
            raise ValueError('for film geometry, sample_diameter must be set\
                             to None')
        # randomly choose positions on interval [0,1]
        r0.sel(event=0).loc["x":"y"] = rng.random((2, ntraj))

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
            theta = rng.uniform(incidence_theta_min, incidence_theta_max,
                                ntraj)

        if incidence_phi_data is not None:
            if len(incidence_phi_data) != ntraj:
                raise ValueError("length of incidence_phi_data must be equal "
                                 "to number of trajectories")
            phi = incidence_phi_data
        else:
            incidence_phi_min = incidence_phi_min.to('rad').magnitude
            incidence_phi_max = incidence_phi_max.to('rad').magnitude
            phi = rng.uniform(incidence_phi_min, incidence_phi_max, ntraj)

        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

    if boundary == 'sphere':

        # raise error if user forgets to input a value for the sphere diameter
        if sample_diameter is None:
            raise ValueError("for sphere geometry, sample_diameter must be "
                             "a physical quantity, not None")

        # randomly choose r on interval [0,1] and multiply by spot size radius
        r = np.sqrt(rng.random(ntraj)) * spot_size_magnitude/2

        # randomly choose th on interval [0,2*pi]
        th = 2*np.pi*rng.random(ntraj)

        # convert to x and y, so that the points are randomly distributed
        # across the cross sectional area of the sphere
        # for details, see: https://mathworld.wolfram.com/DiskPointPicking.html
        x = r * np.cos(th)
        y = r * np.sin(th)
        # calculate z-positions from x- and y-positions
        z = sample_radius - np.sqrt(sample_radius**2 - x**2 - y**2)
        r0.loc[dict(event=0)] = [x, y, z]

        # find the minus normal vectors of the sphere at the initial positions
        r0_magnitude = np.sqrt(x**2 + y**2 + (z - sample_radius)**2)
        # neg_normal should have shape [3, ntraj]
        neg_normal = xr.zeros_like(r0.sel(event=0, drop=True))
        neg_normal.loc["x":"z"] = np.array([-x / r0_magnitude,
                                            -y / r0_magnitude,
                                            -(z - sample_radius)/r0_magnitude])
        # solve for theta and phi for these samples
        theta = np.arccos(neg_normal.loc['z'])
        cosphi, sinphi = neg_normal.loc["x":"y"] / np.sin(theta)

    # If there is no coarse roughness (e.g. surface is flat)
    if coarse_roughness == 0:
        # Refraction of incident light upon entering the sample
        # TODO: only real part of n_sample should be used
        # for the calculation of angles of integration? Or abs(n_sample)?
        theta = refraction(theta, np.abs(n_medium), np.abs(n_sample))

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    # calculate new directions using refracted theta and initial phi
    kx, ky, kz = (sintheta * cosphi), (sintheta * sinphi), costheta
    k0.loc[dict(event=0)] = [kx, ky, kz]

    if coarse_roughness == 0:
        # plot the initial positions and directions of the trajectories
        if plot_initial and (boundary == 'sphere'): # pragma: no cover
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
            X, Y, Z, U, V, W = [x, y, z, kx, ky, kz]
            ax.quiver(X, Y, Z, U, V, W, color = 'g')

            X, Y, Z, U, V, W = [x, y, z, np.zeros(ntraj), np.zeros(ntraj),
                                np.ones(ntraj)]
            ax.quiver(X, Y, Z, U, V, W)

            # draw wireframe hemisphere
            u, v = np.mgrid[0:2*np.pi:20j, np.pi/2:0:10j]
            x = sample_radius*np.cos(u)*np.sin(v)
            y = sample_radius*np.sin(u)*np.sin(v)
            z = sample_radius-sample_radius*np.cos(v)
            ax.plot_wireframe(x, y, z, color=[0.8,0.8,0.8])

        init_traj_props = [r0.to_numpy(), k0.to_numpy(), weight0.to_numpy()]

    # if the surface has coarse roughness
    else:
        k0, kz0_rot, kz0_refl = coarse_roughness_enter(k0, n_medium,
                                                       n_sample,
                                                       coarse_roughness,
                                                       boundary, rng=rng)
        init_traj_props = [r0.to_numpy(), k0.to_numpy(), weight0.to_numpy(),
                           kz0_rot.to_numpy(), kz0_refl.to_numpy()]

    if fields:
        # The field is initialized with nevents+1 because we want to save
        # the value of the field from before the photon enters the sample.
        # Shape should be [3, nevents+1, ntraj)]
        fields0 = xr.zeros_like(r0, dtype = 'complex')
        # initialize for unpolarized, incoherent light
        if coherent:
            phase = np.zeros((2,ntraj))
        else:
            phase = rng.random((2, ntraj))*2*np.pi
        if polarized:
            fields0.sel(event=0).loc['x'] = np.exp(phase[0]*1j)
        else:
            fields0.sel(event=0).loc['x':'y'] = np.exp(phase*1j)

        fields0.loc[dict(event=0)] = normalize(*fields0.sel(event=0))

        # first step into the sample is same
        fields0.loc[dict(event=1)] = fields0.sel(event=0)
        init_traj_props.append(fields0.to_numpy())

    return init_traj_props


def calc_scat(model, wavelen,
              fields = False,
              fine_roughness=0,
              min_angle = 0.01,
              num_angles = 200,
              num_phis = 300):
    """
    Calculates the phase function, scattering coefficient, and absorption
    coefficient

    Parameters
    ----------
    model : `sc.Model` object
        scattering model to use
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
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

    Returns
    -------
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    mu_scat : float or 2-element array (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering
        model. When fine_roughness is larger than 0, mu_scat is a 2-element
        array, where the first element is the scattering coefficient from
        either Mie theory or single scattering model, and the second element is
        the scattering coefficient from Mie theory.
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
    When there is fine roughness, we assume that light goes from the index
    of the matrix to the index of the scatterer. Thus we assume that fine
    roughness particles are not embedded in an effective medium.

    """
    wavelen = wavelen.to_preferred()
    units = wavelen.units

    n_sample = model.index_external(wavelen)

    # calculate the absorption coefficient
    mu_abs = 4*np.pi*n_sample.imag.to_numpy().squeeze()/wavelen

    # Define angles at which phase function will be calculated, based on
    # whether light is polarized or unpolarized
    # Scattering angles (typically from a small angle to pi). A non-zero small
    # angle is needed because in the single scattering model, if the analytic
    # formula is used, S(q=0) returns nan. To prevent any errors or warnings,
    # set the minimum value of angles to be a small value, such as 0.01.
    angles = sc.Quantity(np.linspace(min_angle, np.pi, num_angles), 'rad')

    thetas = angles
    if fields:
        phis = sc.Quantity(np.linspace(min_angle, 2*np.pi, num_phis), 'rad')
    else:
        phis = None

    # calculate scattering quantities using the Model object
    dscat = model.differential_cross_section(wavelen, thetas, phis=phis)
    cscat = model.scattering_cross_section(dscat)
    p = model.phase_function(dscat).to_numpy().squeeze()

    mu_scat = model.number_density * (cscat.loc["avg"].to_numpy().squeeze() *
                                      units**2)

    # Here, the resulting units of mu_scat and mu_abs are nm^2/um^3. Thus, we
    # simplify the units to 1/um
    mu_scat = mu_scat.to('1/um')
    mu_abs = mu_abs.to('1/um')

    # if there is fine surface roughness, also calculate and return the scatt
    # coeff from Mie theory. We assume that fine roughness particles are in the
    # matrix and not in the effective sample medium.
    if fine_roughness > 0.:
        # We use the same form factor and lengthscale from the existing model.
        # Just need to change the external index to that of the matrix and
        # change the structure factor to a constant. We copy the model to avoid
        # modifying the original
        roughness_model = copy.deepcopy(model)
        roughness_model.index_external = roughness_model.index_matrix
        roughness_model.structure_factor = sc.structure.Constant(1.0)

        dscat = roughness_model.differential_cross_section(wavelen, thetas)
        cscat_total_mie = roughness_model.scattering_cross_section(dscat)
        mu_scat_mie = (roughness_model.number_density
                       * (cscat_total_mie.loc["avg"].to_numpy().squeeze()
                          * units**2))

        mu_scat_mie = mu_scat_mie.to('1/um')
        mu_scat = sc.Quantity(np.array([mu_scat.magnitude,
                                        mu_scat_mie.magnitude]), '1/um')

    return p, mu_scat, mu_abs


def sample_angles(nevents, ntraj, p, min_angle=0.01, rng=None):
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
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package

    Returns
    -------
    sintheta, costheta, sinphi, cosphi, theta, phi : ndarray
        Sampled azimuthal and scattering angles, and their sines and cosines.

    """
    if rng is None:
        rng = sc.rng

    if isinstance(p,sc.Quantity):
        p = p.magnitude
    num_theta = len(p)

    # The direction for the first event is defined upon initialization
    # so we only need to sample nevents-1.
    # In previous versions of the code, we sampled nevents, which gave us an
    # extra sampled angle that was never used. While this did not lead to
    # incorrect results, it led to inconsistencies in indexing which had the
    # potential to create future bugs. Sampling one less angle fixes this
    # issue.
    nevents = nevents-1

    # Scattering angles for the phase function calculation (typically from 0 to
    # pi). A non-zero minimum angle is needed because in the single scattering
    # model, if the analytic formula is used, S(q=0) returns nan.
    thetas = sc.Quantity(np.linspace(min_angle, np.pi, num_theta), 'rad')
    thetas = thetas.magnitude

    if len(p.shape)==1: # if p depends only on theta

        # Randomly sample azimuthal angle phi from uniform distribution [0 -
        # 2pi]
        rand = rng.random((nevents,ntraj))
        phi = 2*np.pi*rand

        # make sure probability is normalized
        # prob is integral of p in solid angle
        prob = p * np.sin(thetas)*2*np.pi
         # normalize to make it add up to 1
        prob_norm = prob/sum(prob)

        # Randomly sample scattering angle theta
        theta = rng.choice(thetas, (nevents, ntraj), p = prob_norm)

    if len(p.shape)==2: # if p depends on theta and phi

        # get the number of phis from the shape of the phase function
        num_phi = p.shape[1]

        # sum for theta axis to get phi probabilities
        p_phi = np.sum(p, axis = 0)

        # define phi values from which to sample
        phis = sc.Quantity(np.linspace(min_angle,2*np.pi, num_phi), 'rad')
        phis = phis.magnitude

        # sample indices for phi values
        phi_ind = rng.choice(num_phi, (nevents, ntraj),
                             p = p_phi/np.sum(p_phi))

        # sample thetas based on sampled phi values
        theta_ind = np.zeros((nevents,ntraj))
        theta = np.zeros((nevents,ntraj))
        phi = np.zeros((nevents,ntraj))

        # calculate and normalize p(theta) for each phi, event, and trajectory
        p_theta = p[:, phi_ind] * np.sin(thetas[:, np.newaxis, np.newaxis])
        p_theta_norm = p_theta/np.sum(p_theta, axis=0)

        # It's hard to vectorize this loop because rng.choice works only with a
        # one-dimensional probability vector p.  There may be a way to
        # vectorize using rng.multinomial, which takes an array of probs.
        # However, rng.multinomial might not work with np.random.RandomState
        for i in range(nevents):
            for j in range(ntraj):
                theta_ind[i,j] = rng.choice(num_theta,
                                            p = p_theta_norm[:,i,j])

        # sampled angles
        theta = thetas[theta_ind.astype(int)]
        phi = phis[phi_ind.astype(int)]

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    return sintheta, costheta, sinphi, cosphi, theta, phi


def sample_step(nevents, ntraj, mu_scat, fine_roughness=0., rng=None):
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
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package

    Returns
    -------
    step : ndarray
        Sampled step sizes for all trajectories and scattering events.

    """
    if rng is None:
        rng = sc.rng

    if fine_roughness > 1. or fine_roughness < 0.:
        raise ValueError('fine roughness fraction must be between 0 and 1')

    # check whether mu_scat contains two values
    if len(np.array([mu_scat.magnitude]).flatten()) > 1:
        mu_scat, mu_scat_mie = mu_scat
    else:
        mu_scat_mie = None

    # Generate array of random numbers from 0 to 1
    rand = rng.random((nevents,ntraj)) #uncomment

    # sample step sizes
    step = -np.log(1.0-rand) / mu_scat

    # If there is fine surface roughness, sample the first step from Mie theory
    # for the number of trajectories set by fine_roughness
    if mu_scat_mie is not None:
        ntraj_mie = int(round(ntraj * fine_roughness))
        rand_ntraj = rng.random(ntraj_mie)
        step[0,0:ntraj_mie] = -np.log(1.0-rand_ntraj) / mu_scat_mie

    return step

def coarse_roughness_enter(k, n_medium, n_sample, coarse_roughness, boundary,
                           rng=None):
    '''
    Calculates new initial directions based on the coarse roughness of the
    sample.

    Parameters
    ----------
    k: 3D array-like (structcol.Quantity [dimensionless])
        Directions of propagation. Has shape of (3, number of events,
        number of trajectories). k0[0,:,:] and k0[1,:,:] are initialized to
        zero, and k0[2,0,:] is initialized to 1.
    n_medium: float (structcol.Quantity [dimensionless])
        Refractive index of the medium.
    n_sample: float (structcol.Quantity [dimensionless])
        Refractive index of the sample.
    coarse_roughness : float (can be structcol.Quantity [dimensionless])
        Coarse surface roughness should be included when the roughness is large
        on the scale of the wavelength of light. This means that light
        encounters a locally smooth surface that has a slope relative to the
        z=0 plane. Then the model corrects the Fresnel reflection and
        refraction to account for the different angles of incidence due to the
        roughness. The coarse_roughness parameter is the rms slope of the
        surface. If included, it should be larger than 0. There is no upper
        bound, but when the coarse roughness tends to infinity, the surface
        becomes too "spiky" and light can no longer hit it, which reduces the
        reflectance down to 0.
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options are
        'film' or 'sphere.' Coarse roughness is currently only implemented for
        a film.
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package

    Returns
    -------
    k0_rough: 3D array-like (structcol.Quantity [dimensionless])
        Initial direction of propagation, corrected for coarse roughness.
    kz0_rot : array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the
        surface. The array size is (1, ntraj). Only returned if
        coarse_roughness is set to > 0.
    kz0_refl : array_like (structcol.Quantity [dimensionless])
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global
        coordinate system. The array size is (1, ntraj). Only returned if
        coarse_roughness is set to > 0.

    '''
    if rng is None:
        rng = sc.rng

    if boundary == 'sphere':
        raise ValueError("course roughness not yet implemented for sphere "
                         "boundary")

    # strip units from dimensionless quantities
    if isinstance(n_medium, sc.Quantity):
        n_medium = n_medium
    if isinstance(n_sample, sc.Quantity):
        n_sample = n_sample

    ntraj = k.shape[2]

    # for constructing rotation matrices
    zeros = np.zeros(ntraj)
    ones = np.ones(ntraj)
    rotcoords = [("component2", ["x", "y", "z"]),
                 ("component", ["x", "y", "z"]),
                 ("trajectory", range(ntraj))]

    # get the first event only
    k0 = k.sel(event=0)

    # sample the surface roughness angles theta_a
    theta_a_full = np.linspace(0., np.pi / 2, 500)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_a = (P_theta_a(theta_a_full, coarse_roughness) /
                  sum(P_theta_a(theta_a_full, coarse_roughness)))
        if isinstance(prob_a, sc.Quantity):
            prob_a = prob_a.magnitude
    if np.isnan(prob_a).all():
        theta_a = np.zeros(ntraj)
    else:
        theta_a = np.array([rng.choice(theta_a_full, ntraj, p=prob_a)
                            for i in range(1)]).flatten()

    # In case the surface is rough, then find new coordinates of initial
    # directions after rotating the surface by an angle theta_a around y axis
    sintheta_a = np.sin(theta_a)
    costheta_a = np.cos(theta_a)

    # rotation matrix R_y(theta)
    R = xr.DataArray([[costheta_a, zeros, -sintheta_a],
                      [zeros, ones, zeros],
                      [sintheta_a, zeros, costheta_a]],
                     coords=rotcoords)
    k0_rot = xr.dot(R, k0, dim=("component"))

    # Find the new angles theta and phi between the incident trajectories and
    # the normal to the new surface after the coordinate axis rotation
    norm = (k0**2).sum(dim="component")
    theta_rot = np.arccos(k0_rot.loc['z'] / norm)
    phi_rot = np.arccos(k0_rot.loc['x'] / norm)

    # Refraction of incident light upon entering sample
    # TODO: only real part of n_sample should be used
    # for the calculation of angles of integration? Or abs(n_sample)?
    theta_refr = refraction(theta_rot, n_medium, np.abs(n_sample))

    k0_rot_refr = xr.DataArray([np.sin(theta_refr) * np.cos(phi_rot),
                                np.sin(theta_refr) * np.sin(phi_rot),
                                np.cos(theta_refr)],
                               coords = rotcoords[1:])

    # Rotate the axes back so that the initial refracted directions are in old
    # (global) coordinates by doing an axis rotation around y by 2pi-theta_a
    R = xr.DataArray(
            [[np.cos(2*np.pi-theta_a), zeros, -np.sin(2*np.pi-theta_a)],
             [zeros, ones, zeros],
             [np.sin(2*np.pi-theta_a), zeros, np.cos(2*np.pi-theta_a)]],
        coords = rotcoords)
    k0_refr = xr.dot(R, k0_rot_refr, dim=("component"))

    # Create an empty array of the initial direction cosines of the right size
    # (shape [3, nevents, ntraj])
    k0_rough = xr.zeros_like(k)

    # Fill up the first row (corresponding to the first scattering event) of
    # the direction cosines array with the randomly generated angles:
    k0_rough.sel(event=0).loc["x":"z"] = k0_refr.rename({"component2":
                                                         "component"})

    # Calculate Fresnel reflected directions, which are the same as the initial
    # directions in the local coordinate system but flipping the z sign
    k0_rot_refl = k0_rot.copy()
    k0_rot_refl.loc['z'] = -k0_rot_refl.loc['z']

    # Rotate the axes back so that the reflected directions are in old (global)
    # coordinates by doing an axis rotation around y by 2pi-theta_a
    # We need only the z-component, so perhaps can simplify the expression
    # below
    k0_refl = xr.dot(R, k0_rot_refl, dim=("component"))

    return k0_rough, k0_rot.loc['z'], k0_refl.loc['z']


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
