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

class Simulation:
    """Input parameters and methods for running a Monte Carlo simulation.

    Attributes
    ----------
    nevents : int
        number of scattering events
    ntrajectories : int
        number of trajectories
    p : array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    mu_scat : float or 2-element array (structcol.Quantity [1/length])
        Scattering coefficient from single scattering model.
    mu_scat_mie : float
        When fine_roughness is larger than 0, we also calculate a scattering
        coefficient from Mie theory and store as mu_scat_mie.
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient of the sample as an effective medium.
    initial_state : `xr.Dataset`
        Initial state of the photon packets in the simulation. Contains
        "position" (array of initial position vectors in cartesian
        coordinates), "direction" (array of initial propagation directions),
        "weight" (array of initial photon packet weights). May also contain
        "fields" (initial electric fields of photon packets).  The initial
        positions are randomized within an area on the x-y plane commensurate
        with the diameter of a sphere (if using a sphere geometry). The initial
        z-positions correspond to the top of the film or sphere boundary.  The
        initial directions are set according to the illumination parameters.
        Initial packet weights are set to 1.
    traj : `xr.Dataset`
        Monte Carlo trajectories.  As with initial_state, traj Contains
        "position", "direction", "weight", and (optionally) "fields", but for
        all events in every trajectory.
    coarse_roughness : float
    fine_roughness : float
        Fine and coarse roughness (see constructor).  If coarse_roughness is
        nonzero, the `Simulation.initial_state` and `Simulation.traj` Datasets
        also contains "kz0_rot" and "kz0_refl":
            - kz0_rot: Initial z-directions that are rotated to account for the
              fact that coarse surface roughness changes the angle of incidence
              of light. Thus these are the incident z-directions relative to
              the local normal to the surface.
            - kz0_refl: z-directions of the Fresnel reflected light after it
              hits the sample surface for the first time. These directions are
              in the global coordinate system.

    Methods
    -------
    absorb(mu_abs, step_size)
        calculate absorption at each scattering event with given absorption
        coefficient and step size.
    scatter(sintheta, costheta, sinphi, cosphi)
        calculate directions of propagation after each scattering event with
        given randomly sampled scattering and azimuthal angles.
    move(mu_scat, step_size)
        calculate new positions of the trajectory with given scattering
        coefficient, obtained from either Mie theory or the single scattering
        model.
    run()
    calc_fields()
    plot_coord(ntraj, three_dim=False)
        plot positions of trajectories as a function of number scattering
        events.

    Notes
    -----
    Each event consists of a step of a photon packet along a direction. For
    each event, we keep track of the (1) starting position of the photon
    packet; (2) weight of the packet at the starting position; (3) its
    direction; (4) its step size. In event 0, the weight is equal to 1 and the
    direction is fixed, rather than sampled. In subsequent events, the
    direction and step size are sampled, and the values are used to calculation
    the starting position and weight for the next event. Although there are
    `nevents` steps and directions, there are `nevents+1` positions and weights
    (and fields) because we calculate the position and weight (and field) of
    each packet after the last step.

    The dimension names and coords of the DataArrays in Simulation.traj are as
    follows:
    - position:  component x,y,z, event 0:nevents+1, trajectory 0:ntraj
    - weight:                     event 0:nevents+1, trajectory 0:ntraj
    - direction: component x,y,z, event 0:nevents,   trajectory 0:ntraj
    - step:                       event 0:nevents,   trajectory 0:ntraj
    - field:     component x,y,z, event 0:nevents+1, trajectory 0:ntraj

    Note that the step (or step_size) array is not stored by the class, but
    instead is specified as an argument to the move() and absorb() methods.
    Also note that in the Simulation.traj Dataset, all of these DataArrays are
    aligned, so that nans are inserted where there is no data (e.g. for
    direction.sel(event=nevents+1))

    Note also that the packet weight represents the fraction of that particular
    packet that is propagated through the sample. It does not represent the
    packet's weight relative to other photons. The weight array is initialized
    to 1 because we start with the full weight of the initial photons. To
    make the relative weights of photons different, introduce a new variable
    (e.g., relative intensity), rather than changing the
    intialization of the weights array.

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

    Simulations can be vectorized over wavelength. In this case, the initial
    state is broadcast, so that the same initial state is used for each
    wavelength (and any other leading dimensions).

    """
    def __init__(self, model, wavelen, nevents, ntraj, boundary,
                 rng=None,
                 initial_state=None,
                 incidence_theta_min=sc.Quantity(0.,'rad'),
                 incidence_theta_max=sc.Quantity(0.,'rad'),
                 incidence_theta_data=None,
                 incidence_phi_min=sc.Quantity(0.,'rad'),
                 incidence_phi_max=sc.Quantity(2*np.pi,'rad'),
                 incidence_phi_data=None,
                 min_angle = 0.01,
                 num_thetas = 200,
                 num_phis = 300,
                 plot_initial=False,
                 spot_size=sc.Quantity('1.0 um'),
                 sample_diameter=None,
                 fine_roughness=0.0,
                 coarse_roughness=0.,
                 coherent=False,
                 polarized=True,
                 fields=False):
        """Constructor for Simulation object.

        Sets the trajectories' initial conditions (position, direction,
        weight, and polarization if set to True).
        The initial positions are determined randomly in the x-y plane.

        If boundary is a sphere, the initial z-positions are confined to the
        surface of a sphere. If boundary is a film, the initial z-positions are
        set to zero.

        If incidence_theta_min and incidence_theta_max are both set to 0, the
        initial propagation direction is set to be 1 at z, meaning that the
        photon packets point straight down in z. The initial directions are
        corrected for refraction, for either type of boundary and for any
        incidence angle.

        Parameters
        ----------
        model : `sc.Model` object
            scattering model to use
        wavelen : float (structcol.Quantity [length])
            Wavelength of light in vacuum
        nevents : int
            Number of scattering events in each trajectory
        ntraj : int
            Number of trajectories
        boundary : string
            Geometrical boundary for Monte Carlo calculations. Current options
            are 'film' or 'sphere'
        rng : numpy.random.Generator object (default None)
            If not specified, use the default generator initialized on loading
            the package
        initial_state : `xr.Dataset`
            Initial direction, position, and weights for the simulation.  If
            not provided, these are set according to the incidence parameters.
        incidence_theta_min : float (structcol.Quantity [angle])
            Minimum value for theta when it incides onto the sample.
            Should be >= 0 and < pi/2.
        incidence_theta_max : float (structcol.Quantity [angle])
            Maximum value for theta when it incides onto the sample.
            Should be >= 0 and < pi/2.
        incidence_theta_data : array (structcol.Quantity [angle]) (optional)
            Array of values for the incident theta for each trajectory. Length
            of the array must therefore be the same as number of trajectories.
            If None, the code will randomly sample theta angles from a uniform
            distribution between incidence_theta_min and incidence_theta_max.
            If user does not specify units, values must be in radians.
        incidence_phi_min : float (structcol.Quantity [angle])
            Minimum value for phi when it incides onto the sample.
            Should be >= 0 and <= pi.
        incidence_phi_max : float (structcol.Quantity [angle])
            Maximum value for phi when it incides onto the sample.
            Should be >= 0 and <= pi.
        incidence_phi_data : array (structcol.Quantity [angle]) (optional)
            Array of values for the incident phi for each trajectory. Length of
            the array must therefore be the same as number of trajectories. If
            None, the code will randomly sample phi angles from a uniform
            distribution between incidence_phi_min and incidence_phi_max.  If
            user does not specify units, values must be in radians.
        min_angle : float
            min_angle to prevent numerical artifacts associated with
            calculating structure factor at theta=0
        num_thetas : int
            Sets the number of thetas at which phase function p will be
            calculated.
        num_phis : int
            Sets the number of phis at which phase function p will be
            calculated. Only used if polarization is True.
        plot_initial : boolean
            If plot_initial is set to True, function will create a 3d plot
            showing initial positions and directions of trajectories before
            entering the sphere and directly after refraction correction upon
            entering the sphere.
        spot_size : float (structcol.Quantity [length])
            For film sample, side length of a square spot size. For sphere
            sample diameter of a circular spot size.
        sample_diameter : None or float (structcol.Quantity [length])
            Diameter of the sample. Default is None. Should be None if sample
            geometry is a film. Should be float equal to the sphere diameter if
            sample is a sphere.
        fine_roughness : float
            When the sample has surface roughness that is comparable to the
            wavelength of light, then the first step is calculated with Mie
            theory because light "sees" the Mie scatterer first instead of the
            sample as a whole. After taking the first step, light is inside the
            sample and is scattered in in the usual way, with the phase
            function based on the effective medium approximation. This
            parameter should be between 0 and 1 and corresponds to the fraction
            of the sample area that has fine roughness. For ex, a value of 0.3
            means that 30% of incident light will hit fine surface roughness
            (e.g. will "see" a Mie scatterer first). The rest of the light will
            see a smooth surface, which could be flat or have coarse roughness
            (long in the lengthscale of light).
        coarse_roughness : float
            Coarse surface roughness should be included when the roughness is
            large on the scale of the wavelength of light. This means that
            light encounters a locally smooth surface that has a slope relative
            to the z=0 plane. Then the model corrects the Fresnel reflection
            and refraction to account for the different angles of incidence due
            to the roughness. The coarse_roughness parameter is the rms slope
            of the surface. If included, it should be larger than 0. There is
            no upper bound, but when the coarse roughness tends to infinity,
            the surface becomes too "spiky" and light can no longer hit it,
            which reduces the reflectance down to 0.
        fields : boolean
            If True, also returns the initial fields of trajectories
        coherent : boolean
            If True, assumes the intial relative phases between trajectories
            are zero. If coherent is set to True while fields is set to False,
            then the coherent value is ignored, since there can be no coherence
            without taking into account the fields.

        Returns
        -------
        None

        Notes
        -----
        For sphere boundary, incidence angle currently must be 0

        References
        ----------
        B. van Ginneken, M. Stavridi, J. J. Koenderink, “Diffuse and specular
        reflectance from rough surfaces”, Applied Optics, 37, 1 (1998) (has
        definition of rsm slope of the surface).

        """

        wavelen = wavelen.to_preferred()
        self.wavelen = wavelen
        wavelen_da = xr.DataArray(np.atleast_1d(wavelen.magnitude),
                                  dims=sc.Coord.WAVELEN)

        self.model = model

        n_sample = model.index_external(wavelen)
        n_medium = model.index_medium(wavelen)

        self.coarse_roughness = coarse_roughness
        if fine_roughness > 1 or fine_roughness < 0:
            raise ValueError("fine roughness fraction must be between 0 and 1")
        self.fine_roughness = fine_roughness
        self.min_angle = min_angle
        self.num_thetas = num_thetas
        self.num_phis = num_phis

        # Define angles at which phase function will be calculated, based on
        # whether light is polarized or unpolarized Scattering angles
        # (typically from a small angle to pi). A non-zero small angle may be
        # needed because S(q=0) can be subject to numerical artifacts. To
        # prevent any errors or warnings, set the minimum value of angles to be
        # a small value, such as 0.01.
        angles = np.linspace(min_angle, np.pi, num_thetas)

        thetas = angles
        if fields:
            phis = np.linspace(min_angle, 2*np.pi, num_phis)
        else:
            phis = None

        # calculate scattering quantities using the Model object
        dscat = model.differential_cross_section(wavelen, thetas, phis=phis)
        cscat = model.scattering_cross_section(dscat)
        self.p = model.phase_function(dscat)

        # calculate scattering and absorption coefficients in standard units
        self.mu_scat = (model.number_density.to_preferred().magnitude
                        * cscat.loc["avg"])
        self.mu_abs = 4*np.pi * n_sample.imag / wavelen_da

        # store leading dimensions (any dimensions to broadcast over, such as
        # wavelength or volume fraction).  These are determined by phase func
        p_leading = self.p.isel({sc.Coord.THETAIDX:0}, drop=True)
        if phis is not None:
            p_leading = p_leading.isel({sc.Coord.PHIIDX:0}, drop=True)
        self.leading_coords = p_leading.coords
        self.leading_dims = p_leading.dims
        self.leading_shape = p_leading.shape

        # if there is fine surface roughness, also calculate and return the
        # scatt coeff from Mie theory. We assume that fine roughness particles
        # are in the matrix and not in the effective sample medium.
        if fine_roughness > 0.:
            # We use the same form factor and lengthscale from the existing
            # model. Just need to change the external index to that of the
            # matrix and change the structure factor to a constant. We copy the
            # model to avoid modifying the original
            roughness_model = copy.deepcopy(model)
            roughness_model.index_external = roughness_model.index_matrix
            roughness_model.structure_factor = sc.structure.Constant(1.0)

            dscat = roughness_model.differential_cross_section(wavelen, thetas)
            cscat_total_mie = roughness_model.scattering_cross_section(dscat)
            mu_scat_mie = (roughness_model.number_density.to_preferred()
                           .magnitude * cscat_total_mie.loc["avg"])

            self.mu_scat_mie = mu_scat_mie

        self.nevents = nevents
        self.ntrajectories = ntraj
        self.boundary = boundary

        if rng is None:
            self.rng = sc.rng
        else:
            self.rng = rng

        # get the spot size magnitude to multiply by initial x and y positions
        spot_size_magnitude = spot_size.to_preferred().magnitude

        if boundary == 'film':
            # raise error if user inputs a value for sphere diameter
            if sample_diameter is not None:
                raise ValueError("for film geometry, sample_diameter must be "
                                 "set to None")
            # randomly choose positions on interval [0,1] for x and y
            r0 = self.rng.random((2, ntraj))
            # set z coordinate to 0 for initial position
            r0 = np.concatenate([r0, np.zeros((1, ntraj))])

            # initialize the incident angles theta and phi. The user can input
            # data or sample randomly from a uniform distribution between a min
            # and a max incident angles.
            if incidence_theta_data is not None:
                if len(incidence_theta_data) != ntraj:
                    raise ValueError("length of incidence_theta_data must be "
                                     "equal to number of trajectories")
                theta = incidence_theta_data
            else:
                incidence_theta_min = incidence_theta_min.to('rad').magnitude
                incidence_theta_max = incidence_theta_max.to('rad').magnitude
                theta = self.rng.uniform(incidence_theta_min,
                                         incidence_theta_max, ntraj)

            if incidence_phi_data is not None:
                if len(incidence_phi_data) != ntraj:
                    raise ValueError("length of incidence_phi_data must be "
                                     "equal to number of trajectories")
                phi = incidence_phi_data
            else:
                incidence_phi_min = incidence_phi_min.to('rad').magnitude
                incidence_phi_max = incidence_phi_max.to('rad').magnitude
                phi = self.rng.uniform(incidence_phi_min, incidence_phi_max,
                                       ntraj)

            sinphi = np.sin(phi)
            cosphi = np.cos(phi)

        elif boundary == 'sphere':
            # raise error if user forgets to input a value for sphere diameter
            if sample_diameter is None:
                raise ValueError("for sphere geometry, sample_diameter must "
                                 "be a physical quantity, not None")

            if isinstance(sample_diameter, sc.Quantity):
                sample_radius = sample_diameter.to_preferred().magnitude/2
            else:
                sample_radius = sample_diameter/2

            # randomly choose r on interval [0,1] and scale by spot size radius
            r = np.sqrt(self.rng.random(ntraj)) * spot_size_magnitude/2

            # randomly choose th on interval [0,2*pi]
            th = 2*np.pi*self.rng.random(ntraj)

            # convert to x and y, so that the points are randomly distributed
            # across the cross sectional area of the sphere
            # for details, see:
            # https://mathworld.wolfram.com/DiskPointPicking.html
            x = r * np.cos(th)
            y = r * np.sin(th)
            # calculate z-positions from x- and y-positions
            z = sample_radius - np.sqrt(sample_radius**2 - x**2 - y**2)
            r0 = np.array([x, y, z])

            # find the minus normal vectors of the sphere at initial positions
            r0_magnitude = np.sqrt(x**2 + y**2 + (z - sample_radius)**2)
            # neg_normal should have shape [3, ntraj]
            neg_normal = np.array([-x / r0_magnitude,
                                   -y / r0_magnitude,
                                   -(z - sample_radius)/r0_magnitude])
            # solve for theta and phi for these samples
            theta = np.arccos(neg_normal[2])
            cosphi, sinphi = neg_normal[0:-1] / np.sin(theta)

        else:
            raise ValueError("boundary must be of type 'film' or 'sphere'")

        # If there is no coarse roughness (e.g. surface is flat)
        if coarse_roughness == 0:
            # Refraction of incident light upon entering the sample
            theta = refraction(theta, n_medium, n_sample)
            theta = theta.rename({"dim_0": "trajectory"})
            theta.coords["trajectory"] = range(ntraj)
        else:
            theta = xr.DataArray(theta, coords={"trajectory": range(ntraj)})
            theta = theta.expand_dims(n_sample.coords)

        # Set up position DataArray. Shape is (..., 3, nevents+1,
        # ntrajectories). The last entry is the position after the final event
        position = xr.DataArray(0.0, dims=["component", "event", "trajectory"],
                                coords = {"component": ["x", "y", "z"],
                                          "event": range(nevents+1),
                                          "trajectory": range(ntraj)})
        # add dimensions and coords from refractive index (includes wavelength)
        position = position.expand_dims(n_sample.coords).copy()
        # set initial position. Note that the initial position is broadcast
        # over all the leading dimensions, so that the same initial state is
        # used for (for example) each wavelength
        position.loc[dict(event=0)] = r0

        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        # calculate new directions using refracted theta and initial phi
        kx, ky, kz = (sintheta * cosphi), (sintheta * sinphi), costheta
        k0 = xr.concat([kx, ky, kz], dim = "component")
        k0.coords["component"] = ["x", "y", "z"]

        # set up direction DataArray. Shape is (..., 3, nevents,
        # ntrajectories). Should have one fewer entry than position DataArray
        # because we don't track direction in the last event.
        direction = xr.zeros_like(position.isel(dict(event=slice(0, -1))))
        # initial state should be broadcast over leading dimensions
        direction.loc[dict(event=0)] = k0

        # as noted in docstring, weights are set to 1 for the first event.  The
        # remaining 1s in the array will be overwritten during the simulation
        weight = xr.ones_like(position.sel(component='x', drop=True))

        # set up trajectory Dataset.  Wait to include directions until after
        # we've accounted for coarse roughness
        self.traj = xr.Dataset({"position": position,
                                "weight": weight})

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

        # if the surface has coarse roughness
        else:
            args = [direction, n_medium, n_sample, coarse_roughness, boundary]
            direction, kz0_rot, kz0_refl = coarse_roughness_enter(*args,
                                                                  rng=self.rng)
            self.traj["kz0_rot"] = kz0_rot
            self.traj["kz0_refl"] = kz0_refl

        self.traj["direction"] = direction

        if fields:
            # The field is initialized with nevents+1 because we want to save
            # the value of the field from before the photon enters the sample.
            # Shape should be (..., 3, nevents+1, ntraj)
            fields = xr.zeros_like(position).astype(complex)

            # initialize for unpolarized, incoherent light
            if coherent:
                phase = np.zeros((2,ntraj))
            else:
                phase = self.rng.random((2, ntraj))*2*np.pi
            if polarized:
                fields.loc[dict(event=0, component="x")] = np.exp(phase[0]*1j)
            else:
                fields.loc[dict(event=0, component=slice("x","y"))] = \
                    np.exp(phase*1j)

            fields.loc[dict(event=0)] = normalize(fields.sel(event=0))

            # first step into the sample has the same field vector as before
            fields.loc[dict(event=1)] = fields.sel(event=0)

            # note that same initial values are broadcast to all wavelengths
            # (and other leading dimensions)
            self.traj["fields"] = fields

        # save initial state
        self.initial_state = self.traj.sel(event=0)

    def reset(self):
        """Resets trajectories to initial state.
        """
        # fill existing traj array with nans
        self.traj = xr.full_like(self.traj, np.nan)
        # now copy initial positions
        self.traj.loc[dict(event=0)] = self.initial_state
        if "fields" in self.traj:
            self.traj.fields.loc[dict(event=1)] = self.traj.fields.sel(event=0)

    def run(self, rng=None):
        """
        Run the simulation.

        Parameters
        ----------
        rng: numpy.random.Generator object (default None)
            Random number generator. If not specified, use the generator stored
            in the Simulation object

        Returns
        -------
        None

        """
        # Sample trajectory angles
        angles = self.sample_angles(rng=rng)
        # Sample step sizes
        step = self.sample_step(rng=rng)

        # Update trajectories based on sampled values
        self.scatter(angles)
        self.move(step)
        self.absorb(step)

        # calculate fields if present
        if "fields" in self.traj:
            self.calc_fields(angles, step)

    def sample_angles(self, rng=None):
        """Samples scattering angles (theta) and azimuthal angles (phi)

        if phase function p is 1d, phi is sampled from uniform distribution,
        and theta from phase function distribution.

        if phase function p is 2d, both theta and phi are sampled from p. Note
        that theta must come first in the shape of the phase function

        Parameters
        ----------
        rng : numpy.random.Generator object (default None)
            Random number generator. If not specified, use the generator stored
            in the Simulation object

        Returns
        -------
        sampled_angles : `xr.Dataset` (dims=..., event, trajectory)
            Sampled azimuthal and scattering angles, and their sines and
            cosines ("sintheta", "costheta", "sinphi", "cosphi", "theta",
            "phi")

        """
        if rng is None:
            rng = self.rng

        # The direction for the first event is defined upon initialization
        # so we only need to sample nevents-1.
        nsamples = self.nevents-1
        ntraj = self.ntrajectories

        p = self.p
        thetas = p.coords["theta"]

        if sc.Coord.PHI not in self.p.coords:
            # if p depends only on theta,
            # Randomly sample azimuthal angle phi from uniform distribution
            # [0 - 2pi]
            sampling_shape = self.leading_shape + (nsamples, ntraj)
            rand = rng.random(sampling_shape)
            phi = 2*np.pi*rand

            # make sure probability is normalized
            # prob is integral of p in solid angle
            prob = p * np.sin(thetas)*2*np.pi
            # normalize to make it add up to 1
            prob_norm = prob/prob.sum(sc.Coord.THETAIDX)

            # expand and transpose to ensure proper broadcasting in sc.choice()
            prob_norm = (prob_norm.expand_dims(("event", "trajectory"))
                         .transpose(*self.leading_dims, "event", "trajectory",
                                    ...))

            # Randomly sample scattering angle theta
            theta = sc.choice(thetas, sampling_shape, prob_norm, rng=rng)

        else:
            # if p depends on theta and phi
            # We must sample from the joint distribution
            # p(theta, phi) = p(phi) * p(theta | phi).  First calculate p(phi)
            # by marginalizing over theta:
            p_phi = p.sum(sc.Coord.THETAIDX)

            # expand and transpose to ensure proper broadcasting in sc.choice()
            p_phi_norm = ((p_phi/p_phi.sum(sc.Coord.PHIIDX))
                          .expand_dims(("event", "trajectory"))
                          .transpose(*self.leading_dims, "event", "trajectory",
                                     ...))

            # sample indices for phi values. We need indices to calculate
            # p(theta | phi) later.
            sampling_shape = self.leading_shape + (nsamples, ntraj)
            phi_ind = sc.choice(p_phi.coords[sc.Coord.PHIIDX],
                                sampling_shape,
                                p_phi_norm,
                                rng=rng)
            phi_ind = xr.DataArray(phi_ind,
                                   dims = (self.leading_dims
                                           + ("event", "trajectory")),
                                   coords = {"event": range(1, self.nevents),
                                             "trajectory": range(ntraj)})
            phi_ind = phi_ind.assign_coords(self.leading_coords)

            # and convert to sampled phis
            phi = p_phi.coords[sc.Coord.PHI][phi_ind]

            # Now calculate and normalize p(theta | phi) for each phi, event,
            # and trajectory. Note: this is memory-intensive
            # (nwavelengths * nevents * ntrajectories * ntheta)
            p_theta = (p[..., phi_ind] * np.sin(thetas))
            p_theta_norm = p_theta/p_theta.sum(sc.Coord.THETAIDX)
            p_theta_norm = p_theta_norm.transpose(..., sc.Coord.THETAIDX)

            # sample theta from conditional distribution
            theta = sc.choice(thetas, sampling_shape, p_theta_norm, rng)

        # Assign coords, setting event number correctly (note again that we did
        # not sample angles for event 0)
        sintheta = xr.DataArray(np.sin(theta),
                                dims = (self.leading_dims
                                        + ("event", "trajectory")),
                                coords = {"event": range(1, self.nevents),
                                          "trajectory": range(ntraj)})
        sampled_angles = xr.Dataset(
            {"sintheta": sintheta.assign_coords(self.leading_coords),
             "costheta": xr.DataArray(np.cos(theta), coords=sintheta.coords),
             "sinphi": xr.DataArray(np.sin(phi), coords=sintheta.coords),
             "cosphi": xr.DataArray(np.cos(phi), coords=sintheta.coords),
             "theta": xr.DataArray(theta, coords=sintheta.coords),
             "phi": xr.DataArray(phi, coords=sintheta.coords)})

        return sampled_angles

    def sample_step(self, rng=None):
        """Samples step sizes from exponential distribution.

        Parameters
        ----------
        rng : `numpy.random.Generator` object (default None)
            Random number generator. If not specified, use the generator stored
            in the Simulation object

        Returns
        -------
        step : array-like
            Sampled step sizes for all trajectories and scattering events.

        """
        if rng is None:
            rng = self.rng
        nevents = self.nevents
        ntraj = self.ntrajectories

        # Generate array of random numbers from 0 to 1
        sampling_shape = self.leading_shape + (nevents, ntraj)
        rand = xr.DataArray(rng.random(sampling_shape),
                            dims = self.leading_dims + ("event", "trajectory"))

        # sample step sizes
        step = -np.log(1.0-rand) / self.mu_scat

        # If there is fine surface roughness, sample the first step from Mie
        # theory for the number of trajectories set by fine_roughness
        if self.fine_roughness > 0:
            ntraj_mie = int(round(ntraj * self.fine_roughness))
            rand_ntraj = xr.DataArray(rng.random(ntraj_mie),
                                      dims=["trajectory"])
            step.loc[dict(event=0, trajectory=slice(0, ntraj_mie))] = \
                -np.log(1.0-rand_ntraj) / self.mu_scat_mie

        # assign event, trajectory coords (leading coords should have already
        # been transferred from mu_scat)
        step = step.assign_coords({"event": range(nevents),
                                   "trajectory": range(ntraj)})

        return step

    def absorb(self, step_size):
        """
        Calculates absorption of photon packet due to traveling the sample
        between scattering events. Absorption is modeled as a reduction of a
        photon packet's weight using Beer-Lambert's law.

        Parameters
        ----------
        step_size: array-like
            Step size of packet (sampled from scattering lengths), in units
            specified by sc.LENGTH_UNIT

        """
        # shift event coord so that step size maps correctly onto weight (the
        # weight at event n is determined by the step at event n-1)
        step = xr.DataArray(step_size)
        step.coords["event"] = range(1, self.nevents + 1)

        mu_abs = self.mu_abs

        # beer lambert
        weight = (self.traj["weight"].sel(event=0)
                  * np.exp(-(mu_abs * step.cumsum("event"))))
        self.traj["weight"].loc[dict(event=slice(1, None))] = weight

    def scatter(self, angles):
        """
        Calculates the directions of propagation after scattering (for either
        'scattering plane' or 'cartesian' polarizations).

        At a scattering event, a photon packet adopts a new direction of
        propagation, which is randomly sampled from the phase function. The new
        direction of propagation also changes the polarization direction.

        Parameters
        ----------
        angles : `xr.Dataset` (dims=..., event, trajectory)
            Sines and cosines of scattering (theta) and azimuthal (phi) angles
            sampled from the phase function. Theta and phi are angles that are
            defined with respect to the previous corresponding direction of
            propagation. Thus, they are defined in a local spherical coordinate
            system. All have dimensions of (nevents, ntrajectories).

        """
        sintheta, costheta, sinphi, cosphi = (angles[key] for key in
                                              ["sintheta", "costheta",
                                               "sinphi", "cosphi"])
        kn = self.traj["direction"]

        # Calculate the new propagation direction by rotation about the y-axis
        # by angle theta followed by rotation about the z-axis by angle phi
        # see pg 105 in A.B. Stephenson lab notebook 1 for derivation and
        # notes

        # this is the product of the rotation matrices R_z(phi).R_y(theta)
        # shape of kn is    (3, nevents+1, ntraj)
        # shape of R is  (3, 3, nevents-1, ntraj)
        R = xr.combine_nested([[costheta*cosphi, -sinphi, sintheta*cosphi],
                               [costheta*sinphi, cosphi, sintheta*sinphi],
                               [-sintheta, xr.zeros_like(sinphi), costheta]],
                              concat_dim=["i", "component"])
        R = R.transpose(..., "i", "component", "event", "trajectory")

        # could vectorize this loop if numpy had a cumulative dot product
        # ufunc.  But np.cumprod only does element by element.
        for n in np.arange(1, self.nevents):
            # Take the dot product of the rotation matrix for current event
            # with the wavevector for previous event. We use numpy for
            # performance. The overhead of xr.dot() is too costly in a loop.
            kn.data[..., n, :] = np.einsum("...ijk, ...jk -> ...ik",
                                           R.data[..., n-1, :],
                                           kn.data[..., n-1, :])

            # equivalent xarray code is below. This code is much slower but
            # more explicit. Note R.sel(event=n) is equal to R[.., n-1, :]
            # because R has nevents-1 events and kn has nevents+1.

            # kn.loc[dict(event=n)] = (xr.dot(R.sel(event=n),
            #                                 kn.sel(event=n-1),
            #                             dim=["component"])
            #                          .rename({"i": "component"}))

        # Update all the directions of the trajectories
        self.traj["direction"] = kn

    def calc_fields(self, angles, step, tir_indices=None):
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
        angles : `xr.Dataset` (dims=..., event, trajectory)
            Sines and cosines of scattering (theta) and azimuthal (phi) angles
            sampled from the phase function. Theta and phi are angles that are
            defined with respect to the previous corresponding direction of
            propagation. Thus, they are defined in a local spherical coordinate
            system.
        step : `xr.DataArray` (dims=..., event, trajectory)
            Step sizes of packets (sampled from scattering lengths)
        tir_indices : `xr.DataArray` (dims=..., trajectory)
            Array of event indices for trajectories with TIR before exit

        Returns
        -------
        None, but modifies self.traj["field"] : `xr.DataArray`
            Electric field vector for each trajectory and event in global
            coordinates

        """
        sintheta, costheta, sinphi, cosphi, theta, phi = angles.values()

        n_particle = self.model.sphere.n(self.wavelen)
        n_sample = self.model.index_external(self.wavelen)

        m = sc.index.ratio(n_particle, n_sample)
        x = sc.size_parameter(n_sample, self.model.sphere.radius_q)
        k = sc.wavevector(n_sample)

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
        S = mie.amplitude_scattering_matrix(m.to_numpy(), x.to_numpy(),
                                            theta.to_numpy().ravel())

        # for clarity of indexing (0->1) we add a zero element to the list
        S = [0] + list(S)
        # Reshape to (..., nevents, ntraj)
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
        En = self.traj["fields"]

        # En has shape (3, nevents+1, ntraj)
        Ex = En[..., 0, 0, :]
        Ey = En[..., 1, 0, :]

        # Ex and Ey are the initialized as the incident field vectors. To get
        # the Ex and Ey at each event, we have to multiply by the scattering
        # amplitude matrix, cumulatively for each event.
        # this gives us the local Ex and Ey vectors
        # Reminder: there is one less sampled angle than event number, because
        # the first event propogates straight into the sample.
        # Note: this basis assumes that
        # the direction of propagation is the +z direction.
        for n in np.arange(2, self.nevents + 1):
            Ex = S2[..., n-2, :] * Ex + S3[..., n-2, :] * Ey
            Ey = S4[..., n-2, :] * Ex + S1[..., n-2, :] * Ey
            # 0th event is before sample, the 1st event has no rotation
            En[..., 0, n, :] = Ex
            En[..., 1, n, :] = Ey

        # Deal with tir
        if tir_indices is not None:
            # TODO: need to test this code (currently untested)
            # select the tir event for each trajectory
            theta_1 = select_events(theta, tir_indices - 2)
            kz_tir = select_events(self.traj.direction[2], tir_indices)
            theta_r = np.arccos(kz_tir)
            theta_tir = 2 * (np.pi / 2 - theta_r)
            costheta_tir = np.cos(theta_1 + theta_tir)
            sintheta_tir = np.sin(theta_1 + theta_tir)
            tir_ind_theta = tir_indices - 2
            tir_ind_theta[tir_ind_theta < 0] = 0
            costheta[tir_ind_theta, :] = costheta_tir
            sintheta[tir_ind_theta, :] = sintheta_tir

        # Rotate to global coords

        # this is the product of the rotation matrices R_z(phi).R_y(theta)
        # calculated for each event in each trajectory
        # shape of En is    (..., 3, nevents+1, ntraj)
        # shape of R is  (..., 3, 3, nevents-1, ntraj)
        R = np.array([[costheta*cosphi, -sinphi, sintheta*cosphi],
                      [costheta*sinphi, cosphi, sintheta*sinphi],
                      [-sintheta, np.zeros(sinphi.shape), costheta]])
        # reshape to (..., 3, 3, events, trajectories)
        R = np.moveaxis(R, (0, 1), (-4, -3))

        # Start with event 2 because the 0th event contains the initialized
        # values from before the field enters the sample. The 1st event
        # contains the values for the field after entering the sample, but
        # before scattering
        for n in np.arange(2, self.nevents + 1):
            # Einstein summation to take the dot product of each rotation
            # matrix at each event in each trajectory with the wavevector
            # (the n: ensures that all subsequent fields are also rotated)
            En[..., n:, :] = np.einsum('...ijl,...jkl->...ikl',
                                       R[..., n-2, :], En[..., n:, :])

        # calculate the step propagation factor. Note that step is in units of
        # sc.LENGTH_UNIT and k is in units of 1/sc.LENGTH_UNIT, so the product
        # is the dimensionless accumulated phase k*distance.
        step_cumul = step.cumsum("event")
        step_phase_factor = np.exp(1j*np.abs(k)*step_cumul)
        # shift event coord so that step_phase_factor maps correctly onto field
        step_phase_factor.coords["event"] = range(1, self.nevents + 1)

        # multiply the fields by the phase propagation factor
        # TODO: account for fine roughness by accounting for different phase
        # shift on scattering from fine roughness particles.
        # should multiply by 1 for trajectories do not have fine roughness
        # ntraj_fine = int(round(ntraj * self.fine_roughness))
        En[..., 1:, :] = En[..., 1:, :] * step_phase_factor

        # Normalize
        self.traj["fields"] = normalize(En, return_nan=False)

    def move(self, step):
        """
        Calculates positions of photon packets in all the trajectories.
        After each scattering event, the photon packet gets a new position
        based on the previous position, the step size, and the direction of
        propagation.

        Parameters
        ----------
        step : `xr.DataArray`
            Step sizes between scattering events in each of the trajectories.
            Coordinate `event` should be range(n_events)

        """
        # calculate vector displacement after each event
        disp = self.traj["direction"] * step
        # shift event coord so that displacement maps correctly onto position
        disp.coords["event"] = range(1, self.nevents + 1)

        # The array of positions is a cumulative sum of all of the
        # displacements. Note: we use da.cumsum() because
        # da.cumulative("event").sum() will be much slower if numbagg is not
        # installed. See https://github.com/pydata/xarray/issues/6528
        cumul_disp = disp.cumsum("event")

        # initial position
        r0 = self.traj["position"].sel(event=0)
        # event numbers to update
        events = dict(event=slice(1, None))

        self.traj["position"].loc[events] = (r0 + cumul_disp)

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

        ax[0].plot(np.arange(len(self.traj["position"][0,:,0])),
                   self.traj["position"][0,:,:], '-')
        ax[0].set_title('Positions during trajectories')
        ax[0].set_ylabel('x (' + str(sc.LENGTH_UNIT) + ')')

        ax[1].plot(np.arange(len(self.traj["position"][1,:,0])),
                   self.traj["position"][1,:,:], '-')
        ax[1].set_ylabel('y (' + str(sc.LENGTH_UNIT) + ')')

        ax[2].plot(np.arange(len(self.traj["position"][2,:,0])),
                   self.traj["position"][2,:,:], '-')
        ax[2].set_ylabel('z (' + str(sc.LENGTH_UNIT) + ')')
        ax[2].set_xlabel('scattering event')

        if three_dim:
            fig = plt.figure(figsize=(8,6))
            ax3D = fig.add_subplot(111, projection='3d')
            ax3D.set_xlabel('x (' + str(sc.LENGTH_UNIT) + ')')
            ax3D.set_ylabel('y (' + str(sc.LENGTH_UNIT) + ')')
            ax3D.set_zlabel('z (' + str(sc.LENGTH_UNIT) + ')')
            ax3D.set_title('Positions during trajectories')

            for n in np.arange(ntraj):
                ax3D.scatter(self.traj["position"][0,:,n],
                             self.traj["position"][1,:,n],
                             self.traj["position"][2,:,n],
                             color=next(colors))


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


def coarse_roughness_enter(k, n_medium, n_sample, coarse_roughness, boundary,
                           rng=None):
    """
    Calculates new initial directions based on the coarse roughness of the
    sample.

    Parameters
    ----------
    k : `xr.DataArray`
        Directions of propagation. Has shape of (..., 3, number of events,
        number of trajectories). k0.loc["x"] and k0.loc["y"] are initialized to
        zero, and k0.loc[dict(event=0, component="z")] is initialized to 1.
    n_medium : `xr.DataArray` (dims=...)
        Refractive index of the medium, as output from an `sc.Index` object.
    n_sample : `xr.DataArray` (dims=...)
        Refractive index of the sample, as output from an `sc.Index` object.
    coarse_roughness : float
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
    boundary : string
        Geometrical boundary for Monte Carlo calculations. Current options are
        'film' or 'sphere.' Coarse roughness is currently only implemented for
        a film.
    rng : numpy.random.Generator object (default None)
        If not specified, use the default generator initialized on loading the
        package

    Returns
    -------
    k0_rough : `xr.DataArray`
        Initial direction of propagation, corrected for coarse roughness.
    kz0_rot : `xr.DataArray`
        Initial z-directions that are rotated to account for the fact that
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the
        surface. The array size is (1, ntraj). Only returned if
        coarse_roughness is set to > 0.
    kz0_refl : `xr.DataArray`
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global
        coordinate system. The array size is (1, ntraj). Only returned if
        coarse_roughness is set to > 0.

    """
    if rng is None:
        rng = sc.rng

    if boundary == 'sphere':
        raise ValueError("course roughness not yet implemented for sphere "
                         "boundary")

    ntraj = k.sizes["trajectory"]

    # for constructing rotation matrices
    zeros = np.zeros(ntraj)
    ones = np.ones(ntraj)
    # "i" is a dummy index that we rename after taking the dot product
    rotcoords = {"i": ["x", "y", "z"],
                 "component": ["x", "y", "z"],
                 "trajectory": range(ntraj)}

    # get the first event only
    k0 = k.sel(event=0)

    # sample the surface roughness angles theta_a
    theta_a_full = np.linspace(0., np.pi / 2, 500)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_a = (P_theta_a(theta_a_full, coarse_roughness) /
                  sum(P_theta_a(theta_a_full, coarse_roughness)))
    if np.isnan(prob_a).all():
        theta_a = np.zeros(ntraj)
    else:
        theta_a = sc.choice(theta_a_full, ntraj, p=prob_a, rng=rng)

    # In case the surface is rough, then find new coordinates of initial
    # directions after rotating the surface by an angle theta_a around y axis
    sintheta_a = np.sin(theta_a)
    costheta_a = np.cos(theta_a)

    # rotation matrix R_y(theta)
    R = xr.DataArray([[costheta_a, zeros, -sintheta_a],
                      [zeros, ones, zeros],
                      [sintheta_a, zeros, costheta_a]],
                     coords=rotcoords)
    k0_rot = xr.dot(R, k0, dim=("component")).rename({"i": "component"})

    # Find the new angles theta and phi between the incident trajectories and
    # the normal to the new surface after the coordinate axis rotation
    norm = (k0**2).sum(dim="component")
    theta_rot = np.arccos(k0_rot.loc['z'] / norm).drop_vars("component")
    phi_rot = np.arccos(k0_rot.loc['x'] / norm).drop_vars("component")

    # Refraction of incident light upon entering sample
    theta_refr = refraction(theta_rot, n_medium, n_sample)

    k0_rot_refr = xr.concat([np.sin(theta_refr) * np.cos(phi_rot),
                             np.sin(theta_refr) * np.sin(phi_rot),
                             np.cos(theta_refr)],
                            dim="component")
    k0_rot_refr.coords["component"] = rotcoords["component"]

    # Rotate the axes back so that the initial refracted directions are in old
    # (global) coordinates by doing an axis rotation around y by 2pi-theta_a
    R = xr.DataArray(
            [[np.cos(2*np.pi-theta_a), zeros, -np.sin(2*np.pi-theta_a)],
             [zeros, ones, zeros],
             [np.sin(2*np.pi-theta_a), zeros, np.cos(2*np.pi-theta_a)]],
            coords = rotcoords)
    k0_refr = (xr.dot(R, k0_rot_refr, dim=("component"))
               .rename({"i": "component"}))

    # Create an empty array of the initial direction cosines of the right size
    # (shape [..., 3, nevents, ntraj])
    k0_rough = xr.zeros_like(k)

    # Fill up the first row (corresponding to the first scattering event) of
    # the direction cosines array with the randomly generated angles:
    k0_rough.loc[dict(event=0)] = k0_refr

    # Calculate Fresnel reflected directions, which are the same as the initial
    # directions in the local coordinate system but flipping the z sign
    k0_rot_refl = k0_rot.copy()
    k0_rot_refl.loc['z'] = -k0_rot_refl.loc['z']

    # Rotate the axes back so that the reflected directions are in old (global)
    # coordinates by doing an axis rotation around y by 2pi-theta_a
    # We need only the z-component, so perhaps can simplify the expression
    # below
    k0_refl = (xr.dot(R, k0_rot_refl, dim=("component"))
               .rename({"i": "component"}))

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
    B. van Ginneken, M. Stavridi, J. J. Koenderink, “Diffuse and specular
    reflectance from rough surfaces”, Applied Optics, 37, 1 (1998) (has
    definition of rsm slope of the surface).

    """
    term1 = np.sin(theta_a) / r**2 / (np.cos(theta_a))**3
    term2 = np.exp(-(np.tan(theta_a))**2 / (2*r**2))

    return term1 * term2
