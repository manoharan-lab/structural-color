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
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.random import random as random

import structcol as sc

from . import model
from . import montecarlo as mc
from . import normalize, refraction, select_events

EPS = 1.e-9


def inf_to_large(pos0, pos1, radius):
    '''
    convert two sets of trajectory coordinates from infinite values to a large
    value instead

    Parameters
    ----------
    pos0 : array
        position of first point
    pos1 : array
        position of second point
    radius : float
        value that scales the large value that replaces the infinite value.

    Returns
    -------
    pos0, pos1 : arrays
        coordinates with infinite values replaced with large ones
    '''
    pos0[pos0 > 1e20 * radius] = 100 * radius
    pos1[pos1 > 1e20 * radius] = 100 * radius
    pos0[pos0 < -1e20 * radius] = -100 * radius
    pos1[pos1 < -1e20 * radius] = -100 * radius

    return pos0, pos1


def find_vec_sphere_intersect(pos0, pos1, radius):
    """
    Analytically solves for the point at which an exiting trajectory
    intersects with the boundary of the sphere

    Parameters
    ----------
    pos0 : array
        initial position of each trajectory before exit
    pos1 : array
        position of each trajectory after exit
    radius : float
        radius of spherical boundary

    Returns
    ----------
    pos_int: 2d array
        position where exit trajectories intersect the boundary of the sphere,
        with shape: 3 (coordinate), ntrajectories

    """

    # Find k vector from point inside and outside sphere.
    k = normalize(pos1 - pos0)

    # Solve for intersection of k with sphere surface using parameterization.
    # There will be two solutions for each k vector, corresponding to the two
    # points where a line intersects a sphere.
    # See http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    # or Annie Stephenson lab notebook #3, pg 18 for details.
    a = np.sum(k**2, axis=0)
    b = 2 * np.sum(k * pos0, axis=0)
    c = np.sum(pos0**2, axis=0) - radius ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t_m = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    np.seterr(divide='warn', invalid='warn')
    int_p = pos0 + t_p * k
    int_m = pos0 + t_m * k

    # find the distances between the each solution point and the trajectory
    # point outside the sphere
    # casts nans to zero
    dist_p = np.nan_to_num(np.sum((int_p - pos1) ** 2, axis=0))
    dist_m = np.nan_to_num(np.sum((int_m - pos1) ** 2, axis=0))

    # Find the indices of the smaller distances of the two
    # because the intersection point corresponding to the exiting trajectory
    # must be the intersection point closest to the trajectory's position
    # outside the sphere.
    # In the case of nans for the x_int_p/m, y_int_p/m, z_int_p/m values,
    # both will be cast to 0s by nan_to_num, and neither will be greater.
    ind_p = np.where(dist_p < dist_m)[0]
    ind_m = np.where(dist_m < dist_p)[0]

    # Keep only the intercept closest to the exit point of the trajectory
    # pos_int will be zero when there was no exit_traj.
    pos_int = np.zeros((3, len(pos0[0])))
    pos_int[:, ind_p] = int_p[:, ind_p]
    pos_int[:, ind_m] = int_m[:, ind_m]

    return pos_int


def exit_kz(indices, trajectories, boundary, thickness, n_inside, n_outside):
    '''
    returns kz of exit trajectories, corrected for refraction at the spherical
    boundary. Since sphere trajectories can refract away from the detector,
    this is currently only relevant for the sphere case

    Parameters
    ----------
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options
        are 'film' or 'sphere'.
    thickness: float
        thickness of film or diameter of sphere
    n_inside: float
        refractive index inside sphere boundary
    n_outside: float
        refractive index outside sphere boundary

    Returns
    -------
    k2z: 1D array (length ntraj)
        z components of refracted kz upon trajectory exit

    '''
    trajectories = mc.NumpyTrajectory(trajectories)

    # Get the angles between trajectories and the normal,
    # along with their normal vectors.
    theta_1, norm = get_angles(indices, boundary, trajectories, thickness)

    # Take cross product of k1 and sphere normal vector to find vector
    # to rotate around.
    k1 = select_events(trajectories.direction, indices)
    kr = np.transpose(np.cross(np.transpose(k1), np.transpose(norm)))

    # Use Snell's law to calculate angle between k2 and normal vector.
    # theta_2 is nan if photon is totally internally reflected.
    theta_2 = refraction(theta_1, n_inside, n_outside).to_numpy()

    # The angle to rotate around is theta_2-theta_1.
    theta = theta_2 - theta_1

    # Perform the rotation.
    _, _, k2z = rotate_refract(norm * thickness / 2, kr, k1, theta)

    # If k2z is nan, leave uncorrected.
    # Since nan means the trajectory was totally internally reflected, the
    # exit kz doesn't matter, but in order to calculate the fresnel reflection
    # back into the sphere, we still need it to count as a potential exit
    # hence we leave the kz unchanged.

    return k2z


def rotate_refract(point, axis, vec, alpha):
    '''
    Rotates vector <vec> by angle alpha about the unit vector <uvw>,
    where (a,b,c) is a point on the vector we are rotating about.

    Parameters
    ----------
    point : array
        point on the vector <uvw> to rotate about
    axis : array
        vector to rotate about
    vec : array
        vector to rotate
    alpha : 1d array
        angle by which to rotate <vec>

    length of each of these arrays is number of trajectories being rotated

    Returns
    -------
    newvec : array
        rotated vector

    Notes
    -----
    This rotation matrix was derived by Glenn Murray
    and it's derivation is explained here:
    https://sites.google.com/site/glennmurray/Home/rotation-matrices
    -and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
    '''

    a, b, c = point
    # normalize because the rotation matrix formula assumes <uvw> has length 1
    u, v, w = normalize(axis, return_nan=False)

    # (x,y,z) is a physical point on the k vector
    # we find the point by adding a,b,c to the normalized k vector
    # (this seems to just make the origins of both vec and axis the same --
    # could achieve the same thing by moving "point" to the origin of the
    # coordinate system)
    x, y, z = point + vec

    # rotation matrix
    x_rot = (a * (v**2 + w**2) \
             - u * (b * v + c * w - u * x - v * y - w * z)) \
             * (1 - np.cos(alpha)) + x * np.cos(alpha) \
             + (-c * v + b * w - w * y + v * z) * np.sin(alpha)
    y_rot = (b * (u**2 + w**2) \
             - v * (a * u + c * w - u * x - v * y - w * z)) \
             * (1 - np.cos(alpha)) + y * np.cos(alpha) \
             + (c * u - a * w + w * x - u * z) * np.sin(alpha)
    z_rot = (c * (u**2 + v**2) \
             - w * (a * u + b * v - u * x - v * y - w * z)) \
             * (1 - np.cos(alpha)) + z * np.cos(alpha) \
             + (-b * u + a * v - v * x + u * y) * np.sin(alpha)

    # to recover the k vector from the point rotated in space, we must subtract
    # a,b,c
    newvec = np.array([x_rot, y_rot, z_rot]) - point

    return newvec


def get_angles(indices, boundary, trajectories, thickness,
               init_dir=None, plot_exits=False):
    '''
    Returns angles relative to vector normal to boundary
    (either film or sphere) at point on boundary.

    Parameters
    ----------
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options
        are 'film' or 'sphere'.
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    thickness: float
        thickness of film or diameter of sphere
    init_dir: None or 1D array of pint quantities (length ntrajectories)
        kz of trajectories before they enter sample
    plot_exits : boolean
        If set to True, function will plot the last point of trajectory inside
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits,
        making one plot for reflection and one plot for transmission

    Returns
    -------
    angles: 1D array of pint quantities (length ntrajectories)
        angle between k1 and the normal vector at the exit point of the
        trajectory
    norm: 2D array of shape (3, ntrajectories)
        vector normal to the surface at the exit point of the trajectory
    '''

    if boundary == 'sphere':
        pos = trajectories.position.copy()
        radius = thickness / 2

        # Subtract radius from z to center the sphere at 0,0,0. This makes the
        # following calculations much easier.
        pos[2] = pos[2] - radius

        # If incident light..
        if init_dir is not None:
            # TODO implement capability for diffuse illumination of sphere
            k = trajectories.direction.copy()

            # Select initialized trajectory positions.
            k1 = select_events(k, indices)

            # Initial positions are on the sphere boundary.
            # Multiply by minus sign to flip normal vector so the dot product
            # with k1 has the right sign.
            intersect = -select_events(pos, indices)
        else:
            # Get positions outside of sphere boundary from after exit
            # (or entrance if this is for first event).
            # Index starting with 1 so we pick point after exit event, since
            # it skips the initial point.
            pos1 = select_events(pos[:, 1:, :], indices)

            # Get positions inside sphere boundary from before exit
            # Index starting with len(x)-1 so we pick point before
            # exit event, since it starts with initial point before
            # the first event.
            pos0 = select_events(pos[:, :-1, :], indices)

            # Make sure there are no infinite values in the coordinates.
            # This prevents an error for the extreme case
            # where mu_scat is infinite, which means that the index contrast
            # between the particle and matrix is 0
            pos0, pos1 = inf_to_large(pos0, pos1, radius)

            # calculate the normalized k1 vector from the positions
            # inside and outside (X0,y0,z0) and (x1,y1,z1)
            k1 = normalize(pos1 - pos0)

            # get positions at sphere boundary from exit
            intersect = find_vec_sphere_intersect(pos0, pos1, radius)

        # calculate the vector normal to the sphere boundary at the exit
        norm = normalize(intersect)

        # calculate the dot product between the normal vector
        # and the exit vector
        dot_norm = np.sum(norm * k1, axis=0)

        # calulate the angle between the normal vector and the exit vector
        angles = np.arccos(np.nan_to_num(dot_norm))

        # plot the points before exit, after exit, and on exit boundary
        if plot_exits:  # pragma: no cover
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(select_x0,select_y0,select_z0, c = 'b')
            # ax.scatter(select_x1,select_y1,select_z1, c = 'g')
            ax.scatter(intersect[0], intersect[1], intersect[2], c='r')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim([-1.5 * radius, 1.5 * radius])
            ax.set_ylim([-1.5 * radius, 1.5 * radius])
            ax.set_zlim([-1.5 * radius, 1.5 * radius])

            u, v = np.mgrid[0:2 * np.pi:20j, np.pi:0:10j]
            x = radius * np.cos(u) * np.sin(v)
            y = radius * np.sin(u) * np.sin(v)
            z = radius * (-np.cos(v))
            ax.plot_wireframe(x, y, z, color=[0.8, 0.8, 0.8])

    if boundary == 'film':
        kz = trajectories.direction[2].copy()

        # If incident light, use init dir instead of kz to get direction.
        # before entering sample.
        if init_dir is not None:
            cosz = init_dir
        else:
            # Select scattering events resulted in exit.
            cosz = select_events(kz, indices)

        # Calculate angle to normal from cos_z component (only want magnitude).
        angles = np.arccos(np.abs(cosz))

        # Calculate the normal vector.
        norm = np.zeros((3, kz.shape[0], kz.shape[1]))
        norm[2, :, :] = np.sign(cosz)

    # Turn nan values to zeros.
    norm = np.nan_to_num(norm)

    return angles, norm


def fresnel_pass_frac(indices, n_before, n_inside, n_after, boundary,
                      trajectories, thickness,
                      init_dir=None,
                      plot_exits=False):
    '''
    Returns weights of interest reduced by fresnel reflection across two
    interfaces, For example passing through a coverslip.

    Note: if n_inside = None, returns weights of interest reduced accross one
    interface. This code has not been tested for case of some sort of coverslip
    covering a sphere. It is currently only used for case of passing from
    sphere directly to air.

    Parameters
    ----------
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    n_before: float
        Refractive index of the medium light is coming from
    n_inside: float
        Refractive index of the boundary material (e.g. glass coverslip)
    n_after: float
        Refractive index of the medium light is going into
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    trajectories:Trajectory object
        Trajectory object used in Monte Carlo simulation
    thickness: float
        thickness of film or diameter of sphere
    init_dir: None or 1D array of pint quantities (length ntrajectories)
        kz of trajectories before they enter sample
    plot_exits : boolean
        if set to True, function will plot the last point of trajectory inside
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits,
        making one plot for reflection and one plot for transmission

    Returns
    -------
    fresnel_pass:1D array (length ntraj)
        trajectory weights reduced by fresnel reflection
    norm: 1D array of shape (3, ntrajectories)
        vector normal to the surface at the exit point of the trajectory


    '''
    # Allow single interface by passing in None as n_inside.
    if n_inside is None:
        n_inside = n_before

    # Get angles between trajectory direction and normal vector.
    theta_before, norm = get_angles(indices, boundary, trajectories,
                                    thickness,
                                    init_dir=init_dir,
                                    plot_exits=plot_exits)

    # Find angles inside.
    theta_inside = refraction(theta_before, n_before, n_inside).to_numpy()

    # If theta_inside is nan (because the trajectory doesn't exit due to TIR),
    # then replace it with pi/2 (the trajectory goes sideways infinitely) to
    # avoid errors during the calculation of stuck trajectories.
    theta_inside[np.isnan(theta_inside)] = np.pi / 2.0

    # Find fraction passing through both interfaces.
    _, trans_1 = model.fresnel_coeffs(n_before, n_inside, theta_before)
    trans_s1, trans_p1 = trans_1.to_numpy()
    _, trans_2 = model.fresnel_coeffs(n_inside, n_after, theta_inside)
    trans_s2, trans_p2 = trans_2.to_numpy()
    fresnel_trans = (trans_s1 + trans_p1) * (trans_s2 + trans_p2) / 4.

    # Find fraction reflected off both interfaces before transmission.
    refl_1, _ = model.fresnel_coeffs(n_inside, n_after, theta_inside)
    refl_s1, refl_p1 = refl_1.to_numpy()
    refl_2, _ = model.fresnel_coeffs(n_inside, n_before, theta_inside)
    refl_s2, refl_p2 = refl_2.to_numpy()
    fresnel_refl = (refl_s1 + refl_p1) * (refl_s2 + refl_p2) / 4.

    # Any number of higher order reflections off the two interfaces.
    # Use converging geometric series 1 + a + a ** 2 + a ** 3... = 1 / (1 - a)
    fresnel_pass = fresnel_trans / (1 - fresnel_refl + EPS)

    return fresnel_pass, norm


def detect_correct(indices, trajectories, weights, n_before, n_after, boundary,
                   thickness, thresh_angle, init_dir=None):
    '''
    Returns weights of interest within detection angle

    Parameters
    ----------
    indices: 1D array
        Length ntraj. Values represent events of interest in each trajectory
    trajectories:Trajectory object
        Trajectory object used in Monte Carlo simulation
    weights: 2D array
        weights values, with axes corresponding to events, trajectories
    n_before: float
        Refractive index of the medium light is coming from
    n_after: float
        Refractive index of the medium light is going into
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    thickness: float
        thickness of film or diameter of sphere
    thresh_angle: float
        Detection angle to compare with output angles

    Returns
    -------
    filtered weights: 1D array (length ntraj)
        trajectory weights within detection angle

    '''

    # Find angles when crossing interface.
    angles, _ = get_angles(indices, boundary, trajectories, thickness,
                           init_dir=init_dir)

    theta = refraction(angles, n_before, n_after).to_numpy()
    theta[np.isnan(theta)] = np.inf  # This avoids a warning.

    # Choose only the ones inside detection angle.
    filtered_weights = copy.deepcopy(weights)
    filtered_weights[theta > thresh_angle] = 0

    return filtered_weights


def find_exits(n_sample, n_medium, thickness, z_low, boundary, trajectories):
    """
    Find booleans describing valid exits for each event and trajectory. Value
    of 1 indicates a valid exit, value of 0 indicates no valid exit.

    Parameters
    ----------
    n_sample : `xr.DataArray`
        Refractive index of the sample, returned from an `sc.Index` object
    n_medium : `xr.DataArray`
        Refractive index of the medium, returned from an `sc.Index` object
    thickness: float
        thickness of film or diameter of sphere
    z_low: float
        starting z_values for trajectores in Monte Carlo simulation, usually
        set to 0
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    trajectories : `xr.Dataset`
        trajectories from a Simulation object

    Returns
    -------
    `xr.Dataset` : shape (..., ntraj)
        "exit"  : array of event indices for exit, irrespective of direction
        "refl"  : array of event indices for reflected trajectories
        "trans" : array of event indices for transmitted trajectories
        "stuck" : array of event indices for stuck trajectories
        "tir"   : array of event indices for trajectories with TIR before exit

    """

    if boundary == 'film':

        # We consider a periodic extension of the film as follows:
        #                                                             floor
        #
        #  bot   ---------------------------------------------------  -1
        #
        #
        #  top   -------------*-------------------------------------  0
        #                      \
        #                       \
        #  bot   ---------------------------------------------------  1
        #                         \
        #                          \
        #  top   ---------------------------------------------------  2
        #
        # With this scheme, an internally reflected trajectory, like the one
        # shown, can be considered to continue to move in the same direction.
        # Trajectories that are not internally reflected but cross a boundary
        # have exited the sample.

        # Get variables we need from trajectories.
        zdir = trajectories["direction"].sel(component="z")
        z = trajectories["position"].sel(component="z")

        # Rescale z in terms of integer numbers of sample thickness.
        z_floors = np.floor((z - z_low) / (thickness - z_low))

        # Potential exits occur whenever trajectories cross any boundary.
        potential_exits = ~(z_floors.diff("event") == 0)

        # Find all kz with magnitude large enough to exit.
        # todo-get particle in here?
        # we use .roll to shift zdir to the direction that got us to current z
        no_tir = ((np.abs(zdir) > np.cos(np.arcsin(n_medium / n_sample)))
                  .roll(event=1))

        # Exit in positive direction (transmission) occurs
        # iff crossing odd boundary.
        ##
        ## Truth table
        ## -----------
        ## let a = floor at event n-1
        ##     b = floor at event n - floor at event n-1
        ## a odd,  b > : (odd  +  1) % 2 = 0 = False
        ## a odd,  b = : (odd  +  0) % 2 = 1 = True
        ## a odd,  b < : (odd  +  0) % 2 = 1 = True
        ## a even, b > : (even +  1) % 2 = 1 = True
        ## a even, b = : (even +  0) % 2 = 0 = False
        ## a even, b < : (even +  0) % 2 = 0 = False
        ##
        ## after logical AND with potential_exits, the equality cases are
        ## rendered false, so we only have to deal with two cases where pos_dir
        ## is True:
        ##
        ## a odd,  b < : (odd  +  0) % 2 = 1 = True
        ## a even, b > : (even +  1) % 2 = 1 = True
        ##
        ## in both cases, True means that the trajectory has crossed an odd
        ## boundary, meaning it has come out the bottom of the film
        #
        pos_dir = (np.mod(z_floors.shift(event=1)
                          + 1 * (z_floors > z_floors.shift(event=1)), 2)
                   .astype(bool))
        # equivalent numpy code
        # pos_dir = np.mod(z_floors[:-1]
        #                  + 1*(z_floors[1:] > z_floors[:-1]), 2).astype(bool)

        # Construct boolean arrays of all valid exits in pos & neg directions.
        exits_pos_dir = potential_exits & no_tir & pos_dir
        exits_neg_dir = potential_exits & no_tir & ~pos_dir

        # Construct boolean array to describe whether a trajectory gets
        # totally internally reflected at any event.
        # (the ~pos_dir seems to be placed here with the assumption that only
        # reflected TIR trajectories will be affected by TIR and that double
        # TIR events are rare)
        tir_refl_bool = potential_exits & ~no_tir & ~pos_dir

    if boundary == 'sphere':

        # Get variables we need from trajectories.
        x, y, z = trajectories["position"].transpose("component", ...)

        # Define sphere radius.
        radius = thickness / 2

        # Potential exits occur whenever trajectories are outside
        # sphere boundary.
        potential_exits = (x[..., 1:, :] ** 2 + y[..., 1:, :] ** 2
                           + (z[..., 1:, :] - radius) ** 2) > radius ** 2
        potential_exit_indices = (potential_exits.where(potential_exits)
                                  .idxmin("event", fill_value=0))

        # kz_correct will be nan if trajectory is totally internally reflected.
        kz_correct = exit_kz(potential_exit_indices.to_numpy().squeeze(),
                             trajectories, boundary,
                             thickness, n_sample.to_numpy().squeeze(),
                             n_medium.to_numpy().squeeze())
        # no_tir is calculated to match film case
        # for use in event_distribution.py
        no_tir = ~np.isnan(kz_correct)

        # Exit in positive direction is transmission,
        # and exit in negative direction is reflection.
        # kz_correct will be nan if trajectory is totally internally reflected.
        pos_dir = kz_correct > 0
        neg_dir = kz_correct < 0  # Can't just do ~pos_dir to exclude TIR.

        # Construct boolean arrays of all valid exits in pos & neg directions.
        exits_pos_dir = potential_exits & pos_dir
        exits_neg_dir = potential_exits & neg_dir

        # Construct boolean array to describe whether a trajectory gets
        # totally internally reflected at any event
        # for the sphere.
        tir_refl_bool = potential_exits & ~no_tir

    nevents = trajectories.coords["event"].shape[0] - 1
    # find first valid exit of each trajectory in each direction
    # note we convert to 2 1D arrays with len = Ntraj
    # note that exits_neg_dir and exits_pos_dir contain values of 0 or 1,
    # and exits_neg_dir.where(exits_neg_dir) converts False to nan.
    # Then idxmin returns the *first* instance of the min (1), in cases
    # where there are multiple 1's
    # ("low" here appears to mean low in the sense of small z)
    low_event = exits_neg_dir.where(exits_neg_dir).idxmin("event",
                                                          fill_value=0)

    high_event = exits_pos_dir.where(exits_pos_dir).idxmin("event",
                                                           fill_value=0)

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
    ## set stuck indices to nevents for each stuck trajectory
    stuck_indices = never_exit * nevents

    # correct tir_refl_bool to be True only before exit
    ## we can add these arrays because they are 0 when the trajectory does not
    ## fall into a category (reflected, transmitted, stuck), and the three
    ## categories are mutually exclusive
    exit_indices = refl_indices + trans_indices + stuck_indices
    ## following will select only indices on or after exit events
    exited = tir_refl_bool.coords["event"] > exit_indices
    tir_refl_bool = tir_refl_bool & ~exited

    ## following will find first (i.e., minimum) index at which TIR occurs
    tir_indices = tir_refl_bool.where(tir_refl_bool).idxmin("event",
                                                            fill_value=0)

    # return Dataset. "exit" is all exit indices excluding stuck trajectories
    indices = xr.Dataset({"exit": refl_indices + trans_indices,
                          "refl": refl_indices,
                          "trans": trans_indices,
                          "stuck": stuck_indices,
                          "tir": tir_indices})

    return indices.drop_vars("component")


def calc_outcome_weights(inc_fraction, refl_indices, trans_indices,
                         stuck_indices, weights):
    '''
    Calculates trajectory weight contributions to reflection, tranmission,
    stuck, and absorption

    Parameters
    ----------
    inc_fraction: 1d array (length: ntraj)
        fraction of incident light that is fresnel reflected
        at the medium-sample interface
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    stuck_indices: 1d array (length: ntraj)
        array of event indices for stuck trajectories
    weights: 2d array (shape: nevents, ntraj)
        weights values, with axes corresponding to events, trajectories

    Returns
    -------
    refl_weights: 1d array (length: ntraj)
        weights of reflected trajectories
    trans_weights: 1d array (length: ntraj)
        weights of transmitted trajectories
    stuck_weights: 1d array (length: ntraj)
        weights of stuck trajectories
    absorb_weights: 1d array (length: ntraj)
        weight absorbed for each trajectory
    '''

    # calculate outcome weights from all trajectories
    refl_weights = inc_fraction * select_events(weights, refl_indices)
    trans_weights = inc_fraction * select_events(weights, trans_indices)
    stuck_weights = inc_fraction * select_events(weights, stuck_indices)
    absorb_weights = (inc_fraction - refl_weights - trans_weights
                      - stuck_weights)

    # warn user if too many trajectories got stuck
    stuck_frac = np.sum(stuck_weights) / np.sum(inc_fraction) * 100
    stuck_traj_warn = (" \n{0}% of trajectories did not exit the sample. "
                       "Increase Nevents to improve "
                       "accuracy.").format(str(stuck_frac))
    if stuck_frac >= 20:
        warnings.warn(stuck_traj_warn)

    return refl_weights, trans_weights, stuck_weights, absorb_weights


def fresnel_correct_enter(n_medium, n_front, n_sample, boundary, thickness,
                          trajectories, fresnel_traj, kz0_rot):
    '''
    Corrects weights for fresnel reflection when light enters the sample,
    taking into account the refractive index of the medium, a material in
    front of the sample such as a sample chamber, and of the sample itself.
    The returned value is the fraction of light that passes, which can
    be multiplied by the initial trajectory weights to find the weights that
    pass.

    Parameters
    ----------
    n_medium: float
        Refractive index of the medium
    n_front: float
        Refractive index of the boundary material (e.g. glass coverslip)
    n_sample: float
        Refractive index of the sample
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    thickness: float
        thickness of film or diameter of sphere
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    fresnel_traj: boolean
        describes whether trajectories object passed in represents
        trajectories that have been fresnel reflected back into the sample
    kz0_rot : None or array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the
        surface. The array size is (1, ntraj).
    kz0_refl : None or array_like (structcol.Quantity [dimensionless])
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global
        coordinate system. The array size is (1, ntraj).

    Returns
    -------
    init_dir: None or 1D array-like (length ntrajectories)
        kz of trajectories before they enter sample
    inc_pass_frac: 1D array-like (length ntrajectories)
        weight fraction of trajectory that passes through interface
        upon entering sample
    '''

    # Get variables to use throughout function.
    kz = trajectories.direction[2]
    ntraj = kz.shape[1]
    indices = np.ones(ntraj)

    if boundary == 'film':
        # Calculate initial weights (=inc_fraction) that actually enter
        # the sample after Fresnel reflection.
        if kz0_rot is None:
            # init_dir is reverse-corrected for refraction.
            # init_dir = kz before medium/sample interface.
            angles, _ = get_angles(indices, boundary, trajectories, thickness)
            init_dir = np.cos(refraction(angles, n_sample, n_medium)
                              .to_numpy())
        else:
            kz0_rot = np.squeeze(kz0_rot)
            init_dir = kz0_rot

    if boundary == 'sphere':
        # init_dir is reverse-corrected for refraction.
        # init_dir = kz before medium/sample interface.
        # For now, we assume initial direction is in +z.
        # TODO add capability for diffuse illumination
        init_dir = np.ones(ntraj)

    # Calculate initial weights that actually enter the sample after Fresnel.
    if not fresnel_traj:
        inc_pass_frac, _ = fresnel_pass_frac(indices, n_medium, n_front,
                                             n_sample, boundary, trajectories,
                                             thickness, init_dir=init_dir)

    else:
        # If fresnel_traj is true, the trajectories start inside the sample
        # and no Fresnel correction is needed.
        inc_pass_frac = np.ones(ntraj)

    return init_dir, inc_pass_frac


def fresnel_correct_exit(n_sample, n_medium, n_front, n_back, refl_indices,
                         trans_indices, refl_weights, trans_weights,
                         absorb_weights, boundary, thickness, trajectories,
                         fresnel_traj, plot_exits):
    '''
    Corrects weights for fresnel reflection when light exits the sample,
    taking into account the refractive index of the medium, a material in
    front of and behind the sample such as a sample chamber, and of the sample
    itself.

    Parameters
    ----------
    n_sample: float
        Refractive index of the sample
    n_medium: float
        Refractive index of the medium
    n_front: float
        Refractive index of the front boundary material (e.g. glass coverslip)
    n_back: float
        Refractive index of the back boundary material (e.g. glass coverslip)
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    refl_weights: 1d array (length: ntraj)
        weights of reflected trajectories
    trans_weights: 1d array (length: ntraj)
        weights of transmitted trajectories
    absorb_weights: 1d array (length: ntraj)
        weight absorbed for each trajectory
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    thickness: float
        thickness of film or diameter of sphere
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    fresnel_traj: boolean
        describes whether trajectories object passed in represents
        trajectories that have been fresnel reflected back into the sample
    plot_exits: boolean
        instructs whether or not to plot the exits

    Returns
    -------
    refl_frac: float
        fraction of trajectory weights that are successfully reflected, meaning
        not fresnel reflected back into the sample
    trans_frac: float
        fraction of trajectory weights that are successfully transmitted,
        meaning not fresnel reflected back into the sample
    refl_weights_pass: 1d array (length: ntraj)
        weights of reflected trajectories, corrected for fresnel reflection
        upon exit. Zero values indicate trajectory was totally internally
        reflected back into the sample, or not reflected in the first place
    trans_weights_pass: 1d array (length: ntraj)
        weights of transmitted trajectories, corrected for fresnel reflection
        upon exit. Zero values indicate trajectory was totally internally
        reflected back into the sample, or not transmited in the first place
    refl_fresnel: 1d array (length: ntraj)
        weights of reflected trajectories that are fresnel reflected
        back into the sample
    trans_fresnel: 1d array (length: ntraj)
        weights of transmitted trajectories that are fresnel reflected back
        into the sample
    norm_vec_refl: 2d array (shape: 3, ntrajectories)
        vector normal to the surface at the exit point
        of reflected trajectories
    norm_vec_trans: 2d array (shape: 3, ntrajectories)
        vector normal to the surface at the exit point
        of transmitted trajectories

    '''

    # Calculate the pass fraction for reflected trajectories
    # fresnel_pass_frac_refl will be 0 for a trajectory that is totally
    # internally reflected upon reaching the interface. It will be 1 for a
    # trajectory that passes through the inferace with no fresnel reflection
    # (this would only happen if there is no index contrast).
    (fresnel_pass_frac_refl,
     norm_vec_refl) = fresnel_pass_frac(refl_indices,
                                        n_sample,
                                        n_front,
                                        n_medium,
                                        boundary,
                                        trajectories,
                                        thickness,
                                        plot_exits=plot_exits)

    # set up axes if plot_exits is true
    if plot_exits:  # pragma: no cover
        plt.gca().set_title('Reflected exits')
        plt.gca().view_init(-164, -155)

    # Calculate the pass fraction for transmitted trajectories
    # fresnel_pass_frac_trans will be 0 for a trajectory that is totally
    # internally reflected upon reaching the interface. It will be 1 for a
    # trajectory that passes through the inferace with no fresnel reflection
    # (this would only happen if there is no index contrast).
    (fresnel_pass_frac_trans,
     norm_vec_trans) = fresnel_pass_frac(trans_indices,
                                         n_sample,
                                         n_back,
                                         n_medium,
                                         boundary,
                                         trajectories,
                                         thickness,
                                         plot_exits=plot_exits)

    # set up axes if plot_exits is true
    if plot_exits:  # pragma: no cover
        plt.gca().set_title('Transmitted exits')
        plt.gca().view_init(-164, -155)

    # Multiply the pass fraction by the weights in order to get the weights
    # of the passed trajectories
    refl_weights_pass = refl_weights * fresnel_pass_frac_refl
    trans_weights_pass = trans_weights * fresnel_pass_frac_trans

    # Subtract the passed weights from the weights before the interface
    # to find the weights of trajectores that are fresnel reflected
    # back into the sample
    refl_fresnel = refl_weights - refl_weights_pass
    trans_fresnel = trans_weights - trans_weights_pass

    # Calculate fraction that are successfully transmitted or reflected,
    # meaning not fresnel reflected back into the sample.
    # When running fresnel trajectories, we must calculate refl_frac and
    # trans_frac as fractions of total trajectories because known_outcomes
    # will change as we run run more fresnel trajectories through recursion
    if fresnel_traj:
        ntraj = len(refl_weights_pass)
        refl_frac = np.sum(refl_weights_pass) / ntraj
        trans_frac = np.sum(trans_weights_pass) / ntraj

    else:
        # calculate fraction that are successfully transmitted or reflected
        known_outcomes = np.sum(absorb_weights + refl_weights_pass
                                + trans_weights_pass)
        refl_frac = np.sum(refl_weights_pass) / known_outcomes
        trans_frac = np.sum(trans_weights_pass) / known_outcomes

    return (refl_frac, trans_frac, refl_weights_pass, trans_weights_pass,
            refl_fresnel, trans_fresnel, norm_vec_refl, norm_vec_trans)


def detect_corrected_traj(inc_pass_frac, n_sample, n_medium,
                          refl_indices, trans_indices,
                          refl_weights_pass, trans_weights_pass, trajectories,
                          boundary, thickness, detection_angle, EPS, kz0_rot,
                          kz0_refl):
    '''
    Corrects trajectories for detection aperture spanning angles less than
    or equal to the detection angle.

    Parameters
    ----------
    inc_pass_frac: 1d array (length: ntraj)
        weights of trajectories that pass through the medium-sample interface
    n_sample: float
        Refractive index of the sample
    n_medium: float
        Refractive index of the medium
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    refl_weights_pass: 1d array (length: ntraj)
        weights of reflected trajectories, corrected for fresnel reflection
        upon exit. Zero values indicate trajectory was totally internally
        reflected back into the sample, or not reflected in the first place
    trans_weights_pass: 1d array (length: ntraj)
        weights of transmitted trajectories, corrected for fresnel reflection
        upon exit. Zero values indicate trajectory was totally internally
        reflected back into the sample, or not transmited in the first place
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    thickness: float
        thickness of film or diameter of sphere
    detection_angle: float
        largest trajectory angle that can be detected. Detection angle
        is defined relative to the z-axis and must be between 0 and pi/2.
        If 0, this means that the detection range is infinitesimal.
        If pi/2, all backscattering angles are detected.
    EPS: float
        small number used to prevent a divide-by-zero error when calculating
        trans_det_frac and refl_det_frac
    kz0_rot : None or array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the
        surface. The array size is (1, ntraj).
    kz0_refl : None or array_like (structcol.Quantity [dimensionless])
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global
        coordinate system. The array size is (1, ntraj).

    Returns
    -------
    inc_refl_detected: 1d array (length: ntraj)
        detected weights of trajectories reflected at sample interface
    trans_detected: 1d array (length: ntraj)
        detected weights of transmitted trajectories
    refl_detected: 1d array (length: ntraj)
        detected weights of reflected trajectories
    trans_det_frac: float
        fraction of the successfully transmitted light that is detected
    refl_det_frac: float
        fraction of the successfully reflected light that is detected
    '''
    kz = trajectories.direction[2]
    ntraj = kz.shape[1]

    inc_refl = (1 - inc_pass_frac)  # fresnel reflection incident on sample

    # Calculate the detected weights of the trajectories reflected at the
    # sample interface.
    # TODO: the inc_refl_detected does not work for sphere case. Requires
    # more complicated math in detect_correct()
    if (kz0_rot is not None) and (kz0_refl is not None):
        kz0_refl = np.squeeze(kz0_refl)
        angles_from_kz0_refl = np.arccos(kz0_refl)
        # can't use detect_correct() because it uses get_angles(), which always
        # returns an angle that is always on the same side as the detector (the
        # angles returned are between 0 and np.pi/2 and those are the angles
        # that the detector can cover). Since in this case the Fresnel
        # reflected angles can be pointing in the transmission direction,
        # I manually eliminate the weights of the Fresnel reflected
        # trajectories that reflect outside of the detected angles
        # (including the trajectories that go towards the transmission
        # direction) and can never be detected.
        inc_refl_detected = inc_refl
        inc_refl_detected[angles_from_kz0_refl < np.pi-detection_angle] = 0
    else:
        inc_refl_detected = detect_correct(np.ones(ntraj), trajectories,
                                           inc_refl, n_medium, n_medium,
                                           boundary, thickness,
                                           detection_angle,
                                           init_dir=kz[0, :])

    # calculate the detected weights of the transmitted trajectories
    trans_detected = detect_correct(trans_indices, trajectories,
                                    trans_weights_pass,
                                    n_sample, n_medium, boundary,
                                    thickness, detection_angle)

    # calculate the detected weights of the reflected trajectories
    refl_detected = detect_correct(refl_indices, trajectories,
                                   refl_weights_pass,
                                   n_sample, n_medium, boundary,
                                   thickness, detection_angle)

    # calculate the fraction of the successfully transmitted light that is
    # detected
    trans_det_frac = (np.max([np.sum(trans_detected), EPS])
                      / np.max([np.sum(trans_weights_pass), EPS]))

    # calculate the fraction of the successfully reflected light that is
    # detected
    refl_det_frac = (np.max([np.sum(refl_detected), EPS]) /
                     np.max([np.sum(refl_weights_pass), EPS]))
    return (inc_refl_detected,
            trans_detected, refl_detected,
            trans_det_frac, refl_det_frac)


def distribute_ambig_traj_weights(refl_fresnel, trans_fresnel,
                                  refl_frac, trans_frac,
                                  refl_det_frac, trans_det_frac,
                                  refl_detected, trans_detected,
                                  stuck_weights, inc_refl_detected, boundary,
                                  detector):
    '''
    Distribute the stuck trajectory weights among reflected and transmitted

    Parameters
    ----------
    refl_fresnel: 1d array (length: ntraj)
        weights of reflected trajectories that are fresnel reflected
        back into the sample
    trans_fresnel: 1d array (length: ntraj)
        weights of transmitted trajectories that are fresnel reflected back
        into the sample
    refl_frac: float
        fraction of trajectory weights that are successfully reflected, meaning
        not fresnel reflected back into the sample
    trans_frac: float
        fraction of trajectory weights that are successfully transmitted,
        meaning not fresnel reflected back into the sample
    refl_det_frac: float
        fraction of the successfully reflected light that is detected
    trans_det_frac: float
        fraction of the successfully transmitted light that is detected
    refl_detected: 1d array (length: ntraj)
        detected weights of reflected trajectories
    trans_detected: 1d array (length: ntraj)
        detected weights of transmitted trajectories
    stuck_weights: 1d array (length: ntraj)
        weights of stuck trajectories
    inc_refl_detected: 1d array (length: ntraj)
        detected weights of trajectories reflected at sample interface
    boundary: string
        geometrical boundary, current options are 'film' or 'sphere'
    detector: boolean
        Set to true if you want to calculate reflection while using
        a goniometer detector (detector at a specified angle).

    Returns
    -------
    reflectance: float
        fraction of light reflected, including corrections for fresnel and
        detector
    transmittance: float
        fraction of light transmitted, including corrections for fresnel and
        detector
    refl_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel
        contributions
    trans_per_traj: 1d array (length:ntraj)
        transmittance distributed to each trajectory, including fresnel
        contributions
    '''
    ntraj = len(refl_fresnel)

    if boundary == 'film':
        # Stuck are 50/50 reflected/transmitted since they are randomized.
        # non-TIR fresnel are treated as new trajectories at the appropriate
        # interface. This means reversed R/T ratios for fresnel reflection
        # at transmission interface.
        extra_refl = (refl_fresnel * refl_frac
                      + trans_fresnel * trans_frac + stuck_weights * 0.5)
        extra_trans = (trans_fresnel * refl_frac
                       + refl_fresnel * trans_frac + stuck_weights * 0.5)

    if boundary == 'sphere':
        # TODO these approximations work best if run_fresnel_traj = True
        # otherwise should add option to use approximations for film

        # Stuck are 50/50 reflected/transmitted since they are randomized.
        # non-TIR fresnel are treated as new trajectories at the appropriate
        # interface. This means reversed R/T ratios for fresnel reflection at
        # transmission interface.
        # extra_refl = (refl_fresnel * refl_frac
        #               + trans_fresnel * trans_frac + stuck_weights * 0.5
        # extra_trans = (trans_fresnel * refl_frac
        #                + refl_fresnel * trans_frac + stuck_weights * 0.5
        extra_refl = 0.5*(refl_fresnel + trans_fresnel + stuck_weights)
        extra_trans = 0.5*(trans_fresnel + refl_fresnel + stuck_weights)

    if detector:
        # TODO make this work for specular angles by checking incident angle
        # assumes detection angle is not specular
        inc_refl_detected = 0

        # TODO check whether this makes sense for both film and sphere
        # TODO add a geometrical correction factor to deal with trans_fresnel
        # and stuck based on size of detector compared to backscattering
        # hemisphere
        extra_refl = refl_fresnel * refl_frac

    # calculate transmitted and reflected weights for each traj
    trans_weights = trans_detected + extra_trans * trans_det_frac
    refl_weights = (refl_detected + extra_refl * refl_det_frac
                    + inc_refl_detected)

    # divide by ntraj to get refl and trans per traj
    refl_per_traj = refl_weights / ntraj
    trans_per_traj = trans_weights / ntraj

    # sum to calculate reflectance and transmittance
    transmittance = np.sum(trans_per_traj)
    reflectance = np.sum(refl_per_traj)

    return reflectance, transmittance, refl_per_traj, trans_per_traj


def calc_refracted_direction(dir, pos, n1, n2, plot):
    '''
    TODO: make this work for transmission
    TODO: make this work for sphere

    refracts <dir> across an interface of two refractive indices, n1 and n2

    Parameters
    ----------
    dir : array
        initial direction vector
    pos : array
        position before trajectory exit
    n1 : float
        index of refraction of initial medium
    n2 : float
        index of refraction of medium to enter
    plot : boolean
        If True, plots the intersection point with film incident plane
        and k refraction

    Returns
    -------
    newdir : array
        refracted direction vector
    point : array
        point of intersection of direction vector and incident plane

    all arrays have length of number of trajectories
    '''

    # Find point on the vector around which to rotate.
    # We choose the point where the plane and line intersect
    # see Annie Stephenson lab notebook #3 pg 91 for derivation.
    with np.errstate(divide='ignore', invalid='ignore'):
        x_plane = -pos[2] / dir[2] * dir[0] + pos[0]
        y_plane = -pos[2] / dir[2] * dir[1] + pos[1]
    # any point on film incident plane is z = 0
    point = np.array([x_plane, y_plane, np.zeros_like(x_plane)])

    # Negate positive kz for reflection.
    # Remember that trajectories with positie kz can count as reflected
    # due to the imposed periodic boundary conditions imposed
    # in the trajectories in the film case.
    pos_kz = np.where(dir[2] > 0)
    dir[2][pos_kz] = -dir[2][pos_kz]

    # Find vector around which to rotate: k1 X - z.
    axis = normalize(np.array([-dir[1], dir[0], np.zeros_like(dir[0])]))

    # Calculate the angle with respect to the normal at which the trajectories
    # leave
    theta_1 = np.arccos(np.abs(dir[2]))
    theta_2 = refraction(theta_1, n1, n2).to_numpy()

    # Find angle by which to rotate trajectory direction.
    alpha = - (theta_2 - theta_1)

    # Rotate exit direction by refracted angle.
    newdir = rotate_refract(point, axis, dir, alpha)

    if plot:    # pragma: no cover
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('intesection points of exit vector and film plane')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(point[0], point[1], point[2], s=5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('k refraction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(dir[0], dir[1], dir[2], s=10, label='k1')
        ax.scatter(newdir[0], newdir[1], newdir[3], s=20, label='k2')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.legend()

    return newdir, point


def calc_indices_detected(indices, trajectories, det_theta, det_len, det_dist,
                          nsample, nmedium, plot):
    """
    TODO: make this work for transmission

    Detector function.

    Takes in exit event indices and removes indices that do not fit within the
    bounds of the detector, replacing the event number in the array with a
    zero.

    Parameters
    ----------
    indices: 1d array
        array of length ntraj where elements correspond to event number of an
        exit event for the trajectory corresponding to the index of the
        array. An element value of zero means that there was no exit
        event for the trajectory.
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    det_theta: float-like
        angle between the normal to the sample (-z axis) and the center of the
        detector
    det_len: float-like
        side length of the of the detector, assuming it is a square
    det_dist: float-like
        distance from the sample to the detector
    nsample: float
        refractive index of the sample
    nmedium: float
        refractive index of the medium
    plot: boolean
        if True, will plot exit and detected trajectories

    Returns
    -------
    indices_detected: 1d array
        array of same shape as indices, where elements corresponding to
        trajectories that did not make it into the detector are replaced with
        zero.

    """
    # get variables we need from trajectories
    x, y, z = trajectories.position

    # detector parameters
    if isinstance(det_theta, sc.Quantity):
        det_theta = det_theta.to('radians').magnitude
    if isinstance(det_dist, sc.Quantity):
        det_dist = det_dist.to_preferred().magnitude
    if isinstance(det_len, sc.Quantity):
        det_len = det_len.to_preferred().magnitude

    # coordinates and directions at exit events for all trajectories
    pos0 = select_events(trajectories.position[:, 1:, :], indices)
    dir0 = select_events(trajectories.direction, indices)

    # Calculate new directions from refraction at exit interface.
    dir, pos = calc_refracted_direction(dir0, pos0, nsample, nmedium,
                                        plot=False)

    # Get the radius of the detection hemisphere.
    det_rad = np.sqrt(det_dist**2 + (det_len / 2) ** 2)

    # Get x_min, x_max, y_min, y_max of detector based on geometry.
    # See pg 86 in Annie Stephenson lab notebook #3 for details.
    delta_x = det_len * np.cos(det_theta)
    x_center = det_dist * np.sin(det_theta)
    x_min = x_center - delta_x / 2
    x_max = x_center + delta_x / 2

    delta_y = det_len
    y_center = 0
    y_min = y_center - delta_y / 2
    y_max = y_center + delta_y / 2

    # Solve for the intersection of the scattering hemisphere at the detector
    # arm length and the exit trajectories using parameterization.
    # See Annie Stephenson lab notebook #3, pg 18 for details
    a = np.sum(dir**2, axis=0)
    b = 2 * np.sum(dir * pos, axis=0)
    c = np.sum(pos**2, axis=0) - det_rad ** 2
    t_p = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    t_m = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    if det_theta < np.pi / 2:
        t = t_p
    else:
        t = t_m

    intersect = pos + t * dir

    # check whether trajectory positions at detector hemisphere fall within
    # the detector limits, and update indices_detected to reflect this
    indices_detected = np.zeros(indices.size)
    int_detected = np.zeros_like(intersect)
    for i in range(indices.size):
        if (intersect[0, i] < x_max and intersect[0, i] > x_min)\
           and (intersect[1, i] < y_max and intersect[1, i] > y_min)\
           and (intersect[2, i] < 0):
            indices_detected[i] = indices[i]
            int_detected[:, i] = intersect[:, i]

    if plot:    # pragma: no cover
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('exit and detected trajectories')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-1.2 * det_rad, 1.2 * det_rad])
        ax.set_ylim([-1.2 * det_rad, 1.2 * det_rad])
        ax.set_zlim([-1.2 * det_rad, 1.2 * det_rad])
        ax.scatter(x, y, z, s=5)  # plot last position in film before exit
        ax.scatter(intersect[0], intersect[1], intersect[2],
                   s=3, c='b', label='exit traj')
        ax.scatter(int_detected[0], int_detected[1], int_detected[2],
                   s=20, label='detected traj')
        ax.view_init(elev=-148., azim=-112)
        plt.legend()

    return indices_detected


def select_tir_traj(inarray, events):
    '''
    selects a given value (inarray) for all trajectories,
    replacing that value with the negative of the previous one
    for TIR'd trajectories

    intended to work only for films
    '''
    ntraj = events.size
    nevents = inarray.shape[0]

    valid_events = (events > 0)
    ev = events[valid_events].astype(int) - 1
    tr = np.where(valid_events)[0]
    outarray = np.zeros((nevents, ntraj))
    outarray[ev, tr] = inarray[ev, tr]
    outarray[0, :] = inarray[0, :]
    # logic for negating TIR events below is buggy. Idea is to flip the sign of
    # the z component for both the direction and position, but only after TIR
    # events. As written, though, the code flips the signs of all the
    # directions and all positions after event 2, even if no TIR.
    for i in range(1, nevents):
        nonzero = outarray[i-1, :] != 0
        nonzero = nonzero.astype(int)
        ind = np.where(nonzero == 1)[0]
        outarray[i, ind] = -inarray[i, ind]

        # Find where still zeros and replace those with inarray.
        ind_zeros = np.where(outarray[i, :] == 0)[0]
        outarray[i, ind_zeros] = inarray[i, ind_zeros]
    return outarray


def shift_traj_tir(trajectories, tir_indices):
    '''
    Change the z-position and z-direction for all TIR trajectories to be
    the TIR direction, and keep same for all other trajectories
    '''
    # Select kz and z0 for all trajectories,
    # changing them for TIR events to be the negative of the previous event
    kz_incl_tir_refl = select_tir_traj(trajectories.direction[2, :, :],
                                       tir_indices)
    z0_incl_tir_refl = select_tir_traj(trajectories.position[2, :, :],
                                       tir_indices)

    # Set the kz and z0 in the trajectories object
    trajectories.position[2, :, :] = z0_incl_tir_refl
    trajectories.direction[2, :, :] = kz_incl_tir_refl


def calc_refl_trans(sim,
                    thickness,
                    z_low=0,
                    detection_angle=np.pi / 2,
                    n_front=None,
                    n_back=None,
                    return_extra=False,
                    run_fresnel_traj=False,
                    fresnel_traj=False,
                    call_depth=0,
                    max_call_depth=20,
                    max_stuck=0.01,
                    plot_exits=False,
                    detector=False,
                    det_theta=None,
                    det_len=None,
                    det_dist=None,
                    plot_detector=False,
                    save_stuck_weights=False,
                    rng=None):
    """
    Calculates the weight fraction of reflected and transmitted trajectories
    (reflectance and transmittance).Identifies which trajectories are reflected
    or transmitted, and at which scattering event. Includes corrections for
    Fresnel reflections and for a detector.

    Parameters
    ----------
    sim : `sc.Simulation` object
        Monte Carlo simulation with results for trajectories
    thickness : float (structcol.Quantity [length])
        thickness of film or diameter of sphere
    z_low : float (structcol.Quantity [length])
        Initial z-position of sample closest to incident light source.
        Should normally be set to 0.
    detection_angle: float
        Range of angles of detection. Only the packets that come out of the
        sample within this range will be detected and counted. Should be
        0 < detection_angle <= pi/2, where 0 means that no angles are detected,
        and pi/2 means that all the backscattering angles are detected.
    n_front: float (structcol.Quantity [dimensionless])
        Refractive index of the front cover of the sample (default None)
    n_back: float (structcol.Quantity [dimensionless])
        Refractive index of the back cover of the sample (default None)
    return_extra: boolean
        determines whether to return a host of extra variables that are used
        for additional calculations using the trajectories
    run_fresnel_traj: boolean
        If set to True, function will calculate new trajectories for weights
        that are fresnel reflected back into the sphere upon exit (There is
        almost always at least some small weight that is reflected back into
        sphere). If set to False, fresnel reflected trajectories are evenly
        distributed to reflectance and transmittance.
    fresnel_traj: boolean
        This argument is not intended to be set by the user. Its purpose is to
        keep track of whether calc_refl_trans_sphere() is running for the
        trajectories initially being sent into the sphere or for the fresnel
        reflected (tir) trajectories that are trapped in the sphere. It's
        default value is False, and it is changed to True when
        calc_refl_trans_sphere() is recursively called for calculating the
        reflectance from fresnel reflected trajectories
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
    max_stuck:float
        The maximum weight of stuck trajectories to leave in the sample
        without creating new trajectories to rerun. This argument is only used
        if run_fresnel_traj is True.
    plot_exits: boolean
        If set to True, function will plot the last point of trajectory inside
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits,
        making one plot for reflection and one plot for transmission
    detector: boolean
        Set to true if you want to calculate reflection while using a
        goniometer detector (detector at a specified angle).
        If True, must also specify det_theta, det_len, and det_dist.
    det_theta: float-like
        angle between the normal to the sample (-z axis) and the center of the
        detector
    det_len: float-like
        side length of the of the detector, assuming it is a square
    det_dist: float-like
        distance from the sample to the detector
    plot_detector: boolean
        if True, will plot refraction plots and exit and detected trajectories
    save_stuck_weights: boolean (optional)
        default is False. If set to True, saves a file with the stuck weight
        fraction. Useful for testing whether event number is sufficiently high.
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package.  The RNG is needed only when run_fresnel_traj is True.

    Returns
    -------
    refl_trans_result: tuple
        contains a tuple of other variables. The contents of refl_trans_result
        are different based on whether return_extra is set to True or False.

        Returns if return_extra is False
        --------------------------------
        reflectance: float
            Fraction of reflected trajectories, including the Fresnel
            correction but not considering the range of the detector.
        transmittance: float
            Fraction of transmitted trajectories, including the Fresnel
            correction but not considering the range of the detector.

        Returns if return_extra is True
        -------------------------------
        (refl_indices, trans_indices, inc_refl_detected/ntraj,
        refl_weights_pass/ntraj, trans_weights_pass/ntraj, trans_frac,
        refl_frac, refl_fresnel/ntraj, trans_fresnel/ntraj, norm_vec_refl,
        norm_vec_trans, reflectance, transmittance)
            These variables can be used to do various extra calculations using
            the trajectories

    Note
    ----
        absorptance of the sample can be found by 1 - reflectance -
        transmittance

    """
    n_med = sim.model.index_medium(sim.wavelen)
    n_samp = sim.model.index_external(sim.wavelen)

    boundary = sim.boundary
    fine_roughness = sim.fine_roughness

    if fine_roughness > 0:
        n_matrix = sim.model.index_matrix(sim.wavelen).to_numpy()
        try:
            n_particle = sim.model.sphere.n(sim.wavelen).to_numpy()
            # TODO use xr.where to handle vectorization over wavelength
            if n_particle < n_matrix:
                n_tir = (fine_roughness
                         * n_matrix + (1 - fine_roughness) * n_samp)
            else:
                n_tir = (fine_roughness
                         * n_particle + (1 - fine_roughness) * n_samp)
        except AttributeError:
            warnings.warn("n_particle and n_matrix not specified; using "
                          "n_sample for n_tir instead of roughness correction",
                          category=UserWarning, stacklevel=2)
            n_tir = n_samp
    else:
        n_tir = n_samp

    # set up values as floats to be used throughout function
    ntraj = sim.ntrajectories
    if isinstance(z_low, sc.Quantity):
        z_low = z_low.to_preferred().magnitude
    if isinstance(thickness, sc.Quantity):
        thickness = thickness.to_preferred().magnitude

    # Find event indices for each trajectory outcome
    exits = find_exits(n_tir, n_med, thickness, z_low, boundary, sim.traj)

    # convert to numpy for further processing
    refl_indices = exits["refl"].to_numpy().squeeze()
    trans_indices = exits["trans"].to_numpy().squeeze()
    stuck_indices = exits["stuck"].to_numpy().squeeze()
    tir_indices = exits["tir"].to_numpy().squeeze()
    trajectories = mc.NumpyTrajectory(sim.traj)
    kz0_rot = trajectories.kz0_rot
    kz0_refl = trajectories.kz0_refl
    # until this function is refactored to use xarray, convert all indices to
    # numpy arrays
    n_med = n_med.to_numpy()
    # drop VOLFRAC dimension, which will be included in all effective index
    # calculations.
    if sc.Coord.VOLFRAC in n_samp.coords:
        n_samp = n_samp.isel({sc.Coord.VOLFRAC: 0}, drop=True)
        n_tir = n_tir.isel({sc.Coord.VOLFRAC: 0}, drop=True)
        n_samp = n_samp.to_numpy()
        n_tir = n_tir.to_numpy()

    # Correct indices to account for detector.
    # TODO make this work for trans_indices as well
    if detector:
        shift_traj_tir(trajectories, tir_indices)
        refl_indices = calc_indices_detected(refl_indices, trajectories,
                                             det_theta, det_len, det_dist,
                                             n_samp, n_med,
                                             plot_detector)

    # Find fraction and direction of light that enters sample.
    init_dir, inc_pass_frac = fresnel_correct_enter(n_med, n_front, n_tir,
                                                    boundary, thickness,
                                                    trajectories, fresnel_traj,
                                                    kz0_rot)

    # Calculate outcome weights of trajectories.
    (refl_weights,
     trans_weights,
     stuck_weights,
     absorb_weights) = calc_outcome_weights(inc_pass_frac, refl_indices,
                                            trans_indices, stuck_indices,
                                            trajectories.weight)

    # Correct for Fresnel reflection upon exiting.
    (refl_frac, trans_frac,
     refl_weights_pass,
     trans_weights_pass,
     refl_fresnel, trans_fresnel,
     norm_vec_refl, norm_vec_trans) = fresnel_correct_exit(n_tir, n_med,
                                                           n_front, n_back,
                                                           refl_indices,
                                                           trans_indices,
                                                           refl_weights,
                                                           trans_weights,
                                                           absorb_weights,
                                                           boundary, thickness,
                                                           trajectories,
                                                           fresnel_traj,
                                                           plot_exits)

    # Correct for effect of detection angle upon leaving sample.
    (inc_refl_detected,
     trans_detected, refl_detected,
     trans_det_frac, refl_det_frac) = detect_corrected_traj(inc_pass_frac,
                                                            n_samp, n_med,
                                                            refl_indices,
                                                            trans_indices,
                                                            refl_weights_pass,
                                                            trans_weights_pass,
                                                            trajectories,
                                                            boundary,
                                                            thickness,
                                                            detection_angle,
                                                            EPS, kz0_rot,
                                                            kz0_refl)

    total_stuck = np.sum(refl_fresnel + trans_fresnel + stuck_weights) / ntraj

    # If we want to run Fresnel reflected as new trajectories
    # (only implemented for sphere boundary)
    if (run_fresnel_traj and call_depth < max_call_depth
        and total_stuck > max_stuck):

        # Calculate the reflectance and transmittance per trajectory
        # without Fresnel weights.
        refl_per_traj_nf = (refl_detected + inc_refl_detected) / ntraj
        trans_per_traj_nf = trans_detected / ntraj


        # Rerun Fresnel reflected components of trajectories.
        # We have to pass the original DataArrays for n_medium and n_sample
        # because run_sphere_fresnel_traj sets up a new simulation
        (reflectance,
         transmittance,
         refl_per_traj,
         trans_per_traj) = run_sphere_fresnel_traj(refl_per_traj_nf,
                                                   trans_per_traj_nf,
                                                   refl_fresnel,
                                                   trans_fresnel,
                                                   stuck_weights,
                                                   refl_indices,
                                                   trans_indices, tir_indices,
                                                   sim, thickness, z_low,
                                                   max_stuck,
                                                   call_depth, plot_exits,
                                                   rng=rng)
    else:
        # Distribute ambiguous trajectory weights.
        # Note that when run_fresnel_traj == True, this statement is only
        # executed at the final recursive call of cal_refl_trans(). At this
        # call, the reflectance returned here will be tiny, much less than 1%,
        # but it will be added to the reflectance calculated from all of the
        # previous calls, so this is expected and desired.
        (reflectance,
         transmittance,
         refl_per_traj,
         trans_per_traj) = distribute_ambig_traj_weights(refl_fresnel,
                                                         trans_fresnel,
                                                         refl_frac, trans_frac,
                                                         refl_det_frac,
                                                         trans_det_frac,
                                                         refl_detected,
                                                         trans_detected,
                                                         stuck_weights,
                                                         inc_refl_detected,
                                                         boundary, detector)

    # Return desired results.
    if return_extra:
        refl_trans_result = (refl_indices,
                             trans_indices,
                             stuck_indices,
                             tir_indices,
                             inc_refl_detected / ntraj,
                             refl_weights_pass / ntraj,
                             trans_weights_pass / ntraj,
                             refl_per_traj,
                             trans_per_traj,
                             trans_frac,
                             refl_frac,
                             refl_fresnel / ntraj,
                             trans_fresnel / ntraj,
                             reflectance,
                             transmittance,
                             norm_vec_refl,
                             norm_vec_trans)

    else:
        refl_trans_result = reflectance, transmittance

    # For performance testing, print the stuck weights to text file
    if save_stuck_weights:
        if boundary == 'film':  # pragma: no cover
            with open('stuck_weights_small.txt', 'a') as f:
                print(total_stuck, file=f)
            f.close()

    return refl_trans_result


def run_sphere_fresnel_traj(refl_per_traj_nf, trans_per_traj_nf,
                            refl_fresnel, trans_fresnel, stuck_weights,
                            refl_indices, trans_indices,
                            tir_indices, sim, thickness, z_low,
                            max_stuck, call_depth, plot_exits,
                            rng=None):
    '''
    For the sphere case, there are many trajectories that are totally
    internally reflected or partially reflected back into the sample

    This function takes the weights of the trajectory components reflected back
    into the sample (whether it's the whole weight through tir, or just partial
    through fresnel) and re-runs them as new trajectories, until most of them
    exit through reflectance, transmittance, or are absorbed. Note that it does
    not re-run stuck trajectories (trajectories that have not yet attempted an
    exit.)

    Parameters
    ----------
    refl_per_traj_nf: 1d array (length: ntraj)
        reflectance per trajectory, not counting trajectory weights fresnel
        reflected back into the sample
    trans_per_traj_nf: 1d array (length: ntraj)
        transmittance per trajectory, not counting trajectory weights fresnel
        reflected back into the sample
    refl_fresnel: 1d array (length: ntraj)
        weights of reflected trajectories that are fresnel reflected
        back into the sample
    trans_fresnel: 1d array (length: ntraj)
        weights of transmitted trajectories that are fresnel reflected back
        into the sample
    stuck_weights: 1d array (length: ntraj)
        weights of stuck trajectories
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    tir_indices: 1d array (length: ntraj)
        array of event indices for totally internally reflected trajectories
    stuck_indices: 1d array (length: ntraj)
        array of event indices for stuck trajectories
    sim : `sc.Simulation` object
        Monte Carlo simulation with results
    thickness : float (structcol.Quantity [length])
        thickness of film or diameter of sphere
    z_low : float (structcol.Quantity [length])
        Initial z-position of sample closest to incident light source.
        Should normally be set to 0.
    max_stuck:float
        The maximum weight of stuck trajectories to leave in the sample
        without creating new trajectories to rerun. This argument is only used
        if run_fresnel_traj is True.
    call_depth: int
        This argument is not intended to be set by the user. Call_depth keeps
        track of the recursion call_depth. It's default value is 0, and upon
        each recursive call to calc_refl_trans_sphere(), it is increased by 1.
    plot_exits: boolean
        If set to True, function will plot the last point of trajectory inside
        the sphere, the first point of the trajectory outside the sphere,
        and the point on the sphere boundary at which the trajectory exits,
        making one plot for reflection and one plot for transmission
    rng: numpy.random.Generator object (default None) random number generator.
        If not specified, use the default generator initialized on loading the
        package

    Returns
    -------
    reflectance_fresnel + reflectance_no_fresnel: float
        new reflectance after re-running fresnel reflected trajectories. Adds
        the previous reflectance (reflectance_no_fresnel) and the new addition
        to the reflectance (reflectance_fresnel)
    transmittance_fresnel + transmittance_no_fresnel: float
        new transmittance after re-running fresnel reflected trajectories. Adds
        the previous transmittance (transmittance_no_fresnel) and the new
        addition to the transmittance (transmittance_fresnel)

    '''

    if rng is None:
        rng = sc.rng

    # Set up values to use throughout function.
    trajectories = mc.NumpyTrajectory(sim.traj)
    if isinstance(z_low, sc.Quantity):
        z_low = z_low.to_preferred().magnitude
    diameter = thickness
    if isinstance(diameter, sc.Quantity):
        diameter = diameter.to_preferred().magnitude

    radius = diameter / 2
    kx, ky, kz = trajectories.direction
    nevents = trajectories.position.shape[1]
    ntraj = trajectories.position.shape[2]

    # New weights are the weights that are fresnel reflected
    # back into the sphere.
    weights_fresnel = xr.DataArray(np.ones((nevents+1, ntraj)),
                                   coords = {"event": range(nevents+1),
                                             "trajectory": range(ntraj)})
    weights_fresnel[:,:] = refl_fresnel + trans_fresnel + stuck_weights

    # Add refl and trans indices for all attempted or successful exit indices
    # (this works because refl_indices is 0 for any trajectory that is not
    # reflected out of the sample; similar for trans and tir; also no
    # trajectory should have a nonzero element in more than one of the arrays)
    indices = refl_indices + trans_indices + tir_indices

    # Get positions outside of sphere boundary from after exit
    pos1 = select_events(trajectories.position[:, 1:, :], indices)

    # get positions inside sphere boundary from before exit
    pos0 = select_events(trajectories.position[:, :-1, :], indices)

    # make sure none of the coordinates are infinite
    pos0, pos1 = inf_to_large(pos0, pos1, radius)

    # get radius vector to subtract from select_z
    select_radius = select_events(radius * np.ones((nevents, ntraj)), indices)

    # shift z for intersect finding
    pos0[2] = pos0[2] - select_radius
    pos1[2] = pos1[2] - select_radius

    # get positions at sphere boundary from exit
    intersect = find_vec_sphere_intersect(pos0, pos1, radius)

    # shift z back to global coordinates
    intersect[2] = intersect[2] + select_radius

    # define vectors for reflection inside sphere
    k_out = select_events(trajectories.direction, indices)
    normal = np.array([intersect[0] / radius, intersect[1] / radius,
                       (intersect[2] - select_radius) / radius])

    # calculate reflected direction inside sphere
    k_refl = rotate_reflect(k_out, normal)

    # set the initial directions as the reflected directions
    directions = xr.DataArray(np.zeros((3, nevents, ntraj)),
                              coords = {"component": ["x", "y", "z"],
                                        "event": range(nevents),
                                        "trajectory": range(ntraj)})
    directions[:, 0, :] = k_refl

    # set the initial positions at the sphere boundary
    positions = xr.DataArray(np.zeros((3, nevents + 1, ntraj)),
                             coords = {"component": ["x", "y", "z"],
                                       "event": range(nevents+1),
                                       "trajectory": range(ntraj)})
    positions[:, 0, :] = intersect

    # TODO: get rid of trajectories whose initial weights are 0
    # find indices where initial weights are 0
#        indices = np.where(weights_fresnel[0,:] == 0)
#        if indices[0].size > 0:
#            weights_fresnel = np.delete(weights_fresnel,indices)
#            positions = np.delete(positions, indices, axis = 0)
#            directions = np.delete(directions, indices,axis = 0)

    # create new simulation object and insert our existing trajectories
    # note that we don't pass the rng to the Simulation object because it would
    # generate random numbers for the initial trajectories, which we already
    # have calculated. The generation of new initial trajectories would change
    # the state of the rng, making it hard to compare to results of regression
    # tests. Instead we pass the rng to the run() method
    sim_fresnel = mc.Simulation(sim.model, sim.wavelen, nevents, ntraj,
                                sim.boundary, sample_diameter = thickness*2)
    trajectories_fresnel = xr.Dataset({"position": positions,
                                       "direction": directions,
                                       "weight": weights_fresnel})
    # insert our previously calculated values into the simulation
    n_sample = sim.model.index_external(sim.wavelen)
    sim_fresnel.traj = trajectories_fresnel.expand_dims(n_sample.coords).copy()

    # Run photons
    sim_fresnel.run(rng=rng)

    # Calculate reflection and transmition
    (_, trans_indices_fresnel, _, _, _, _, _,
     refl_per_traj_fresnel,
     trans_per_traj_fresnel,
     _, _, _, _,
     reflectance_fresnel,
     transmittance_fresnel,
     norm_refl_f, norm_trans_f) = calc_refl_trans(sim_fresnel, thickness,
                                                  z_low = z_low,
                                                  plot_exits = plot_exits,
                                                  run_fresnel_traj = True,
                                                  fresnel_traj = True,
                                                  call_depth = call_depth + 1,
                                                  max_stuck = max_stuck,
                                                  return_extra = True,
                                                  rng=rng)

    # Calculate reflectance and transmittance without fresnel
    reflectance_no_fresnel = np.sum(refl_per_traj_nf)
    transmittance_no_fresnel = np.sum(trans_per_traj_nf)

    return (reflectance_fresnel + reflectance_no_fresnel,
            transmittance_fresnel + transmittance_no_fresnel,
            refl_per_traj_fresnel + refl_per_traj_nf,
            trans_per_traj_fresnel + trans_per_traj_nf)


def rotate_reflect(k_out, normal):
    '''
    Find the reflection vector given an initial vector and the normal vector
    to the surface at the reflection.

    see http://mathworld.wolfram.com/Reflection.html for more details

    Paremeters
    ----------
    k_out: array
        3d vector to be reflected. Usually refers to vector describing
        trajectory direction as it exits sample.
    normal: array
        3d vector normal to the surface on which to reflect

    Returns
    -------
    k_refl: array
        3d vector that is reflected.
    '''

    dot_k_out_normal = np.sum(k_out * normal, axis=0)
    k_refl = k_out - 2 * dot_k_out_normal * normal

    return k_refl


def normalize_refl_goniometer(refl, det_dist, det_len, det_theta):
    '''
    calculates the reflectance renormalized for goniometer measurement

    This normalization scheme makes several key assumptions:
    (1) The area of the detection hemisphere spanned by the detector aperture
        is a square. As the detector size approaches the diameter of the
        detection hemisphere, this assumption becomes worse. In reality, the
        detection hemisphere area spanned by the detector is the projection of
        a square on the sphere surface, which looks like a curved square patch.
    (2) The reference reflector (maximum reflectance) is that of a lambertian
        reflector, meaning the reflectance is uniform over the detection
        hemisphere and that the integrated reflected intensity is equal to the
        intensity of the incident beam. This means that if the sample has a
        specular component, the reflectance could be greater than one for the
        specular angle.

    The normalization formula:

    refl_renormlized = ((area of detection hemisphere)/(area detected)
                        * reflectance)

    We are just scaling up the reflectance based on the area detected relative
    to the total possible area that can be detected.

    vocab
    -----
    detection hemisphere: the hemisphere surrounding the sample, having radius
                          of the detector distance. In the case of reflectance
                          measurements, it is a reflection hemisphere.
    '''

    # calculate the area of the detection hemisphere divided by the area of the
    # detector
    area_frac = (2 * np.pi * det_dist ** 2) / det_len ** 2

    #now multiply by cosine to scale with a lambertian reflector
    coeff = area_frac * np.cos(det_theta)

    # multiply the area fraction by the input reflectance
    refl_renormalized = coeff * refl

    return refl_renormalized

def calc_haze(trajectories, trans_per_traj, transmittance, trans_indices,
              cutoff_angle=sc.Quantity(4.5, 'deg')):
    '''
    Calculates haze, the fraction of diffuse transmittance over total
    transmittance:

    H = T_{diffuse}/T_{total}

    For diffuse transmittance, we use a cutoff of 4.5 degrees, meaning we
    call anything scattered more than 4.5 degrees from the forward direction
    diffuse transmittance. The 4.5 degree cuttoff comes from:

    W. B. Rogers, M. Corbett, S. Magkiriadou, P. Guarillof, V. N. Manoharan.
    Optical Materials Express. 4,12, 2621 (2014)

    Parameters
    ----------
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    trans_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel
        contributions
    transmittance: float
        fraction of light transmitted, including corrections for fresnel and
        detector
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    cutoff_angle: float-like, sc.Quantity
        angle greater than which light is considered diffusely transmitted

    Returns
    -------
    haze: float
        the fraction of diffuse transmittance over total transmittance, can
        have values from 0 to 1.
    '''

    # note: only implemented for film currently

    kz = trajectories.direction[2]
    cosz = select_events(kz, trans_indices)

    # calculate angle to normal from cos_z component (only want magnitude)
    angles = sc.Quantity(np.arccos(np.abs(cosz)),'')

    trans_ind_forward = np.where(angles < cutoff_angle)[0]

    trans_forward = np.sum(trans_per_traj[trans_ind_forward])

    trans_diffuse = transmittance - trans_forward

    haze = trans_diffuse / transmittance

    return haze
