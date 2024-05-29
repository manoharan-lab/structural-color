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
from . import model
from . import montecarlo as mc
from . import refraction
from . import normalize
from . import select_events
import numpy as np
from numpy.random import random as random
import structcol as sc
import matplotlib.pyplot as plt
import warnings

EPS = 1.e-9


def inf_to_large(x0, y0, z0, x1, y1, z1, radius):
    '''
    convert two sets of trajectory coordinates from infinite values to a large 
    value instead

    Parameters
    ----------
    x0: array
        1st cooordinate of first point
    y0: array
        2nd cooordinate of first point
    z0: array
        3rd cooordinate of first point
    x1: array
        1st cooordinate of second point
    y1: array
        2nd cooordinate of second point
    z1: array
        3rd cooordinate of second point
    radius: float
        value that scales the large value that replaces the infinite value.
    
    Returns
    -------
    x0, y0, z0, x1, y1, z1: arrays
        coordinates with infinite values replaced with large ones
    '''
    x0[x0 > 1e20 * radius] = 100 * radius
    y0[y0 > 1e20 * radius] = 100 * radius
    z0[z0 > 1e20 * radius] = 100 * radius
    x1[x1 > 1e20 * radius] = 100 * radius
    y1[y1 > 1e20 * radius] = 100 * radius
    z1[z1 > 1e20 * radius] = 100 * radius
    x0[x0 < -1e20 * radius] = -100 * radius
    y0[y0 < -1e20 * radius] = -100 * radius
    x1[x1 < -1e20 * radius] = -100 * radius
    y1[y1 < -1e20 * radius] = -100 * radius
    z1[z1 < -1e20 * radius] = -100 * radius

    return x0, y0, z0, x1, y1, z1


def find_vec_sphere_intersect(x0, y0, z0, x1, y1, z1, radius):
    """
    Analytically solves for the point at which an exiting trajectory
    intersects with the boundary of the sphere

    Parameters
    ----------
    x0: 1d array
        initial x-position of each trajectory before exit
    y0: 1d array
        initial y-position of trajectory before exit
    z0: 1d array
        initial z-position of trajectory before exit
    x1: 1d array
        x-position of trajectory after exit
    y1: 1d array
        y-position of trajectory after exit
    z1: 1d array
        z-position of trajectory after exit
    radius : float
        radius of spherical boundary

    Returns
    ----------
    pos_int: 2d array
        position where exit trajectories intersect the boundary of the sphere,
        with shape: 3 (coordinate), ntrajectories

    """

    # Find k vector from point inside and outside sphere.
    kx, ky, kz = normalize(x1 - x0, y1 - y0, z1 - z0)

    # Solve for intersection of k with sphere surface using parameterization.
    # There will be two solutions for each k vector, corresponding to the two
    # points where a line intersects a sphere.
    # See http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    # or Annie Stephenson lab notebook #3, pg 18 for details.
    a = kx ** 2 + ky ** 2 + kz ** 2
    b = 2 * (kx * x0 + ky * y0 + kz * z0)
    c = x0 ** 2 + y0 ** 2 + z0 ** 2 - radius ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t_m = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    np.seterr(divide='warn', invalid='warn')
    x_int_p = x0 + t_p * kx
    y_int_p = y0 + t_p * ky
    z_int_p = z0 + t_p * kz

    x_int_m = x0 + t_m * kx
    y_int_m = y0 + t_m * ky
    z_int_m = z0 + t_m * kz

    # find the distances between the each solution point and the trajectory
    # point outside the sphere
    # casts nans to zero
    dist_p = np.nan_to_num((x_int_p - x1) ** 2 + (y_int_p - y1) ** 2
                           + (z_int_p - z1) ** 2)
    dist_m = np.nan_to_num((x_int_m - x1) ** 2 + (y_int_m - y1) ** 2
                           + (z_int_m - z1) ** 2)

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
    pos_int = np.zeros((3, len(x0)))
    pos_int[:, ind_p] = x_int_p[ind_p], y_int_p[ind_p], z_int_p[ind_p]
    pos_int[:, ind_m] = x_int_m[ind_m], y_int_m[ind_m], z_int_m[ind_m]

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

    # Get the angles between trajectories and the normal,
    # along with their normal vectors.
    theta_1, norm = get_angles(indices, boundary, trajectories, thickness)

    # Take cross product of k1 and sphere normal vector to find vector
    # to rotate around.
    k1 = select_events(trajectories.direction, indices)
    kr = np.transpose(np.cross(np.transpose(k1), np.transpose(norm)))

    # Use Snell's law to calculate angle between k2 and normal vector.
    # theta_2 is nan if photon is totally internally reflected.
    theta_2 = refraction(theta_1, n_inside, n_outside)

    # The angle to rotate around is theta_2-theta_1.
    theta = theta_2 - theta_1

    # Perform the rotation.
    _, _, k2z = rotate_refract(norm[0] * thickness / 2,
                               norm[1] * thickness / 2,
                               norm[2] * thickness / 2,
                               kr[0], kr[1], kr[2],
                               k1[0], k1[1], k1[2], theta)

    # If k2z is nan, leave uncorrected.
    # Since nan means the trajectory was totally internally reflected, the
    # exit kz doesn't matter, but in order to calculate the fresnel reflection
    # back into the sphere, we still need it to count as a potential exit
    # hence we leave the kz unchanged.

    return k2z


def rotate_refract(a, b, c, u, v, w, kx_1, ky_1, kz_1, alpha):
    '''
    Rotates vector <k1> by angle alpha about the unit vector <uvw>,
    where (a,b,c) is a point on the vector we are rotating about.

    Parameters
    ----------
    a: 1d array
        x-coordinate of point on the vector <uvw> to rotate about
    b: 1d array
        y-coordinate of point on the vector <uvw> to rotate about
    c: 1d array
        z-coordinate of point on the vector <uvw> to rotate about 
    u: 1d array
        x-component of vector to rotate about
    v: 1d array
        y-component of vector to rotate about
    w: 1d array
        z-component of vector to rotate about
    kx_1: 1d array
        x-component of vector to rotate
    ky_1: 1d array
        y-component of vector to rotate
    kz_1: 1d array
        z-component of vector to rotate
    alpha: 1d array
        angle by which to rotate <k1>

    length of each of these arrays is number of trajectories being rotated

    Returns
    -------
    kx_2: 1d array
        x-component of vector to rotate
    ky_2: 1d array
        y-component of vector to rotate
    kz_2: 1d array
        z-component of vector to rotate

    Notes
    -----
    This rotation matrix was derived by Glenn Murray
    and it's derivation is explained here: 
    https://sites.google.com/site/glennmurray/Home/rotation-matrices
    -and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
    '''

    # (x,y,z) is a physical point on the k vector
    # we find the point by adding a,b,c to the normalized k vector
    x = a + kx_1
    y = b + ky_1
    z = c + kz_1

    # rotation matrix
    x_rot = (a * (v ** 2 + w ** 2) - u * (b * v + c * w - u * x - v * y - w * z)) * (1 - np.cos(alpha)) + x * np.cos(alpha) + (-c * v + b * w - w * y + v * z) * np.sin(alpha)
    y_rot = (b * (u ** 2 + w ** 2) - v * (a * u + c * w - u * x - v * y - w * z)) * (1 - np.cos(alpha)) + y * np.cos(alpha) + (c * u - a * w + w * x - u * z) * np.sin(alpha)
    z_rot = (c * (u ** 2 + v ** 2) - w * (a * u + b * v - u * x - v * y - w * z)) * (1 - np.cos(alpha)) + z * np.cos(alpha) + (-b * u + a * v - v * x + u * y) * np.sin(alpha)

    # to recover the k vector from the point rotated in space, we must subtract
    # a,b,c
    kx_2 = x_rot - a
    ky_2 = y_rot - b
    kz_2 = z_rot - c

    return kx_2, ky_2, kz_2


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
        x, y, z = trajectories.position
        radius = thickness / 2

        # Subtract radius from z to center the sphere at 0,0,0. This makes the
        # following calculations much easier.
        z = z - radius

        # If incident light..
        if init_dir is not None:
            # TODO implement capability for diffuse illumination of sphere
            kx, ky, kz = trajectories.direction

            # Select initialized trajectory positions.
            select_kx1 = select_events(kx, indices)
            select_ky1 = select_events(ky, indices)
            select_kz1 = select_events(kz, indices)

            # Combine into one vector.
            k1 = np.array([select_kx1, select_ky1, select_kz1])

            # Initial positions are on the sphere boundary.
            # Multiply by minus sign to flip normal vector so the dot product
            # with k1 has the right sign.
            x_inter = -select_events(x, indices)
            y_inter = -select_events(y, indices)
            z_inter = -select_events(z, indices)
        else:

            # Get positions outside of sphere boundary from after exit
            # (or entrance if this is for first event).
            # Index starting with 1 so we pick point after exit event, since
            # it skips the initial point.
            select_x1 = select_events(x[1:, :], indices)
            select_y1 = select_events(y[1:, :], indices)
            select_z1 = select_events(z[1:, :], indices)

            # Get positions inside sphere boundary from before exit
            # Index starting with len(x)-1 so we pick point before
            # exit event, since it starts with initial point before
            # the first event.
            select_x0 = select_events(x[:len(x) - 1, :], indices)
            select_y0 = select_events(y[:len(y) - 1, :], indices)
            select_z0 = select_events(z[:len(z) - 1, :], indices)

            # Make sure there are no infinite values in the coordinates.
            # This prevents an error for the extreme case
            # where mu_scat is infinite, which means that the index contrast 
            # between the particle and matrix is 0
            (select_x0,
             select_y0,
             select_z0,
             select_x1,
             select_y1,
             select_z1) = inf_to_large(select_x0, select_y0, select_z0,
                                       select_x1, select_y1, select_z1, radius)

            # calculate the normalized k1 vector from the positions
            # inside and outside (X0,y0,z0) and (x1,y1,z1)
            k1 = normalize(select_x1 - select_x0, select_y1 - select_y0,
                           select_z1 - select_z0)

            # get positions at sphere boundary from exit
            x_inter, y_inter, z_inter = find_vec_sphere_intersect(select_x0,
                                                                  select_y0,
                                                                  select_z0,
                                                                  select_x1,
                                                                  select_y1,
                                                                  select_z1,
                                                                  radius)

        # calculate the vector normal to the sphere boundary at the exit
        norm = normalize(x_inter, y_inter, z_inter)

        # calculate the dot product between the normal vector
        # and the exit vector
        dot_norm = (norm[0, :] * k1[0, :] + norm[1, :] * k1[1, :]
                    + norm[2, :] * k1[2, :])

        # calulate the angle between the normal vector and the exit vector
        angles = np.arccos(np.nan_to_num(dot_norm))
        angles = sc.Quantity(angles, '')

        # plot the points before exit, after exit, and on exit boundary
        if plot_exits:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(select_x0,select_y0,select_z0, c = 'b')
            # ax.scatter(select_x1,select_y1,select_z1, c = 'g')
            ax.scatter(x_inter, y_inter, z_inter, c='r')
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
        kz = trajectories.direction[2]

        # If incident light, use init dir instead of kz to get direction.
        # before entering sample.
        if init_dir is not None:
            cosz = init_dir
        else:
            # Select scattering events resulted in exit.
            cosz = select_events(kz, indices)

        # Calculate angle to normal from cos_z component (only want magnitude).
        angles = sc.Quantity(np.arccos(np.abs(cosz)), '')

        # Calculate the normal vector.
        norm = np.zeros((3, kz.shape[0], kz.shape[1]))
        if isinstance(cosz, sc.Quantity):
            cosz = cosz.magnitude
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
    theta_inside = refraction(theta_before, n_before, n_inside)

    # If theta_inside is nan (because the trajectory doesn't exit due to TIR),
    # then replace it with pi/2 (the trajectory goes sideways infinitely) to
    # avoid errors during the calculation of stuck trajectories.
    theta_inside[np.isnan(theta_inside)] = np.pi / 2.0

    # Find fraction passing through both interfaces.
    trans_s1, trans_p1 = model.fresnel_transmission(n_before, n_inside,
                                                    theta_before)
    trans_s2, trans_p2 = model.fresnel_transmission(n_inside,
                                                    n_after,
                                                    theta_inside)
    fresnel_trans = (trans_s1 + trans_p1) * (trans_s2 + trans_p2) / 4.

    # Find fraction reflected off both interfaces before transmission.
    refl_s1, refl_p1 = model.fresnel_reflection(n_inside, n_after,
                                                theta_inside)
    refl_s2, refl_p2 = model.fresnel_reflection(n_inside, n_before,
                                                theta_inside)
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

    theta = refraction(angles, n_before, n_after)
    theta[np.isnan(theta)] = np.inf  # This avoids a warning.

    # Choose only the ones inside detection angle.
    filtered_weights = copy.deepcopy(weights)
    filtered_weights[theta > thresh_angle] = 0

    return filtered_weights


def set_up_values(n_sample, trajectories, z_low, thickness):
    '''
    Takes the quantities relevant to sample and trajectories
    and converts them to numpy arrays of the proper units for
    more efficient operations

    Parameters
    ----------
    n_sample: float-like
        Refractive index of the sample
    trajectories:Trajectory object
        Trajectory object used in Monte Carlo simulation
    z_low: float-like
        starting z_values for trajectores in Monte Carlo simulation, usually
        set to 0
    thickness: float-ike
        thickness of film or diameter of sphere

    Returns
    -------
    (n_sample, trajectories, z_low, thickness): tuple of numpy arrays

    '''

    # if the particle has a complex refractive index, the n_sample will be
    # complex too and the code will give lots of warning messages. Better to
    # take only the absolute value of n_sample from the beggining
    n_sample = np.abs(n_sample)

    # create a copy of drajectories object to modify within the function.
    # this should not affect the trajectories object passed by the user
    trajectories = copy.deepcopy(trajectories)

    # set up the values we need as numpy arrays for efficiency
    if isinstance(trajectories.position, sc.Quantity):
        trajectories.position = trajectories.position.to('um').magnitude
    if isinstance(trajectories.direction, sc.Quantity):
        trajectories.direction = trajectories.direction.magnitude
    if isinstance(trajectories.weight, sc.Quantity):
        trajectories.weight = trajectories.weight.magnitude
    if isinstance(z_low, sc.Quantity):
        z_low = z_low.to('um').magnitude
    if isinstance(thickness, sc.Quantity):
        thickness = thickness.to('um').magnitude

    return (n_sample, trajectories, z_low, thickness)


def find_valid_exits(n_sample, n_medium, thickness, z_low, boundary,
                     trajectories):
    '''
    Find booleans describing valid exits for each event and trajectory. Value
    of 1 indicates a valid exit, value of 0 indicates no valid exit.

    Parameters
    ----------
    n_sample: float
        Refractive index of the sample
    n_medium: float
        Refractive index of the medium
    thickness: float
        thickness of film or diameter of sphere
    z_low: float
        starting z_values for trajectores in Monte Carlo simulation, usually
        set to 0
    boundary: string 
        geometrical boundary, current options are 'film' or 'sphere' 
    trajectories:Trajectory object
        Trajectory object used in Monte Carlo simulation

    Returns
    -------
    exits_pos_dir: 2d array (shape: nevents, ntraj)
        boolean for positive exits. Value of 1 means the trajectory exited in 
        the positive (transmission) direction for that event.
    exits_neg_dir: 2d array (shape: nevents, ntraj)
        boolean for negative exits. Value of 1 means the trajectory exited in
        the negative (reflection) direction for that event
        trajectory and event.
    tir_refl_bool: 2d array of booleans (shape: nevents, ntraj)
        describe whether a trajectory gets totally internally reflected at any 
        event and also exits in the negative direction to contribute to reflectance
    '''

    if boundary == 'film':

        # Get variables we need from trajectories.
        kz = trajectories.direction[2]
        z = trajectories.position[2]

        # Rescale z in terms of integer numbers of sample thickness.
        z_floors = np.floor((z - z_low) / (thickness - z_low))

        # Potential exits occur whenever trajectories cross any boundary.
        potential_exits = ~(np.diff(z_floors, axis=0) == 0)

        # Find all kz with magnitude large enough to exit.
        # todo-get particle in here?
        no_tir = abs(kz) > np.cos(np.arcsin(n_medium / n_sample))

        # Exit in positive direction (transmission) occurs
        # iff crossing odd boundary.
        pos_dir = np.mod(z_floors[:-1] + 1 * (z_floors[1:] > z_floors[:-1]), 2).astype(bool)

        # Construct boolean arrays of all valid exits in pos & neg directions.
        exits_pos_dir = potential_exits & no_tir & pos_dir
        exits_neg_dir = potential_exits & no_tir & ~pos_dir

        # Construct boolean array to describe whether a trajectory gets
        # totally internally reflected at any event.
        tir_refl_bool = potential_exits & ~no_tir.astype(bool) & ~pos_dir

    if boundary == 'sphere':

        # Get variables we need from trajectories.
        x, y, z = trajectories.position

        # Get number of trajectories.
        ntraj = z.shape[1]

        # Define sphere radius.
        radius = thickness / 2

        # Potential exits occur whenever trajectories are outside
        # sphere boundary.
        potential_exits = (x[1:, :] ** 2 + y[1:, :] ** 2
                           + (z[1:, :] - radius) ** 2) > radius ** 2
        potential_exit_indices = np.argmax(np.vstack([np.zeros(ntraj),
                                                     potential_exits]), axis=0)

        # kz_correct will be nan if trajectory is totally internally reflected.
        kz_correct = exit_kz(potential_exit_indices, trajectories, boundary,
                             thickness, n_sample, n_medium)
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
        tir_refl_bool = potential_exits & ~no_tir.astype(bool)

    return exits_pos_dir, exits_neg_dir, tir_refl_bool


def find_event_indices(exits_neg_dir, exits_pos_dir, tir_refl_bool):
    '''
    Parameters
    ----------
    exits_neg_dir: 2d array (shape: nevents, ntraj)
        boolean for negative exits. Value of 1 means the trajectory exited in
        the negative (reflection) direction for that event
        trajectory and event.
    exits_pos_dir: 2d array (shape: nevents, ntraj)
        boolean for positive exits. Value of 1 means the trajectory exited in
        the positive (transmission) direction for that event.
    tir_refl_bool: 2d array of booleans (shape: nevents, ntraj)
        describe whether a trajectory gets totally internally reflected
        at any event and also exits in the negative direction
        to contribute to reflectance

    Returns
    -------
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    stuck_indices: 1d array (length: ntraj)
        array of event indices for stuck trajectories
    tir_indices: 1d array (length: ntraj)
        array of event indices for stuck trajectories
    '''

    nevents = exits_neg_dir.shape[0]
    ntraj = exits_neg_dir.shape[1]

    # find first valid exit of each trajectory in each direction
    # note we convert to 2 1D arrays with len = Ntraj
    # need vstack to reproduce earlier behaviour:
    # an initial row of zeros is used to distinguish no events case
    # note that exits_neg_dir and exits_pos_dir contain values of 0 or 1,
    # and argmax returns the *first* instance of the max (1), in cases
    # where there are multiple 1's
    low_event = np.argmax(np.vstack([np.zeros(ntraj), exits_neg_dir]), axis=0)
    high_event = np.argmax(np.vstack([np.zeros(ntraj), exits_pos_dir]), axis=0)

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
    stuck_indices = never_exit * nevents
    tir_indices = np.argmax(np.vstack([np.zeros(ntraj), tir_refl_bool]),
                            axis=0)

    return refl_indices, trans_indices, stuck_indices, tir_indices


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
    absorb_weights = inc_fraction - refl_weights - trans_weights - stuck_weights

    # warn user if too many trajectories got stuck
    stuck_frac = np.sum(stuck_weights) / np.sum(inc_fraction) * 100
    stuck_traj_warn = " \n{0}% of trajectories did not exit the sample. Increase Nevents to improve accuracy.".format(str(stuck_frac))
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
            init_dir = np.cos(refraction(angles, n_sample, n_medium))
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
    if plot_exits:
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
    if plot_exits:
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
    trans_det_frac = np.max([np.sum(trans_detected), EPS]) / np.max([np.sum(trans_weights_pass), EPS])

    # calculate the fraction of the successfully reflected light that is
    # detected
    refl_det_frac = np.max([np.sum(refl_detected), EPS]) / np.max([np.sum(refl_weights_pass), EPS])
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
        extra_refl = refl_fresnel * refl_frac + trans_fresnel * trans_frac + stuck_weights * 0.5
        extra_trans = trans_fresnel * refl_frac + refl_fresnel * trans_frac + stuck_weights * 0.5

    if boundary == 'sphere':
        # TODO these approximations work best if run_fresnel_traj = True
        # otherwise should add option to use approximations for film

        # Stuck are 50/50 reflected/transmitted since they are randomized.
        # non-TIR fresnel are treated as new trajectories at the appropriate
        # interface. This means reversed R/T ratios for fresnel reflection at 
        # transmission interface.
        # extra_refl = refl_fresnel * refl_frac + trans_fresnel * trans_frac + stuck_weights * 0.5
        # extra_trans = trans_fresnel * refl_frac + refl_fresnel * trans_frac + stuck_weights * 0.5
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
    refl_weights = refl_detected + extra_refl * refl_det_frac + inc_refl_detected
    
    # divide by ntraj to get refl and trans per traj
    refl_per_traj = refl_weights / ntraj
    trans_per_traj = trans_weights / ntraj
    
    # sum to calculate reflectance and transmittance
    transmittance = np.sum(trans_per_traj)
    reflectance = np.sum(refl_per_traj)

    return reflectance, transmittance, refl_per_traj, trans_per_traj


def calc_refracted_direction(kx_1, ky_1, kz_1, x_1, y_1, z_1, n1, n2, plot):
    '''
    TODO: make this work for transmission
    TODO: make this work for sphere
    
    refracts <k1> across an interface of two refractive indeces, n1 and n2
    
    Parameters
    ----------
    kx_1: 1d array
        x-component of initial direction vector
    ky_1: 1d array
        y-component of initial direction vector
    kz_1: 1d array
        z-component of initial direction vector
    x_1: 1d array
        x-position before trajectory exit
    y_1: 1d array
        y-position before trajectory exit
    z_1: 1d array
        z-position before trajectory exit
    n1: float
        index of refraction of initial medium
    n2: float
        index of refraction of medium to enter
    plot: boolean
        If True, plots the intersection point with film incident plane
        and k refraction
    
    Returns
    -------
    kx_2: 1d array
        x-component of refracted direction vector
    ky_2: 1d array
        y-component of refracted direction vector
    kz_2: 1d array
        z-component of refracted direction vector
    x_plane: 1d array
        x-coordinate of intersection of direction vector and incident plane
    y_plane: 1d array
        y-coordinate of intersection of direction vector and incident plane
    z_plane: 1d array
        z-coordinate of intersection of direction vector and incident plane
    
    all 1d arrays have length of number of trajectories
    '''
    
    # Find point on the vector around which to rotate.
    # We choose the point where the plane and line intersect
    # see Annie Stephenson lab notebook #3 pg 91 for derivation.
    with np.errstate(divide='ignore', invalid='ignore'):
        x_plane = -z_1 / kz_1 * kx_1 + x_1
        y_plane = -z_1 / kz_1 * ky_1 + y_1
    z_plane = np.zeros((x_plane.shape))  # any point on film incident plane is z = 0 
    
    # Negate positive kz for reflection.
    # Remember that trajectories with positie kz can count as reflected
    # due to the imposed periodic boundary conditions imposed
    # in the trajectories in the film case.
    pos_kz = np.where(kz_1 > 0)
    kz_1[pos_kz] = -kz_1[pos_kz]
    
    # Find vector around which to rotate: k1 X - z.
    u = -ky_1
    v = kx_1
    w = np.zeros((v.shape))
    u, v, w = normalize(u, v, w)

    # Calculate the angle with respect to the normal at which the trajectories leave    
    theta_1 = np.arccos(np.abs(kz_1))
    theta_2 = refraction(theta_1, n1, n2)

    # Find angle by which to rotate trajectory direction.
    alpha = - (theta_2 - theta_1)

    # Rotate exit direction by refracted angle.
    kx_2, ky_2, kz_2 = rotate_refract(x_plane, y_plane, z_plane,
                                      u, v, w, kx_1, ky_1, kz_1, alpha)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('intesection points of exit vector and film plane')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(x_plane, y_plane, z_plane, s=5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('k refraction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(kx_1, ky_1, kz_1, s=10, label='k1')
        ax.scatter(kx_2, ky_2, kz_2, s=20, label='k2')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.legend()

    return kx_2, ky_2, kz_2, x_plane, y_plane, z_plane    


def calc_indices_detected(indices, trajectories, det_theta, det_len, det_dist,
                          nsample, nmedium, plot):
    """
    TODO: make this work for transmission
    
    Detector function.
    
    Takes in exit event indices and removes indices that do not fit within the 
    bounds of the detector, replacing the event number in the array with a zero.
    
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
    kx, ky, kz = trajectories.direction
     
    # detector parameters
    if isinstance(det_theta, sc.Quantity):
        det_theta = det_theta.to('radians').magnitude
    if isinstance(det_dist, sc.Quantity):
        det_dist = det_dist.to('um').magnitude
    if isinstance(det_len, sc.Quantity):
        det_len = det_len.to('um').magnitude
    
    # coordinates and directions at exit events for all trajectories
    x0 = select_events(x[1:], indices)
    y0 = select_events(y[1:], indices)
    z0 = select_events(z[1:], indices)

    kx0 = select_events(kx, indices)
    ky0 = select_events(ky, indices)
    kz0 = select_events(kz, indices)
 
    # Calculate new directions from refraction at exit interface. 
    kx, ky, kz, x, y, z = calc_refracted_direction(kx0, ky0, kz0, x0, y0, z0,
                                                   nsample, nmedium, 
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
    a = kx ** 2 + ky ** 2 + kz ** 2
    b = 2 * (kx * x + ky * y + kz * z)
    c = x ** 2 + y ** 2 + z ** 2 - det_rad ** 2
    t_p = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    t_m = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    if det_theta < np.pi / 2:
        t = t_p
    else:
        t = t_m
        
    x_int = x + t * kx
    y_int = y + t * ky
    z_int = z + t * kz
    
    # check whether trajectory positions at detector hemisphere fall within 
    # the detector limits, and update indices_detected to reflect this
    indices_detected = np.zeros(indices.size)
    x_int_detected = np.zeros(x_int.size)
    y_int_detected = np.zeros(y_int.size)
    z_int_detected = np.zeros(z_int.size)
    for i in range(indices.size):
        if (x_int[i] < x_max and x_int[i] > x_min)\
           and (y_int[i] < y_max and y_int[i] > y_min)\
           and (z_int[i] < 0):
            indices_detected[i] = indices[i]
            x_int_detected[i] = x_int[i]
            y_int_detected[i] = y_int[i]
            z_int_detected[i] = z_int[i]
    
    if plot:
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
        ax.scatter(x_int.magnitude, y_int.magnitude, z_int.magnitude, s=3, c='b', label='exit traj')        
        ax.scatter(x_int_detected, y_int_detected, z_int_detected, s=20, label='detected traj')
        ax.view_init(elev=-148., azim=-112)
        plt.legend()
            
    return indices_detected
    

def select_tir_traj(inarray, events):
    '''
    selects a given value (inarray) for all trajectories,
    replacing that value with the previous one
    for TIR'd trajectories
    '''
    ntraj = events.size
    nevents = inarray.shape[0]

    valid_events = (events > 0)
    ev = events[valid_events].astype(int) - 1
    tr = np.where(valid_events)[0]
    outarray = np.zeros((nevents, ntraj))
    outarray[ev, tr] = inarray[ev, tr]
    outarray[0, :] = inarray[0, :]
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


def calc_refl_trans(trajectories, thickness, n_medium, n_sample, boundary,
                    z_low=0,
                    detection_angle=np.pi / 2,
                    n_front=None,
                    n_back=None,
                    p=None,
                    return_extra=False,
                    run_fresnel_traj=False,
                    fresnel_traj=False,
                    call_depth=0, 
                    max_call_depth=20,
                    max_stuck=0.01, 
                    plot_exits=False, 
                    mu_scat=None, 
                    mu_abs=None,
                    detector=False, 
                    det_theta=None, 
                    det_len=None, 
                    det_dist=None,
                    plot_detector=False, 
                    kz0_rot=None,
                    kz0_refl=None,
                    save_stuck_weights=False,
                    fine_roughness=0,
                    n_particle=None,
                    n_matrix=None):
    """
    Calculates the weight fraction of reflected and transmitted trajectories
    (reflectance and transmittance).Identifies which trajectories are reflected
    or transmitted, and at which scattering event. Includes corrections for
    Fresnel reflections and for a detector.

    Parameters
    ----------
    trajectories : Trajectory object
        Trajectory object used in Monte Carlo simulation
    thickness: float (structcol.Quantity [length])
        thickness of film or diameter of sphere
    n_medium: float (structcol.Quantity [dimensionless])
        Refractive index of the medium.
    n_sample: float (structcol.Quantity [dimensionless])
        Refractive index of the sample.
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options 
        are 'film' or 'sphere'.
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
    p: array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
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
        This argument is not intended to be set by the user. It's purpose is to 
        keep track of whether calc_refl_trans_sphere() is running for the trajectories
        initially being sent into the sphere or for the fresnel reflected (tir)
        trajectories that are trapped in the sphere. It's default value is
        False, and it is changed to True when calc_refl_trans_sphere() is 
        recursively called for calculating the reflectance from fresnel 
        reflected trajectories
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
    mu_scat : float (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering model.
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient from Mie theory.
    detector: boolean
        Set to true if you want to calculate reflection while using a goniometer
        detector (detector at a specified angle).
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
    kz0_rot : None or array_like (structcol.Quantity [dimensionless])
        Initial z-directions that are rotated to account for the fact that  
        coarse surface roughness changes the angle of incidence of light. Thus
        these are the incident z-directions relative to the local normal to the 
        surface. The array size is (1, ntraj).  
    kz0_refl : None or array_like (structcol.Quantity [dimensionless])
        z-directions of the Fresnel reflected light after it hits the sample
        surface for the first time. These directions are in the global 
        coordinate system. The array size is (1, ntraj).
    save_stuck_weights: boolean (optional)
        default is False. If set to True, saves a file with the stuck weight
        fraction. Useful for testing whether event number is sufficiently high. 
    
    Returns
    -------
    refl_trans_result: tuple
        contains a tuple of other variables. The contents of refl_trans_result
        are different based on whether return_extra is set to True or False.
    
        Returns if return_extra is False
        --------------------------------
        reflectance: float
            Fraction of reflected trajectories, including the Fresnel correction
            but not considering the range of the detector.
        transmittance: float
            Fraction of transmitted trajectories, including the Fresnel correction
            but not considering the range of the detector.
            
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
        absorptance of the sample can be found by 1 - reflectance - transmittance
    
    """
  
    # make sure roughness-related values make sense
    if (kz0_rot is None and kz0_refl is not None) or (kz0_rot is not None and kz0_refl is  None):
        raise ValueError('when including coarse surface roughness, must specify both kz0_rot and kz0_refl')
    
    # set up values as floats and numpy arrays to be used throughout function
    ntraj = trajectories.position[2].shape[1]
    n_sample, trajectories, z_low, thickness = set_up_values(n_sample,
                                                              trajectories, 
                                                              z_low, thickness)
    
    # construct booleans for positive and negative exits
    # TODO: confirm this
    if n_particle is not None:
        if n_matrix is not None:
            if n_particle < n_matrix:
                n_tir = fine_roughness * n_matrix + (1 - fine_roughness) * n_sample
            else:
                n_tir = fine_roughness * n_particle + (1 - fine_roughness) * n_sample
    else:
        n_tir = n_sample
    exits_pos_dir, exits_neg_dir, tir_refl_bool = find_valid_exits(n_tir, 
                                                                   n_medium, 
                                                                   thickness, 
                                                                   z_low, 
                                                                   boundary, 
                                                                   trajectories)     
    
    # Find event indices for each trajectory outcome.
    (refl_indices, 
     trans_indices, 
     stuck_indices,
     tir_indices) = find_event_indices(exits_neg_dir, exits_pos_dir,tir_refl_bool)

    # Correct indices to account for detector.
    # TODO make this work for trans_indices as well
    if detector == True:
        shift_traj_tir(trajectories, tir_indices)
        refl_indices = calc_indices_detected(refl_indices, trajectories, 
                                             det_theta, det_len, det_dist, 
                                             n_sample, n_medium, 
                                             plot_detector)
    
    # Find fraction and direction of light that enters sample.
    init_dir, inc_pass_frac = fresnel_correct_enter(n_medium, n_front, n_tir, 
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
     norm_vec_refl, norm_vec_trans) = fresnel_correct_exit(n_tir, n_medium,
                                                n_front, n_back, refl_indices, 
                                                trans_indices, refl_weights, 
                                                trans_weights, absorb_weights,
                                                boundary, thickness, trajectories,
                                                fresnel_traj, plot_exits)

    # Correct for effect of detection angle upon leaving sample.
    (inc_refl_detected, 
     trans_detected, refl_detected, 
     trans_det_frac, refl_det_frac) = detect_corrected_traj(inc_pass_frac, 
                                                            n_sample, n_medium,
                                                            refl_indices, 
                                                            trans_indices,
                                                            refl_weights_pass, 
                                                            trans_weights_pass,
                                                            trajectories,
                                                            boundary, thickness,
                                                            detection_angle, EPS,
                                                            kz0_rot, kz0_refl)
      
    total_stuck = np.sum(refl_fresnel + trans_fresnel + stuck_weights) / ntraj

    # If we want to run Fresnel reflected as new trajectories 
    # (only implemented for sphere boundary)     
    if run_fresnel_traj and call_depth < max_call_depth and total_stuck > max_stuck:

        # Calculate the reflectance and transmittance per trajectory
        # without Fresnel weights.
        refl_per_traj_nf = (refl_detected + inc_refl_detected) / ntraj
        trans_per_traj_nf = trans_detected / ntraj

        
        # Rerun Fresnel reflected components of trajectories.
        (reflectance, 
         transmittance,
         refl_per_traj,
         trans_per_traj) = run_sphere_fresnel_traj(refl_per_traj_nf,
                                                  trans_per_traj_nf, 
                                                  refl_fresnel, 
                                                  trans_fresnel,stuck_weights,
                                                  trajectories,
                                                  refl_indices, 
                                                  trans_indices,
                                                  tir_indices,
                                                  thickness, boundary, z_low, p,
                                                  n_medium, n_sample, mu_scat, mu_abs,
                                                  max_stuck, call_depth, plot_exits)  
    else:
        # Distribute ambiguous trajectory weights.
        # Note that when run_fresnel_traj == True, this statement is only executed
        # at the final recursive call of cal_refl_trans(). At this call,
        # the reflectance returned here will be tiny, much less than 1%, but it 
        # will be added to the reflectance calculated from all of the previous
        # calls, so this is expected and desired. 
        (reflectance, 
         transmittance,
         refl_per_traj, 
         trans_per_traj) = distribute_ambig_traj_weights(refl_fresnel, trans_fresnel, 
                                                        refl_frac, trans_frac,
                                                        refl_det_frac, trans_det_frac,
                                                        refl_detected, trans_detected,
                                                        stuck_weights, inc_refl_detected, 
                                                        boundary, detector)
                                        
    # Return desired results.
    if return_extra:
        refl_trans_result = (refl_indices, 
                             trans_indices, 
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
                             tir_refl_bool,
                             norm_vec_refl, 
                             norm_vec_trans)
        
    else:
        refl_trans_result = reflectance, transmittance
    
    # For performance testing, print the stuck weights to text file
    if save_stuck_weights:
        if boundary == 'film':
            with open('stuck_weights_small.txt', 'a') as f:
                print(total_stuck, file=f)
            f.close()
    
    return refl_trans_result

    
def run_sphere_fresnel_traj(refl_per_traj_nf, trans_per_traj_nf, 
                             refl_fresnel, trans_fresnel, stuck_weights,
                             trajectories, refl_indices, trans_indices, 
                             tir_indices, thickness, boundary, z_low, 
                             p, n_medium, n_sample, 
                             mu_scat, mu_abs, 
                             max_stuck, call_depth, plot_exits):
    '''
    For the sphere case, there are many trajectories that are totally internally
    reflected or partially reflected back into the sample
    
    This function takes the weights of the trajectory components reflected back
    into the sample (whether it's the whole weight through tir, or just partial
    through fresnel) and re-runs them as new trajectories, until most of them
    exit through reflectance, transmittance, or are absorbed. Note that it does 
    not re-run stuck trajectories (trajectories that have not yet attempted
    an exit.)
        
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
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    trans_indices: 1d array (length: ntraj)
        array of event indices for transmitted trajectories
    tir_indices: 1d array (length: ntraj)
        array of event indices for totally internally reflected trajectories
    stuck_indices: 1d array (length: ntraj)
        array of event indices for stuck trajectories
    thickness: float (structcol.Quantity [length])
        thickness of film or diameter of sphere
    boundary: string
        Geometrical boundary for Monte Carlo calculations. Current options 
        are 'film' or 'sphere'.
    z_low : float (structcol.Quantity [length])
        Initial z-position of sample closest to incident light source.
        Should normally be set to 0.
    p: array_like (structcol.Quantity [dimensionless])
        Phase function from either Mie theory or single scattering model.
    n_medium: float (structcol.Quantity [dimensionless])
        Refractive index of the medium.
    n_sample: float (structcol.Quantity [dimensionless])
        Refractive index of the sample.
    mu_scat : float (structcol.Quantity [1/length])
        Scattering coefficient from either Mie theory or single scattering model.
    mu_abs : float (structcol.Quantity [1/length])
        Absorption coefficient from Mie theory.    
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
    
    Returns
    -------
    reflectance_fresnel + reflectance_no_fresnel: float
        new reflectance after re-running fresnel reflected trajectories. Adds
        the previous reflectance (reflectance_no_fresnel) and the new addition
        to the reflectance (reflectance_fresnel)
    transmittance_fresnel + transmittance_no_fresnel: float
        new transmittance after re-running fresnel reflected trajectories. Adds
        the previous transmittance (transmittance_no_fresnel) and the new addition
        to the transmittance (transmittance_fresnel)
        
    '''
    
    # Set up values to use throughout function.
    n_sample, trajectories, z_low, diameter = set_up_values(n_sample, 
                                                            trajectories, 
                                                            z_low, 
                                                            thickness)
    radius = diameter / 2
    x,y,z = trajectories.position
    kx, ky, kz = trajectories.direction
    nevents = x.shape[0]
    ntraj = x.shape[1]

    # New weights are the weights that are fresnel reflected 
    # back into the sphere.
    weights_fresnel = np.zeros((nevents, ntraj))
    weights_fresnel[:,:] = refl_fresnel + trans_fresnel + stuck_weights
    weights_fresnel = sc.Quantity(weights_fresnel, '')
    
    # Add refl and trans indices for all attempted or successful exit indices
    indices = refl_indices + trans_indices + tir_indices
    
    # Get positions outside of sphere boundary from after exit
    select_x1 = select_events(x[1:,:], indices)
    select_y1 = select_events(y[1:,:], indices)
    select_z1 = select_events(z[1:,:], indices)   
    
    # get positions inside sphere boundary from before exit
    select_x0 = select_events(x[:len(x)-1,:], indices)
    select_y0 = select_events(y[:len(y)-1,:], indices)
    select_z0 = select_events(z[:len(z)-1,:], indices)
    
    # make sure none of the coordinates are infinite
    (select_x0, 
     select_y0,
     select_z0,
     select_x1,
     select_y1,
     select_z1) = inf_to_large(select_x0, select_y0, select_z0,
                               select_x1, select_y1, select_z1, radius)
    
    # get radius vector to subtract from select_z
    select_radius = select_events(radius * np.ones((nevents, ntraj)), indices)

    # shift z for intersect finding
    select_z0 = select_z0 - select_radius
    select_z1 = select_z1 - select_radius
    
    # get positions at sphere boundary from exit
    x_inter, y_inter, z_inter = find_vec_sphere_intersect(select_x0,
                                                          select_y0,
                                                          select_z0,
                                                          select_x1,
                                                          select_y1,
                                                          select_z1, 
                                                          radius)
    # shift z back to global coordinates
    z_inter = z_inter + select_radius
    
    # define vectors for reflection inside sphere
    select_kx = select_events(kx, indices)
    select_ky = select_events(ky, indices)
    select_kz = select_events(kz, indices)
    k_out = np.array([select_kx, select_ky, select_kz])
    normal = np.array([x_inter / radius, y_inter / radius,
                       (z_inter - select_radius) / radius])
    
    # calculate reflected direction inside sphere
    k_refl = rotate_reflect(k_out, normal)
    
    # set the initial directions as the reflected directions
    directions = np.zeros((3 , nevents , ntraj))
    directions = sc.Quantity(directions, '')
    directions[:, 0, :] = k_refl
    
    # set the initial positions at the sphere boundary
    positions = np.zeros((3, nevents + 1, ntraj))
    positions[:, 0, :] = x_inter, y_inter, z_inter

    # TODO: get rid of trajectories whose initial weights are 0
    # find indices where initial weights are 0
#        indices = np.where(weights_fresnel[0,:] == 0)
#        if indices[0].size > 0:
#            weights_fresnel = np.delete(weights_fresnel,indices)
#            positions = np.delete(positions, indices, axis = 0)
#            directions = np.delete(directions, indices,axis = 0)
    
    # create new trajectories object
    trajectories_fresnel = mc.Trajectory(positions, directions, weights_fresnel)

    # Generate a matrix of all the randomly sampled angles first 
    sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents, ntraj, p)

    # Create step size distribution
    step = mc.sample_step(nevents, ntraj, mu_scat)

    # Run photons
    trajectories_fresnel.absorb(mu_abs, step)
    trajectories_fresnel.scatter(sintheta, costheta, sinphi, cosphi)         
    trajectories_fresnel.move(step)

    # Calculate reflection and transmition                                                       
    (_, trans_indices_fresnel, _, _, _,
     refl_per_traj_fresnel, 
     trans_per_traj_fresnel,
     _, _, _, _, 
     reflectance_fresnel, 
     transmittance_fresnel,_, 
     norm_refl_f, norm_trans_f) = calc_refl_trans(trajectories_fresnel, 
                                                      thickness, n_medium,
                                                      n_sample, boundary,
                                                      p = p, mu_abs = mu_abs,
                                                      mu_scat = mu_scat, 
                                                      plot_exits = plot_exits,
                                                      run_fresnel_traj = True,
                                                      fresnel_traj = True, 
                                                      call_depth = call_depth + 1,
                                                      max_stuck = max_stuck,
                                                      return_extra = True)                                             

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
        3d vector to be reflected. Usually refers to vector describing trajectory
        direction as it exits sample.
    normal: array
        3d vector normal to the surface on which to reflect
        .
    Returns
    -------
    k_refl: array
        3d vector that is reflected. 
    '''
            
    dot_k_out_normal = (k_out[0] * normal[0] + k_out[1] * normal[1] + 
                        k_out[2] * normal[2])
            
    k_refl = k_out - 2 * dot_k_out_normal * normal
            
    return k_refl


def normalize_refl_goniometer(refl, det_dist, det_len, det_theta):
    '''
    calculates the reflectance renormalized for goniometer measurement
    
    This normalization scheme makes several key assumptions:
    (1) The area of the detection hemisphere spanned by the detector aperture is
        a square. As the detector size approaches the diameter of the detection
        hemisphere, this assumption becomes worse. In reality, the detection hemisphere
        area spanned by the detector is the projection of a square on the sphere
        surface, which looks like a curved square patch.
    (2) The reference reflector (maximum reflectance) is that of a lambertian
        reflector, meaning the reflectance is uniform over the detection hemisphere
        and that the integrated reflected intensity is equal to the intensity 
        of the incident beam. This means that if the sample has a specular 
        component, the reflectance could be greater than one for the specular angle.
        
    The normalization formula:

    refl_renormlized = (area of detection hemisphere)/(area detected) * reflectance   

    We are just scaling up the reflectance based on the area detected relative 
    to the total possible area that can be detected.      
        
    vocab
    -----
    detection hemisphere: the hemisphere surrounding the sample, 
                          having radius of the detector distance. In the case
                          of reflectance measurements, it is a reflection hemisphere.
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
