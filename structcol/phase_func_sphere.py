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
Created on Thu Feb  8 12:34:06 2018

@author: stephenson
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import structcol as sc
from . import select_events
from scipy.special import factorial
import warnings

def get_exit_pos(norm_refl, norm_trans, radius):
    '''
    find the exit points of trajectories sent into a sphere

    Parameters
    ----------
    norm_refl: 2d array-like, shape (3, number of trajectories)
        array of normal vectors for trajectories at their
        reflection exit from the sphere
    norm_trans: 2d array-like, shape (3, number of trajectoires)
        array of normal vectors for trajectories at their
        transmission exit from the sphere
        norm_trans
    radius: float-like
        radius of the spherical boundary

    Returns
    -------
    x_inter: array-like
        x-coordinates of exit positions of trajectories
    y_inter: array-like
        y-coordinates of exit positions of trajectories
    z_inter:array-like
        z-coordinates of exit positions of trajectories
    '''
    # add the normal vectors for reflection and transmission to get
    # normal vectors for all exits
    norm = norm_trans + norm_refl

    # get the x-coordinate
    x_inter = norm[0,:]
    x_inter = x_inter * radius

    # get the y-coordinate
    y_inter = norm[1,:]
    y_inter = y_inter * radius

    # get the z-coordinate
    z_inter = norm[2,:]
    z_inter = z_inter * radius

    return x_inter, y_inter, z_inter

def conv_circ(signal, ker ):
    '''
    signal: real 1D array
    ker: real 1D array
    signal and ker must have same shape
    '''
    return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(ker)))

def calc_pdf(x, y, z, radius,
             refl_per_traj,
             trans_per_traj,
             refl_indices,
             trans_indices,
             plot=False,
             phi_dependent=False,
             nu_range=np.linspace(0.01, 1, 200),
             phi_range=np.linspace(0, 2 * np.pi, 300),
             kz=None,
             kernel_bin_width='silverman'):

    '''
    Calculates kernel density estimate of probability density function
    as a function of nu or nu and phi for a given set of x,y, and z coordinates

    x, y, and z are the points on the sphere at which a trajectory exits
    the sphere

    Parameters
    ----------
    x: 1d array-like
        x-coordinate of each trajectory at exit event
    y: 1d array-like
        y-coordinate of each trajectory at exit event
    z: 1d array-like
        z-coordinate of each trajectory at exit event
    radius: float
        radius of sphere boundary
    refl_per_traj: 1d array
        array of trajectory weights that exit through reflection, normalized
        by the total number of trajectories
    trans_per_traj: 1d array
        array of trajectory weights that exit through transmission, normalized
        by the total number of trajectories
    refl_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through reflection
    trans_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through transmission
    plot: boolean
        If set to True, the intermediate and final pdfs will be plotted
    phi_dependent: boolean (optional)
        If set to True, the returned pdf will require both a nu and a phi
        input
    nu_range: 1d array (optional)
        the nu values for which the pdf
    phi_range: 1d array (optional)
        the phi values for which to calculate the pdf, if the pdf is
        phi-dependent
    kz: 1d array or None (optional)
        the kz values at the exit events for all the trajectories
    kernel_bin_width: string or scalar or callable (optional)
        determines the bin width for the gaussian kde used to calculate the
        structured sphere phase function. See scipy's gaussian_kde() function
        for more details. Default is 'silverman'

    Returns
    -------
    pdf_array: 1d or 2d array
        probability density function values as function of nu if
        phi_dependent = False, and as a function of nu and phi if
        phi_depenedent = True

    Notes
    -----
    the probability density function is calculated as a function of nu and phi
    instead of theta and phi to correct for the inequal areas on the sphere
    surface for equal spacing in theta. Theta is related to nu by:

        Theta = arccos(2 * nu - 1)

    see http://mathworld.wolfram.com/SpherePointPicking.html for more details

    '''
    trans_indices_scat = np.copy(trans_indices)
    trans_indices_scat[trans_indices == 1] = 0

    # calculate thetas for each exit point
    # If optional parameter kz is specified, we calculate theta based on kz.
    # If not, we calculate theta based on the z exit position
    if kz is not None:
        # remove the values of kz=0 because they mean that there
        # was no reflection or transmission event
        kz = select_events(kz, refl_indices + trans_indices_scat)
        kz = kz[kz != 0]
        theta = np.arccos(kz)
    else:
        # turn indices into booleans
        refl_indices[refl_indices != 0] = 1
        trans_indices_scat[trans_indices_scat != 0] = 1

        # multiply by _per_traj to get only the weights of the exited photons
        # need to include this because of TIR
        refl_weights_z = refl_per_traj * refl_indices
        trans_weights_z = trans_per_traj * trans_indices_scat

        # turn back into boolean
        refl_weights_z[refl_weights_z != 0] = 1
        trans_weights_z[trans_weights_z != 0] = 1

        # multiply z's by 0 if there is no exit
        z = z*(refl_weights_z + trans_weights_z)

        # get rid of the z's that don't exit
        z = z[z != 0]
        theta = np.arccos(z / radius)

    # since we don't care about event number, change all non-zero values to 1
    refl_indices[refl_indices != 0] = 1
    trans_indices_scat[trans_indices_scat != 0] = 1

    # mutiply per_traj by indices to get rid of intensities where there is no
    # actual scattering exit event
    refl_weights = refl_per_traj * refl_indices
    trans_weights = trans_per_traj * trans_indices_scat

    # add weights to get scattered weights per traj
    weights = refl_weights + trans_weights

    # remove zeros to match kz size
    weights = weights[weights != 0]

    # convert thetas to nus
    nu = (np.cos(theta) + 1) / 2

    # add reflections of data on to ends to prevent dips in distribution
    # due to edges
    nu_edge_correct = np.hstack((-nu, nu, -nu + 2))
    weights_edge_correct = np.hstack((weights, weights, weights))

    if not phi_dependent:
        if nu_edge_correct.size == 0:
            warnings.warn("No trajectories reflected or transmitted. "
                          "Check sample parameters")
            pdf_array = np.nan

        else:
            # calculate the pdf kernel density estimate
            pdf = gaussian_kde(nu_edge_correct.magnitude,
                               bw_method=kernel_bin_width,
                               weights=weights_edge_correct)

            # calculate the pdf for specific nu values
            theta = np.linspace(0.01, np.pi, 200)
            nu = (np.cos(theta) + 1) / 2
            pdf_array = pdf(nu)
            pdf_array = pdf_array / np.sum(pdf_array)

        if plot:    # pragma: no cover
            # plot the distribution from data, with edge correction, and kde
            plot_dist_1d(nu_range, nu, nu_edge_correct, pdf(nu_range))
            plt.xlabel(r'$\nu$')

    else:
        # calculate phi for each exit point
        phi = np.arctan2(y, x) + np.pi

        # add reflections of data to ends to prevent dips in distribution
        # due to edges
        phi_edge_correct = np.tile(np.hstack((-phi, phi, -phi + 4 * np.pi)), 3)
        nu_edge_correct = np.hstack((np.tile(-nu, 3), np.tile(nu, 3),
                                     np.tile(-nu + 2, 3)))

        # calculate the pdf kernel density estimate
        pdf = gaussian_kde(np.vstack([nu_edge_correct, phi_edge_correct]))

        # calculate the pdf for specific nu and phi values
        theta = np.linspace(0.01, np.pi, 200)
        nu = (np.cos(theta) + 1) / 2
        pdf_array = pdf(nu, phi_range)

        if plot:    # pragma: no cover
            # plot the the calculated kernel density estimate in phi
            nu_2d, phi_2d = np.meshgrid(nu_range, phi_range)
            angle_range = np.vstack([nu_2d.ravel(), phi_2d.ravel()])
            pdf_vals = np.reshape(pdf(angle_range), nu_2d.shape)
            pdf_marg_nu = np.sum(pdf_vals, axis = 0)

            # plot the nu distribution from data, with edge correction, and kde
            plot_dist_1d(nu_range, nu, nu_edge_correct, pdf_marg_nu)
            plt.xlabel(r'$\nu$')

            # plot the phi distribution from data, with edge correction, and
            # kde
            pdf_marg_phi = np.sum(pdf_vals, axis = 1)
            plot_dist_1d(phi_range, phi, phi_edge_correct, pdf_marg_phi)
            plt.xlabel(r'$\phi$')

    return pdf_array

def plot_phase_func(pdf, nu=np.linspace(0, 1, 200), phi=None,
                    save=False):            # pragma: no cover
    '''
    Plots a given probability density function (pdf)

    If the provided probability density is a function of only nu,
    then the pdf is plotted against theta. We convert nu to theta because theta
    is the more commonly used physical parameter in spherical coordinates.

    If the provided probability density is a function of nu and phi,
    then the pdf is plotted against theta and phi as a heatmap.

    Parameters
    ----------
    pdf: function, 1 or 2 arguments
        probability density function that requires an input of nu values
        or nu and phi values
    nu: 1d array-like (optional)
        y-coordinate of each trajectory at exit event
    phi: None or 1d array-like (optional)
        z-coordinate of each trajectory at exit event
    save: boolean (optional)
        tells whether or not to save the plot

    Notes
    -----
    see http://mathworld.wolfram.com/SpherePointPicking.html for more details
    on conversion between theta and nu

    '''
    # convert nu to theta
    theta = np.arccos(2 * nu - 1)

    if phi is None:
        # calculate the phase function for theta points
        phase_func = pdf(nu) / np.sum(pdf(nu) * np.diff(nu)[0])

        # make polar plot in theta
        plt.figure()
        ax = plt.subplot(111,projection = 'polar')
        ax.set_title(r'phase function in $\theta$')
        ax.plot(theta, phase_func, linewidth=3, color=[0.45, 0.53, 0.9])
        ax.plot(-theta, phase_func, linewidth=3, color=[0.45, 0.53, 0.9])
    else:
        # calculate the phase function for points in theta and phi
        theta_2d, phi_2d = np.meshgrid(theta, phi)
        nu_2d = (np.cos(theta_2d) + 1) / 2
        angles = np.vstack([nu_2d.ravel(), phi_2d.ravel()])
        pdf_vals = np.reshape(pdf(angles), theta_2d.shape)
        phase_func = pdf_vals / np.sum(pdf_vals * np.diff(phi)[0]
                                       * np.diff(theta)[0])

        # make heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(phase_func, cmap=plt.cm.gist_earth_r,
                        extent=[theta[0], theta[-1], phi[0], phi[-1]])
        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
        ax.set_xlim([theta[0], theta[-1]])
        ax.set_ylim([phi[0], phi[-1]])
        fig.colorbar(cax)

    if save:
        plt.savefig('phase_fun.pdf')
        np.save('phase_function_data',phase_func)


def plot_dist_1d(var_range, var_data, var_data_edge_correct,
                 pdf_var_vals):                 # pragma: no cover
    '''
    plots the probability distribution of a variable of interest

    Parameters
    ----------
    var_range: 1d array
        array of values of variable whose pdf you want to find. Should sweep
        whole range of interest.
    var_data: 1d array-like
        values of the variable of interest from data.
    var_data_edge_correct: 1d array-like
        values of the variable of interest corrected for edge effects in the
        probability distribution
    pdf_var_vals: 1d array
        probability density values for variable values of var_range
        if pdf is 2d, this array is marginalized over the other variable

    '''
    plt.figure()

    # plot the kde using seaborn, from the raw data
    sns.distplot(var_data, rug=True, hist=False,
                 label='distribution from data')

    # plot the kde using seaborn, from edge corrected data
    sns.distplot(var_data_edge_correct, rug=True, hist=False,
                 label='distribution with edge correction')

    # renormalize the pdf
    pdf_norm = pdf_var_vals / np.sum(pdf_var_vals*np.diff(var_range)[0])

    # plot
    plt.plot(var_range, pdf_norm,
             label = 'kernel density estimate, correctly normalized')
    plt.legend()
    plt.xlim([var_range[0],var_range[-1]])
    plt.ylabel('probability density')

def calc_directions(theta_sample, phi_sample, x_inter,y_inter, z_inter, k1,
                    radius):
    '''
    Calculates directions of exit trajectories

    Parameters
    ---------_
    theta_sample: 1d array
        sampled thetas of exit trajectories
    phi_sample: 1d array
        sampled phis of exit trajectories
    x_inter: 1d array-like
        x-coordinate of each trajectory at exit event
    y_inter: 1d array-like
        y-coordinate of each trajectory at exit event
    z_inter: 1d array-like
        z-coordinate of each trajectory at exit event
    k1: 2d array
        direction vector for trajectories
    radius: float-like
        radius of sphere boundary

    Returns
    -------
    k1: 2d array
        direction vector for exit trajectories

    '''
    z_sample = radius*np.cos(theta_sample)
    y_sample = radius*np.sin(phi_sample) * np.sin(theta_sample)
    x_sample = radius*np.cos(phi_sample) * np.sin(theta_sample)

    xa = np.vstack((x_sample, y_sample, z_sample)).T
    xb = np.vstack((x_inter, y_inter, z_inter)).T

    distances = cdist(xa, xb)
    ind = np.argmin(distances, axis = 1)

    return k1[:,ind]

def plot_exit_points(x, y, z, radius, plot_dimension='3d'): # pragma: no cover
    '''
    plots data corresponding to the x,y,z and radius inputs

    the main use of this function is to plot data points over the heatmap
    of the pdf for validation of the kernel density estimate

    Parameters
    ----------
    x: 1d array-like
        x-coordinate of each trajectory at exit event
    y: 1d array-like
        y-coordinate of each trajectory at exit event
    z: 1d array-like
        z-coordinate of each trajectory at exit event
    radius: float-like
        radius of sphere boundary
    plot_dimension: string (optional)
        If set to '3d' the plot is a 3d plot on a sphere. If set to '2d', plots
        on a 2d plot theta vs phi
    '''

    unit = '' # initialize unit to empty string
    if isinstance(x, sc.Quantity):
        unit = x.units # save unit for later use
        x = x.to('um').magnitude
    if isinstance(y, sc.Quantity):
        y = y.to('um').magnitude
    if isinstance(z, sc.Quantity):
        z = z.to('um').magnitude
    if isinstance(radius, sc.Quantity):
        radius = radius.to('um').magnitude


    if plot_dimension == '2d':
        # calculate thetas for each exit point
        theta = np.arccos(z/radius)

        # calculate phi for each exit point
        phi = np.arctan2(y,x) + np.pi

        # plot
        plt.plot(theta, phi, '.')
        plt.xlim([0, np.pi])
        plt.ylim([0, 2 * np.pi])

    if plot_dimension == '3d':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x ' + str(unit))
            ax.set_ylabel('y ' + str(unit))
            ax.set_zlabel('z ' + str(unit))
            ax.set_title('exit positions')
            ax.view_init(-164, -155)
            ax.plot(x, y, z, '.')

def calc_d_avg(volume_fraction, radius):
    '''
    Calculates the average spacing between structured spheres in a bulk film,
    given their volume fraction

    Parameters
    ----------
    volume_fraction: float-like
        volume fraction of structured spheres in a bulk film
    radius: float-like
        radius of structured spheres in a bulk film

    Returns
    -------
    d_avg: float-like
        average spacing between structured spheres in a bulk film

    '''

    # calculate the number density
    number_density = volume_fraction / (4 / 3 * np.pi * radius**3)

    # calculate the average interparticle spacing
    d_avg = 2 * (3 / (4 * np.pi * number_density))**(1 / 3)

    return d_avg

def calc_mu_scat_abs(refl_per_traj, trans_per_traj, refl_indices,
                     trans_indices, volume_fraction, radius, n_sample,
                     wavelength):
    '''
    Calculates scattering coefficient and absorption coefficient using results
    of the Monte Carlo calc_refl_trans() function

    calculates the scattering length from the formula:

        mu_scat = number density * total scattering cross section

    where the scattering length is the inverse of the above expression:

        lscat = 1/(number density * total scattering cross section)

    and the total scattering cross section is found by integrating the
    fraction of scattered light and multiplying by the initial area:

        total scattering cross section = power scattered / incident intensity
                           = power scattered / (incident power / incident area)
                           = power scattered / incident power * 2*pi*radius**2
                           = (scattered fraction)*2*pi*radius**2

    calculates the absorption length from the formula:

        mu_abs = number density * total absorption cross section

    where the absorption length is the inverse of the above expression:

        l_abs = 1/(number density * total absorption cross section)

    and the total absorption cross section is found by subtracting the

        total absorption cross section = power absorbed / incident intensity
                    = power absorbed / (incident power / indcident area)
                    = power absorbed / incident power *2*pi*radius**2
                    = (absobred fraction)*2*pi*radius**2

    Parameters
    ----------
    refl_per_traj: 1d array
        array of trajectory weights that exit through reflection, normalized
        by the total number of trajectories
    trans_per_traj: 1d array
        array of trajectory weights that exit through transmission, normalized
        by the total number of trajectories
    refl_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through reflection
    trans_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through transmission
    volume_fraction: float-like
        volume fraction of structured spheres in a bulk film
    radius: float-like
        radius of structured spheres in a bulk film
    n_sample: float-like
        refractive index of the material surrounding the sphere, oven referred
        to as the bulk matrix
    wavelength: float-like
        source light wavelength

    Returns
    -------
    mu_scat: float-like
        scattering coefficient for bulk film of structured spheres
    mu_abs: float-like
        absorption coefficient for bulk film of structured spheres

    '''

    # calculate the number density
    number_density = volume_fraction / (4 / 3 * np.pi * radius**3)

    # calculate the total absorption cross section
    # assumes no stuck
    tot_abs_cross_section = ((1 - np.sum(refl_per_traj + trans_per_traj))
                             * np.pi * radius**2)

    # remove transmission contribution from trajectories that did not scatter
    trans_per_traj_scat = np.copy(trans_per_traj)
    trans_per_traj_scat[trans_indices == 1] = 0

    tot_scat_cross_section = (np.sum(refl_per_traj + trans_per_traj_scat)
                              * 2 * np.pi * radius**2)

    # calculate mu_scat, mu_abs using the sphere
    mu_scat = number_density * tot_scat_cross_section
    mu_abs_sphere = number_density * tot_abs_cross_section

    # don't need to include volume fraction for mu_abs_sphere component
    # because already included in number_density
    mu_abs_matrix = 4 * np.pi * np.imag(n_sample) / wavelength
    mu_abs = mu_abs_sphere + mu_abs_matrix * (1 - volume_fraction)

    return mu_scat, mu_abs

def calc_scat_bulk(refl_per_traj,
                   trans_per_traj,
                   refl_indices,
                   trans_indices,
                   norm_refl,
                   norm_trans,
                   volume_fraction,
                   diameter,
                   n_sample,
                   wavelength,
                   plot=False, phi_dependent=False,
                   nu_range=np.linspace(0.01, 1, 200),
                   phi_range=np.linspace(0, 2 * np.pi, 300),
                   kz=None,
                   kernel_bin_width='silverman'):
    '''
    Parameters
    ----------
    refl_per_traj: 1d array
        array of trajectory weights that exit through reflection, normalized
        by the total number of trajectories
    trans_per_traj: 1d array
        array of trajectory weights that exit through transmission, normalized
        by the total number of trajectories
    refl_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through reflection
    trans_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through transmission
    norm_refl: 2d array-like, shape (3, number of trajectories)
        array of normal vectors for trajectories at their
        reflection exit from the sphere
    norm_trans: 2d array-like, shape (3, number of trajectoires)
        array of normal vectors for trajectories at their
        transmission exit from the sphere
        norm_trans
    volume_fraction: float-like
        volume fraction of structured spheres in a bulk film
    diameter: float-like
        diameter of structured spheres in a bulk film
    n_sample: float-like
        refractive index of the material surrounding the sphere, often referred
        to as the bulk matrix
    wavelength: float-like
        source light wavelength
    plot: boolean (optional)
        If set to True, the intermediate and final pdfs will be plotted
    phi_dependent: boolean (optional)
        If set to True, the returned pdf will require both a nu and a phi
        input
    nu_range: 1d array (optional)
        the nu values for which the pdf
    phi_range: 1d array (optional)
        the phi values for which to calculate the pdf, if the pdf is
        phi-dependent
    kz: None or 1d array (optional)
        the kz values at the exit events of the trajectories
    kernel_bin_width: string or scalar or callable (optional)
        determines the bin width for the gaussian kde used to calculate the
        structured sphere phase function. See scipy's gaussian_kde() function
        for more details. Default is 'silverman'

    Returns
    -------
    p: 1d array
        phase function for bulk film
    mu_scat: float-like
        scattering coefficient for bulk film
    mu_abs: float-like
        absorption coefficient for bulk film

    '''
    # until refactoring, convert back to numpy
    if isinstance(n_sample, xr.DataArray):
        n_sample = n_sample.to_numpy()

    # get radius from diameter
    radius = diameter / 2

    # calculate the lscat of the microsphere for use in the bulk simulation
    mu_scat, mu_abs = calc_mu_scat_abs(refl_per_traj, trans_per_traj,
                                       refl_indices, trans_indices,
                                       volume_fraction, radius, n_sample,
                                       wavelength)

    # find the points on the sphere where trajectories exit
    x_inter, y_inter, z_inter = get_exit_pos(norm_refl, norm_trans, radius)

    # calculate the probability density function as a function of nu, which
    # depends on the scattering angle
    p = calc_pdf(x_inter, y_inter, z_inter, radius,
                 refl_per_traj,
                 trans_per_traj,
                 refl_indices,
                 trans_indices,
                 plot=plot,
                 phi_dependent=phi_dependent,
                 nu_range=nu_range,
                 phi_range=phi_range,
                 kz=kz,
                 kernel_bin_width=kernel_bin_width)

    return p, mu_scat, mu_abs

def size_distribution(diameter_range, mean, t):
    '''
    Depricated in Mie-separated branch. A nearly identical function exists in
    model.py. The only difference is that this function is not normalized

    Parameters
    ----------
    diameter_range: array
        Range of diameters of the distribution.
    mean: 1-element array
        Mean diameter of the distribution.
    t: 1-element array
        'Width' of the distribution. t = (1 - p**2) / p**2, where p is the
        polydispersity index.

    Returns
    -------
    distr: array (same length as diameter_range)
        Schulz distribution as a fuction of diameter.

    '''
    if t <= 100:
        schulz = (((t + 1) / mean)**(t + 1) * diameter_range**t
                  / factorial(t) * np.exp(- diameter_range / mean * (t + 1)))
        distr = schulz
    else:
        std_dev = diameter_range / np.sqrt(t + 1)
        distr = (np.exp(-(diameter_range - mean)**2 / (2 * std_dev**2))
                 / np.sqrt(2 * np.pi * std_dev**2))
        #distr = distr/np.sum(distr)
    return(distr)

def calc_diam_list(num_diam, diameter_mean, pdi,
                   equal_spacing=False, plot=True, num_pdf_points=600):
    '''
    Calculate the list of radii to sample from for a given polydispersity and
    number of radii. This function is used specifically to calculate a list of
    radii to sample in the polydisperse bulk Monte Carlo model.

    Parameters
    ----------
    num_diam: int
        number of diameters
    diam_mean: float, sc.Quantity
        mean radius of the distribution
    pdi: float
        polydispersity index of the distribution
    equal_spacing: boolean
        If True, the calculated list of radii is equally spaced, instead of
        choosing points based on FWHM
    plot: boolean
        if True, the probability density function is plotted as a function of
        radius, as well as the list of radii points
    num_pdf_points: int
        number of points at which to calculate the probability density function
        should not need to change this value

    Returns
    -------
    diam_list: 1d numpy array
        list of diameters from which to sample in polydisperse bulk Monte Carlo
    '''
    # get radius from diameter
    radius_mean = diameter_mean / 2

    # calculate the range of diameters at which to calculate the pdf
    diam_range = (np.linspace(1, 4 * radius_mean.magnitude, num_pdf_points)
                  * radius_mean.units)

    # calculate the radii at equal spacings
    if equal_spacing:
        rad_mean = radius_mean.magnitude
        num_half = int(np.round((num_diam + 1) / 2))
        rad_list = (np.unique(np.hstack((np.linspace(rad_mean / 100,
                                                     rad_mean, num_half),
                                         np.linspace(rad_mean,
                                                    2 * rad_mean, num_half))))
                    * radius_mean.units)
    # calculate the radii based on FWHM
    else:
        # calculate pdf
        t = (1 - pdi**2) / pdi**2
        pdf_range = size_distribution(diam_range, 2 * radius_mean, t).magnitude
        rad_range = diam_range.magnitude / 2

        # find radius at maximum of pdf
        max_rad_ind = np.argmax(pdf_range)
        max_rad = rad_range[max_rad_ind]

        # calculate the list of radii
        # This algorithm starts by finding the radius at the FWHM on either
        # side of the maximum. Then is finds the radius at the FW(3/4)M, then
        # FW(1/4)M, then FW(7/8)M, then FW(5/8)M, then FW(3/8)M...
        rad_list = [max_rad]
        num = 1
        denom = 2
        for i in range(0, num_diam - 1, 2):
            rad_ind_1 = np.argmin(np.abs(pdf_range[0:int(num_pdf_points/2)]
                                         - (num / denom) * np.max(pdf_range)))
            rad_ind_2 = np.argmin(np.abs(pdf_range[int(num_pdf_points/2):]
                                         - (num / denom) * np.max(pdf_range)))
            rad_list.append(rad_range[rad_ind_1])
            rad_list.append(rad_range[300 + rad_ind_2])
            if num == 1:
                denom = 2 * denom
                num = denom - 1
            else:
                num = num - 2

        # put the list in order and make it into a numpy array
        rad_list.sort()
        rad_list = np.array(rad_list)*radius_mean.units

    # plot the radii over the pdf
    if plot:    # pragma: no cover
        # calculate t for distrubtion
        t = (1 - pdi**2) / pdi**2

        # calculate pdf
        pdf = size_distribution(2*rad_list, 2 * radius_mean, t)
        if equal_spacing:
            pdf_range = size_distribution(diam_range, 2 * radius_mean, t)

        plt.figure()
        plt.scatter(2 * rad_list, pdf, s=45, color=[0.8,0.3,0.3])
        plt.plot(diam_range, pdf_range, linewidth=2.5)
        plt.xlabel('diameter (' + str(radius_mean.units) + ')')
        plt.ylabel('probability density')

    # calc diameter from radius
    diam_list = 2 * rad_list

    return diam_list

def sample_diams(pdi, diam_list, diam_mean, ntrajectories_bulk, nevents_bulk,
                 rng=None):
    '''
    Sample the radii to simulate polydispersity in the bulk Monte Carlo
    simulation

    Parameters
    ----------
    pdi: float
        polydispersity index of the distribution
    diam_list: 1d numpy array
        list of diams from which to sample in polydisperse bulk Monte Carlo
    diam_mean: float, sc.Quantity
        mean diameter of the distribution
    ntrajectories_bulk: int
        number of trajectories in the bulk Monte Carlo simulation
    nevents_bulk: int
        number of trajectories in the bulk Monte Carlo simulation
    rng: numpy.random.Generator object (default None)
        random number generator.  If not specified, use the default
        generator initialized on loading the package

    Returns
    -------
    diams_sampled: 2d array (shape nevents_bulk, ntrajectories_bulk)
        array of the samples microsphere diameters for polydisperity in the
        bulk Monte Carlo calculations
    '''

    if rng is None:
        rng = sc.rng

    # calculate t for distrubtion
    t = (1 - pdi**2) / pdi**2

    # calculate pdf
    pdf = size_distribution(diam_list, diam_mean, t)
    pdf_norm = pdf / np.sum(pdf)

    # sample diameter distribution
    diams_sampled = np.reshape(rng.choice(diam_list.magnitude,
                                          ntrajectories_bulk*nevents_bulk,
                                          p=pdf_norm),
                               (nevents_bulk, ntrajectories_bulk))

    return diams_sampled

def sample_concentration(p, ntrajectories_bulk, nevents_bulk, rng=None):
    '''
    Sample the radii to simulate polydispersity in the bulk Monte Carlo
    simulation using pre-calculated probabilities

    Parameters
    ----------
    p: 1d numpy array
        probability distribution of parameters in rad_list
    ntrajectories_bulk: int
        number of trajectories in the bulk Monte Carlo simulation
    nevents_bulk: int
        number of trajectories in the bulk Monte Carlo simulation
    rng: numpy.random.Generator object (default None)
        random number generator.  If not specified, use the default
        generator initialized on loading the package

    Returns
    -------
    params_sampled: 2d array (shape nevents_bulk, ntrajectories_bulk)
        array of the sample parameter for polydisperity in the bulk
        Monte Carlo calculations
    '''
    if rng is None:
        rng = sc.rng

    # sample distribution
    param_list = np.arange(np.size(p)) + 1

    params_sampled = np.reshape(rng.choice(param_list,
                                           (ntrajectories_bulk
                                            *nevents_bulk), p=p),
                                (nevents_bulk, ntrajectories_bulk))

    return params_sampled


def sample_angles_step_poly(nevents_bulk, ntrajectories_bulk, p_sphere,
                            params_sampled, mu_scat_bulk, param_list=None,
                            rng=None):
    '''
    Calculate the list of radii to sample from for a given polydispersity and
    number of radii. This function is used specifically to calculate a list of
    radii to sample in the polydisperse bulk Monte Carlo model.

    Parameters
    ----------
    ntrajectories_bulk: int
        number of trajectories in the bulk Monte Carlo simulation
    nevents_bulk: int
        number of trajectories in the bulk Monte Carlo simulation
    p_sphere: 2d array (shape number of sphere types, number of angles)
        phase function for a sphere, found from a Monte Carlo simulation
        with spherical boundary conditions
    params_sampled: 2d array (shape nevents_bulk, ntrajectories_bulk)
        array of the sampled microsphere parameters (could be radius or
        diameter) for polydisperity in the bulk Monte Carlo calculations
    mu_scat_bulk: 1d array (sc.Quantity, length number of sphere types)
        scattering coefficient for a sphere, calculated using Monte Carlo
        simulation with spherical boundary conditions
    param_list: 1d numpy array
        list of parameters (usually radius or diameter) from which to sample
        in polydisperse bulk Monte Carlo
    rng: numpy.random.Generator object (default None)
        random number generator.  If not specified, use the default
        generator initialized on loading the package

    Returns
    -------
    sintheta, costheta, sinphi, cosphi: ndarray
        Sampled scattering and azimuthal angles sines and cosines.
    step:ndarray
        Sampled step sizes for all trajectories and scattering events
    theta, phi: ndarray
        Sampled scattering and azimuthal angles
    '''
    if rng is None:
        rng = sc.rng

    # get param_list
    if param_list is None:
        param_list = np.arange(p_sphere.shape[0]) + 1
    elif isinstance(param_list, sc.Quantity):
        param_list = param_list.magnitude

    # get scattering length from scattering coefficient
    lscat = 1/mu_scat_bulk

    # Sample phi angles
    rand = rng.random((nevents_bulk,ntrajectories_bulk))
    phi = 2 * np.pi * rand
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    # Sample theta angles and calculate step size based on sampled radii
    theta = np.zeros((nevents_bulk, ntrajectories_bulk))
    lscat_rad_samp = np.zeros((nevents_bulk, ntrajectories_bulk))
    angles = np.linspace(0.01, np.pi, p_sphere.shape[1])

    # loop through all the radii, finding the positions of each radius
    # in the sampled radii array, and assigning the appropriate phase fun and
    # lscat for each one
    for j in range(p_sphere.shape[0]):
        ind_ev, ind_tr = np.where(params_sampled == param_list[j])

        if ind_ev.size == 0:
            continue
        prob = p_sphere[j,:] * np.sin(angles) * 2 * np.pi
        prob_norm = prob / np.sum(prob)

        # sample step sizes
        rand = rng.random(ind_ev.size)
        lscat_rad_samp[ind_ev, ind_tr] = (-np.log(1.0 - rand)
                                          * lscat[j].magnitude)

        # sample angles
        theta[ind_ev, ind_tr] = rng.choice(angles, ind_ev.size, p=prob_norm)

    # calculate sines, cosines, and step
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    step = (lscat_rad_samp * np.ones((nevents_bulk, ntrajectories_bulk))
            * lscat.units)

    # This function samples one extra step for each angle than is needed.
    # Whereas sample_angles was corrected to use nevents = nevents-1, this
    # function was not. So we correct the lengths of the arrays for the angles
    # before returning them. To ensure that MC tests give the same resuts
    # (which requires the same number of random numbers to be generated, we do
    # the correction here (for now).

    return sintheta[:-1], costheta[:-1], sinphi[:-1], \
        cosphi[:-1], step, theta[:-1], phi[:-1]
