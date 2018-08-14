# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:34:06 2018

@author: stephenson
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist 
import structcol as sc

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
    x_inter = x_inter[x_inter!=0]*radius
    
    # get the y-coordinate
    y_inter = norm[1,:]
    y_inter = y_inter[y_inter!=0]*radius
    
    # get the z-coordinate
    z_inter = norm[2,:]
    z_inter = z_inter[z_inter!=0]*radius
    
    return x_inter, y_inter, z_inter

def calc_pdf(x, y, z, radius, plot = False, phi_dependent = False, 
             nu_range = np.linspace(0.01, 1, 200), 
             phi_range = np.linspace(0, 2*np.pi, 300)):
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
    plot: boolean
        If set to True, the intermediate and final pdfs will be plotted
    phi_dependent: boolean
        If set to True, the returned pdf will require both a nu and a phi
        input
    
    Returns
    -------
    pdf: function, 1 or 2 arguments
        probability density function that requires an input of nu values if
        phi_dependent = False, and an input of nu and phi values is 
        phi_depenedent = True
        
    Notes
    -----
    the probability density function is calculated as a function of nu and phi
    instead of theta and phi to correct for the inequal areas on the sphere
    surface for equal spacing in theta. Theta is related to nu by:
        
        Theta = arccos(2*nu-1)
        
    see http://mathworld.wolfram.com/SpherePointPicking.html for more details
    
    '''
    
    # calculate thetas for each exit point
    theta = np.arccos(z/radius)
    
    # convert thetas to nus
    nu = (np.cos(theta) + 1) / 2
    
    # add reflections of data on to ends to prevent dips in distribution
    # due to edges
    nu_edge_correct = np.hstack((-nu, nu, -nu + 2))
    
    if not phi_dependent:
        # calculate the pdf kernel density estimate
        pdf = gaussian_kde(nu_edge_correct)
        
        if plot == True:
            # plot the distribution from data, with edge correction, and kde
            plot_dist_1d(nu_range, nu, nu_edge_correct, pdf(nu_range))
            plt.xlabel(r'$\nu$')
    
    else:
        # calculate phi for each exit point
        phi = np.arctan2(y,x) + np.pi
        
        # add reflections of data to ends to prevent dips in distribution 
        # due to edges
        phi_edge_correct = np.tile(np.hstack((-phi, phi, -phi + 4*np.pi)),3)
        nu_edge_correct = np.hstack((np.tile(-nu,3), np.tile(nu,3), np.tile(-nu+2,3)))        
        
        # calculate the pdf kernel density estimate
        pdf = gaussian_kde(np.vstack([nu_edge_correct,phi_edge_correct]))
        
        if plot == True:
            # plot the the calculated kernel density estimate in phi
            nu_2d, phi_2d = np.meshgrid(nu_range, phi_range)
            angle_range = np.vstack([nu_2d.ravel(), phi_2d.ravel()])
            pdf_vals = np.reshape(pdf(angle_range), nu_2d.shape)
            pdf_marg_nu = np.sum(pdf_vals, axis = 0)
            
            # plot the nu distribution from data, with edge correction, and kde
            plot_dist_1d(nu_range, nu, nu_edge_correct, pdf_marg_nu)
            plt.xlabel(r'$\nu$')

            # plot the phi distribution from data, with edge correction, and kde
            pdf_marg_phi = np.sum(pdf_vals, axis = 1)
            plot_dist_1d(phi_range, phi, phi_edge_correct, pdf_marg_phi)
            plt.xlabel(r'$\phi$')
        
    return pdf

def plot_phase_func(pdf, nu=np.linspace(0, 1, 200), phi=None, save=False):
    '''
    plots a given probability density function (pdf)
    
    if the provided probability density is a function of only nu,
    then the pdf is plotted against theta. We convert nu to theta because theta
    is the more commonly used physical parameter in spherical coordinates.
    
    if the provided probability density is a function of nu and phi,
    then the pdf is plotted against theta and phi as a heatmap.
    
    Parameters
    ----------
    pdf: function, 1 or 2 arguments
        probability density function that requires an input of nu values 
        or nu and phi values
       
    nu: 1d array-like
        y-coordinate of each trajectory at exit event
    phi: None or 1d array-like
        z-coordinate of each trajectory at exit event
    save: boolean
        
    Notes
    -----
    see http://mathworld.wolfram.com/SpherePointPicking.html for more details
    on conversion between theta and nu
    
    '''
    # convert nu to theta
    theta = np.arccos(2*nu-1)
    
    if phi is None:
        # calculate the phase function for theta points
        phase_func = pdf(nu)/np.sum(pdf(nu)*np.diff(nu)[0])    
        
        # make polar plot in theta
        plt.figure()
        ax = plt.subplot(111,projection = 'polar')
        ax.set_title(r'phase function in $\theta$')
        ax.plot(theta, phase_func, linewidth =3, color = [0.45, 0.53, 0.9])
        ax.plot(-theta, phase_func, linewidth =3, color = [0.45, 0.53, 0.9])
    else:
        # calculate the phase function for points in theta and phi
        theta_2d, phi_2d = np.meshgrid(theta, phi)
        nu_2d = (np.cos(theta_2d) + 1)/2
        angles = np.vstack([nu_2d.ravel(), phi_2d.ravel()])
        pdf_vals = np.reshape(pdf(angles), theta_2d.shape)
        phase_func = pdf_vals/np.sum(pdf_vals*np.diff(phi)[0]*np.diff(theta)[0])
        
        # make heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(phase_func, cmap = plt.cm.gist_earth_r, extent = [theta[0], theta[-1], phi[0], phi[-1]])
        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
        ax.set_xlim([theta[0], theta[-1]])
        ax.set_ylim([phi[0], phi[-1]])
        fig.colorbar(cax)
        
    if save==True:
        plt.savefig('phase_fun.pdf')
        np.save('phase_function_data',phase_func)
        

    
def plot_dist_1d(var_range, var_data, var_data_edge_correct, pdf_var_vals):
    '''

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
    sns.distplot(var_data, rug = True, hist = False, 
                 label = 'distribution from data')
    
    # plot the kde using seaborn, from edge corrected data
    sns.distplot(var_data_edge_correct, rug = True, hist = False, 
                 label = 'distribution with edge correction')
    
    # renormalize the pdf 
    pdf_norm = pdf_var_vals/np.sum(pdf_var_vals*np.diff(var_range)[0])
    
    # plot
    plt.plot(var_range, pdf_norm, 
             label = 'kernel density estimate, correctly normalized')
    plt.legend()
    plt.xlim([var_range[0],var_range[-1]])
    plt.ylabel('probability density')
    

def calc_directions(theta_sample, phi_sample, x_inter,y_inter, z_inter, k1, microsphere_radius):
    '''
    calculates directions of exit trajectories
    
    
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
    radius: float-like
        radius of sphere boundary
        
    Returns
    -------
    k1: 2d array
        direction vector for trajectories
    
    '''
    z_sample = microsphere_radius*np.cos(theta_sample)
    y_sample = microsphere_radius*np.sin(phi_sample)*np.sin(theta_sample)
    x_sample = microsphere_radius*np.cos(phi_sample)*np.sin(theta_sample)

    xa = np.vstack((x_sample,y_sample,z_sample)).T
    xb = np.vstack((x_inter,y_inter, z_inter)).T

    distances = cdist(xa,xb)
    ind = np.argmin(distances, axis=1)

    return k1[:,ind]

def plot_exit_points(x, y, z, radius, plot_dimension = '3d'):
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
    plot_dimension: string
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
        plt.ylim([0, 2*np.pi])
        
    if plot_dimension == '3d':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x ' + str(unit))
            ax.set_ylabel('y ' + str(unit))
            ax.set_zlabel('z ' + str(unit))
            ax.set_title('exit positions')
            ax.view_init(-164,-155)
            ax.plot(x, y, z, '.')
            
            
def calc_d_avg(volume_fraction, radius):
    '''
    calculates the average spacing between structured spheres in a bulk film,
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
    number_density = volume_fraction/(4/3*np.pi*radius**3)
    
    # calculate the average interparticle spacing
    d_avg = 2*(3/(4*np.pi*number_density))**(1/3)
    
    return d_avg
    
def calc_lscat(refl_per_traj, trans_per_traj, trans_indices, volume_fraction, radius):
    '''
    calculates the scattering length from the formula:
        
        lscat = 1/(number density * total scattering cross section)
        
    where the total scattering cross section is found by integrating the 
    fraction of scattered light and multiplying by the initial are
    
    total scattering cross section = power scattered / incident intensity
                                   = power scattered / (incident power / incident area)
                                   = power scattered / incident power * 2*pi*radius**2
                                   = (scattered fraction)*2*pi*radius**2
    
    Parameters
    ----------
    refl_per_traj: 1d array
        array of trajectory weights that exit through reflection, normalized
        by the total number of trajectories
    trans_per_traj: 1d array
        array of trajectory weights that exit through transmission, normalized
        by the total number of trajectories
    trans_indices: 1d array
        array of event indices at which trajectories exit structured sphere
        through transmission
    volume_fraction: float-like
        volume fraction of structured spheres in a bulk film
    radius: float-like
        radius of structured spheres in a bulk film
        
    Returns
    -------
    lscat: float-like
        scattering length for bulk film of structured spheres
    
    '''
    
    # calculate the number density
    number_density = volume_fraction/(4/3*np.pi*radius**3)
    
    # remove transmission contribution from trajectories that did not scatter
    trans_per_traj[trans_indices == 1] = 0
    
    # calculate the total scattering cross section
    tot_scat_cross_section = np.sum(refl_per_traj + trans_per_traj)*2*np.pi*radius**2
    
    # calculate the scattering length
    lscat = 1/(number_density*tot_scat_cross_section)
    
    return lscat
    
    
    
    
    
    
    
    
    
    
    