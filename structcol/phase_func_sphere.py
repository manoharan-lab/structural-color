# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:34:06 2018

@author: stephenson
"""

import numpy as np
import structcol as sc
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from numpy.random import random as random
from numpy.random import choice as choice
from scipy.spatial.distance import cdist

def calc_random_points(radius, ntraj):
    # randomly choose x-positions within sphere radius
    data_x = 2*radius*(random((1,ntraj))-.5)
    data_x = data_x[0]

    # randomly choose y-positions within sphere radius contrained by x-positions
    data_y = np.zeros(ntraj)
    for i in range(ntraj):    
        data_y[i] = 2*np.sqrt(radius**2-data_x[i]**2) * (random((1))-.5)
        
        # calculate z-positions from x- and y-positions
        data_z = np.sqrt(radius**2 - data_x**2 - data_y**2)*choice([-1,1], size = len(data_x))
    
    return (data_x, data_y, data_z)
    
def plot_exit_points(data_x, data_y, data_z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')
    ax.set_title('exit positions')
    ax.view_init(-164,-155)
    ax.plot(data_x, data_y, data_z, '.')

def calc_pdf(x, y, z, radius, plot = False):
    # calculate pdf
    theta_phi = np.zeros((2, len(x)))

    theta_phi[0,:] = np.arccos(z/radius) # theta
    theta_phi[1,:] = np.arctan2(y,x) + np.pi # phi
    
    # nu with reflections
    nu_phi = np.zeros((2, 3*len(x)))
    nu_phi[0,len(x):2*len(x)] = (np.cos(theta_phi[0,:]) + 1)/2
    nu_phi[0,0:len(x)] = (-(np.cos(theta_phi[0,:])+1)-1 + 1)/2
    nu_phi[0,2*len(x):3*len(x)] = (-(np.cos(theta_phi[0,:])-1)+1 + 1)/2
    
    if plot == True:
        # phi with reflections
        nu_phi[1, len(x):2*len(x)] = theta_phi[1,:]
        nu_phi[1, 0:len(x)] = -theta_phi[1,:] 
        nu_phi[1, 2*len(x):3*len(x)] = -(theta_phi[1,:]-2*np.pi) + 2*np.pi
        
        plt.figure()
        plt.title('phi plus reflections')
        sns.distplot(nu_phi[1,:], rug = True, hist = False)
        
        plt.figure()
        ax = plt.subplot(111,projection = 'polar')
        ax.set_title('phase function phi')
        pdf_phi = gaussian_kde(nu_phi[1,:])
        phi = np.linspace(0.01, 2*np.pi, 200)
        phase_func_phi = pdf_phi(phi)
        print(np.sum(phase_func_phi))
        ax.plot(phi, phase_func_phi, linewidth = 3, color = [0.45, 0.53, 0.9])
        
        plt.figure()
        plt.title('costheta')
        sns.distplot(2*nu_phi[0,:]-1, rug = True, hist = False)
        
        plt.figure()
        plt.title('theta')
        sns.distplot(theta_phi[0,:]*180/np.pi, rug = True, hist = False)
        
        plt.figure()
        plt.title('phi')
        sns.distplot(theta_phi[1,:], rug = True, hist = False)
    
    return gaussian_kde(nu_phi[0,:])

def plot_colormap_pdf(pdf):
    theta, phi = np.mgrid[0:np.pi:200j, 0:2*np.pi:200j]
    nu = (np.cos(theta)+1)/2
    positions = np.vstack([nu.ravel(), phi.ravel()])

    Z = np.reshape(pdf(positions).T, theta.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap = plt.cm.gist_earth_r, extent = [0, np.pi, 0, 2*np.pi])
    print(Z.shape)
    ax.set_xlabel('theta')
    ax.set_xlim([0, np.pi])
    ax.set_ylabel('phi')
    ax.set_ylim([0, 2*np.pi])

def plot_pdf_theta(pdf, save=False):
        
    # plot pdf 1d
    plt.figure()
    ax = plt.subplot(111,projection = 'polar')
    ax.set_title('phase function microsphere')
    theta = np.linspace(.01,np.pi,200)
    nu = (np.cos(theta) + 1)/2
    phase_func = pdf(nu)/np.sum(pdf(nu))
    ax.plot(theta, phase_func, linewidth =3, color = [0.45, 0.53, 0.9])
    ax.plot(-theta, phase_func, linewidth =3, color = [0.45, 0.53, 0.9])
    
    if save==True:
        plt.savefig('phase_fun.pdf')
        print(phase_func.shape)
        np.save('phase_function_data',phase_func)

def calc_directions(theta_sample, phi_sample, x_inter,y_inter, z_inter, k1, microsphere_radius):
    z_sample = microsphere_radius*np.cos(theta_sample)
    y_sample = microsphere_radius*np.sin(phi_sample)*np.sin(theta_sample)
    x_sample = microsphere_radius*np.cos(phi_sample)*np.sin(theta_sample)

    xa = np.vstack((x_sample,y_sample,z_sample)).T
    xb = np.vstack((x_inter,y_inter, z_inter)).T

    distances = cdist(xa,xb)
    ind = np.argmin(distances, axis=1)

    return k1[:,ind]
    