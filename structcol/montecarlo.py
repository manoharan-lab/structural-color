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

class Trajectory():   
    """    
    Class that describes trajectories of photons packets in a scattering
    and/or absorbing medium.

    Attributes
    ----------
    r : ndarray (structcol.Quantity [length]) 
        array of position vectors in cartesian coordinates of n trajectories 
    k : ndarray (structcol.Quantity [dimensionless]) 
        array of direction of propagation vectors in cartesian coordinates 
        of n trajectories after every scattering event 
    W : ndarray (structcol.Quantity [dimensionless]) 
        array of photon packet weights during for n trajectories
    nevents : int
        number of scattering events

    Methods
    -------
    absorb(mua, mus)
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
    
    def __init__(self, r, k, W, nevents):
        
        self.r = r 
        self.k = k  
        self.W = W  
        self.nevents = nevents          

    
    def absorb(self, mua, mus):
        """
        Calculates absorption of packet.
        
        mua: absorption coefficient
        mus: scattering coefficient
        """
        
        mut = mua + mus   
        dW = self.W * mua / mut    
        
        self.W = self.W - dW
        
        return self.W
    
    
    def scatter(self, sintheta, costheta, sinphi, cosphi):
        """
        Calculates the directions of propagation after scattering.
        
        sintheta, costheta, sinphi, cosphi: scattering and azimuthal angles 
        sampled from the phase function. 
        """
                
        kn = self.k
        eps = 1.e-4
        
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
        self.k = kn
        
        return self.k                        


    def move(self, lscat):
        """
        Calculates new positions in trajectories. 
        
        lscat: scattering length from Mie theory (step size in trajectories). 
        """
        
        displacement = self.r
        displacement[:, 1:, :] = lscat * self.k 
        
        self.r[0] = np.cumsum(displacement[0,:,:], axis=0)
        self.r[1] = np.cumsum(displacement[1,:,:], axis=0)
        self.r[2] = np.cumsum(displacement[2,:,:], axis=0)
                
        return self.r

    
    def plot_coord(self, ntraj, three_dim=False):
        """
        Plots the trajectories' cartesian coordinates as a function of 
        the number of scattering events (or 'time').
        three_dim = True: plots the coordinates in 3D.       
        """
        
        colormap = plt.cm.gist_ncar
        colors = itertools.cycle([colormap(i) for i in np.linspace(0, 0.9, ntraj)])
        
        f, ax = plt.subplots(3, figsize=(8,17), sharex=True)
        
        ax[0].plot(np.arange(len(self.r[0,:,0])), self.r[0,:,:], '-')
        ax[0].set_title('Positions during trajectories')
        ax[0].set_ylabel('x (' + str(self.r.units) + ')')
        
        ax[1].plot(np.arange(len(self.r[1,:,0])), self.r[1,:,:], '-')
        ax[1].set_ylabel('y (' + str(self.r.units) + ')')
        
        ax[2].plot(np.arange(len(self.r[2,:,0])), self.r[2,:,:], '-')
        ax[2].set_ylabel('z (' + str(self.r.units) + ')')
        ax[2].set_xlabel('scattering event')
  
        if three_dim == True:
            fig = plt.figure(figsize = (8,6))
            ax3D = fig.add_subplot(111, projection='3d')
            ax3D.set_xlabel('x (' + str(self.r.units) + ')')
            ax3D.set_ylabel('y (' + str(self.r.units) + ')')
            ax3D.set_zlabel('z (' + str(self.r.units) + ')')
            ax3D.set_title('Positions during trajectories')
            
            for n in np.arange(ntraj):
                ax3D.scatter(self.r[0,:,n], self.r[1,:,n], self.r[2,:,n], color=next(colors)) 
        #plt.show()

def initialize(nevents, ntraj):
    """
    Sets the trajectory's initial conditions (position, direction, and weight).
    
    nevents: number of scattering events
    ntraj: number of trajectories
    """
    
    # initial position
    r0 = np.zeros((3, nevents+1, ntraj))
    r0[0,0,:] = random((1,ntraj))
    r0[1,0,:] = random((1,ntraj))

    # initial direction
    eps = 1.e-9
    k0 = np.zeros((3, nevents, ntraj))
    k0[2,0,:] = 1. - eps
    
    # initial weight
    W0 = np.zeros((nevents, ntraj))
    W0[0,:] = 0.001                        # (figure out how to determine this) 
    
    return r0, k0, W0


def phase_function(radius, n_particle, n_matrix, angles, wavelen):
    """
    Calculates the phase function from Mie theory. 
    
    wavelen, radius, angles: must be entered as Quantity    
    angles: scattering angles (typically from 0 to pi)
    p: phase_function 
        
    p = diff. scatt. cross section / cscat    (Bohren and Huffmann 13.3) 
    diff. scat. cross section = S11 / k^2 
    p = S11 / (k^2 * cscat)
    """
        
    angles = angles.to('rad')
    ksquared = (2 * np.pi *n_matrix / wavelen)**2

    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)

    S2squared, S1squared = mie.calc_ang_dist(m, x, angles)
    S11 = (S1squared + S2squared)/2
    cscat = mie.calc_cross_sections(m, x, wavelen)[0]

    p = S11 / (ksquared * cscat)      
    
    return p


def scat_abs_length(radius, n_particle, n_matrix, volume_fraction, wavelen):
    """
    Calculates the scattering and absorption lengths from Mie theory. 
    
    radius, n_particle, n_matrix, wavelen: must be entered as Quantity to allow 
    specifying units
    """
    
    number_density = 3.0 * volume_fraction / (4.0 * np.pi * radius**3)
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    
    cscat = mie.calc_cross_sections(m, x, wavelen)[0]
    cabs = mie.calc_cross_sections(m, x, wavelen)[2]

    lscat = 1 / (cscat * number_density)
    lscat = lscat.to('um')
    
    labs = 1 / (cabs * number_density)
    labs = labs.to('um')
    
    return lscat, labs

    
def sampling(nevents, ntraj, p, angles):
    """
    Samples azimuthal angles from uniform distribution, 
    and scattering angles from phase function. 
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