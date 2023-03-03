# -*- coding: utf-8 -*-
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
This module 

References
----------
[1] 

.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>

"""

import pymie as pm
import copy
from pymie import mie, size_parameter, index_ratio
from pymie import multilayer_sphere_lib as msl
from . import model
from . import montecarlo as mc
from . import phase_func_sphere as pfs
from . import refraction
from . import normalize
from . import select_events
import numpy as np
from numpy.random import random as random
import structcol as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import warnings
from scipy.optimize import fsolve
import seaborn as sns


def calc_pol_frac(trajectories, indices):
    '''
    calculates polarization contribution to the event type indicate by indices
    (usually reflection or transmission) for each polariztion component,
    in global, cartesian coordinates
    
    Parameters
    ----------
    trajectories: Trajectory object
        trajectories from Monte Carlo calculation
    indices: 1d array (length ntrajectories)
        event indices of interest, often indices of reflected or transmitted 
        events
    
    Returns
    -------
    pol_frac_x: float
        x-polarized intensity fraction
    pol_frac_y: float
        y-polarized intensity fraction
    pol_frac_z: float
        z-polarized intensity fraction
    '''
    polarization = trajectories.polarization
    if isinstance(polarization, sc.Quantity):
        polarization = polarization.magnitude
        
    pol_x = select_events(polarization[0,:,:], indices) 
    pol_y = select_events(polarization[1,:,:], indices)
    pol_z = select_events(polarization[2,:,:], indices)
    
    ntrajectories = len(indices)
    pol_frac_x = np.sum(np.abs(pol_x)**2)/ntrajectories
    pol_frac_y = np.sum(np.abs(pol_y)**2)/ntrajectories
    pol_frac_z = np.sum(np.abs(pol_z)**2)/ntrajectories
    
    return pol_frac_x, pol_frac_y, pol_frac_z
    
def calc_pol_frac_phase(trajectories, indices, refl_per_traj):
    '''
    calculates polarization contribution to the event type indicate by indices
    (usually reflection or transmission) for each polariztion component,
    in global, cartesian coordinates
    
    Parameters
    ----------
    trajectories: Trajectory object
        trajectories from Monte Carlo calculation
    indices: 1d array (length ntrajectories)
        event indices of interest, often indices of reflected or transmitted 
        events
    refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory from Monte Carlo simulation.
        Sum should be total reflectance from Monte Carlo calculation, 
        without corrections for Fresnel reflected and stuck weights.
    event_dist: boolean
        determines whether to sum fractions over all events or leave as a function
        of events
    Returns
    -------
    pol_frac_x: float
        x-polarized intensity fraction
    pol_frac_y: float
        y-polarized intensity fraction
    pol_frac_z: float
        z-polarized intensity fraction
    '''
    
    #( _,_,
    #refl_x_ev,
    #refl_y_ev,
    #refl_z_ev) = calc_phase_refl_trans_event(refl_per_traj, np.array([0]), np.array([0]), 
    #                      indices, np.array([0]), trajectories)
    calc_refl_phase_time(traj_time, trajectories, refl_indices, refl_per_traj,
                         bin_width=sc.Quantity(40,'fs'),
                         convolve=False, components=False)
                                                       
    pol_frac_x = np.sum(refl_x)
    pol_frac_y = np.sum(refl_y)
    pol_frac_z = np.sum(refl_z)
    
    return pol_frac_x, pol_frac_y, pol_frac_z

  
def calc_refl_co_cross_phase(trajectories, step, indices, radius, 
                             volume_fraction, n_particle, n_sample, 
                             wavelength, det_theta, refl_per_traj,
                             traj_time,
                             bin_width=sc.Quantity('40 fs'),
                             convolve=False,
                             concentration=None,
                             radius2=None):
    '''
    Calculates the co, cross, and perp polarized reflectance, where 'perp' in this 
    case refers to light polarized in the direction perpendicular to the co/cross
    plane. Only a small fraction of light of the light should be perpendicular 
    to the co/cross plane, since light cannot be polarized in the direction of
    propagation, and the detected signal should be composed mostly of light 
    propagating perpendicularly to the co/cross plane. 
    
    Parameters
    ----------
    trajectories: Trajectory object
        trajectories from Monte Carlo calculation
    indices: 1d array (length ntrajectories)
        event indices of interest, often indices of reflected or transmitted 
        events
    det_theta: float-like
        angle between the normal to the sample (-z axis) and the center of the 
        detector 
    refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory from Monte Carlo simulation.
        Sum should be total reflectance from Monte Carlo calculation, 
        without corrections for Fresnel reflected and stuck weights.
    event_dist: boolean
        determines whether returns will be arrays to show reflectance values
        as a function of event
    Returns
    -------
    refl_co: float or 1d array 
        co-polarized reflectance
    refl_cr: float or 1d array
        cross-polarized reflectance
    refl_perp: float or 1d array
        reflectance perpendicularly polarized to co and cross
    '''
    
    (pol_frac_x, 
     pol_frac_y, 
     pol_frac_z) = calc_refl_phase_time(traj_time, trajectories, indices, 
                                        refl_per_traj,
                                        bin_width=bin_width,
                                        convolve=convolve, components=True)
    # incorporate geometry of the goniometer setup
    refl_co = pol_frac_z*np.sin(det_theta) + pol_frac_x*np.cos(det_theta)
    refl_cr = pol_frac_y
    refl_perp = -pol_frac_z*np.cos(det_theta) + pol_frac_x*np.sin(det_theta)
    
    
    return (refl_co, refl_cr, refl_perp)
 

  
def calc_refl_co_cross(trajectories, indices, det_theta):
    '''
    Calculates the co, cross, and perp polarized reflectance, where 'perp' in this 
    case refers to light polarized in the direction perpendicular to the co/cross
    plane. Only a small fraction of light of the light should be perpendicular 
    to the co/cross plane, since light cannot be polarized in the direction of
    propagation, and the detected signal should be composed mostly of light 
    propagating perpendicularly to the co/cross plane.
    
    Parameters
    ----------
    trajectories: Trajectory object
        trajectories from Monte Carlo calculation
    indices: 1d array (length ntrajectories)
        event indices of interest, often indices of reflected or transmitted 
        events
    det_theta: float-like
        angle between the normal to the sample (-z axis) and the center of the 
        detector 
        
    Returns
    -------
    refl_co: float
        co-polarized reflectance
    refl_cr: float
        cross-polarized reflectance
    refl_perp: float
        reflectance perpendicularly polarized to co and cross
    refl_co_per_traj: 1d array (length: ntraj)
        co-polarized reflectance for each trajectory
    refl_cr_per_traj: 1d array (length: ntraj)
        cross-polarized reflectance for each trajectory
    refl_perp_per_traj: 1d array (length: ntraj)
        perpendicularly polarized reflectance for each trajectory
    '''
    # first calculate refl co/cross per trajctory
    
    # calculate polarization intensity per traj
    ntrajectories = len(indices)
    pol_x_abs = np.abs(select_events(trajectories.polarization[0,:,:], indices))**2/ntrajectories 
    pol_y_abs = np.abs(select_events(trajectories.polarization[1,:,:], indices))**2/ntrajectories
    pol_z_abs = np.abs(select_events(trajectories.polarization[2,:,:], indices))**2/ntrajectories
    
    # this comes from the geometry of the goniometer setup
    refl_co_per_traj = pol_z_abs*np.sin(det_theta) + pol_x_abs*np.cos(det_theta)
    refl_cr_per_traj = pol_y_abs
    refl_perp_per_traj = -pol_z_abs*np.cos(det_theta) + pol_x_abs*np.sin(det_theta)    
    
    # calculate pol fraction
    pol_frac_x, pol_frac_y, pol_frac_z = calc_pol_frac(trajectories, indices)
    
    # incorporate geometry of the goniometer setup
    refl_co = pol_frac_z*np.sin(det_theta) + pol_frac_x*np.cos(det_theta)
    refl_cr = pol_frac_y
    refl_perp = -pol_frac_z*np.cos(det_theta) + pol_frac_x*np.sin(det_theta)
    
    
    return (refl_co, refl_cr, refl_perp, refl_co_per_traj, 
            refl_cr_per_traj, refl_perp_per_traj)
   
def calc_refl_pol_angle(angle, refl_co, refl_cr, 
                        refl_events_co=None, refl_events_cr=None):
    '''
    Calculates reflectance for spectrum angle between co and cross    
    
    Parameters
    ----------
    angle: float-like (sc.Quantity)
        angle between co and cross polarized, where 0 deg is co and 90 deg is cross
    refl_co: float
        co-polarized reflectance
    refl_cr: float
        cross-polarized reflectance
    refl_events_co: 1d array (length: ntraj)
        co-polarized reflectance for each event
    refl_events_cr: 1d array (length: ntraj)
        cross-polarized reflectance for each event

    Returns
    -------
    refl_pol_ang: float
        reflectance for the polarization angle
    refl_pol_ang_events: 1d array (length nevents)
        reflectance for the polarization angle as a function of event number
    
    '''
    if isinstance(angle, sc.Quantity):
        angle = angle.to('rad').magnitude
    
    refl_pol_ang = refl_co*(np.cos(angle))**2 + refl_cr*(np.sin(angle))**2
    
    if refl_events_co is not None:
        refl_pol_ang_events = (refl_events_co*(np.cos(angle))**2
                                    + refl_events_cr*(np.sin(angle))**2)
        return (refl_pol_ang, refl_pol_ang_events)
    else:
        return refl_pol_ang

def calc_phase_refl_trans_event(refl_per_traj, inc_refl_per_traj, trans_per_traj, 
                          refl_indices, trans_indices, trajectories):
    '''
    DEPRECATED
    
    Returns reflectance and transmittance as a function of event number
    
    Parameters
    ----------
    refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory from Monte Carlo simulation.
        Sum should be total reflectance from Monte Carlo calculation, 
        without corrections for Fresnel reflected and stuck weights.
    inc_refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory at the sample interface. 
        This contribution comes from the Fresnel reflection as the light
        enters the sample
    trans_per_traj: 1d array (length: ntrajectories)
        Transmittance contribution for each trajectory from Monte Carlo simulation.
        Sum should be total transmittance from Monte Carlo calculation,
        without corrections for Fresnel reflected and stick weights.
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event. 
    trans_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    nevents: int
        number of events for which Monte Carlo Calculation is run 
    
    Returns
    -------
    refl_intensity_phase_events: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event. 
    trans_events: 1d array (length: 2*nevents + 1)
        transmittance contribution for each event.
        
    Note: This function is located in detector.py instead of 
    event_distribution.py because it is an essental step for reflectance detection
    including phase. It could be argued that this function belongs instead 
    in event_distribution.py, but this requires moving around a variety of other
    helper functions in order to avoid cyclic importing between event_distribution.py 
    and detector.py. For simplicity, we've chosen for now to leave this function 
    detector.py.
    '''
    nevents = trajectories.nevents
    ntraj = len(trajectories.polarization[0,0,:])

    # write expression for unweighted field 
    traj_field_x =  trajectories.polarization[0,:,:]*np.exp(trajectories.phase[0,:,:]*1j) 
    traj_field_y =  trajectories.polarization[1,:,:]*np.exp(trajectories.phase[1,:,:]*1j) 
    traj_field_z =  trajectories.polarization[2,:,:]*np.exp(trajectories.phase[2,:,:]*1j)  
    refl_events = np.zeros(2*nevents + 1)
    tot_field_x_ev = np.zeros(2*nevents + 1, dtype=complex)
    tot_field_y_ev = np.zeros(2*nevents + 1, dtype=complex)
    tot_field_z_ev = np.zeros(2*nevents + 1, dtype=complex)
    trans_events = np.zeros(2*nevents + 1)
    # add fresnel reflection at first interface

    refl_events[0] = np.sum(inc_refl_per_traj)

    #loop through all events
    for ev in range(1, nevents):
        # find trajectories that were reflected/transmitted at this event
        traj_ind_refl_ev = np.where(refl_indices == ev)[0]
        traj_ind_trans_ev = np.where(trans_indices == ev)[0]
        
        # write expression for field including weight 
        # since the trajectory weights are in units of intensity, we take the
        # square root to find the amplitude for the field
        w = np.sqrt(refl_per_traj[traj_ind_refl_ev]*ntraj)
        
        # add reflectance/transmittance due to trajectories 
        # reflected/transmitted at this event
        tot_field_x_ev[ev] += np.sum(w*traj_field_x[ev,traj_ind_refl_ev])
        tot_field_y_ev[ev] += np.sum(w*traj_field_y[ev,traj_ind_refl_ev])
        tot_field_z_ev[ev] += np.sum(w*traj_field_z[ev,traj_ind_refl_ev])

        # TODO fix this for transmittance
        trans_events[ev] += np.sum(trans_per_traj[traj_ind_trans_ev])
        
    # calculate intensity as E*E
    intensity_x_ev = np.conj(tot_field_x_ev)*tot_field_x_ev
    intensity_y_ev = np.conj(tot_field_y_ev)*tot_field_y_ev
    intensity_z_ev = np.conj(tot_field_z_ev)*tot_field_z_ev

    
    # add the x,y, and z intensity
    refl_intensity_phase_events = intensity_x_ev + intensity_y_ev + intensity_z_ev
    
    # normalize
    intensity_incident = np.sum(trajectories.weight[0,:]) # assumes normalized light is incoherent
    refl_phase_events = refl_intensity_phase_events/intensity_incident
    refl_x_ev = intensity_x_ev/intensity_incident
    refl_y_ev = intensity_y_ev/intensity_incident
    refl_z_ev = intensity_z_ev/intensity_incident
    
    return refl_phase_events, refl_x_ev, refl_y_ev, refl_z_ev
  
def calc_refl_phase(trajectories, refl_indices, refl_per_traj):
    '''
    DEPRECATED
    
    Calculates the reflectance including contributions from phase as a function 
    of events.
    
    Paramters
    ---------
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event. 
    refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory from Monte Carlo simulation.
        Sum should be total reflectance from Monte Carlo calculation, 
        without corrections for Fresnel reflected and stuck weights.
    
    Returns
    -------
    refl_phase: float
        reflectance including contributions from phase
    refl_phase_events: 1d array (length: 2*nevents + 1)
        reflectance including contributions from phase as a function of events
    
    '''     
    
    # get the reflectance per event
    refl_phase_events, _, _, _, _ = calc_phase_refl_trans_event(refl_per_traj, np.array([0]), np.array([0]), 
                          refl_indices, np.array([0]), trajectories)
    
    # sum to get the total reflectance
    refl_phase = np.sum(refl_phase_events)
    
    return refl_phase, refl_phase_events
    
def calc_traj_time(step, exit_indices, radius, volume_fraction, 
                   n_particle, n_sample, mu_scat, wavelength, concentration=None,
                   radius2=None):
    '''
    Calculates the amount of time each trajectory spends scattering in the 
    sample before exit    
    
    parameters:
    ----------
    step: 2d array (structcol.Quantity [length])
        Step sizes between scattering events in each of the trajectories.
    exit_indices: 1d array (length: ntrajectories)
        event number at exit for each trajectory. Input refl_indices if you want
        to only consider reflectance and trans_indices if you want to only
        consider transmittance. Input refl_indices + trans_indices if you
        want to consider both
    radius: float (structcol.Quantity [length])
        Radius of particle.
    volume_fraction: float
        Volume fraction of particles.
    n_particle: float
        Index of refraction of particle.
    n_sample: float
        Index of refraction of sample.
    wavelength: float (structcol.Quantity [length])
        Wavelength.
    concentration: 2-element array (structcol.Quantity [dimensionless])
        Concentration of each scatterer if the system is binary. For 
        polydisperse monospecies systems, specify the concentration as 
        [0., 1.]. The concentrations must add up to 1. If system is monodisperse
        and monospecies, do not specify.
    radius2: float (structcol.Quantity [length])
        Mean radius of secondary scatterer. Specify only if the system is 
        binary, meaning that there are two mean particle radii (for example,
        one small and one large).
    
    returns:
    -------
    traj_time: 1d array (structcol.Quantity [time], length ntraj)
        time each trajectory spends scattering inside the sample before exit
    travel_time: 1d array (structcol.Quantity [time], length ntraj)
        time each trajectory spends travelling inside the sample before exit
    dwell_time: float (structcol.Quantity [time])
        time duration of scattering inside a particle
    '''    
    
    # calculate the path length
    ntraj = len(exit_indices)
    path_length_traj = sc.Quantity(np.zeros(ntraj),'um')
    
    for i in range(0, ntraj):
        path_length_traj[i] = np.sum(step[:exit_indices[i],i])
    stuck_traj_ind = np.where(path_length_traj.magnitude==0)[0]

    # calculate the time passed based on distance travelled
    c = sc.Quantity(2.99792e8,'m/s')
    velocity = c/np.real(n_sample.magnitude)
    travel_time = path_length_traj/velocity
    
    # calculate the dwell time in a scatterer    
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelength, n_sample, radius)
    nstop = mie._nstop(x)
    W = mie.calc_energy(radius, n_sample, m, x, nstop)
    rho=model._number_density(volume_fraction, radius)
    
    if len(mu_scat)>1:
        mu_scat = mu_scat[0]
    cscat = mu_scat/rho
    dwell_time = W/(cscat*c)
        
    # add the dwell times and travel times
    traj_time = travel_time + dwell_time
    
    # set traj_time = 0 for stuck trajectories
    traj_time[stuck_traj_ind]=sc.Quantity(0,'fs')
    
    # change units to femtoseconds and discard imaginary part
    traj_time = traj_time.to('fs')
    traj_time = np.real(traj_time.magnitude)
    traj_time = sc.Quantity(traj_time,'fs')
    
    return traj_time, travel_time, dwell_time
    

def calc_refl_phase_time(traj_time, trajectories, refl_indices, refl_per_traj,
                         bin_width=sc.Quantity(40,'fs'),
                         bin_width_pos = sc.Quantity(10, 'um'),
                         convolve=False, components=False):
    '''
    Calculates the reflectance including phase, by considering trajectories
    that exit at the same time to be coherent. To do this we, must bin trajectories
    with similar exit times and add their fields. Then we convolve the 
    reflectance as a function of time with a step function in order to 
    give a steady state value for the reflectance
    
    parameters:
    ----------
    traj_time: 1d array (structcol.Quantity [time], length ntraj)
        time each trajectory spends traversing inside the sample before exit
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    refl_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel 
        contributions
    bin_width: float (structcol.Quantity [time])
        size of time bins for creating field versus time. Should be set equal to
        coherence time of source
   convolve: boolean
        determines whether the reflectance vs time curve should be convolved 
        with a step function input signal to simulate a constant input signal
        instead of a pulse input signal. Note that this is not currently fully
        implemented and will likely return a noisy result with a magnitude
        smaller than expected. 
    components: boolean
    
    returns:
    -------
    refl_steady: float
        steady state reflectance including contributions from phase
    refl_phase_per_traj: ndarray (length: ntraj)
        phase corrected reflectance as a function of event number. The cross
        terms for interfering trajectories are distributed to the different 
        trajectories by relative weight.
    '''
    ntraj = len(trajectories.polarization[0,0,:]) 
    nevents = len(trajectories.polarization[0,:,0]) 
    traj_time = traj_time.to('fs').magnitude
    bin_width = int(bin_width.to('fs').magnitude)
    bin_width_pos = int(bin_width_pos.to('um').magnitude) 
    
    # create the historgram  
    traj_time_cut = traj_time[traj_time>0]
    
    no_refl_warn = "No trajectories were reflected. Check sample parameters or increase number of trajectories."
    if len(traj_time_cut)==0:
        warnings.warn(no_refl_warn)
        n_bins = 0
    else:
        #bin_range=np.cumsum(bin_width*(np.arange(0,nevents+1)))+1#range(1,int(round(max(traj_time_cut))+ bin_width), bin_width)
        bin_range = range(1,int(round(max(traj_time_cut))+ bin_width), bin_width)     
        hist, bin_edges = np.histogram(traj_time_cut, bins = bin_range)   
        bin_min = bin_edges[0::1]
        bin_max = bin_edges[1::1]
        n_bins = len(hist)
        
    pos_x = select_events(trajectories.position[0,:,:].magnitude, refl_indices)
    pos_y = select_events(trajectories.position[1,:,:].magnitude, refl_indices)
    pos_x_cut = pos_x[pos_x!=0]
    pos_y_cut = pos_y[pos_y!=0]    
    
    bin_range_x = range(int(round(min(pos_x_cut))),int(round(max(pos_x_cut))+ bin_width_pos), bin_width_pos)
    bin_range_y = range(int(round(min(pos_y_cut))),int(round(max(pos_y_cut))+ bin_width_pos), bin_width_pos)
    hist_pos, bin_edges_x, bin_edges_y =  np.histogram2d(pos_x_cut, pos_y_cut, bins = [bin_range_x, bin_range_y])
    x_bin_min = bin_edges_x[0::1]
    x_bin_max = bin_edges_x[1::1]
    y_bin_min = bin_edges_y[0::1]
    y_bin_max = bin_edges_y[1::1]
    n_bins_x = hist_pos.shape[0]
    n_bins_y = hist_pos.shape[1]

    # write expression for unweighted field 
    traj_field_x =  np.abs(trajectories.polarization[0,:,:])*np.exp((trajectories.phase[0,:,:])*1j) 
    traj_field_y =  np.abs(trajectories.polarization[1,:,:])*np.exp(trajectories.phase[1,:,:]*1j) 
    traj_field_z =  np.abs(trajectories.polarization[2,:,:])*np.exp(trajectories.phase[2,:,:]*1j)  
    tot_field_x_tm = np.zeros((n_bins, n_bins_x, n_bins_y), dtype=complex)
    tot_field_y_tm = np.zeros((n_bins, n_bins_x, n_bins_y), dtype=complex)
    tot_field_z_tm = np.zeros((n_bins, n_bins_x, n_bins_y), dtype=complex)
    Ix_per_traj_phase = np.zeros(ntraj)
    Iy_per_traj_phase = np.zeros(ntraj)
    Iz_per_traj_phase = np.zeros(ntraj)
    
    
    # loop through the time bins of the histogram
    for i in range(n_bins):
    
        # find trajectories that were reflected/transmitted at this time bin
        traj_ind_refl_time = np.where((traj_time>=bin_min[i])& (traj_time < bin_max[i]))[0]
        
        for k in range(n_bins_x):
            traj_ind_refl_pos_x = np.where((pos_x>=x_bin_min[k])& (pos_x < x_bin_max[k]))[0]
            if len(traj_ind_refl_pos_x)==0:
                continue
            for l in range(n_bins_y):
                traj_ind_refl_pos_y = np.where((pos_y>=y_bin_min[l])& (pos_y < y_bin_max[l]))[0] 
                if len(traj_ind_refl_pos_y)==0:
                    continue
             
                traj_ind_refl_1 = np.intersect1d(traj_ind_refl_time, traj_ind_refl_pos_x)
                traj_ind_refl = np.intersect1d(traj_ind_refl_1, traj_ind_refl_pos_y) 
                
                w = np.sqrt(refl_per_traj[traj_ind_refl]*ntraj)
                
                refl_field_x = w*traj_field_x[refl_indices[traj_ind_refl]-1,traj_ind_refl]
                refl_field_y = w*traj_field_y[refl_indices[traj_ind_refl]-1,traj_ind_refl]
                refl_field_z = w*traj_field_z[refl_indices[traj_ind_refl]-1,traj_ind_refl]
         
                # add reflectance/transmittance due to trajectories 
                # reflected/transmitted at this time bin
                tot_field_x_tm[i, k, l] += np.sum(refl_field_x)
                tot_field_y_tm[i, k, l] += np.sum(refl_field_y)
                tot_field_z_tm[i, k, l] += np.sum(refl_field_z)
            
                # loop through trajectories in the bin
                # todo add capabilities for coherence length in this
                traj_ind_refl = [] # need to take this line out later
                for j in range(len(traj_ind_refl)):
                    field_weight_x = np.abs(refl_field_x[j])/(np.abs(refl_field_x[j]) + np.abs(refl_field_x))
                    field_weight_y = np.abs(refl_field_y[j])/(np.abs(refl_field_y[j]) + np.abs(refl_field_y))
                    field_weight_z = np.abs(refl_field_z[j])/(np.abs(refl_field_z[j]) + np.abs(refl_field_z))
                    
                    coherence_terms_x = 2*np.abs(refl_field_x[j])*np.abs(refl_field_x) 
                    coherence_terms_y = 2*np.abs(refl_field_y[j])*np.abs(refl_field_y) 
                    coherence_terms_z = 2*np.abs(refl_field_z[j])*np.abs(refl_field_z)           
                    
                    rel_phase_x = np.real(trajectories.phase[0,refl_indices[traj_ind_refl[j]]-1,traj_ind_refl[j]]
                                    -trajectories.phase[0,refl_indices[traj_ind_refl]-1,traj_ind_refl])
                    rel_phase_y = (trajectories.phase[1,refl_indices[traj_ind_refl[j]]-1,traj_ind_refl[j]]
                                    -trajectories.phase[1,refl_indices[traj_ind_refl]-1,traj_ind_refl]) 
                    rel_phase_z = (trajectories.phase[2,refl_indices[traj_ind_refl[j]]-1,traj_ind_refl[j]]
                                    -trajectories.phase[2,refl_indices[traj_ind_refl]-1,traj_ind_refl]) 
                    
                    Ix_per_traj_phase[traj_ind_refl[j]] = np.sum(coherence_terms_x*field_weight_x*np.cos(rel_phase_x))
                                            
                    Iy_per_traj_phase[traj_ind_refl[j]] = np.sum(coherence_terms_y*field_weight_y*np.cos(rel_phase_y))
                                            
                    Iz_per_traj_phase[traj_ind_refl[j]] = np.sum(coherence_terms_z*field_weight_z*np.cos(rel_phase_z))
            

    # redefine n_bins and tot_fields to handle no reflectance case
    if n_bins==0:
        n_bins=1
        tot_field_x_tm = np.zeros(n_bins, dtype=complex)
        tot_field_y_tm = np.zeros(n_bins, dtype=complex)
        tot_field_z_tm = np.zeros(n_bins, dtype=complex)
        
    if convolve:
    # TODO 
    # note convolution code not  fully implemented
    # current code results in improper normalization
    
        # define step function to convolve with
        step_func = np.ones(n_bins)
        
        # convolve to get steady state
        field_x_steady_tm = np.convolve(tot_field_x_tm, step_func)/bin_width
        field_y_steady_tm = np.convolve(tot_field_y_tm, step_func)/bin_width
        field_z_steady_tm = np.convolve(tot_field_z_tm, step_func)/bin_width
        
        # take field value after it's reached steady state
        field_x_steady = field_x_steady_tm[n_bins-1]
        field_y_steady = field_y_steady_tm[n_bins-1]
        field_z_steady = field_z_steady_tm[n_bins-1]
        
    else:
        field_x_steady = tot_field_x_tm
        field_y_steady = tot_field_y_tm
        field_z_steady = tot_field_z_tm
    
    # calculate intensity as E*E
    intensity_x = np.conj(field_x_steady)*field_x_steady
    intensity_y = np.conj(field_y_steady)*field_y_steady
    intensity_z = np.conj(field_z_steady)*field_z_steady

    # add the x,y, and z intensity
    refl_intensity_phase_events = np.sum(intensity_x + intensity_y + intensity_z)
    refl_intensity_phase_per_traj = Ix_per_traj_phase + Iy_per_traj_phase + Iz_per_traj_phase
    
    # normalize
    intensity_incident = np.sum(trajectories.weight[0,:]).magnitude # assumes normalized light is incoherent
    refl_steady = np.real(refl_intensity_phase_events/intensity_incident)
    refl_phase_per_traj = refl_intensity_phase_per_traj/intensity_incident
    
    refl_x = np.sum(intensity_x)/intensity_incident
    refl_y = np.sum(intensity_y)/intensity_incident
    refl_z = np.sum(intensity_z)/intensity_incident

    if components ==True:
        return refl_x, refl_y, refl_z
    else:
        return refl_steady, refl_phase_per_traj

def calc_coherence(phase):
    if isinstance(phase, sc.Quantity):
        phase = phase.magnitude    
    
    phase_diffs = []
    for i in range(phase.size):
        for j in range(phase.size):
            if i == j:
                continue
            phase_diffs.append(np.abs(phase[i] - phase[j]))
    coherence = np.cos(np.array(phase_diffs))
    # cos(pi) = -1
    # cos(0) = 1
    return np.ndarray.flatten(coherence)
    
def calc_coherence_refl_fields(fields, refl_indices):
    phase = np.angle(fields)
    
    coh_x, coh_y, coh_z = calc_refl_coherence(phase, refl_indices)
    
    return coh_x, coh_y, coh_z
    
def calc_refl_coherence(phase, refl_indices):
    if isinstance(phase, sc.Quantity):
        phase = phase.magnitude
    
    refl_phase_x = select_events(phase[0,:,:], refl_indices)
    refl_phase_x = refl_phase_x[refl_phase_x!=0]   
    
    refl_phase_y = select_events(phase[1,:,:], refl_indices)
    refl_phase_y = refl_phase_y[refl_phase_y!=0] 
    
    refl_phase_z = select_events(phase[2,:,:], refl_indices)
    refl_phase_z = refl_phase_z[refl_phase_z!=0] 
    
    coherence_refl_x = np.mean(calc_coherence(refl_phase_x))
    coherence_refl_y = np.mean(calc_coherence(refl_phase_y))
    coherence_refl_z = np.mean(calc_coherence(refl_phase_z))
    
    return coherence_refl_x, coherence_refl_y, coherence_refl_z

def calc_refl_phase_test(trajectories, refl_indices, refl_per_traj,
                         components=False):
    '''
    Calculates the reflectance including phase, by considering trajectories
    that exit at the same time to be coherent. To do this we must bin trajectories
    with similar exit times and add their fields. Then we convolve the 
    reflectance as a function of time with a step function in order to 
    give a steady state value for the reflectance
    
    parameters:
    ----------
    traj_time: 1d array (structcol.Quantity [time], length ntraj)
        time each trajectory spends traversing inside the sample before exit
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    refl_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel 
        contributions
    bin_width: float (structcol.Quantity [time])
        size of time bins for creating field versus time. Should be set equal to
        coherence time of source
   convolve: boolean
        determines whether the reflectance vs time curve should be convolved 
        with a step function input signal to simulate a constant input signal
        instead of a pulse input signal. Note that this is not currently fully
        implemented and will likely return a noisy result with a magnitude
        smaller than expected. 
    components: boolean
    
    returns:
    -------
    refl_steady: float
        steady state reflectance including contributions from phase
    refl_phase_per_traj: ndarray (length: ntraj)
        phase corrected reflectance as a function of event number. The cross
        terms for interfering trajectories are distributed to the different 
        trajectories by relative weight.
    '''
  
    
    ntraj = len(trajectories.polarization[0,0,:]) 
    nevents = len(trajectories.polarization[0,:,0]) 
    
    no_refl_warn = "No trajectories were reflected. Check sample parameters or increase number of trajectories."

    # write expression for field 

    # what if we didn't use polarization?
    # but instead used direction??
    # polarization is orthogonal to direction
    w = np.sqrt(refl_per_traj*ntraj)
    traj_field_x =  w*np.abs(trajectories.polarization[0,:,:])*np.exp((trajectories.phase[0,:,:])*1j) 
    traj_field_y =  w*np.abs(trajectories.polarization[1,:,:])*np.exp(trajectories.phase[1,:,:]*1j) 
    traj_field_z =  w*np.abs(trajectories.polarization[2,:,:])*np.exp(trajectories.phase[2,:,:]*1j)  

    #w = np.sqrt(refl_per_traj[refl_indices]*ntraj)
    
    
    # select traj_field values only for the reflected indices
    refl_field_x = select_events(traj_field_x, refl_indices)
    refl_field_y = select_events(traj_field_y, refl_indices)
    refl_field_z = select_events(traj_field_z, refl_indices)
    
    ## plot distribution of phase refl x,y,z (remove zeros) 
    trajectories_phase_y = trajectories.phase[1,:,:].magnitude
    phase_y = np.mod(np.real(trajectories_phase_y), 2*np.pi)
    phase_y_refl = select_events(phase_y, refl_indices)
    plt.figure()
    sns.distplot(phase_y_refl[phase_y_refl!=0])    
    
    ## plot distribution of refl_field x,y,z (remove zeros)
    # should span from -1 to +1 (in the real part)
    plt.figure()
    sns.distplot(refl_field_y[refl_field_y!=0]) 
    plt.figure()
 
    # plot distribution of phase back calculated from refl_field. Should
    # span -pi to pi
    plt.figure()
    phase_field_y = np.angle(refl_field_y)
    sns.distplot(phase_field_y[refl_field_y!=0])  
    
    coherence = calc_coherence(phase_y_refl[refl_field_y!=0])
    print('coherence: ' + str(np.mean(coherence)))
    
    # add reflectance/transmittance due to trajectories 
    # reflected/transmitted at this time bin
    tot_field_x = np.sum(refl_field_x)
    tot_field_y = np.sum(refl_field_y)
    tot_field_z = np.sum(refl_field_z)
    
    non_phase_int_x = np.conj(refl_field_x)*refl_field_x
    non_phase_int_y = np.conj(refl_field_y)*refl_field_y
    non_phase_int_z = np.conj(refl_field_z)*refl_field_z
    refl_non_phase = np.sum(non_phase_int_x + non_phase_int_y + non_phase_int_z)

    ## print tot_field x,y,z
    #print(np.sum(refl_field_x[refl_field_x!=0]))

    # calculate intensity as E*E
    intensity_x = np.conj(tot_field_x)*tot_field_x
    intensity_y = np.conj(tot_field_y)*tot_field_y
    intensity_z = np.conj(tot_field_z)*tot_field_z

    # add the x,y, and z intensity
    refl_intensity = np.sum(intensity_x + intensity_y + intensity_z)
    
    # normalize
    intensity_incident = np.sum(trajectories.weight[0,:]).magnitude # assumes normalized light is incoherent
    #refl_intensity = refl_intensity.magnitude
    refl = np.real(refl_intensity/intensity_incident)
    print('refl_phase:' + str(refl))
    print('refl_non_phase: ' + str(refl_non_phase/intensity_incident))
    
    refl_x = np.sum(intensity_x)/intensity_incident
    refl_y = np.sum(intensity_y)/intensity_incident
    refl_z = np.sum(intensity_z)/intensity_incident

    if components ==True:
        return refl_x, refl_y, refl_z
    else:
        return refl
        
###############################################################################       
########## to keep ############################################################    
###############################################################################
        
def calc_refl_phase_fields(trajectories, refl_indices, refl_per_traj,
                         components=False):
    '''
    Calculates the reflectance including phase, by considering trajectories
    that exit at the same time to be coherent. To do this we, must bin trajectories
    with similar exit times and add their fields. Then we convolve the 
    reflectance as a function of time with a step function in order to 
    give a steady state value for the reflectance
    
    parameters:
    ----------
    traj_time: 1d array (structcol.Quantity [time], length ntraj)
        time each trajectory spends traversing inside the sample before exit
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    refl_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel 
        contributions
    bin_width: float (structcol.Quantity [time])
        size of time bins for creating field versus time. Should be set equal to
        coherence time of source
   convolve: boolean
        determines whether the reflectance vs time curve should be convolved 
        with a step function input signal to simulate a constant input signal
        instead of a pulse input signal. Note that this is not currently fully
        implemented and will likely return a noisy result with a magnitude
        smaller than expected. 
    components: boolean
    
    returns:
    -------
    refl_steady: float
        steady state reflectance including contributions from phase
    refl_phase_per_traj: ndarray (length: ntraj)
        phase corrected reflectance as a function of event number. The cross
        terms for interfering trajectories are distributed to the different 
        trajectories by relative weight.
    '''
  
    
    ntraj = len(trajectories.direction[0,0,:]) 
    nevents = len(trajectories.direction[0,:,0]) 
    
    if np.all(refl_indices==0):
        no_refl_warn = "No trajectories were reflected. Check sample parameters or increase number of trajectories."
        warnings.warn(no_refl_warn)
    if isinstance(trajectories.weight, sc.Quantity):
        weights = trajectories.weight.magnitude
    else:
        weights = trajectories.weight

    # write expression for field 
    # 0th event is before entering sample, so we start from 1, for later use with 
    # select_events
    w = np.sqrt(refl_per_traj*ntraj) #0 for not reflected traj, but that's fine since we only care about refl
    traj_field_x =  w*trajectories.fields[0,1:,:] 
    traj_field_y =  w*trajectories.fields[1,1:,:] 
    traj_field_z =  w*trajectories.fields[2,1:,:]
    #print(w.shape)
    #print(traj_field_x)

    # select traj_field values only for the reflected indices
    # TODO: may need to fix indexing here since fields have an extra event index
    refl_field_x = select_events(traj_field_x, refl_indices)
    refl_field_y = select_events(traj_field_y, refl_indices)
    refl_field_z = select_events(traj_field_z, refl_indices)
    #print(refl_field_x)
    
    # add reflected fields
    tot_field_x = np.sum(refl_field_x)
    tot_field_y = np.sum(refl_field_y)
    tot_field_z = np.sum(refl_field_z)
    
    # for comparing without phase
    non_phase_int_x = np.conj(refl_field_x)*refl_field_x
    non_phase_int_y = np.conj(refl_field_y)*refl_field_y
    non_phase_int_z = np.conj(refl_field_z)*refl_field_z
    refl_non_phase = np.sum(non_phase_int_x + non_phase_int_y + non_phase_int_z)
    #refl_non_phase = np.sum(refl_per_traj) # remove this line
    #print('refl_per_traj 2: ' + str(np.sum(refl_per_traj)))

    # calculate intensity as E*E
    intensity_x = np.conj(tot_field_x)*tot_field_x
    intensity_y = np.conj(tot_field_y)*tot_field_y
    intensity_z = np.conj(tot_field_z)*tot_field_z

    # add the x,y, and z intensity
    refl_intensity = np.sum(intensity_x + intensity_y + intensity_z)
    
    # normalize
    intensity_incident = ntraj#np.sum(weights[0,:]) # assumes normalized light is incoherent. This line was causing the silica sample issue
    #print('intensity_incident: ' + str(intensity_incident))
    refl_fields = np.real(refl_intensity/intensity_incident)
    
    refl_x = np.sum(intensity_x)/intensity_incident
    refl_y = np.sum(intensity_y)/intensity_incident
    refl_z = np.sum(intensity_z)/intensity_incident

    if components ==True:
        return tot_field_x, tot_field_y, tot_field_z, refl_fields, refl_non_phase/intensity_incident
    else:
        return refl_fields, refl_non_phase/intensity_incident        
        
def calc_refl_co_cross_fields(trajectories, refl_indices, refl_per_traj, det_theta):
    '''
    Goniometer detector size should already be taken account in calc_refl_trans()
    so the refl_indices will only include trajectories that exit within the detector
    area. 
    
    Muliplying by the sines and cosines of the detector theta is an approximation, 
    since the goniometer detector area is usually small enough such that the detector
    size is not that big. Should check that this approximation is reasonable. The
    alternative would be to keep track of the actual exit theta of each trajectory,
    using the direction property. 
    
    '''
    
    (tot_field_x, 
     tot_field_y, 
     tot_field_z,
     refl_field,
     refl_intensity) = calc_refl_phase_fields(trajectories, refl_indices, refl_per_traj,
                         components=True)
    
    # incorporate geometry of the goniometer setup                 
    # rotate the total x, y, z fields to the par/perp detector basis
    # this is based on a clockwise rotation about the y-axis of angle det_theta                     
    tot_field_co = tot_field_x*np.cos(det_theta) + tot_field_z*np.sin(det_theta)# E field polarization is co-polarized (mostly x)
    tot_field_cr = tot_field_y# E field polarization is cross-polarized  (mostly y)
    tot_field_perp = -tot_field_x*np.sin(det_theta) + tot_field_z*np.cos(det_theta)# E field polarization is perp to detector plane
                         
    # take the modulus to get intensity
    refl_co = np.conj(tot_field_co)*tot_field_co#refl_z*np.sin(det_theta) + refl_x*np.cos(det_theta) # refl_x
    refl_cr = np.conj(tot_field_cr)*tot_field_cr
    refl_perp = np.conj(tot_field_perp)*tot_field_perp# refl_z*np.cos(det_theta) - refl_x*np.sin(det_theta) # refl_z
    
    return (refl_co, refl_cr, refl_perp, refl_field, refl_intensity)

    
    
    
    