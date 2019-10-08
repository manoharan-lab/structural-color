# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:38:47 2018

@author: stephenson
"""

import numpy as np
import matplotlib.pyplot as plt
import structcol as sc
from structcol.detector import select_events
from structcol.detector import fresnel_pass_frac
    
def calc_refl_trans_event(refl_per_traj, inc_refl_per_traj, trans_per_traj, 
                          refl_indices, trans_indices, nevents):
    '''
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
    refl_events: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event. 
    trans_events: 1d array (length: 2*nevents + 1)
        transmittance contribution for each event.
    '''
    refl_events = np.zeros(2*nevents + 1)
    trans_events = np.zeros(2*nevents + 1)
    
    # add fresnel reflection at first interface
    refl_events[0] = np.sum(inc_refl_per_traj)
    
    #loop through all events
    for ev in range(1, nevents + 1):
        # find trajectories that were reflected/transmitted at this event
        traj_ind_refl_ev = np.where(refl_indices == ev)[0]
        traj_ind_trans_ev = np.where(trans_indices == ev)[0]
        
        # add reflectance/transmittance due to trajectories 
        # reflected/transmitted at this event
        refl_events[ev] += np.sum(refl_per_traj[traj_ind_refl_ev])
        trans_events[ev] += np.sum(trans_per_traj[traj_ind_trans_ev])
    return refl_events, trans_events
    
    
def calc_refl_trans_event_traj(refl_per_traj, inc_refl_per_traj, trans_per_traj, 
                          refl_indices, trans_indices, nevents, ntraj=100):
    '''
    Returns reflectance and transmittance as a function of event number
    and trajectory
    
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
    ntraj: int
        number of trajectories to keep track of. If this number is too high, 
        the arrays will be way too large. 
    
    Returns
    -------
    refl_events_traj: 2d array (shape: 2*nevents + 1, ntraj)
        reflectance contribution for each event and trajectory. 
    trans_events_taj: 2d array (shape: 2*nevents + 1, ntraj)
        transmittance contribution for each event and trajectory.
    '''
    refl_events_traj = np.zeros((2*nevents + 1, ntraj))
    trans_events_traj = np.zeros((2*nevents + 1, ntraj))
    
    # shorten parameters to just look at number of trajectores specified
    refl_per_traj = refl_indices[0:ntraj]
    inc_refl_per_traj = refl_indices[0:ntraj]
    trans_per_traj = refl_indices[0:ntraj]
    refl_indices = refl_indices[0:ntraj]
    trans_indices = refl_indices[0:ntraj]
    
    # add fresnel reflection at first interface
    refl_events_traj[0,:] = inc_refl_per_traj
    
    #loop through all events
    for ev in range(1, nevents + 1):
        # find trajectories that were reflected/transmitted at this event
        traj_ind_refl_ev = np.where(refl_indices == ev)[0]
        traj_ind_trans_ev = np.where(trans_indices == ev)[0]
        
        # add reflectance/transmittance due to trajectories 
        # reflected/transmitted at this event
        refl_events_traj[ev, traj_ind_refl_ev] += np.sum(refl_per_traj[traj_ind_refl_ev])
        trans_events_traj[ev, traj_ind_trans_ev] += np.sum(trans_per_traj[traj_ind_trans_ev])
        
    return refl_events_traj, trans_events_traj

def calc_thetas_event_traj(theta, refl_indices, nevents, ntraj = 100):
    '''
    Returns array of thetas at reflection for every event and trajectory. If
    trajectory is not reflected at a particular event, theta value is 0.
    
    Parameters
    ----------
    theta: 2d array (shape: nevents, ntrajectories)
        sampled thetas used in Monte Carlo. Includes all thetas for all trajectories.
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event. 
    nevents: int
        number of events for which Monte Carlo calculation is run 
    ntraj: int, optional
        number of trajectories from Monte Carlo to examine. Usually want a smaller
        number than the full number of trajectories run because size of matrix
        will be too big. Default value is 100. 
    
    Returns
    -------
    theta_event_traj: 2d array (shape: nevents, ntrajectories)
        thetas at reflection for every event and trajectory. If trajectory is
        not reflected at a particular event, theta value is 0 for that event and
        trajectory. 
    '''

    theta_event_traj = np.zeros((nevents, ntraj))
    
    # shorted refl_indices to just look at number of trajectores specified
    refl_indices = refl_indices[0:ntraj]
    
    # loop through events
    for ev in range(1,nevents):
        
        # find trajectory indeces where a reflection took place
        traj_ind_refl_ev = np.where(refl_indices == ev)[0]
        
        # add the thetas corresponding to reflection to the theta_event_traj
        # array
        theta_event_traj[ev, traj_ind_refl_ev] = theta[ev, traj_ind_refl_ev]
    
    return theta_event_traj

def calc_tir(tir_refl_bool, refl_indices, trans_indices, inc_refl_per_traj, 
             n_sample, n_medium, boundary, trajectories, thickness):
    '''
    Returns weights of various types of totally internally reflected trajectories
    as a function of event number
    
    Parameters
    ----------
    tir_refl_bool: boolean (shape: nevents, ntrajectories)
        Boolean describing whether a trajectory is totally internally reflected
        at a given event
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event. 
    trans_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    inc_refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory at the sample interface. 
        This contribution comes from the Fresnel reflection as the light
        enters the sample
    n_sample: float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the sample.
    n_medium: float (structcol.Quantity [dimensionless] or 
        structcol.refractive_index object)
        Refractive index of the medium.
    boundary: string 
        geometrical boundary, current options are 'film' or 'sphere'
    trajectories:Trajectory object
        Trajectory object used in Monte Carlo simulation
    thickness: float
        thickness of film or diameter of sphere
    
    Returns
    -------
    tir_all_events: 1d array (length: nevents)
        summed weights of trajectories that are totally internally reflected at 
        any event, regardeless of whether they are reflected, transmitted, or stuck.
        The event index of the array corresponds to the event at which they are 
        totally internally reflected. 
    tir_all_refl_events: 1d array (length: nevents)
        summed weights of trajectories that are totally internally reflected at any 
        event, but only those which eventually contribute to reflectance. The
        event index of the array corresponds to the event at which they are
        reflected. 
    tir_single_events: 1d array (length: nevents)
        summed weights of trajectories that are totally internally reflected 
        after the first scattering event, regardless of whether they are 
        reflected, transmitted, or stuck. The event index corresponds to the 
        event at which they are totally internally reflected
    tir_single_refl_events: 1d array (length: nevents)
        summed weights of trajectories that are totally internally reflected
        adter the first scattering event and eventually contribute to reflectance. 
        The event index corresponds to the event at which they are reflected.
    tir_indices_single_events: 1d array (length: nevents)
        The event indices of trajectories that are totally internally reflected 
        after a single scattering event.
    '''
    
    weights = trajectories.weight
    nevents = trajectories.nevents
    ntraj = trajectories.direction.shape[2]
    if isinstance(weights, sc.Quantity):
        weights = weights.magnitude
    if isinstance(n_sample, sc.Quantity):
        n_sample = np.abs(n_sample.magnitude)
    if isinstance(n_medium, sc.Quantity):
        n_medium = n_medium.magnitude
    
    ### tir for all events ###
    
    # get the event indices for which trajectories are tir'd
    tir_indices = np.argmax(np.vstack([np.zeros(ntraj),tir_refl_bool]), axis=0)
    
    # make event indices of zero larger than possible nevents
    # so that refl_events of 0 never have a smaller number than any other events
    refl_ind_inf = np.copy(refl_indices)
    refl_ind_inf[refl_ind_inf == 0] = nevents*10
    trans_ind_inf = np.copy(trans_indices)
    trans_ind_inf[trans_ind_inf == 0] = nevents*10
    
    # find  tir indices where trajectories are tir'd before getting reflected
    # or transmitted
    tir_indices[np.where(tir_indices>refl_ind_inf)[0]] = 0
    tir_indices[np.where(tir_indices>trans_ind_inf)[0]] = 0
    tir_all = (1-inc_refl_per_traj) * select_events(weights, tir_indices)/ntraj
    
    ### tir for all events that gets reflected eventually ###
    
    # find event indices where tir'd trajectories are reflected
    tir_ev_ind = np.where(tir_indices!=0)
    tir_indices_refl = np.zeros(ntraj)
    tir_indices_refl[tir_ev_ind] = refl_indices[tir_ev_ind]
    
    # find the tir reflectance at each event
    tir_all_refl = ((1-inc_refl_per_traj) * select_events(weights, tir_indices_refl)*
                   fresnel_pass_frac(tir_indices_refl, n_sample, None, n_medium,
                                     boundary, trajectories, thickness)[0])/ntraj
    
    ### tir for only single scat event ###
    
    # find the event indices where single scat trajectories are tir'd
    tir_indices_single = np.copy(tir_indices)
    tir_indices_single[np.where(tir_indices!=2)] = 0
    tir_single = (1-inc_refl_per_traj) * select_events(weights, tir_indices_single)/ntraj
    
    ### tir for only single scat event that gets reflected eventually ###
    
    # find event indices where single scat tir'd trajectories are reflected
    tir_ev_sing_ind = np.where(tir_indices_single == 2)
    tir_indices_single_refl = np.zeros(ntraj)
    tir_indices_single_refl[tir_ev_sing_ind] = refl_indices[tir_ev_sing_ind]

    # calculate the single scat tir'd reflectance at each event
    tir_single_refl = ((1-inc_refl_per_traj) * select_events(weights, tir_indices_single_refl)*
                      fresnel_pass_frac(tir_indices_single_refl, n_sample, None, n_medium,
                                        boundary, trajectories, thickness)[0])/ntraj
                                        
    #loop through all events
    tir_all_events = np.zeros(2*nevents + 1)
    tir_all_refl_events = np.zeros(2*nevents + 1)
    tir_single_events = np.zeros(2*nevents + 1)
    tir_single_refl_events = np.zeros(2*nevents + 1)
    for ev in range(1, nevents + 1):
        # find trajectories that were reflected/transmitted at this event
        traj_ind_tir_ev = np.where(tir_indices == ev)[0]
        traj_ind_tir_refl_ev = np.where(tir_indices_refl == ev)[0]
        traj_ind_tir_sing_ev = np.where(tir_indices_single == ev)[0]
        traj_ind_tir_sing_refl_ev = np.where(tir_indices_single_refl == ev)[0]
        
        # add reflectance/transmittance due to trajectories 
        # reflected/transmitted at this event
        tir_all_events[ev] += np.sum(tir_all[traj_ind_tir_ev])
        tir_all_refl_events[ev] += np.sum(tir_all_refl[traj_ind_tir_refl_ev])
        tir_single_events[ev] += np.sum(tir_single[traj_ind_tir_sing_ev])
        tir_single_refl_events[ev] += np.sum(tir_single_refl[traj_ind_tir_sing_refl_ev])
     
    return (tir_all_events, 
            tir_all_refl_events, 
            tir_single_events, 
            tir_single_refl_events, 
            tir_indices_single)
    
    
    
def calc_pdf_scat(refl_events, trans_events, nevents):
    '''
    Calculates probability density function of reflection and transmission at 
    each event. 
    
    Parameters
    ----------
    refl_events: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event
    trans_events: 1d array (length: 2*nevents + 1)
        transmittance contribution for each event
    nevents: int
        number of events for which Monte Carlo calculation is run 
       
    Returns
    -------
    pdf_refl: 1d array (length: 2*nevents + 1)
        probability of reflection at each event
    pdf_trans: 1d array (length: 2*nevents +1)
        probability of transmission at each event
    '''
    # 0th event: reflection due to fresnel at interface
    # 1st event: reflection exits after 1st step into sample (always 0 
    # because cannot)
    # 2nd event: "singly scattered" in the sense that has scattered once inside
    # the sample, so could exit
    # why "nevents + 1" ? because we added an extra "event" by including 
    # the fresnel reflection as the 0th event
    
    pdf_refl = refl_events[2:nevents + 1]/np.sum(refl_events[2:nevents + 1])
    pdf_trans = trans_events[1:nevents +1]/np.sum(trans_events[1:nevents +1])
    return pdf_refl, pdf_trans

def calc_refl_event_fresnel_pdf(refl_events, pdf_refl, pdf_trans, refl_indices, 
                            trans_indices, refl_fresnel, trans_fresnel,
                            refl_frac, trans_frac, nevents):
    '''
    Calculates the reflectance contribution from fresnel reflected trajectory
    weights and adds it to the total reflectance contribution for a sampled event
    at which the fresnel trajectory exits
    
    
    Parameters
    ----------
    refl_events: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event.
    pdf_refl: 1d array (length: 2*nevents + 1)
        probability of reflection at each event
    pdf_trans: 1d array (length: 2*nevents +1)
        probability of transmission at each event
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event. 
    trans_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    refl_fresnel: 2d array (shape: nevents, ntrajectories)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal reflection. 
    trans_fresnel: 2d array (shape: nevents, ntrajectories)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal reflection.
    refl_frac: 2d array (shape: nevents, ntrajectories)
        fraction of trajectory weights that are reflected normalized by the 
        known outcomes of trajectories
    trans_frac: 2d array (shape: nevents, ntrajectories)
        fraction of trajectory weights that are transmitted normalized by the 
        known outcomes of trajectories
    nevents: int
        number of events for which Monte Carlo calculation is run 
       
    Returns
    -------
    refl_events + fresnel_samp: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event added to the fresnel reflectance
        contribution for each event. 
    '''
    # sample reflection and transmission event numbers
    sampled_refl_event = np.random.choice(np.arange(2, nevents + 1), 
                                          size = nevents+1,
                                          p = pdf_refl)
    sampled_trans_event = np.random.choice(np.arange(1, nevents + 1), 
                                          size = nevents+1,
                                          p = pdf_trans)
    
    # add the frensel reflected trajectory event to the sampled event of reflection
    # or transmission
    fresnel_samp = np.zeros(2*nevents + 1)
    for ev in range(1, nevents + 1):
        traj_ind_event_refl = np.where(refl_indices == ev)[0]
        traj_ind_event_trans = np.where(trans_indices == ev)[0]
        # sampled_refl_event has a size nevents + 1, and this loop has size nevents + 1
        # sampled_trans_event has a size nevents + 1, even though it includes an extra event to sample

        fresnel_samp[int(ev + sampled_refl_event[ev])] += refl_frac*np.sum(refl_fresnel[traj_ind_event_refl])
        fresnel_samp[int(ev + sampled_trans_event[ev])] += trans_frac*np.sum(trans_fresnel[traj_ind_event_trans])
    return refl_events + fresnel_samp

def calc_refl_event_fresnel_avg(refl_events, refl_indices, trans_indices, 
                            refl_fresnel, trans_fresnel,
                            refl_frac, trans_frac, nevents):
    '''
    Calculates the reflectance contribution from fresnel reflected trajectory
    weights and adds it to the total reflectance contribution for the average 
    event at which the fresnel trajectory exits
    
    Parameters
    ----------
    refl_events: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event.
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event. 
    trans_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    refl_fresnel: 2d array (shape: nevents, ntrajectories)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal reflection. 
    trans_fresnel: 2d array (shape: nevents, ntrajectories)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal reflection.
    refl_frac: 2d array (shape: nevents, ntrajectories)
        fraction of trajectory weights that are reflected normalized by the 
        known outcomes of trajectories
    trans_frac: 2d array (shape: nevents, ntrajectories)
        fraction of trajectory weights that are transmitted normalized by the 
        known outcomes of trajectories
    nevents: int
        number of events for which Monte Carlo calculation is run 
       
    Returns
    -------
    refl_events + fresnel_samp: 1d array (length: 2*nevents + 1)
        reflectance contribution for each event added to the fresnel reflectance
        contribution for each event. 
    '''
    # find average event at which reflection or transmission occurs
    avg_refl_event = np.round(np.average(refl_indices[refl_indices!=0]))
    avg_trans_event = np.round(np.average(trans_indices[trans_indices!=0]))
    
    fresnel_avg = np.zeros(2*nevents + 1)
    # add the frensel reflected trajectory event to the average event of reflection
    # or transmission
    for ev in range(1, nevents + 1): 
        traj_ind_event_refl = np.where(refl_indices == ev)[0]
        traj_ind_event_trans = np.where(trans_indices == ev)[0]
        fresnel_avg[int(ev + avg_refl_event)] += refl_frac*np.sum(refl_fresnel[traj_ind_event_refl])
        fresnel_avg[int(ev + avg_trans_event)] += trans_frac*np.sum(trans_fresnel[traj_ind_event_trans])
    return refl_events + fresnel_avg

def plot_refl_event(wavelengths, refl_events, event):
    '''
    Plot the reflectance spectrum for a given event(s)
    
    Parameters
    ----------
    wavelengths: 1d array-like
        wavelengths at which reflectance is calculated
    refl_events: 2d array (shape: wavelengths.length, nevents)
        reflectance as a function of event number
    event: 1d array 
        event or events of interest for which to plot the reflectance. If array
        with more than one element, reflectance is plotted for each event
    '''
    if isinstance(wavelengths, sc.Quantity):
        wavelengths = wavelengths.to('nm').magnitude
    
    plt.figure()
    for ev in range(0, len(event)):    
        plt.plot(wavelengths, refl_events[:, event[ev]], label = event[ev], linewidth = 3)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    
def plot_refl_event_norm(wavelengths, refl_events, event):
    '''
    Plot the reflectance spectrum for a given event(s),
    normalized by the amount of light still in the sample at each event
    
    Parameters
    ----------
    wavelengths: 1d array-like
        wavelengths at which reflectance is calculated
    refl_events: 2d array (shape: wavelengths.length, nevents)
        reflectance as a function of event number
    event: 1d array 
        event or events of interest for which to plot the reflectance. If array
        with more than one element, reflectance is plotted for each event
    '''
    if isinstance(wavelengths, sc.Quantity):
        wavelengths = wavelengths.to('nm').magnitude
    
    plt.figure()
    for ev in range(0, len(event)):    
        events_before = np.arange(0,event[ev])
        plt.plot(wavelengths, refl_events[:, event[ev]]/(1-np.sum(refl_events[:,events_before], axis = 1)), label = event[ev], linewidth = 3)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    
def plot_refl_event_sum(wavelengths, refl_events, events, label = ''):
    '''
    Plot the summed reflectance spectrum for a given range of events event(s)
    
    Parameters
    ----------
    wavelengths: 1d array-like
        wavelengths at which reflectance is calculated
    refl_events: 2d array (shape: wavelengths.length, nevents)
        reflectance as a function of event number
    event: 1d array 
        event or events of interest for which to plot the reflectance.
    label: string
        label for legend()
    '''
    if isinstance(wavelengths, sc.Quantity):
        wavelengths = wavelengths.to('nm').magnitude
    plt.plot(wavelengths, np.sum(refl_events[:,events], axis = 1), label = label, linewidth = 3)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.ylabel('Reflectance')
    plt.xlabel('Wavelength (nm)')
    
def plot_refl_event_sum_norm(wavelengths, refl_events, events, label = ''):
    '''
    Plot the summed reflectance spectrum for a given range of events event(s)
    normalized by the amount of light still in the sample at each event
    
    Parameters
    ----------
    wavelengths: 1d array-like
        wavelengths at which reflectance is calculated
    refl_events: 2d array (shape: wavelengths.length, nevents)
        reflectance as a function of event number
    event: 1d array 
        event or events of interest for which to plot the reflectance.
    label: string
        label for legend()
    '''
    if isinstance(wavelengths, sc.Quantity):
        wavelengths = wavelengths.to('nm').magnitude
    refl_events_norm = np.zeros((len(wavelengths), len(events)))
    for ev in range(0, len(events)):
        events_before = np.arange(0, events[ev])
        refl_events_norm[:, ev] = refl_events[:, events[ev]]/(1-np.sum(refl_events[:,events_before], axis = 1))
    plt.plot(wavelengths, np.sum(refl_events_norm, axis = 1), label = label, linewidth = 3)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.ylabel('Reflectance')
    plt.xlabel('Wavelength (nm)')
    
def plot_refl_dist(wavelengths, refl_events, wavelength):
    '''
    Plot the distribution of reflectance as a function of event number
    at a given wavelength
    
    Parameters
    ----------
    wavelengths: 1d array-like
        wavelengths at which reflectance is calculated
    refl_events: 2d array (shape: wavelengths.length, nevents)
        reflectance as a function of event number
    wavelengths: 1d array-like
        wavelengths at which to plot the reflectance distributions
    '''
    if isinstance(wavelengths, sc.Quantity):
        wavelengths = wavelengths.to('nm').magnitude
    events = np.arange(0, refl_events.shape[1])
    
    plt.figure()
    for wl in range(0, len(wavelength)):    
        wavelength_ind = np.where(wavelengths == wavelength[wl])[0][0]
        plt.semilogx(events-1, refl_events[wavelength_ind,:]/np.sum(refl_events[wavelength_ind,:]), label = wavelength[wl], 
                     marker = '.', markersize = 12, linewidth = 2)
    plt.xlim([1, events[-1]])
    plt.xlabel('Scattering Event Number')
    plt.ylabel('Reflectance Contribution')
    plt.legend()
        
    plt.figure()
    for wl in range(0, len(wavelength)):
        wavelength_ind = np.where(wavelengths == wavelength[wl])[0][0]
        plt.loglog(events-1, refl_events[wavelength_ind,:]/np.sum(refl_events[wavelength_ind,:]), label = wavelength[wl],
                   marker = '.', markersize = 12, linewidth = 0)
    plt.xlim([1, events[-1]])
    plt.xlabel('Scattering Event Number')
    plt.ylabel('Reflectance Contribution')
    plt.legend()
    
    plt.figure()
    for wl in range(0, len(wavelength)):
        wavelength_ind = np.where(wavelengths == wavelength[wl])[0][0]
        plt.semilogx(events-1, refl_events[wavelength_ind,:], label = wavelength[wl],
                   marker = '.', markersize = 12, linewidth = 2)
    plt.xlim([1, events[-1]])
    plt.xlabel('Scattering Event Number')
    plt.ylabel('Reflectance')
    plt.legend()

def save_data(particle, matrix, particle_radius, volume_fraction, thickness, reflectance, 
              refl_events, wavelengths, nevents, ntrajectories, theta_event_traj = None, 
              refl_events_fresnel_samp = None, refl_events_fresnel_avg = None, 
              zpos = None, kz = None, theta_range = None, tir_single = None,
              tir_single_refl = None, tir_all = None, tir_all_refl = None,
              tir_indices_single = None):
    '''
    Saves data as a .npz file. Generates file name using sample parameters.
    '''

    filename = particle +\
        '_in_' + matrix +\
        '_ntraj' + str(ntrajectories) +\
        '_nevent' + str(nevents) +\
        '_rad' + str(particle_radius.magnitude) +\
        '_vf' + str(volume_fraction.magnitude) +\
        '_thick' + str(thickness.magnitude) +\
        '_numwl' + str(wavelengths.size)
        
    np.savez(filename, 
             reflectance = reflectance,
             refl_events = refl_events, 
             refl_events_fresnel_samp = refl_events_fresnel_samp,
             refl_events_fresnel_avg = refl_events_fresnel_avg,
             zpos = zpos, 
             kz = kz,
             wavelengths = wavelengths,
             theta_event_traj = theta_event_traj,
             theta_range = theta_range,
             tir_all = tir_all,
             tir_all_refl = tir_all_refl,
             tir_single = tir_single,
             tir_single_refl = tir_single_refl,
             tir_indices_single = tir_indices_single)
    

