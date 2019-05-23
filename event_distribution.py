# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:38:47 2018

@author: stephenson
"""

import numpy as np
import matplotlib.pyplot as plt
import structcol as sc
from structcol.montecarlo import select_events
from structcol.montecarlo import fresnel_pass_frac
    
def calc_refl_trans_event(refl_per_traj, inc_refl_per_traj, trans_per_traj, 
                          refl_indices, trans_indices, nevents):
    refl_events = np.zeros(2*nevents + 1)
    trans_events = np.zeros(2*nevents + 1)
    
    refl_events[0] = np.sum(inc_refl_per_traj) # add fresnel reflection 
    for ev in range(1, nevents + 1):
        traj_ind_refl_ev = np.where(refl_indices == ev)[0]
        traj_ind_trans_ev = np.where(trans_indices == ev)[0]
        
        refl_events[ev] += np.sum(refl_per_traj[traj_ind_refl_ev])
        trans_events[ev] += np.sum(trans_per_traj[traj_ind_trans_ev])
    return refl_events, trans_events

def calc_thetas_event_traj(theta, refl_indices, nevents, ntraj = 100):
    # answers the question:
    # What is the theta for a given event and given trajectory that is 
    # reflected at that event
    theta_event_traj = np.zeros((nevents, ntraj))
    refl_indices = refl_indices[0:ntraj]
    
    for ev in range(1,nevents):
        traj_ind_refl_ev = np.where(refl_indices == ev)[0]
        theta_event_traj[ev, traj_ind_refl_ev] = theta[ev, traj_ind_refl_ev]
    
    return theta_event_traj

def calc_tir(tir_refl_bool, refl_indices, trans_indices, inc_refl_per_traj, weights, kz, ntraj, n_sample, n_medium):
    
    if isinstance(weights, sc.Quantity):
        weights = weights.magnitude
    if isinstance(kz, sc.Quantity):
        kz = kz.magnitude
    if isinstance(n_sample, sc.Quantity):
        n_sample = np.abs(n_sample.magnitude)
    if isinstance(n_medium, sc.Quantity):
        n_medium = n_medium.magnitude
        

   
    #tir = np.logical_not(no_tir)*1#~no_tir
    
    # tir for all events
    tir_indices = np.argmax(np.vstack([np.zeros(ntraj),tir_refl_bool]), axis=0)
    
    refl_ind_inf = np.copy(refl_indices)
    refl_ind_inf[refl_ind_inf == 0] = ntraj*10
    trans_ind_inf = np.copy(trans_indices)
    trans_ind_inf[trans_ind_inf == 0] = ntraj*10
    
    tir_indices[np.where(tir_indices>refl_ind_inf)[0]] = 0
    tir_indices[np.where(tir_indices>trans_ind_inf)[0]] = 0
    tir_all = np.sum((1-inc_refl_per_traj) * select_events(weights, tir_indices))
    
    # tir for all events that gets reflected eventually
    tir_ev_ind = np.where(tir_indices!=0)
    tir_indices_refl = np.zeros(ntraj)
    tir_indices_refl[tir_ev_ind] = refl_indices[tir_ev_ind]
    
    tir_all_refl = np.sum((1-inc_refl_per_traj) * select_events(weights, tir_indices_refl)*
                   fresnel_pass_frac(kz, tir_indices_refl, n_sample, None, n_medium))
    
    # tir for only single scat event
    tir_indices_single = np.copy(tir_indices)
    tir_indices_single[np.where(tir_indices!=2)] = 0
    tir_single = np.sum((1-inc_refl_per_traj) * select_events(weights, tir_indices_single))
    
    # tir for only single scat event that gets reflected eventually
    tir_ev_sing_ind = np.where(tir_indices_single == 2)
    tir_indices_single_refl = np.zeros(ntraj)
    tir_indices_single_refl[tir_ev_sing_ind] = refl_indices[tir_ev_sing_ind]
    #print(tir_indices_single_refl)
    tir_single_refl = np.sum((1-inc_refl_per_traj) * select_events(weights, tir_indices_single_refl)*
                      fresnel_pass_frac(kz, tir_indices_single_refl, n_sample, None, n_medium))
     
    return tir_all, tir_all_refl, tir_single, tir_single_refl, tir_indices_single
    
    
    
def calc_pdf_scat(refl_events, trans_events, nevents):
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
                            refl_fresnel, trans_fresnel,
                            refl_frac, trans_frac, nevents):
    
    sampled_refl_event = np.random.choice(np.arange(2, nevents + 1), 
                                          size = nevents+1,
                                          p = pdf_refl)
    sampled_trans_event = np.random.choice(np.arange(1, nevents + 1), 
                                          size = nevents+1,
                                          p = pdf_trans)
    fresnel_samp = np.zeros(2*nevents + 1)
    for ev in range(1, nevents + 1):
        traj_ind_event = np.where(refl_indices == ev)[0]
        # sampled_refl_event has a size nevents + 1, and this loop has size nevents + 1
        # sampled_trans_event has a size nevents + 1, even though it includes and extra event to sample

        fresnel_samp[int(ev + sampled_refl_event[ev])] += refl_frac*np.sum(refl_fresnel[traj_ind_event])
        fresnel_samp[int(ev + sampled_trans_event[ev])] += trans_frac*np.sum(trans_fresnel[traj_ind_event])
    return refl_events + fresnel_samp

def calc_refl_event_fresnel_avg(refl_events, refl_indices, trans_indices, 
                            refl_fresnel, trans_fresnel,
                            refl_frac, trans_frac, nevents):
    avg_refl_event = np.round(np.average(refl_indices[refl_indices!=0]))
    print('avg_refl_event: ' + str(avg_refl_event))
    avg_trans_event = np.round(np.average(trans_indices[trans_indices!=0]))
    print('avg_trans_event: ' + str(avg_trans_event))
    
    fresnel_avg = np.zeros(2*nevents + 1)
    for ev in range(1, nevents + 1): 
        traj_ind_event = np.where(refl_indices == ev)[0]
        fresnel_avg[int(ev + avg_refl_event)] += refl_frac*np.sum(refl_fresnel[traj_ind_event])
        fresnel_avg[int(ev + avg_trans_event)] += trans_frac*np.sum(trans_fresnel[traj_ind_event])
    return refl_events + fresnel_avg

def plot_refl_event(wavelengths, refl_events, event):
    
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
    if isinstance(wavelengths, sc.Quantity):
        wavelengths = wavelengths.to('nm').magnitude
    plt.plot(wavelengths, np.sum(refl_events[:,events], axis = 1), label = label, linewidth = 3)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.ylabel('Reflectance')
    plt.xlabel('Wavelength (nm)')
    
def plot_refl_event_sum_norm(wavelengths, refl_events, events, label = ''):
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
    

