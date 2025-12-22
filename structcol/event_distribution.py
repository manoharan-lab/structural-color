# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Copyright 2025 Anna B. Stephenson, Vinothan N. Manoharan
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
Notes
-----
Event coordinates have length (2*nevents+1) rather than (nevents+1) because of
how fresnel trajectories are counted as events. The exit event of a fresnel
trajectory is added to the exit event of a random reflection or the average
exit event of all reflections. Thus the exit event of the fresnel trajectory
can be larger than the largest event in the simulation.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import structcol as sc
from . import select_events
from structcol.detector import fresnel_pass_frac
from structcol import detector as det

def sum_per_event_number(weights, events, nevents):
    """Calculate summed weights (over trajectories) as a function of event
    number.

    Parameters
    ----------
    weights : `xr.DataArray` (dims=..., trajectory)
        weights at each event of interest
    events : `xr.DataArray` (dims=..., trajectory)
        indices of events
    nevents : int
        total number of events in MC simulation

    """
    # to avoid loop over event indices, first expand weights array to include
    # event coord and broadcast values over all events at each trajectory
    event_dim = dict(event=range(2*nevents+1))
    weights_expanded = weights.expand_dims(event_dim).copy()
    event_coord = weights_expanded.coords["event"]

    # next, select weights only at the events in the index array and sum
    weights_by_event = (weights_expanded.where(event_coord == events)
                        .fillna(0.0).sum("trajectory"))

    return weights_by_event

def calc_refl_trans_event(refl_per_traj, inc_refl_per_traj, trans_per_traj,
                          refl_indices, trans_indices, nevents):
    """Returns reflectance and transmittance as a function of event number

    Parameters
    ----------
    refl_per_traj : `xr.DataArray` (dims=..., trajectory)
        Reflectance contribution for each trajectory from Monte Carlo
        simulation. Sum should be total reflectance from Monte Carlo
        calculation, without corrections for Fresnel reflected and stuck
        weights.
    inc_refl_per_traj : `xr.DataArray` (dims=..., trajectory)
        Reflectance contribution for each trajectory at the sample interface.
        This contribution comes from the Fresnel reflection as the light
        enters the sample
    trans_per_traj : `xr.DataArray` (dims=..., trajectory)
        Transmittance contribution for each trajectory from Monte Carlo
        simulation. Sum should be total transmittance from Monte Carlo
        calculation, without corrections for Fresnel reflected and stick
        weights.
    refl_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event.
    trans_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    nevents : int
        number of events for which Monte Carlo Calculation is run

    Returns
    -------
    refl_events : `xr.DataArray` (dims=..., event)
        reflectance contribution for each event.
    trans_events : `xr.DataArray` (dims=..., event)
        transmittance contribution for each event.

    """
    refl_events = sum_per_event_number(refl_per_traj, refl_indices, nevents)
    trans_events = sum_per_event_number(trans_per_traj, trans_indices, nevents)

    # zeroth event should include only fresnel reflection
    # need to use reindex_like() to select only those events in refl_events
    refl_events.loc[dict(event=0)] = \
        inc_refl_per_traj.reindex_like(refl_events).sum("trajectory")
    trans_events.loc[dict(event=0)] = xr.zeros_like(trans_events.sel(event=0))

    return(refl_events, trans_events)


def calc_path_length(step, exit_indices):
    '''
    Returns reflectance and transmittance as a function of event number

    Parameters
    ----------
    step: 2d array (shape: nevents, ntrajectories)
        Sampled step sizes for all events and trajectories in Monte
        Carlo model
    exit_indices: 1d array (length: ntrajectories)
        event number at exit for each trajectory. Input refl_indices if you
        want to only consider reflectance and trans_indices if you want to only
        consider transmittance. Input refl_indices + trans_indices if you want
        to consider both
    Returns
    -------
    path_length_traj: 1d array (length: ntrajectories)
        path length travelled before exit for each trajectory.

    '''
    ntraj = len(exit_indices)
    path_length_traj = sc.Quantity(np.zeros(ntraj), sc.LENGTH_UNIT)

    for i in range(0, ntraj):
        path_length_traj[i] = np.sum(step[:exit_indices[i],i])

    return path_length_traj


def calc_refl_trans_event_traj(refl_per_traj, inc_refl_per_traj,
                          trans_per_traj, refl_indices, trans_indices, nevents,
                          ntraj=100):
    '''
    Returns reflectance and transmittance as a function of event number
    and trajectory

    Parameters
    ----------
    refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory from Monte Carlo
        simulation. Sum should be total reflectance from Monte Carlo
        calculation, without corrections for Fresnel reflected and stuck
        weights.
    inc_refl_per_traj: 1d array (length: ntrajectories)
        Reflectance contribution for each trajectory at the sample interface.
        This contribution comes from the Fresnel reflection as the light
        enters the sample
    trans_per_traj: 1d array (length: ntrajectories)
        Transmittance contribution for each trajectory from Monte Carlo
        simulation. Sum should be total transmittance from Monte Carlo
        calculation, without corrections for Fresnel reflected and stick
        weights.
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
        refl_events_traj[ev, traj_ind_refl_ev] += \
            np.sum(refl_per_traj[traj_ind_refl_ev])
        trans_events_traj[ev, traj_ind_trans_ev] += \
            np.sum(trans_per_traj[traj_ind_trans_ev])

    return refl_events_traj, trans_events_traj


def calc_thetas_event_traj(theta, refl_indices, nevents, ntraj = 100):
    '''
    Returns array of thetas at reflection for every event and trajectory. If
    trajectory is not reflected at a particular event, theta value is 0.

    Parameters
    ----------
    theta: 2d array (shape: nevents, ntrajectories)
        sampled thetas used in Monte Carlo. Includes all thetas for all
        trajectories.
    refl_indices: 1d array (length: ntrajectories)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event.
    nevents: int
        number of events for which Monte Carlo calculation is run
    ntraj: int, optional
        number of trajectories from Monte Carlo to examine. Usually want a
        smaller number than the full number of trajectories run because size of
        matrix will be too big. Default value is 100.

    Returns
    -------
    theta_event_traj: 2d array (shape: nevents, ntrajectories)
        thetas at reflection for every event and trajectory. If trajectory is
        not reflected at a particular event, theta value is 0 for that event
        and trajectory.
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


def calc_tir(tir_indices, refl_indices, trans_indices, inc_refl_per_traj,
             n_sample, n_medium, boundary, trajectories, thickness):
    """
    Returns weights of various types of totally internally reflected
    trajectories as a function of event number

    Parameters
    ----------
    tir_indices : `xr.DataArray` (dims=..., trajectory)
        array of event indices for trajectories with TIR before exit
    refl_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event.
    trans_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    inc_refl_per_traj : `xr.DataArray` (dims=...)
        Reflectance contribution for each trajectory at the sample interface.
        This contribution comes from the Fresnel reflection as the light
        enters the sample
    n_sample : `xr.DataArray` (dims=...)
        Refractive index of the sample, as returned by an Index object
    n_medium : `xr.DataArray` (dims=...)
        Refractive index of the medium, as returned by an Index object
    boundary : string
        geometrical boundary, current options are 'film' or 'sphere'
    trajectories : `xr.Dataset` (dims=..., component, event, trajectory)
        Trajectories from a Monte Carlo simulation
    thickness : float
        thickness of film or diameter of sphere

    Returns
    -------
    tir_all_events : `xr.DataArray` (dims=..., event)
        summed weights of trajectories that are totally internally reflected at
        any event, regardeless of whether they are eventually reflected,
        transmitted, or stuck. The event index of the array corresponds to the
        event at which they are totally internally reflected.
    tir_all_refl_events : `xr.DataArray` (dims=..., event)
        summed weights of trajectories that are totally internally reflected at
        any event, but only those which eventually contribute to reflectance.
        The event index of the array corresponds to the event at which they are
        reflected.
    tir_single_events : `xr.DataArray` (dims=..., event)
        summed weights of trajectories that are totally internally reflected
        after the first scattering event, regardless of whether they are
        reflected, transmitted, or stuck. The event index corresponds to the
        event at which they are totally internally reflected
    tir_single_refl_events : `xr.DataArray` (dims=..., event)
        summed weights of trajectories that are totally internally reflected
        adter the first scattering event and eventually contribute to
        reflectance. The event index corresponds to the event at which they are
        reflected.
    tir_indices_single : `xr.DataArray` (dims=..., trajectory)
        The event indices of trajectories that are totally internally reflected
        after a single scattering event.

    """
    weights = trajectories["weight"]
    nevents = trajectories.sizes["event"] - 1
    ntraj = trajectories.sizes["trajectory"]

    # ### tir for all events ###

    # make event indices of zero larger than possible nevents so that
    # refl_events of 0 never have a smaller number than any other events
    refl_ind_inf = xr.where(refl_indices == 0, nevents*10, refl_indices)
    trans_ind_inf = xr.where(trans_indices == 0, nevents*10, trans_indices)

    # find  tir indices where trajectories are tir'd before getting reflected
    # or transmitted
    tir_indices = xr.where(tir_indices > refl_ind_inf, 0, tir_indices)
    tir_indices = xr.where(tir_indices > trans_ind_inf, 0, tir_indices)
    tir_all = (1-inc_refl_per_traj) * select_events(weights, tir_indices)/ntraj

    ### tir for all events that gets reflected eventually ###

    # find event indices where tir'd trajectories are reflected
    tir_indices_refl = xr.where(tir_indices != 0, refl_indices, 0)
    # find the tir reflectance at each event
    tir_all_refl = ((1-inc_refl_per_traj)
                    * select_events(weights, tir_indices_refl)
                    * fresnel_pass_frac(tir_indices_refl, n_sample, None,
                                     n_medium, boundary, trajectories,
                                     thickness)[0]) / ntraj

    # find the event indices where single scat trajectories are tir'd
    tir_indices_single = xr.where(tir_indices != 2, 0, tir_indices)
    tir_single = ((1-inc_refl_per_traj)
                  * select_events(weights, tir_indices_single) / ntraj)

    ### tir for only single scat event that gets reflected eventually ###

    # find event indices where single scat tir'd trajectories are reflected
    tir_indices_single_refl = xr.where(tir_indices_single == 2,
                                       refl_indices, 0)

    # calculate the single scat tir'd reflectance at each event
    tir_single_refl = ((1-inc_refl_per_traj)
                       * select_events(weights, tir_indices_single_refl)
                       * fresnel_pass_frac(tir_indices_single_refl, n_sample,
                                           None, n_medium, boundary,
                                           trajectories, thickness)[0]) / ntraj

    # add up reflectance/transmittance due to trajectories
    # reflected/transmitted at each event
    tir_all_events = sum_per_event_number(tir_all, tir_indices, nevents)
    tir_all_refl_events = sum_per_event_number(tir_all_refl, tir_indices_refl,
                                               nevents)
    tir_single_events = sum_per_event_number(tir_single, tir_indices_single,
                                             nevents)
    tir_single_refl_events = sum_per_event_number(tir_single_refl,
                                                  tir_indices_single_refl,
                                                  nevents)

    return (tir_all_events,
            tir_all_refl_events,
            tir_single_events,
            tir_single_refl_events,
            tir_indices_single)


def calc_tir_phase_event_input(tir_indices, step, refl_indices, radius,
                               volume_fraction, n_particle, n_sample,
                               wavelength, trajectories, refl_per_traj,
                               bin_width=sc.Quantity(40,'fs')):
    '''
    Calculates the parameters needed to input into calc_refl_trans_event
    in order to calculate the total internal reflected weights as a function
    of events number, including phase calculations (coherence effects)

    Parameters
    ----------
    tir_indices: array (shape: ntraj)
        array of event indices for trajectories with TIR before exit
    step: 2d array (structcol.Quantity [length])
        Step sizes between scattering events in each of the trajectories.
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
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
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel
        contributions
    bin_width: float (structcol.Quantity [time])
        size of time bins for creating field versus time. Should be set equal
        to coherence time of source

    Returns
    -------
    tir_per_traj_phase: 1d array (length:ntraj)
        totally internnaly reflected weights for each trajectory
    tir_indices_refl: 1d array (length:ntraj)
        event indices at which trajectories are totally internally reflected

    '''

    nevents = trajectories.nevents
    ntraj = len(refl_indices)

    traj_times_tir,_,_ = det.calc_traj_time(step, refl_indices, radius,
                                            volume_fraction, n_particle,
                                            n_sample, wavelength)
    _, tir_per_traj_phase = det.calc_refl_phase_time(traj_times_tir,
                                                 trajectories, refl_indices,
                                                 refl_per_traj,
                                                 bin_width=bin_width)


    # make event indices of zero larger than possible nevents so that
    # refl_events of 0 never have a smaller number than any other events
    refl_ind_inf = np.copy(refl_indices)
    refl_ind_inf[refl_ind_inf == 0] = nevents*10
    # find  tir indices where trajectories are tir'd before getting reflected
    tir_indices[np.where(tir_indices>refl_ind_inf)[0]] = 0
    tir_ev_ind = np.where(tir_indices!=0)
    tir_indices_refl = np.zeros(ntraj)
    tir_indices_refl[tir_ev_ind] = refl_indices[tir_ev_ind]

    return tir_per_traj_phase, tir_indices_refl


def calc_pdf_scat(refl_events, trans_events, nevents):
    """
    Calculates probability density function of reflection and transmission at
    each event.

    Parameters
    ----------
    refl_events : `xr.DataArray` (dims=..., event)
        reflectance contribution for each event
    trans_events: `xr.DataArray` (dims=..., event)
        transmittance contribution for each event
    nevents : int
        number of events for which Monte Carlo calculation is run

    Returns
    -------
    pdf_refl : `xr.DataArray` (dims=..., event)
        probability of reflection at each event
    pdf_trans : `xr.DataArray` (dims=..., event)
        probability of transmission at each event
    """
    # 0th event: reflection due to fresnel at interface
    # 1st event: reflection exits after 1st step into sample (always 0
    # because cannot)
    # 2nd event: "singly scattered" in the sense that has scattered once inside
    # the sample, so could exit
    # why "nevents + 1" ? because we added an extra "event" by including
    # the fresnel reflection as the 0th event

    pdf_refl = refl_events.isel(event=slice(2, nevents+1))
    pdf_trans = trans_events.isel(event=slice(1, nevents+1))

    # returned normalized
    return pdf_refl/pdf_refl.sum("event"), pdf_trans/pdf_trans.sum("event")


def calc_refl_event_fresnel_pdf(refl_events, pdf_refl, pdf_trans, refl_indices,
                                trans_indices, refl_fresnel, trans_fresnel,
                                refl_frac, trans_frac, nevents, rng=None):
    """
    Calculates the reflectance contribution from fresnel reflected trajectory
    weights and adds it to the total reflectance contribution for a sampled
    event at which the fresnel trajectory exits


    Parameters
    ----------
    refl_events : `xr.DataArray` (dims=..., event)
        reflectance contribution for each event.
    pdf_refl : `xr.DataArray` (dims=..., event)
        probability of reflection at each event
    pdf_trans : `xr.DataArray` (dims=..., event)
        probability of transmission at each event
    refl_indices : 'xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event.
    trans_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    refl_fresnel : `xr.DataArray` (dims=..., trajectory)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal
        reflection.
    trans_fresnel : `xr.DataArray` (dims=..., trajectory)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal
        reflection.
    refl_frac : `xr.DataArray` (dims=...)
        fraction of trajectory weights that are reflected normalized by the
        known outcomes of trajectories
    trans_frac : `xr.DataArray` (dims=...)
        fraction of trajectory weights that are transmitted normalized by the
        known outcomes of trajectories
    nevents : int
        number of events for which Monte Carlo calculation is run
    rng : numpy.random.Generator object (default None)
        If not specified, use the default generator initialized on loading the
        package

    Returns
    -------
    refl_events + fresnel_samp : `xr.DataArray` (dims: ..., event)
        reflectance contribution for each event added to the fresnel
        reflectance contribution for each event.

    Notes
    -----
    As with calc_refl_event_fresnel_avg(), we consider a fresnel trajectory
    generated at event x, but here we sample the number of events it takes to
    exit (y) from the exit events of existing trajectories. The trajectory then
    exits at event x + y. All fresnel trajectories generated at event x are
    associated with the same y.

    """
    if rng is None:
        rng = sc.rng

    # find the weights of the fresnel reflected trajectories at each event
    refl_weights = refl_frac * sum_per_event_number(refl_fresnel, refl_indices,
                                                    nevents)
    trans_weights = trans_frac * sum_per_event_number(trans_fresnel,
                                                      trans_indices, nevents)

    # sample reflection and transmission event numbers
    # (for each possible event index, we pick an offset index from the actual
    # trajectories)
    pdf_refl = pdf_refl.transpose(..., "event")
    pdf_trans = pdf_trans.transpose(..., "event")
    # save coords to add back to arrays after conversion to numpy for sampling
    coords = pdf_refl.drop_vars("event").coords
    # calculate shape of array to be sampled based on leading dims in pdf
    leading_dims_shape = pdf_refl.shape[:-1]
    sample_size = leading_dims_shape + (nevents+1,)

    sampled_refl_event = sc.choice(np.arange(2, nevents + 1),
                                   sample_size, pdf_refl.to_numpy(),
                                   rng=rng)
    sampled_trans_event = sc.choice(np.arange(1, nevents + 1),
                                    sample_size, pdf_trans.to_numpy(),
                                    rng=rng)

    # convert to DataArrays
    sampled_refl_event = xr.DataArray(sampled_refl_event,
                                      coords=dict(**coords,
                                                  event=range(nevents+1)))
    sampled_trans_event = xr.DataArray(sampled_trans_event,
                                       coords=dict(**coords,
                                                   event=range(nevents+1)))

    # add the frensel reflected trajectory event to the sampled event of
    # reflection or transmission
    fresnel_samp = xr.DataArray(np.zeros(2*nevents + 1),
                                coords={"event": range(2*nevents + 1)})
    for ev in range(1, nevents + 1):
        # sampled_refl_event has a size nevents + 1, and this loop has size
        # nevents + 1
        # sampled_trans_event has a size nevents + 1, even though it includes
        # an extra event to sample
        new_refl_event = (ev + sampled_refl_event.isel(event=ev))
        fresnel_samp.loc[dict(event=new_refl_event)] = \
            (fresnel_samp.isel(event=new_refl_event)
             + refl_weights.isel(event=ev))
        new_trans_event = (ev + sampled_trans_event.isel(event=ev))
        fresnel_samp.loc[dict(event=new_trans_event)] = \
            (fresnel_samp.isel(event=new_trans_event)
             + trans_weights.isel(event=ev))

    # addition will restore leading dimensions to the returned array
    return refl_events + fresnel_samp


def calc_refl_event_fresnel_avg(refl_events, refl_indices, trans_indices,
                            refl_fresnel, trans_fresnel,
                            refl_frac, trans_frac, nevents):
    """
    Calculates the reflectance contribution from fresnel reflected trajectory
    weights and adds it to the total reflectance contribution for the average
    event at which the fresnel trajectory exits

    Parameters
    ----------
    refl_events : `xr.DataArray` (dims=..., events)
        reflectance contribution for each event.
    refl_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is reflected. Value of 0 means
        trajectory is not reflected at any event.
    trans_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices at which each trajectory is transmitted. Value of 0 means
        trajectory is not transmitted at any event
    refl_fresnel : `xr.DataArray` (dims=..., trajectory)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal
        reflection.
    trans_fresnel : `xr.DataArray` (dims=..., trajectory)
        weights of trajectories that are Fresnel reflected back into the sample
        when a trajectory exits. This does not include total internal
        reflection.
    refl_frac : `xr.DataArray` (dims=...)
        fraction of trajectory weights that are reflected normalized by the
        known outcomes of trajectories
    trans_frac : `xr.DataArray` (dims=...)
        fraction of trajectory weights that are transmitted normalized by the
        known outcomes of trajectories
    nevents : int
        number of events for which Monte Carlo calculation is run

    Returns
    -------
    refl_events + fresnel_samp : `xr.DataArray` (dims=..., events)
        reflectance contribution for each event added to the fresnel
        reflectance contribution for each event.

    Notes
    -----
    Consider a fresnel reflection that occurs at an event x. If we do not want
    to simulate the resulting trajectory, we can assume it exits after some
    number of events y. In this function, we assume y is the average number of
    events it takes for any reflected trajectory to exit. So the trajectory
    from fresnel reflection exits at event x + y. One disadvantage of this
    approach is that the calculated event distribution will show a sudden
    increase at the average event y. Advantage is that it is vectorized over
    wavelength.

    """
    # find average event at which reflection or transmission occurs
    avg_refl_event = (refl_indices.where(refl_indices!=0, drop=True)
                      .mean("trajectory").round().astype(int))
    avg_trans_event = (trans_indices.where(trans_indices!=0, drop=True)
                       .mean("trajectory").round().astype(int))

    # find the weights of the fresnel reflected trajectories at each event
    refl_weights = refl_frac * sum_per_event_number(refl_fresnel, refl_indices,
                                                    nevents)
    trans_weights = trans_frac * sum_per_event_number(trans_fresnel,
                                                      trans_indices, nevents)
    event_coord = refl_weights.coords["event"]

    # shift event coord by the average event number.  Note that avg_refl_event
    # and avg_trans_events are arrays, so we can't use xr.DataArray.shift,
    # which only works with scalar shifts.  Instead we use advanced indexing.
    refl_index_shift = (event_coord - avg_refl_event).clip(0, 2*nevents+1)
    refl_weights = refl_weights.transpose("event",...).loc[refl_index_shift]
    trans_index_shift = (event_coord - avg_trans_event).clip(0, 2*nevents+1)
    trans_weights = trans_weights.transpose("event",...).loc[trans_index_shift]

    # add to get total weights
    ntraj = refl_indices.sizes["trajectory"]
    alltraj = xr.DataArray(np.zeros(ntraj),
                           coords={"trajectory": range(ntraj)})
    fresnel_weights = (refl_weights.reindex_like(alltraj, fill_value=0.0)
                       + trans_weights.reindex_like(alltraj, fill_value=0.0)
                       + refl_events)

    return fresnel_weights


def plot_refl_event(wavelengths, refl_events, event):   # pragma: no cover
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
        plt.plot(wavelengths, refl_events[:, event[ev]], label = event[ev],
                 linewidth = 3)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()

def plot_refl_event_norm(wavelengths, refl_events, event): # pragma: no cover
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
        plt.plot(wavelengths, (refl_events[:, event[ev]] /
                               (1 - np.sum(refl_events[:,events_before],
                                           axis = 1))),
                 label = event[ev], linewidth = 3)
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()

def plot_refl_event_sum(wavelengths, refl_events, events,
                        label = ''):        # pragma: no cover
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
    plt.plot(wavelengths, np.sum(refl_events[:,events], axis = 1),
             label = label, linewidth = 3)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.ylabel('Reflectance')
    plt.xlabel('Wavelength (nm)')

def plot_refl_event_sum_norm(wavelengths, refl_events, events,
                             label = ''):       # pragma: no cover
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
        refl_events_norm[:, ev] = (refl_events[:, events[ev]]
                                   / (1 - np.sum(refl_events[:,events_before],
                                                 axis = 1)))
    plt.plot(wavelengths, np.sum(refl_events_norm, axis = 1),
             label = label, linewidth = 3)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.ylabel('Reflectance')
    plt.xlabel('Wavelength (nm)')

def plot_refl_dist(wavelengths, refl_events, wavelength): # pragma: no cover
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
        plt.semilogx(events-1, (refl_events[wavelength_ind,:]
                                / np.sum(refl_events[wavelength_ind,:])),
                     label = wavelength[wl], marker = '.', markersize = 12,
                     linewidth = 2)
    plt.xlim([1, events[-1]])
    plt.xlabel('Scattering Event Number')
    plt.ylabel('Reflectance Contribution')
    plt.legend()

    plt.figure()
    for wl in range(0, len(wavelength)):
        wavelength_ind = np.where(wavelengths == wavelength[wl])[0][0]
        plt.loglog(events-1, (refl_events[wavelength_ind,:] /
                              np.sum(refl_events[wavelength_ind,:])),
                   label = wavelength[wl], marker = '.', markersize = 12,
                   linewidth = 0)
    plt.xlim([1, events[-1]])
    plt.xlabel('Scattering Event Number')
    plt.ylabel('Reflectance Contribution')
    plt.legend()

    plt.figure()
    for wl in range(0, len(wavelength)):
        wavelength_ind = np.where(wavelengths == wavelength[wl])[0][0]
        plt.semilogx(events-1, refl_events[wavelength_ind,:],
                     label = wavelength[wl], marker = '.', markersize = 12,
                     linewidth = 2)
    plt.xlim([1, events[-1]])
    plt.xlabel('Scattering Event Number')
    plt.ylabel('Reflectance')
    plt.legend()

def save_data(particle, matrix, particle_radius, volume_fraction, thickness,
              reflectance, refl_events, wavelengths, nevents, ntrajectories,
              theta_event_traj = None, refl_events_fresnel_samp = None,
              refl_events_fresnel_avg = None, zpos = None, kz = None,
              theta_range = None, tir_single = None, tir_single_refl = None,
              tir_all = None, tir_all_refl = None,
              tir_indices_single = None):   # pragma: no cover
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
