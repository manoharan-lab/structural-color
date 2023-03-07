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
This module provides functions for detecting properties of
trajectories simulated by the Monte Carlo model that are
related to it's field properties: polarization and phase.


.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>

"""
from pymie import mie
from . import select_events
from . import LIGHT_SPEED_VACUUM
import numpy as np
import structcol as sc
import warnings


def calc_refl_phase_fields(trajectories, refl_indices, refl_per_traj,
                           components=False):
    '''
    Calculates the reflectance including phase, by considering trajectories
    that exit at the same time to be coherent. To do this, we must bin
    trajectories with similar exit times and add their fields. Then
    we convolve the reflectance as a function of time with a step function
    in order to give a steady state value for the reflectance.

    Parameters
    ----------
    trajectories: Trajectory object
        Trajectory object used in Monte Carlo simulation
    refl_indices: 1d array (length: ntraj)
        array of event indices for reflected trajectories
    refl_per_traj: 1d array (length: ntraj)
        reflectance distributed to each trajectory, including fresnel
        contributions
    components: boolean

    Returns
    -------
    if components == True:
        return tot_field_x, tot_field_y, tot_field_z, refl_fields,
        refl_non_phase / intensity_incident
    else:
        return refl_fields, refl_non_phase / intensity_incident
    '''

    ntraj = len(trajectories.direction[0, 0, :])

    if np.all(refl_indices == 0):
        no_refl_warn = '''No trajectories were reflected.
                          Check sample parameters or increase number
                          of trajectories.'''
        warnings.warn(no_refl_warn)
    if isinstance(trajectories.weight, sc.Quantity):
        weights = trajectories.weight.magnitude
    else:
        weights = trajectories.weight

    # Get the amplitude of the field
    # The expression below gives 0 for not reflected traj, but that's fine
    # since we only care about reflected trajectories.
    w = np.sqrt(refl_per_traj * ntraj)

    # Write expression for field.
    # 0th event is before entering sample, so we start from 1,
    # for later use with select_events.
    traj_field_x = w * trajectories.fields[0, 1:, :]
    traj_field_y = w * trajectories.fields[1, 1:, :]
    traj_field_z = w * trajectories.fields[2, 1:, :]

    # Select traj_field values only for the reflected indices.
    refl_field_x = select_events(traj_field_x, refl_indices)
    refl_field_y = select_events(traj_field_y, refl_indices)
    refl_field_z = select_events(traj_field_z, refl_indices)

    # Add reflected fields from all trajectories.
    tot_field_x = np.sum(refl_field_x)
    tot_field_y = np.sum(refl_field_y)
    tot_field_z = np.sum(refl_field_z)

    # Calculate the incoherent reflectance for comparison.
    non_phase_int_x = np.conj(refl_field_x) * refl_field_x
    non_phase_int_y = np.conj(refl_field_y) * refl_field_y
    non_phase_int_z = np.conj(refl_field_z) * refl_field_z
    refl_non_phase = np.sum(non_phase_int_x + non_phase_int_y
                            + non_phase_int_z)

    # Calculate intensity as E^*E.
    intensity_x = np.conj(tot_field_x) * tot_field_x
    intensity_y = np.conj(tot_field_y) * tot_field_y
    intensity_z = np.conj(tot_field_z) * tot_field_z

    # Add the x,y, and z intensity.
    refl_intensity = np.sum(intensity_x + intensity_y + intensity_z)

    # Normalize, assuming incident light is incoherent.
    intensity_incident = ntraj  # np.sum(weights[0,:])
    refl_fields = np.real(refl_intensity / intensity_incident)

    refl_x = np.sum(intensity_x) / intensity_incident
    refl_y = np.sum(intensity_y) / intensity_incident
    refl_z = np.sum(intensity_z) / intensity_incident

    if components:
        return (tot_field_x, tot_field_y, tot_field_z, refl_fields,
                refl_non_phase / intensity_incident)
    else:
        return refl_fields, refl_non_phase / intensity_incident


def calc_refl_co_cross_fields(trajectories, refl_indices, refl_per_traj,
                              det_theta):
    '''
    Goniometer detector size should already be taken account
    in calc_refl_trans() so the refl_indices will only include trajectories
    that exit within the detector area.

    Muliplying by the sines and cosines of the detector theta is an
    approximation, since the goniometer detector area is usually small
    enough such that the detector size is not that big. Should check that
    this approximation is reasonable. The alternative would be to keep track
    of the actual exit theta of each trajectory, using the direction property.

    '''

    (tot_field_x,
     tot_field_y,
     tot_field_z,
     refl_field,
     refl_intensity) = calc_refl_phase_fields(trajectories, refl_indices,
                                              refl_per_traj,
                                              components=True)

    # Incorporate geometry of the goniometer setup.
    # Rotate the total x, y, z fields to the par/perp detector basis,
    # by performing a clockwise rotation about the y-axis by angle det_theta.
    # Co-polarized field is mostly x-polarized.
    # Cross-polarized field is mostly y-polarized.
    # Field perpendicular to scattering plane is mostly z-polarized.
    tot_field_co = (tot_field_x * np.cos(det_theta) + tot_field_z
                    * np.sin(det_theta))
    tot_field_cr = tot_field_y
    tot_field_perp = (-tot_field_x * np.sin(det_theta) + tot_field_z
                      * np.cos(det_theta))

    # Take the modulus to get intensity.
    refl_co = np.conj(tot_field_co) * tot_field_co
    refl_cr = np.conj(tot_field_cr) * tot_field_cr
    refl_perp = np.conj(tot_field_perp) * tot_field_perp

    return (refl_co, refl_cr, refl_perp, refl_field, refl_intensity)


def calc_traj_time(step, exit_indices, radius,
                   n_particle, n_sample, wavelength,
                   min_angle=0.01,
                   num_angles=200):
    '''
    Calculates the amount of time each trajectory spends scattering in the
    sample before exit

    TODO: make this work for polydisperse, core-shell, and bispecies

    parameters:
    ----------
    step: 2d array (structcol.Quantity [length])
        Step sizes between scattering events in each of the trajectories.
    exit_indices: 1d array (length: ntrajectories)
        event number at exit for each trajectory. Input refl_indices if you
        want to only consider reflectance and trans_indices if you want to only
        consider transmittance. Input refl_indices + trans_indices if you
        want to consider both
    radius: float (structcol.Quantity [length])
        Radius of particle.
    n_particle: float
        Index of refraction of particle.
    n_sample: float
        Index of refraction of sample.
    wavelength: float (structcol.Quantity [length])
        Wavelength.
    min_angle: float (in radians)
        minimum angle to integrate over for total cross section
    num_angles: float
        number of angles to integrate over for total cross section

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
    path_length_traj = sc.Quantity(np.zeros(ntraj), 'um')

    for i in range(0, ntraj):
        path_length_traj[i] = np.sum(step[:exit_indices[i], i])
    stuck_traj_ind = np.where(path_length_traj.magnitude == 0)[0]

    # calculate the time passed based on distance travelled
    velocity = LIGHT_SPEED_VACUUM / np.real(n_sample.magnitude)
    travel_time = path_length_traj / velocity

    # calculate the dwell time in a scatterer
    dwell_time = mie.calc_dwell_time(radius, n_sample, n_particle, wavelength,
                                     min_angle=min_angle,
                                     num_angles=num_angles)

    # add the dwell times and travel times
    traj_time = travel_time + dwell_time

    # set traj_time = 0 for stuck trajectories
    traj_time[stuck_traj_ind] = sc.Quantity(0, 'fs')

    # change units to femtoseconds and discard imaginary part
    traj_time = traj_time.to('fs')
    traj_time = np.real(traj_time.magnitude)
    traj_time = sc.Quantity(traj_time, 'fs')

    return traj_time, travel_time, dwell_time
