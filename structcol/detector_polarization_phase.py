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
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>

"""
import structcol as sc
from pymie import mie
from . import select_events
from . import LIGHT_SPEED_VACUUM
import numpy as np
import xarray as xr
import warnings


def calc_refl_fields(trajectories, refl_indices, refl_per_traj,
                     components=False):
    """
    Calculates the reflectance by adding fields coherently

    Parameters
    ----------
    trajectories : `xr.Dataset` (dims=..., component, event, trajectory)
        Trajectories from a `sc.Simulation` object
    refl_indices : `xr.DataArray` (dims=..., trajectory)
        Event indices for reflected trajectories
    refl_per_traj : `xr.DataArray` (dims=..., trajectory)
        Reflectance distributed to each trajectory, including fresnel
        contributions
    components : boolean
        If True, return total field in addition to reflectances

    Returns
    -------
    if components == True:
        tot_field, refl_fields, refl_non_phase / intensity_incident
    else:
        refl_fields, refl_non_phase / intensity_incident

    """
    ntraj = trajectories.sizes["trajectory"]

    if (refl_indices == 0).all():
        no_refl_warn = '''No trajectories were reflected.
                          Check sample parameters or increase number
                          of trajectories.'''
        warnings.warn(no_refl_warn)

    # Get the amplitude of the field
    # The expression below gives 0 for not reflected traj, but that's fine
    # since we only care about reflected trajectories.
    w = np.sqrt(refl_per_traj * ntraj)

    # Write expression for field.
    traj_field = w * trajectories.fields

    # Select traj_field values only for the reflected indices.
    refl_field = select_events(traj_field, refl_indices)

    # Add reflected fields from all trajectories.
    tot_field = refl_field.sum("trajectory")
    # Calculate intensity as E^*E.
    intensity = tot_field.real**2 + tot_field.imag**2
    # Add the x,y, and z intensity.
    refl_intensity = intensity.sum("component")
    # Normalize, assuming incident light is incoherent.
    intensity_incident = ntraj  # np.sum(weights[0,:])
    refl_fields = (refl_intensity / intensity_incident).real

    # Calculate the incoherent reflectance for comparison. This is done by
    # first finding the intensity per trajectory then summing over all
    # trajectories where as when interfering between trajectories we first
    # add the field components of all trajectories then sum over components
    refl_incoherent = ((refl_field.real**2 + refl_field.imag**2)
                       .sum(["component", "trajectory"]))
    refl_intensity_tot = (refl_incoherent / intensity_incident).real

    if components:
        return tot_field, refl_fields, refl_intensity_tot
    else:
        return refl_fields, refl_intensity_tot


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

    (tot_field, refl_field, refl_intensity) = \
        calc_refl_fields(trajectories, refl_indices, refl_per_traj,
                         components=True)

    if isinstance(det_theta, sc.Quantity):
        det_theta = det_theta.to('radians').magnitude

    # Incorporate geometry of the goniometer setup.
    # Rotate the total x, y, z fields to the par/perp detector basis,
    # by performing a clockwise rotation about the y-axis by angle det_theta.
    # Co-polarized field is mostly x-polarized.
    # Cross-polarized field is mostly y-polarized.
    # Field perpendicular to scattering plane is mostly z-polarized.
    tot_field_co = (tot_field.sel(component="x") * np.cos(det_theta)
                    + tot_field.sel(component="z") * np.sin(det_theta))
    tot_field_cr = tot_field.sel(component="y")
    tot_field_perp = (-tot_field.sel(component="x") * np.sin(det_theta)
                      + tot_field.sel(component="z") * np.cos(det_theta))

    # convert to Dataset
    tot_field = xr.Dataset({"co": tot_field_co,
                            "cross": tot_field_cr,
                            "perp": tot_field_perp})

    # Take the modulus to get intensity.
    refl = tot_field.real**2 + tot_field.imag**2
    refl["field"] = refl_field
    refl["intensity"] = refl_intensity

    return refl

# this function is not used (likely deprecated since the introduction of the
# fields model)
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
    path_length_traj = sc.Quantity(np.zeros(ntraj), sc.LENGTH_UNIT)

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
    traj_time[stuck_traj_ind] = sc.Quantity(0.0, 'fs')

    # change units to femtoseconds and discard imaginary part
    traj_time = traj_time.to('fs')
    traj_time = np.real(traj_time.magnitude)
    traj_time = sc.Quantity(traj_time, 'fs')

    return traj_time, travel_time, dwell_time
