# Copyright 2016, Vinothan N. Manoharan, Annie Stephenson
#
# This file is part of the structural-color python package.
#
# This package is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This package is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this package. If not, see <http://www.gnu.org/licenses/>.
"""
Tests for the montecarlo bulk model

TODO: consider moving tests to test_detector_sphere()

.. moduleauthor:: Anna B. Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import numpy as np
import structcol as sc
from structcol import montecarlo as mc
from structcol import detector as det
from structcol import phase_func_sphere as pfs
from numpy.testing import assert_almost_equal
import xarray as xr
import pytest

### Set parameters ###

# Properties of source
wavelength = sc.Quantity('600.0 nm') # wavelengths at which to calculate reflectance

# Geometric properties of sample
#
# radius of the sphere particles
particle_radius = sc.Quantity('0.130 um')
# volume fraction of the particles in the sphere boundary
volume_fraction_particles = 0.6
# volume fraction of the spheres in the bulk film
volume_fraction_bulk = 0.55
# diameter of the sphere boundary
sphere_boundary_diameter = sc.Quantity(10.0, "um").to_preferred()
boundary = 'sphere'
boundary_bulk = 'film'

# Refractive indices
#
# refractive index of particle
index_particle = sc.index.vacuum
particle = sc.Sphere(index_particle, particle_radius)
# refractive index of matrix
index_matrix = sc.index.polystyrene
# refractive index of the bulk matrix
index_matrix_bulk = sc.index.vacuum
n_matrix_bulk = index_matrix_bulk(wavelength)
# refractive index of medium outside the bulk sample.
index_medium = sc.index.vacuum

# Monte Carlo parameters
#
# number of trajectories to run with a spherical boundary
ntrajectories = 2000
# number of scattering events for each trajectory in a spherical boundary
nevents = 300


def calc_sphere_mc():
    # set up a seeded random number generator that will give consistent results
    # between numpy versions.
    seed = 1
    rng = np.random.RandomState([seed])

    # set up scattering model
    model = sc.model.HardSpheres(particle, volume_fraction_particles,
                                 index_matrix, index_medium)

    # Initialize and run the simulation
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories, boundary,
                        sample_diameter = sphere_boundary_diameter,
                        rng=rng)
    sim.run()

    # sim = mc.Simulation(nevents, ntrajectories, n_matrix_bulk, n_sample,
    #                     boundary,sample_diameter = sphere_boundary_diameter,
    #                     rng=rng)

    # Calculate reflection and transmission
    (refl_indices, trans_indices, stuck_indices, tir_indices,
     _, _, _,
     refl_per_traj, trans_per_traj,
     _,_,_,_,
     reflectance_sphere,
     _,
     norm_refl, norm_trans) = det.calc_refl_trans(sim,
                                                  sphere_boundary_diameter,
                                                  run_fresnel_traj = False,
                                                  return_extra = True)

    return (refl_indices, trans_indices, refl_per_traj, trans_per_traj,
            reflectance_sphere, norm_refl, norm_trans)


def test_mu_scat_abs_bulk():

    # make sure there is no absorption when all refractive indices are real
    (refl_indices, trans_indices,
     refl_per_traj, trans_per_traj,
     reflectance_sphere,
     norm_refl, norm_trans) = calc_sphere_mc()

    ### Calculate phase function and lscat ###
    # use output of calc_refl_trans to calculate phase function, mu_scat,
    # and mu_abs for the bulk
    _, _, mu_abs_bulk = pfs.calc_scat_bulk(refl_per_traj,
                                           trans_per_traj,
                                           refl_indices,
                                           trans_indices,
                                           norm_refl, norm_trans,
                                           volume_fraction_bulk,
                                           sphere_boundary_diameter,
                                           n_matrix_bulk,
                                           wavelength)

    assert_almost_equal(mu_abs_bulk, 0)


    # make sure mu_abs reaches limit when there is no scattering
    with pytest.warns(UserWarning,
                      match="No trajectories reflected or transmitted"):
        _, mu_scat_bulk, mu_abs_bulk = \
            pfs.calc_scat_bulk(xr.zeros_like(refl_per_traj),
                               xr.zeros_like(trans_per_traj),
                               refl_indices,
                               trans_indices,
                               norm_refl, norm_trans,
                               volume_fraction_bulk,
                               sphere_boundary_diameter,
                               n_matrix_bulk,
                               wavelength)

    number_density = volume_fraction_bulk/(4/3*np.pi*
                                        (sphere_boundary_diameter.magnitude/2)**3)
    mu_abs_max = number_density*np.pi*(sphere_boundary_diameter.magnitude/2)**2

    assert_almost_equal(mu_abs_bulk, mu_abs_max)

    # check that mu_scat_bulk is 0 when no scattering
    assert_almost_equal(mu_scat_bulk, 0)


    # check the mu_scat_bulk reaches limit when there is only scattering
    norm_refl.loc[dict(component="z")] = 1/np.sqrt(3)
    norm_refl.loc[dict(component="y")] = 1/np.sqrt(3)
    norm_refl.loc[dict(component="x")] = 1/np.sqrt(3)
    norm_trans.loc[dict(component="z")]= 0
    norm_trans.loc[dict(component="y")] = 0
    norm_trans.loc[dict(component="x")] = 0


    _, mu_scat_bulk, _ = pfs.calc_scat_bulk(1/ntrajectories
                                            * xr.ones_like(refl_per_traj),
                                            xr.zeros_like(trans_per_traj),
                                            xr.ones_like(refl_indices) + 3,
                                            xr.zeros_like(trans_indices),
                                            norm_refl, norm_trans,
                                            volume_fraction_bulk,
                                            sphere_boundary_diameter,
                                            n_matrix_bulk,
                                            wavelength)

    number_density = volume_fraction_bulk/(4/3*np.pi*
                                        (sphere_boundary_diameter.magnitude/2)**3)
    mu_scat_max = number_density*2*np.pi*(sphere_boundary_diameter.magnitude/2)**2

    assert_almost_equal(mu_scat_bulk, mu_scat_max)
