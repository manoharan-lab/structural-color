# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang, Solomon Barkley,
# Annie Stephenson
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
Tests for the montecarlo model (in structcol/montecarlo.py).  Tests of
calculated reflectance are in test_detector.py

.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor:: Solomon Barkley <barkley@g.harvard.edu>
.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import structcol as sc
from .. import montecarlo as mc
import numpy as np
import xarray as xr
from numpy.testing import assert_equal, assert_almost_equal

# Define a system to be used for the tests
nevents = 3
ntrajectories = 4
radius = sc.Quantity('150.0 nm')
volume_fraction = 0.5
volume_fraction_da = xr.DataArray([[0.5, 1-0.5]],
                                  coords={sc.Coord.VOLFRAC: [volume_fraction],
                                          sc.Coord.MAT: range(2)})
angles = sc.Quantity(np.linspace(0.01, np.pi, 200), 'rad')
wavelen = sc.Quantity('400.0 nm')
index_particle = sc.Index.constant(1.5)
sphere = sc.Sphere(index_particle, radius)
n_particle = index_particle(wavelen)
index_matrix = sc.Index.constant(1.0)
n_matrix = index_matrix(wavelen)
index_medium = sc.Index.constant(1.0)
n_medium = index_medium(wavelen)
index_sample = sc.EffectiveIndex([index_particle, index_matrix],
                                 volume_fraction_da)
n_sample = index_sample(wavelen)

# Index of the scattering event and trajectory corresponding to the reflected
# photons
refl_index = np.array([2, 0, 2])


def test_sampling():
    # Test that calc_scat() runs. Since this test just looks to see whether
    # sampling angles and steps works, it's better if we don't give it a seeded
    # random number generator, so that we can ensure that sampling works with
    # the default generator.
    model = sc.model.HardSpheres(sphere, volume_fraction, index_matrix,
                                 index_medium)
    p, mu_scat, mu_abs = mc.calc_scat(model, wavelen)

    sim = mc.Simulation(nevents, ntrajectories, n_medium, n_sample, "film")

    # Test that 'sample_angles' runs
    sim.sample_angles(p)

    # Test that 'sample_step' runs
    sim.sample_step(mu_scat)


def test_simulation():
    """Tests that the Simulation object initializes and that
    Simulation.absorb(), Simulation.scatter(), and Simulation.move() return
    correct results.

    """
    # Initialize runs
    nevents = 2
    ntrajectories = 3

    # Create a Simulation object
    sim = mc.Simulation(nevents, ntrajectories, n_medium, n_sample, "film")

    # Test the absorb function
    mu_abs = 1/sc.Quantity(10.0, 'um')
    # step size is 1 in all events and trajectories
    step = xr.DataArray([[1, 1, 1], [1, 1, 1]],
                        coords={"event": range(nevents),
                                "trajectory": range(ntrajectories)})
    sim.absorb(mu_abs, step)
    # since step size is given (not sampled), this test should produce a
    # deterministic result
    assert_almost_equal(sim.traj["weight"]
                        .squeeze(drop=True).to_numpy(),
                        np.array([[ 1.0, 1.0, 1.0],
                                  [ 0.90483742,  0.90483742,  0.90483742],
                                  [ 0.81873075,  0.81873075,  0.81873075]]))

    # Make up some test theta and phi. Note that event 0 is assumed to be
    # propagation straight into the film, so we specify only event 1
    sintheta = xr.DataArray([[0., 0., 0.]],
                            coords = {"event": [1],
                                      "trajectory": range(ntrajectories)})
    costheta = xr.DataArray([[-1., -1., -1.]],
                            coords=sintheta.coords)
    sinphi = xr.DataArray([[0., 0., 0.]],
                          coords=sintheta.coords)
    cosphi = xr.DataArray([[0., 0., 0.]],
                          coords=sintheta.coords)

    # Test the scatter function. Should also produce a deterministic result
    sim.scatter(sintheta, costheta, sinphi, cosphi)

    # Expected propagation directions
    kx = np.array([[0., 0., 0.], [0., 0., 0.]])
    ky = np.array([[0., 0., 0.], [0., 0., 0.]])
    kz = np.array([[1., 1., 1.], [-1., -1., -1.]])
    k = np.array([kx, ky, kz])

    assert_equal(sim.traj["direction"].dropna("event")
                 .squeeze(drop=True).to_numpy(),
                 k)


    # Test the move function.  Should also produce a deterministic result since
    # step sizes are given.
    sim.move(step)
    assert_equal(sim.traj["position"].sel(component="z")
                 .squeeze(drop=True),
                 np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))

# NOTE: the test below will no longer work, since the
# differential_cross_section() function was removed from model.py (all
# differential cross sections are now evaluated using Model methods).  It
# relied on the syntax of the function, whereby not specifying the wavevector k
# allowed for the use of mie.calc_ang_scat(), even in absorbing media.  The
# object version of the code checks to see if the index is complex and will not
# allow the use of mie.calc_ang_scat() if it is.
#
# TODO: rewrite this to test the underlying pymie functions and add to
# test_mie.py instead.

# def test_phase_function_absorbing_medium():
#     # test that the phase function using the far-field Mie solutions
#     # (mie.calc_ang_scat()) in an absorbing medium is the same as the phase
#     # function using the Mie solutions with the asymptotic form of the
#     # spherical Hankel functions but using a complex k
#     # (mie.diff_scat_intensity_complex_medium() with near_fields=False)
#     wavelen = sc.Quantity('550.0 nm')
#     radius = sc.Quantity('105.0 nm')
#     n_matrix = sc.Index.constant(1.47 + 0.001j)(wavelen)
#     n_particle = sc.Index.constant(1.5 + 1e-1 * 1.0j)(wavelen)
#     m = sc.index.ratio(n_particle, n_matrix)
#     x = sc.size_parameter(n_matrix, radius)
#     k = sc.wavevector(n_matrix)
#     ksquared = np.abs(k)**2

#     ## Integrating at the surface of the particle
#     # with mie.calc_ang_scat() (this is how it's currently implemented in
#     # monte carlo)
#     diff_cscat_par_ff, diff_cscat_perp_ff = \
#         model.differential_cross_section(m, x, angles, volume_fraction,
#                                          structure_type='glass',
#                                          form_type='sphere',
#                                          diameters=radius, wavelen=wavelen,
#                                          n_matrix=n_sample, k=None, distance=radius)
#     cscat_total_par_ff = model._integrate_cross_section(diff_cscat_par_ff,
#                                                       1.0/ksquared, angles)
#     cscat_total_perp_ff = model._integrate_cross_section(diff_cscat_perp_ff,
#                                                       1.0/ksquared, angles)
#     cscat_total_ff = (cscat_total_par_ff + cscat_total_perp_ff)/2.0

#     p_ff = (diff_cscat_par_ff + diff_cscat_perp_ff)/(ksquared * 2 * cscat_total_ff)
#     p_par_ff = diff_cscat_par_ff/(ksquared * 2 * cscat_total_par_ff)
#     p_perp_ff = diff_cscat_perp_ff/(ksquared * 2 * cscat_total_perp_ff)

#     # with mie.diff_scat_intensity_complex_medium()
#     diff_cscat_par, diff_cscat_perp = \
#         model.differential_cross_section(m, x, angles, volume_fraction,
#                                          structure_type='glass',
#                                          form_type='sphere',
#                                          diameters=radius, wavelen=wavelen,
#                                          n_matrix=n_sample, k=k, distance=radius)
#     cscat_total_par = model._integrate_cross_section(diff_cscat_par,
#                                                       1.0/ksquared, angles)
#     cscat_total_perp = model._integrate_cross_section(diff_cscat_perp,
#                                                       1.0/ksquared, angles)
#     cscat_total = (cscat_total_par + cscat_total_perp)/2.0

#     p = (diff_cscat_par + diff_cscat_perp)/(ksquared * 2 * cscat_total)
#     p_par = diff_cscat_par/(ksquared * 2 * cscat_total_par)
#     p_perp = diff_cscat_perp/(ksquared * 2 * cscat_total_perp)

#     # test random values of the phase functions
#     assert_almost_equal(p_ff[3].magnitude, p[3].magnitude, decimal=15)
#     assert_almost_equal(p_par_ff[50].magnitude, p_par[50].magnitude, decimal=15)
#     assert_almost_equal(p_perp[83].magnitude, p_perp_ff[83].magnitude, decimal=15)

#     ### Same thing but with a binary and polydisperse mixture
#     ## Integrating at the surface of the particle
#     # with mie.calc_ang_scat() (this is how it's currently implemented in
#     # monte carlo)
#     radius2 = sc.Quantity('150.0 nm')
#     concentration = sc.Quantity(np.array([0.3, 0.7]), '')
#     pdi = sc.Quantity(np.array([0.1, 0.1]), '')
#     diameters = sc.Quantity(np.array([radius.magnitude, radius2.magnitude])*2,
#                             radius.units)

#     diff_cscat_par_ff, diff_cscat_perp_ff = \
#         model.differential_cross_section(m, x, angles, volume_fraction,
#                                          structure_type='polydisperse',
#                                          form_type='polydisperse',
#                                          diameters=diameters, pdi=pdi,
#                                          concentration=concentration,
#                                          wavelen=wavelen,
#                                          n_matrix=n_sample, k=None,
#                                          distance=diameters/2)
#     cscat_total_par_ff = model._integrate_cross_section(diff_cscat_par_ff,
#                                                       1.0/ksquared, angles)
#     cscat_total_perp_ff = model._integrate_cross_section(diff_cscat_perp_ff,
#                                                       1.0/ksquared, angles)
#     cscat_total_ff = (cscat_total_par_ff + cscat_total_perp_ff)/2.0

#     p_ff2 = (diff_cscat_par_ff + diff_cscat_perp_ff)/(ksquared * 2 * cscat_total_ff)
#     p_par_ff2 = diff_cscat_par_ff/(ksquared * 2 * cscat_total_par_ff)
#     p_perp_ff2 = diff_cscat_perp_ff/(ksquared * 2 * cscat_total_perp_ff)

#     # with mie.diff_scat_intensity_complex_medium()
#     diff_cscat_par, diff_cscat_perp = \
#         model.differential_cross_section(m, x, angles, volume_fraction,
#                                          structure_type='polydisperse',
#                                          form_type='polydisperse',
#                                          diameters=diameters, pdi=pdi,
#                                          concentration=concentration,
#                                          wavelen=wavelen,
#                                          n_matrix=n_sample, k=k,
#                                          distance=diameters/2)
#     cscat_total_par = model._integrate_cross_section(diff_cscat_par,
#                                                      1.0/ksquared, angles)
#     cscat_total_perp = model._integrate_cross_section(diff_cscat_perp,
#                                                       1.0/ksquared, angles)
#     cscat_total = (cscat_total_par + cscat_total_perp)/2.0

#     p2 = (diff_cscat_par + diff_cscat_perp)/(ksquared * 2 * cscat_total)
#     p_par2 = diff_cscat_par/(ksquared * 2 * cscat_total_par)
#     p_perp2 = diff_cscat_perp/(ksquared * 2 * cscat_total_perp)

#     # test random values of the phase functions
#     assert_almost_equal(p_ff2[3].magnitude, p2[3].magnitude, decimal=15)
#     assert_almost_equal(p_par_ff2[50].magnitude, p_par2[50].magnitude,
#                         decimal=15)
#     assert_almost_equal(p_perp2[83].magnitude, p_perp_ff2[83].magnitude,
#                         decimal=15)
