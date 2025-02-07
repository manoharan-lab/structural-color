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
Tests for the montecarlo model (in structcol/montecarlo.py)

.. moduleauthor:: Annie Stephenson <stephenson@g.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import structcol as sc
from structcol import montecarlo as mc
from structcol import refractive_index as ri
from structcol import event_distribution as ed
from structcol import detector as det
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less
import pytest

# Monte Carlo parameters
ntrajectories = 300
# number of scattering events in each trajectory
nevents = 30

# source/detector properties
wavelength = sc.Quantity(np.array(550.0),'nm')

# sample properties
particle_radius = sc.Quantity('140.0 nm')
volume_fraction = sc.Quantity(0.56, '')
thickness = sc.Quantity('10.0 um')
particle = 'ps'
matrix = 'air'
boundary = 'film'

# indices of refraction
#
# Refractive indices can be specified as pint quantities or called from the
# refractive_index module. n_matrix is the # space within sample. n_medium is
# outside the sample.
n_particle = ri.n('polystyrene', wavelength)
n_matrix = ri.n('vacuum', wavelength)
n_medium = ri.n('vacuum', wavelength)

# Calculate the effective refractive index of the sample
n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)

# Calculate the phase function and scattering and absorption coefficients from
# the single scattering model (this absorption coefficient is of the scatterer,
# not of an absorber added to the system)
p, mu_scat, mu_abs = mc.calc_scat(particle_radius, n_particle, n_sample,
                                  volume_fraction, wavelength)
lscat = 1/mu_scat.magnitude # microns

# set up a seeded random number generator that will give consistent results
# between numpy versions.
seed = 1
rng = np.random.RandomState([seed])

# Initialize the trajectories
r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                           boundary, rng=rng)
r0 = sc.Quantity(r0, 'um')
k0 = sc.Quantity(k0, '')
W0 = sc.Quantity(W0, '')

# Generate a matrix of all the randomly sampled angles first
sintheta, costheta, sinphi, cosphi, theta, _ = mc.sample_angles(nevents,
                                                                ntrajectories,
                                                                p, rng=rng)
sintheta = np.sin(theta)
costheta = np.cos(theta)

# Create step size distribution
step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

# Create trajectories object
trajectories = mc.Trajectory(r0, k0, W0)

# Run photons
trajectories.absorb(mu_abs, step)
trajectories.scatter(sintheta, costheta, sinphi, cosphi)
trajectories.move(step)

# following calculation should raise a warning that n_particle and n_matrix are
# not set
with pytest.warns(UserWarning):
    refl_indices, trans_indices,\
        inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
        trans_frac, refl_frac,\
        refl_fresnel,\
        trans_fresnel,\
        reflectance,\
        transmittance,\
        tir_refl_bool,_,_ = det.calc_refl_trans(trajectories, thickness,
                                                n_medium, n_sample, boundary,
                                                return_extra = True)

refl_events, trans_events = ed.calc_refl_trans_event(refl_per_traj,
                                                     inc_refl_per_traj,
                                                     trans_per_traj,
                                                     refl_indices,
                                                     trans_indices,
                                                     nevents)

def test_refl_events():
    '''
    Check that refl_events is consistent with reflectance
    '''

    # sum of refl_events should be less than reflectance because it doesn't
    # contain correction terms for fresnel (and stuck for cases where that
    # matters)
    assert_array_less(np.sum(refl_events), reflectance)

    # trajectories always propagate into the sample for first event, so none
    # can be reflected
    assert_equal(refl_events[1],0)

    # trajectories cannot be transmitted at interface before first scattering
    # event
    assert_equal(trans_events[0],0)

def test_fresnel_events():
    '''
    Check that fresnel corrections make sense
    '''
    refl_events_fresnel_avg = ed.calc_refl_event_fresnel_avg(refl_events,
                                                             refl_indices,
                                                             trans_indices,
                                                             refl_fresnel,
                                                             trans_fresnel,
                                                             refl_frac,
                                                             trans_frac,
                                                             nevents)

    # Below we do not use the seeded number generator because this check should
    # pass for any set of trajectories
    pdf_refl, pdf_trans = ed.calc_pdf_scat(refl_events, trans_events, nevents)
    refl_events_fresnel_samp = ed.calc_refl_event_fresnel_pdf(refl_events,
                                                              pdf_refl,
                                                              pdf_trans,
                                                              refl_indices,
                                                              trans_indices,
                                                              refl_fresnel,
                                                              trans_fresnel,
                                                              refl_frac,
                                                              trans_frac,
                                                              nevents)

    # check that average and sampling give same total
    assert_almost_equal(np.sum(refl_events_fresnel_avg),
                        np.sum(refl_events_fresnel_samp))

    # check that reflectance from monte carlo gives same as fresnel reflected
    # summed reflectance from event distribution
    # TODO these should be equal to more decimals. Need to look into this.
    assert_almost_equal(reflectance, np.sum(refl_events_fresnel_avg), decimal=1)

def test_tir_events():
    '''
    Check that totally internally reflected trajectories make sense
    '''
    tir_all,\
    tir_all_refl,\
    tir_single,\
    tir_single_refl,\
    tir_indices_single = ed.calc_tir(tir_refl_bool, refl_indices,
                                     trans_indices, inc_refl_per_traj,
                                     n_sample,
                                     n_medium,
                                     boundary,
                                     trajectories,
                                     thickness)

    # the reflected tir's should always be less than total tir's
    assert_array_less(np.sum(tir_single_refl), np.sum(tir_single))
    assert_array_less(np.sum(tir_all_refl), np.sum(tir_all))

    # test against the values produced by event_distribution_tutorial notebook.
    tir_all_expected = [0.0, 0.0, 0.08666105640938222, 0.0366642930962771,
                        0.013332470216828034, 0.0033331175542070084,
                        0.006666235108414017, 0.0033331175542070084, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    tir_all_refl_expected = [0.0, 0.0, 0.0, 0.009649742188910855,
                             0.009715329717097099, 0.004804704955320356,
                             0.009682737547457578, 0.0032416313926865526, 0.0,
                             0.0063546152067123045, 0.003235947834579924,
                             0.0032600221457399236, 0.005947867678853345, 0.0,
                             0.0032579112813920755, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0]
    tir_single_expected = [0.0, 0.0, 0.08666105640938222, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0]
    tir_single_refl_expected = [0.0, 0.0, 0.0, 0.009649742188910855,
                                0.006532432062663876, 0.0030791581154449806,
                                0.003171340650803997, 0.0032416313926865526,
                                0.0, 0.0063546152067123045,
                                0.003235947834579924, 0.0,
                                0.0031747545295436907, 0.0,
                                0.0032579112813920755, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0]
    tir_indices_single_expected = [0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                                   2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0,
                                   0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0,
                                   2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2,
                                   0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]

    assert_almost_equal(tir_all, tir_all_expected)
    assert_almost_equal(tir_all_refl, tir_all_refl_expected)
    assert_almost_equal(tir_single, tir_single_expected)
    assert_almost_equal(tir_single_refl, tir_single_refl_expected)
    assert_almost_equal(tir_indices_single, tir_indices_single_expected)


def test_event_distribution_mc():
    """
    Tests whether the event distribution is what we expect from a simulation on
    a film of particles. The parameters, setup, and expected values come from
    the event_distribution_tutorial notebook (the expected values shown here
    are not explicitly calculated in the notebook but can be printed out using
    "print(refl_events.tolist())").

    """

    # first test that reflectance matches
    R_expected = 0.17576865671711203

    assert_almost_equal(reflectance, R_expected)

    # now check event distribution
    refl_events_expected = [0.0194201213692633, 0.0, 0.00983360404656042,
                            0.032463296864638135, 0.024011706226611494,
                            0.011623218993711354, 0.01953268955941859,
                            0.0032561381937666694, 0.0, 0.01600022398968138,
                            0.0032513424623250536, 0.006549327786911493,
                            0.006060515167210846, 0.0, 0.003269875006633953,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    trans_events_expected = [0.0, 0.5720886942602855, 0.056077641883742126,
                             0.05237960106124679, 0.027987887423433493,
                             0.034046732490227306, 0.023688359158163582,
                             0.009551414681607365, 0.009476279307495616,
                             0.015453248545051548, 0.0031733245304951673, 0.0,
                             0.0031953271647305703, 0.006418805759965045, 0.0,
                             0.003212631980579724, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert_almost_equal(refl_events, refl_events_expected)
    assert_almost_equal(trans_events, trans_events_expected)

def test_fresnel_corrections_mc():
    """
    Tests whether the event distribution is what we expect from a simulation on
    a film of particles, using two types of corrections for fresnel-reflected
    trajectories. The parameters, setup, and expected values come from the
    event_distribution_tutorial notebook (the expected values shown here are
    not explicitly calculated in the notebook but can be printed out using
    "print(refl_events.tolist())").

    """

    # test method 1: adding fresnel reflected weights to average event
    refl_events_fresnel_avg = ed.calc_refl_event_fresnel_avg(refl_events,
                                                             refl_indices,
                                                             trans_indices,
                                                             refl_fresnel,
                                                             trans_fresnel,
                                                             refl_frac,
                                                             trans_frac,
                                                             nevents)

    refl_events_fresnel_avg_expected = [0.0194201213692633, 0.0,
                                        0.00983360404656042,
                                        0.04218533059192289,
                                        0.026768857060713573,
                                        0.014809811054872979,
                                        0.0209621984949072,
                                        0.005164001253333037,
                                        0.0024874251009888546,
                                        0.01639571792874288,
                                        0.004011465762255197,
                                        0.007716437950325448,
                                        0.006231582571022886,
                                        1.2520157199962823e-05,
                                        0.003343147437339172,
                                        0.00022648271123504603,
                                        1.3297969273040072e-05,
                                        7.499862158069969e-05,
                                        9.83107499096656e-05, 0.0,
                                        1.0292205317366047e-05, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0]

    assert_almost_equal(refl_events_fresnel_avg,
                        refl_events_fresnel_avg_expected)

    # test method 2: adding fresnel reflected weights to sampled events

    # first calculate PDF of spectra
    pdf_refl, pdf_trans = ed.calc_pdf_scat(refl_events, trans_events, nevents)


    refl_events_fresnel_samp = ed.calc_refl_event_fresnel_pdf(refl_events,
                                                              pdf_refl,
                                                              pdf_trans,
                                                              refl_indices,
                                                              trans_indices,
                                                              refl_fresnel,
                                                              trans_fresnel,
                                                              refl_frac,
                                                              trans_frac,
                                                              nevents, rng=rng)

    refl_events_fresnel_samp_expected = [0.0194201213692633, 0.0,
                                         0.00983360404656042,
                                         0.035220447698740213,
                                         0.03376072744956238,
                                         0.013052727929199964,
                                         0.01953268955941859,
                                         0.006720090986214851,
                                         0.0031454255311411253,
                                         0.01632974346849313,
                                         0.005300315356252918,
                                         0.006644602851852114,
                                         0.006168604447539358,
                                         7.327243070521883e-05,
                                         0.0034015664068135264,
                                         7.57923388714192e-05, 0.0,
                                         0.0009022695893273616,
                                         1.9031006724634793e-05, 0.0,
                                         1.0292205317366047e-05, 0.0,
                                         9.83107499096656e-05, 0.0,
                                         5.5967614856064894e-05, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0]

    assert_almost_equal(refl_events_fresnel_samp,
                        refl_events_fresnel_samp_expected)

def test_event_distribution_wavelength_mc():
    """
    Test calculation of reflectance and event distribution at several
    wavelengths. Based on the parameters, setup, and calculation in
    event_distribution_tutorial.ipynb.

    """
    seed = 1
    rng = np.random.RandomState([seed])
    ntrajectories = 300
    nevents = 20

    wavelengths = sc.Quantity(np.arange(400,810,20),'nm')

    particle_radius = sc.Quantity('140 nm')
    volume_fraction = sc.Quantity(0.56, '')
    thickness = sc.Quantity('10 um')
    boundary = 'film'

    # indices of refraction
    n_particle = ri.n('polystyrene', wavelengths)
    n_matrix = ri.n('vacuum', wavelengths)
    n_medium = ri.n('vacuum', wavelengths)

    # initialize arrays for quantities we want to look at later
    refl_events = np.zeros((wavelengths.size, 2*nevents+1))
    reflectance = np.zeros(wavelengths.size)
    p = sc.Quantity(np.zeros((wavelengths.size, 200)),'')
    lscat = np.zeros(wavelengths.size)
    tir_single_events = np.zeros((wavelengths.size, 2*nevents+1))
    tir_single_refl_events = np.zeros((wavelengths.size, 2*nevents+1))
    tir_all_events = np.zeros((wavelengths.size, 2*nevents+1))
    tir_all_refl_events = np.zeros((wavelengths.size, 2*nevents+1))
    tir_indices_single_events = np.zeros((wavelengths.size, ntrajectories))

    # run monte carlo, reflectance, and event_distribution
    for i in range(wavelengths.size):
        n_sample = ri.n_eff(n_particle[i], n_matrix[i], volume_fraction)

        p[i,:], mu_scat, mu_abs = mc.calc_scat(particle_radius, n_particle[i],
                                               n_sample, volume_fraction,
                                               wavelengths[i])
        lscat[i] = 1/mu_scat.magnitude # microns

        r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium[i],
                                   n_sample, boundary, rng=rng)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')

        ######################################################################
        # Generate a matrix of all the randomly sampled angles first
        sintheta, costheta, sinphi, cosphi, theta, _ = \
            mc.sample_angles(nevents, ntrajectories, p[i,:], rng=rng)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        # Create step size distribution
        step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)

        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

        # Run photons
        trajectories.absorb(mu_abs, step)
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        ################### Calculate reflection and transmission
        with pytest.warns(UserWarning):
            refl_indices, trans_indices,\
                inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
                trans_frac, refl_frac,\
                refl_fresnel, trans_fresnel,\
                reflectance[i], transmittance,\
                tir_refl_bool,_,_ = det.calc_refl_trans(trajectories,
                                                        thickness, n_medium[i],
                                                        n_sample, boundary,
                                                        return_extra = True)

        ################### Calculate event distributions ####################

        refl_events[i,:], trans_events = \
            ed.calc_refl_trans_event(refl_per_traj, inc_refl_per_traj,
                                     trans_per_traj, refl_indices,
                                     trans_indices, nevents)

        # total internal reflection
        tir_all_events[i,:],\
            tir_all_refl_events[i,:],\
            tir_single_events[i,:],\
            tir_single_refl_events[i,:],\
            tir_indices_single_events[i,:] = ed.calc_tir(tir_refl_bool,
                                                         refl_indices,
                                                         trans_indices,
                                                         inc_refl_per_traj,
                                                         n_sample, n_medium[i],
                                                         boundary,
                                                         trajectories,
                                                         thickness)

    R_expected = [0.5000928560480027, 0.42000180405009635,
                  0.414386300857287, 0.3884852956834383,
                  0.30702427627889695, 0.2871769919054884,
                  0.2804997166223931, 0.22422288696504963,
                  0.17653697063894183, 0.21977785300242964,
                  0.22485156570293352, 0.23413683449169456,
                  0.25129796208333255, 0.22516060545875238,
                  0.13951819338808738, 0.15463252035445552,
                  0.1376704975287159, 0.12369895777400604,
                  0.07280222946794919, 0.10194549186501563,
                  0.07024739927354683]

    assert_almost_equal(reflectance, R_expected)

    tir_sum_expected = [0.006135879360055146, 0.00317723529150989, 0.0,
                        0.028587716359315972, 0.01573849613374884,
                        0.012232289602960026, 0.043912873755055475,
                        0.0679964129250502, 0.05015962511920376,
                        0.038441796002235795, 0.031684189009536345,
                        0.009106818105021305, 0.012560792306701905,
                        0.03162472453635139, 0.0, 0.019170718152488872,
                        0.015323365692594773, 0.008300530966422295,
                        0.006963695412474454, 0.009301718632232548,
                        0.006497908757697398]

    assert_almost_equal(np.sum(tir_single_refl_events, axis=1),
                        tir_sum_expected)

def test_event_distribution_angle_mc():
    """
    As in test_event_distribution_wavelength_mc(), but tests reflectance and
    event distribution as a function of angle.

    """
    seed = 1
    rng = np.random.RandomState([seed])

    theta_range = sc.Quantity(np.arange(125., 150, 2),'degrees')

    refl_events = np.zeros((theta_range.size, 2*nevents+1))
    refl_events_fresnel_samp = np.zeros((theta_range.size, 2*nevents+1))
    refl_events_fresnel_avg = np.zeros((theta_range.size, 2*nevents+1))
    reflectance = np.zeros(theta_range.size)

    p, mu_scat, mu_abs = mc.calc_scat(particle_radius, n_particle, n_sample,
                                      volume_fraction, wavelength)
    lscat = 1/mu_scat.magnitude

    # Initialize the trajectories
    r0, k0, W0 = mc.initialize(nevents, ntrajectories, n_medium, n_sample,
                               boundary, rng=rng)
    r0 = sc.Quantity(r0, 'um')
    k0 = sc.Quantity(k0, '')
    W0 = sc.Quantity(W0, '')

    # Create step size distribution
    step = mc.sample_step(nevents, ntrajectories, mu_scat, rng=rng)


    for j in range(theta_range.size):
        # Generate a matrix of all the randomly sampled angles first
        _, _, sinphi, cosphi, _, _ = mc.sample_angles(nevents, ntrajectories,
                                                      p, rng=rng)
        theta = (np.ones((nevents,ntrajectories))
                 * theta_range[j].to('rad').magnitude)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

        # Run photons
        trajectories.absorb(mu_abs, step)
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)
        trajectories.move(step)

        ################### Calculate reflection and transmition
        with pytest.warns(UserWarning):
            refl_indices, trans_indices,\
                inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
                trans_frac, refl_frac,\
                refl_fresnel, trans_fresnel,\
                reflectance[j], _,_,_,_= det.calc_refl_trans(trajectories,
                                                             thickness,
                                                             n_medium,
                                                             n_sample,
                                                             boundary,
                                                             return_extra =
                                                             True)


        ################### Calculate event distribution #####################

        refl_events[j,:], trans_events = \
            ed.calc_refl_trans_event(refl_per_traj, inc_refl_per_traj,
                                     trans_per_traj,
                                     refl_indices,
                                     trans_indices,
                                     nevents)

    # test only the reflectance after a single scattering event (as a function
    # of theta)
    single_scat_exp = [0.0, 0.0, 0.0, 0.09827084454821874, 0.2513208917614578,
                       0.27192699072805343, 0.2803295391319316,
                       0.28492644439070053, 0.2910865640807857,
                       0.29276003413051715, 0.29712732645187817,
                       0.29792258839702507, 0.2983565346564514]

    assert_almost_equal(refl_events[:,2], single_scat_exp)
