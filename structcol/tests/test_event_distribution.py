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
from structcol import event_distribution as ed
from structcol import detector as det
import numpy as np
import xarray as xr
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_less, assert_allclose)

class TestEventDistribution():
    """Tests for the functions in event_distribution.py. This class sets up a
    simulation on a film of particles and runs it.

    """
    # Monte Carlo parameters
    ntrajectories = 300
    # number of scattering events in each trajectory
    nevents = 30

    # source/detector properties
    wavelength = sc.Quantity(np.array(550.0), "nm")

    # sample properties
    particle_radius = sc.Quantity("140.0 nm")
    volume_fraction = 0.56
    thickness = sc.Quantity("10.0 um")
    boundary = "film"

    # indices of refraction
    index_particle = sc.index.polystyrene
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelength)

    # Calculate the phase function and scattering and absorption coefficients
    # from the single scattering model (this absorption coefficient is of the
    # scatterer, not of an absorber added to the system)
    particle = sc.Sphere(index_particle, particle_radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix,
                                 index_medium)

    # set up a seeded random number generator that will give consistent results
    # between numpy versions.
    seed = 1
    rng = np.random.RandomState([seed])

    # Initialize and run the simulation
    sim = mc.Simulation(model, wavelength, nevents, ntrajectories, boundary,
                        rng=rng)
    sim.run()

    trajectories = sim.traj

    # Calculate the effective refractive index of the sample
    n_sample = model.index_external(wavelength)

    refl_indices, trans_indices, stuck_indices, tir_indices,\
        inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
        trans_frac, refl_frac,\
        refl_fresnel,\
        trans_fresnel,\
        reflectance,\
        transmittance,\
        _,_ = det.calc_refl_trans(sim, thickness, return_extra = True)

    refl_events, trans_events = ed.calc_refl_trans_event(refl_per_traj,
                                                         inc_refl_per_traj,
                                                         trans_per_traj,
                                                         refl_indices,
                                                         trans_indices,
                                                         nevents)

    def test_refl_events(self):
        """
        Tests whether the event distribution is what we expect from a
        simulation on a film of particles.
        """
        # sum of refl_events should be less than reflectance because it doesn't
        # contain correction terms for fresnel (and stuck for cases where that
        # matters)
        assert_array_less(self.refl_events.sum("event"), self.reflectance)

        # trajectories always propagate into the sample for first event, so
        # none can be reflected
        refl_1 = self.refl_events.sel(event=1)
        assert_equal(refl_1, xr.zeros_like(refl_1))

        # trajectories cannot be transmitted at interface before first
        # scattering event
        trans_0 = self.trans_events.sel(event=0)
        assert_equal(trans_0, xr.zeros_like(trans_0))

        # test that reflectance matches
        R_expected = 0.17576865671711203

        assert_almost_equal(self.reflectance, R_expected)

        # now check event distribution
        refl_events_expected = [0.0194201213692633, 0.0, 0.00983360404656042,
                                0.032463296864638135, 0.024011706226611494,
                                0.011623218993711354, 0.01953268955941859,
                                0.0032561381937666694, 0.0, 0.01600022398968138,
                                0.0032513424623250536, 0.006549327786911493,
                                0.006060515167210846, 0.0, 0.003269875006633953,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0]
        trans_events_expected = [0.0, 0.5720886942602855, 0.056077641883742126,
                                 0.05237960106124679, 0.027987887423433493,
                                 0.034046732490227306, 0.023688359158163582,
                                 0.009551414681607365, 0.009476279307495616,
                                 0.015453248545051548, 0.0031733245304951673,
                                 0.0, 0.0031953271647305703,
                                 0.006418805759965045, 0.0,
                                 0.003212631980579724, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0]

        assert_allclose(self.refl_events.to_numpy().squeeze(),
                            refl_events_expected)
        assert_allclose(self.trans_events.to_numpy().squeeze(),
                            trans_events_expected)

    def test_fresnel_events(self):
        '''
        Tests whether the event distribution is what we expect from a
        simulation on a film of particles, using two types of corrections for
        fresnel-reflected trajectories.
        '''
        # test method 1: adding fresnel reflected weights to average event
        refl_events_fresnel_avg = \
            ed.calc_refl_event_fresnel_avg(self.refl_events, self.refl_indices,
                                           self.trans_indices,
                                           self.refl_fresnel,
                                           self.trans_fresnel,
                                           self.refl_frac, self.trans_frac,
                                           self.nevents)

        # values from before refactoring the calc_refl_event_fresnel_avg()
        # function to use xarray
        refl_events_fresnel_avg_expected = np.array([
            1.942012136926330e-02, 0.000000000000000e+00,
            9.833604046560420e-03, 4.218533059192289e-02,
            2.676885706071358e-02, 1.480981105487297e-02,
            2.096219849490720e-02, 5.164001253333037e-03,
            2.487425100988859e-03, 1.639571792874288e-02,
            4.011465762255197e-03, 7.716437950325448e-03,
            6.231582571022884e-03, 1.252015719996283e-05,
            3.343147437339172e-03, 2.264827112350465e-04,
            1.329796927304007e-05, 7.499862158069966e-05,
            9.831074990966599e-05, 0.000000000000000e+00,
            1.029220531736605e-05, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00, 0.000000000000000e+00,
            0.000000000000000e+00])

        assert_allclose(refl_events_fresnel_avg.to_numpy().squeeze(),
                        refl_events_fresnel_avg_expected,
                        rtol=1e-14)

        # test method 2: adding fresnel reflected weights to sampled events
        pdf_refl, pdf_trans = ed.calc_pdf_scat(self.refl_events,
                                               self.trans_events, self.nevents)
        refl_events_fresnel_samp = \
            ed.calc_refl_event_fresnel_pdf(self.refl_events,
                                           pdf_refl, pdf_trans,
                                           self.refl_indices,
                                           self.trans_indices,
                                           self.refl_fresnel,
                                           self.trans_fresnel,
                                           self.refl_frac,
                                           self.trans_frac,
                                           self.nevents, rng=self.rng)

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
                                             5.5967614856064894e-05, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # check that values match previous results
        assert_almost_equal(refl_events_fresnel_samp.to_numpy().squeeze(),
                            refl_events_fresnel_samp_expected)

        # next two tests should pass for any random number seed
        # check that average and sampling give same total
        assert_almost_equal(np.sum(refl_events_fresnel_avg),
                            np.sum(refl_events_fresnel_samp))

        # check that reflectance from monte carlo gives same as fresnel
        # reflected summed reflectance from event distribution
        # TODO these should be equal to more decimals. Need to look into this.
        assert_almost_equal(self.reflectance,
                            np.sum(refl_events_fresnel_avg), decimal=1)


    def test_tir_events(self):
        """Check that totally internally reflected trajectories match what we
        expect from a simulation on a film of particles.

        """
        tir_all,\
        tir_all_refl,\
        tir_single,\
        tir_single_refl,\
        tir_indices_single = ed.calc_tir(self.tir_indices, self.refl_indices,
                                         self.trans_indices,
                                         self.inc_refl_per_traj,
                                         self.n_sample, self.n_medium,
                                         self.boundary, self.trajectories,
                                         self.thickness)

        # the reflected tir's should always be less than total tir's
        assert_array_less(np.sum(tir_single_refl), np.sum(tir_single))
        assert_array_less(np.sum(tir_all_refl), np.sum(tir_all))

        # test against the values produced by event_distribution_tutorial
        # notebook.
        tir_all_expected = [0.0, 0.0, 0.08666105640938222, 0.0366642930962771,
                            0.013332470216828034, 0.0033331175542070084,
                            0.006666235108414017, 0.0033331175542070084, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0]
        tir_all_refl_expected = [0.0, 0.0, 0.0, 0.009649742188910855,
                                 0.009715329717097099, 0.004804704955320356,
                                 0.009682737547457578, 0.0032416313926865526,
                                 0.0, 0.0063546152067123045,
                                 0.003235947834579924, 0.0032600221457399236,
                                 0.005947867678853345, 0.0,
                                 0.0032579112813920755, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tir_single_expected = [0.0, 0.0, 0.08666105640938222, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0]
        tir_single_refl_expected = [0.0, 0.0, 0.0, 0.009649742188910855,
                                    0.006532432062663876, 0.0030791581154449806,
                                    0.003171340650803997, 0.0032416313926865526,
                                    0.0, 0.0063546152067123045,
                                    0.003235947834579924, 0.0,
                                    0.0031747545295436907, 0.0,
                                    0.0032579112813920755, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tir_indices_single_expected = [0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0,
                                       0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0,
                                       0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
                                       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2,
                                       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                                       0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                       0]

        assert_allclose(tir_all.to_numpy().squeeze(), tir_all_expected)
        assert_allclose(tir_all_refl.to_numpy().squeeze(),
                        tir_all_refl_expected)
        assert_allclose(tir_single.to_numpy().squeeze(), tir_single_expected)
        assert_allclose(tir_single_refl.to_numpy().squeeze(),
                        tir_single_refl_expected)
        assert_allclose(tir_indices_single.to_numpy().squeeze(),
                        tir_indices_single_expected)

class TestEventDistributionWavelength():
    """Tests that MC simulations at different wavelengths return expected
    results. Based on the parameters, setup, and calculation in
    event_distribution_tutorial.ipynb.

    """
    ntrajectories = 300
    nevents = 20

    wavelengths = sc.Quantity(np.arange(400, 810, 20), "nm")

    particle_radius = sc.Quantity("140 nm")
    volume_fraction = 0.56
    thickness = sc.Quantity("10 um")
    boundary = "film"

    # indices of refraction
    index_particle = sc.index.polystyrene
    index_matrix = sc.index.vacuum
    index_medium = sc.index.vacuum
    n_medium = index_medium(wavelengths)

    particle = sc.Sphere(index_particle, particle_radius)
    model = sc.model.HardSpheres(particle, volume_fraction, index_matrix,
                                 index_medium)

    def test_event_distribution_wavelength_mc(self):
        """Test calculation of reflectance and event distribution at several
        wavelengths.

        """
        seed = 1
        rng = np.random.RandomState([seed])

        # initialize lists for quantities we want to look at later
        reflectance_list = []
        refl_events_list = []
        tir_single_refl_events_list = []

        n_sample_eff = self.model.index_external(self.wavelengths)

        # run monte carlo, reflectance, and event_distribution
        for i in range(self.wavelengths.size):
            n_sample = n_sample_eff.isel(wavelength=i, drop=True)

            sim = mc.Simulation(self.model, self.wavelengths[i], self.nevents,
                                self.ntrajectories, self.boundary, rng=rng)
            sim.run()

            ################### Calculate reflection and transmission
            trajectories = sim.traj
            refl_indices, trans_indices, stuck_indices, tir_indices,\
                inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
                trans_frac, refl_frac,\
                refl_fresnel, trans_fresnel,\
                reflectance, transmittance,\
                _,_ = det.calc_refl_trans(sim, self.thickness,
                                          return_extra = True)

            reflectance_list.append(reflectance)

            ################### Calculate event distributions ####################

            refl_events, trans_events = \
                ed.calc_refl_trans_event(refl_per_traj, inc_refl_per_traj,
                                         trans_per_traj, refl_indices,
                                         trans_indices, self.nevents)
            refl_events_list.append(refl_events)

            # total internal reflection
            tir_all_events,\
                tir_all_refl_events,\
                tir_single_events,\
                tir_single_refl_events,\
                tir_indices_single_events = ed.calc_tir(tir_indices,
                                                        refl_indices,
                                                        trans_indices,
                                                        inc_refl_per_traj,
                                                        n_sample,
                                                        self.n_medium[i],
                                                        self.boundary,
                                                        trajectories,
                                                        self.thickness)

            tir_single_refl_events_list.append(tir_single_refl_events)

        reflectance = xr.concat(reflectance_list, "wavelength", join="outer")
        tir_single_refl_events = xr.concat(tir_single_refl_events_list,
                                           "wavelength", join="outer")

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

        assert_allclose(reflectance.to_numpy().squeeze(), R_expected)

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

        assert_almost_equal((tir_single_refl_events.sum("event"))
                            .to_numpy().squeeze(),
                            tir_sum_expected)

    def test_event_distribution_angle_mc(self):
        """As in test_event_distribution_wavelength_mc(), but tests
        reflectance and event distribution as a function of angle.

        """
        seed = 1
        rng = np.random.RandomState([seed])

        nevents = 30
        wavelength = sc.Quantity(np.array(550.0), "nm")

        theta_range = sc.Quantity(np.arange(125., 150, 2), "degrees")

        refl_events_list = []
        reflectance_list = []

        # Initialize the simulation
        sim = mc.Simulation(self.model, wavelength, nevents,
                            self.ntrajectories, self.boundary, rng=rng)

        # Create step size distribution
        step = sim.sample_step()

        for j in range(theta_range.size):
            # Generate a matrix of all the randomly sampled angles first
            angles = sim.sample_angles()

            # need nevents-1 because the first event doesn't involve a change in
            # direction.
            theta = (np.ones((nevents-1, self.ntrajectories))
                     * theta_range[j].to('rad').magnitude)

            # broadcast to correct coordinates
            sintheta = (xr.ones_like(angles["sinphi"])
                        * xr.DataArray(np.sin(theta),
                                       dims=["event", "trajectory"]))
            costheta = (xr.ones_like(angles["sinphi"])
                        * xr.DataArray(np.cos(theta),
                                       dims=["event", "trajectory"]))
            angles["sintheta"] = sintheta
            angles["costheta"] = costheta

            # reset sim to initial conditions
            sim.reset()

            # Run photons
            sim.absorb(step)
            sim.scatter(angles)
            sim.move(step)

            ################### Calculate reflection and transmition
            refl_indices, trans_indices, stuck_indices, tir_indices,\
                inc_refl_per_traj,_,_, refl_per_traj, trans_per_traj,\
                trans_frac, refl_frac,\
                refl_fresnel, trans_fresnel,\
                reflectance, _,_,_= det.calc_refl_trans(sim, self.thickness,
                                                        return_extra = True)

            reflectance_list.append(reflectance)

            ################### Calculate event distribution #####################

            refl_events, trans_events = \
                ed.calc_refl_trans_event(refl_per_traj, inc_refl_per_traj,
                                         trans_per_traj,
                                         refl_indices,
                                         trans_indices,
                                         nevents)

            refl_events_list.append(refl_events)

        refl_events = xr.concat(refl_events_list, "wavelength", join="outer")

        # test only the reflectance after a single scattering event (as a function
        # of theta)
        single_scat_exp = [0.0, 0.0, 0.0, 0.09827084454821874, 0.2513208917614578,
                           0.27192699072805343, 0.2803295391319316,
                           0.28492644439070053, 0.2910865640807857,
                           0.29276003413051715, 0.29712732645187817,
                           0.29792258839702507, 0.2983565346564514]

        assert_almost_equal(refl_events.sel(event=2).to_numpy().squeeze(),
                            single_scat_exp)
