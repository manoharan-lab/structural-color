# Copyright 2016, Vinothan N. Manoharan
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
Tests for the single-scattering model (in structcol/model.py)

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, index_ratio, size_parameter, np, mie, model
from .. import refractive_index as ri
from .main import Spheres, Film, Source, DetectorSingleScat
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError
import pytest

def test_fresnel():
    # test the fresnel reflection and transmission coefficients
    n1 = Quantity(1.00, '')
    n2 = Quantity(1.5, '')

    # quantities calculated from
    # http://www.calctool.org/CALC/phys/optics/reflec_refrac
    rpar, rperp = model.fresnel_reflection(n1, n2, Quantity('0 deg'))
    assert_almost_equal(rpar, 0.04)
    assert_almost_equal(rperp, 0.04)
    rpar, rperp = model.fresnel_reflection(n1, n2, Quantity('45 deg'))
    assert_almost_equal(rpar, 0.00846646)
    assert_almost_equal(rperp, 0.0920134)

    # test total internal reflection
    rpar, rperp = model.fresnel_reflection(n2, n1, Quantity('45 deg'))
    assert_equal(rpar, 1.0)
    assert_equal(rperp, 1.0)

    # test no total internal reflection (just below critical angle)
    rpar, rperp = model.fresnel_reflection(n2, n1, Quantity('41.810 deg'))
    assert_almost_equal(rpar, 0.972175, decimal=6)
    assert_almost_equal(rperp, 0.987536, decimal=6)

    # test vectorized computation
    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # check for value error
    assert_raises(ValueError, model.fresnel_reflection, n2, n1, angles)
    angles = Quantity(np.linspace(0, 90., 10), 'deg')
    rpar, rperp = model.fresnel_reflection(n2, n1, angles)
    rpar_std = np.array([0.04, 0.0362780, 0.0243938, 0.00460754, 0.100064, 1.0,
                         1.0, 1.0, 1.0, 1])
    rperp_std = np.array([0.04, 0.0438879, 0.0590632, 0.105773, 0.390518, 1.0,
                         1.0, 1.0, 1.0, 1.0])
    assert_array_almost_equal(rpar, rpar_std)
    assert_array_almost_equal(rperp, rperp_std)

    # test transmission
    tpar, tperp = model.fresnel_transmission(n2, n1, angles)
    tpar_std = 1.0-rpar_std
    tperp_std = 1.0-rperp_std
    assert_array_almost_equal(tpar, tpar_std)
    assert_array_almost_equal(tperp, tperp_std)

def test_theta_refraction():
    # test that the detection angles theta are refracted correctly at the 
    # medium-sample interface. When n_sample < n_medium, the scattered angles 
    # in the reflection hemisphere (90-180 deg) are refracted at the interface
    # into a smaller range of angles (>90-180 deg). This test checks that the 
    # the reflectance is close to 0 when the angles between theta_min and 
    # theta_max are outside the range of refracted scattered angles. 
    incident_angle = Quantity('0 deg')   
    wavelength = Quantity(500, 'nm')
    radius = Quantity('100 nm')   
    volume_fraction = Quantity(0.5, '')
    n_particle = Quantity(1.0, '')
    n_matrix = Quantity(1.0, '')
    n_medium = Quantity(2.0, '')
    theta_min = Quantity(np.pi/2,'deg')
    
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness=np.inf, structure=None)
    source = Source(wavelength, polarization=None, incidence_angle=0)
    
    # set theta_max to be slightly smaller than the theta corresponding to 
    # total internal reflection (calculated manually to be 2.61799388)
    theta_max = Quantity(2.617,'deg')  
    detector = DetectorSingleScat(theta_min=theta_min, theta_max=theta_max)
    refl1, _, _, _, _ = model.reflection(system, source, detector)

    # try a different range of thetas (but keeping theta_max < total internal
    # reflection angle)
    theta_max = Quantity(2.,'deg')  
    detector = DetectorSingleScat(theta_min=theta_min, theta_max=theta_max)
    refl2, _, _, _, _ = model.reflection(system, source, detector)

    
    # the reflection should be zero plus the fresnel reflection term    
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
    r_fresnel = model.fresnel_reflection(n_medium, n_sample, incident_angle)
    r_fresnel_avg = (r_fresnel[0] + r_fresnel[1])/2
    
    assert_almost_equal(refl1, r_fresnel_avg)
    assert_almost_equal(refl2, r_fresnel_avg)
    assert_almost_equal(refl1, refl2)

def test_differential_cross_section():
    # Test that the differential cross sections for non-core-shell particles 
    # and core-shells are the same at low volume fractions, assuming that the 
    # particle diameter of the non-core-shells is the same as the core 
    # diameter in the core-shells
    
    #n_sample = Quantity(1.5, '')
    n_matrix = Quantity(1.0, '')
    wavelen = Quantity('500 nm')
    theta = Quantity(np.linspace(np.pi/2, np.pi, 200), 'rad')
    
    # Differential cross section for non-core-shells
    radius = Quantity('100 nm')    
    n_particle = Quantity(1.5, '')
    volume_fraction = Quantity(0.0001, '')              # IS VF TOO LOW?
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)  
    diff = model.differential_cross_section(m, x, theta, volume_fraction)
    
    # Differential cross section for core-shells. Core is equal to 
    # non-core-shell particle, and shell is made of vacuum
    radius_cs = Quantity(np.array([100, 110]), 'nm')  
    n_particle_cs = Quantity(np.array([1.5, 1.0]), '')
    
    volume_fraction_shell = volume_fraction * (radius_cs[1]**3 / radius_cs[0]**3-1)
    volume_fraction_cs = Quantity(np.array([volume_fraction.magnitude, 
                                            volume_fraction_shell.magnitude]), '')
    
    n_sample_cs = ri.n_eff(n_particle_cs, n_matrix, volume_fraction_cs)
    m_cs = index_ratio(n_particle_cs, n_sample_cs).flatten()
    x_cs = size_parameter(wavelen, n_sample_cs, radius_cs).flatten() 
    diff_cs = model.differential_cross_section(m_cs, x_cs, theta, 
                                               np.sum(volume_fraction_cs))

    assert_array_almost_equal(diff, diff_cs, decimal=5)
    
def test_reflection_core_shell():
    # Test reflection, anisotropy factor, and transport length calculations to
    # make sure the values for refl, g, and lstar remain the same after adding
    # core-shell capability into the model
    wavelength = Quantity(500, 'nm')
    thickness = Quantity(15, 'um')
    
    # Non core-shell particles with Maxwell-Garnett effective index
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120 nm')
    n_particle = Quantity(1.5, '')
    n_matrix = Quantity(1.0, '')
    n_medium = n_matrix

    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    source = Source(wavelength, polarization=None, incidence_angle=0)
    detector = DetectorSingleScat()
    refl1, _, _, g1, lstar1 = model.reflection(system, source, detector, 
                                         small_angle=Quantity('5 deg'),
                                         maxwell_garnett=True)
    

    # Non core-shell particles with Bruggeman effective index
    volume_fraction2 = Quantity(0.00001, '')
    species = Spheres(n_particle, radius, volume_fraction2, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    refl2, _, _, g2, lstar2 = model.reflection(system, source, detector, 
                                         small_angle=Quantity('5 deg'),
                                         maxwell_garnett=False)

        
    # Core-shell particles of core diameter equal to non core shell particles, 
    # and shell index of air. With Bruggeman effective index
    n_particle3 = Quantity(np.array([1.5, 1.0]), '')
    radius3 = Quantity(np.array([120, 130]), 'nm')
    volume_fraction3 = volume_fraction2 * (radius3[1]**3 / radius3[0]**3)
    
    species = Spheres(n_particle3, radius3, volume_fraction3, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    refl3, _, _, g3, lstar3 = model.reflection(system, source, detector, 
                                               small_angle=Quantity('5 deg'),
                                               maxwell_garnett=False)
                                         
    
    # Outputs for refl, g, and lstar before adding core-shell capability
    refl = Quantity(0.20772170840902376, '')
    g = Quantity(-0.18931942267032678, '')
    lstar = Quantity(10810.088573316663, 'nm')
    
    
    # Compare old outputs (before adding core-shell capability) and new outputs
    # for a non-core-shell using Maxwell-Garnett
    assert_array_almost_equal(refl, refl1)
    assert_array_almost_equal(g, g1) 
    assert_array_almost_equal(lstar, lstar1)

    # Compare a non-core-shell and a core-shell with shell index of air using
    # Bruggeman
    assert_array_almost_equal(refl2, refl3)
    assert_array_almost_equal(g2, g3, decimal=5)
    assert_array_almost_equal(lstar2.to('mm'), lstar3.to('mm'), decimal=4)
    
    
    # Test that the reflectance is the same for a core-shell that absorbs (with
    # the same refractive indices for all layers) and a non-core-shell that 
    # absorbs with the same index
    
    # Absorbing non-core-shell
    radius4 = Quantity('120 nm')
    n_particle4 = Quantity(1.5+0.001j, '')
    
    species = Spheres(n_particle4, radius4, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    refl4 = model.reflection(system, source, detector, 
                             small_angle=Quantity('5 deg'),
                             maxwell_garnett=False)[0]

    
    # Absorbing core-shell
    n_particle5 = Quantity(np.array([1.5+0.001j, 1.5+0.001j]), '')
    radius5 = Quantity(np.array([110, 120]), 'nm')
        
    species = Spheres(n_particle5, radius5, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness, structure='glass')
    refl5 = model.reflection(system, source, detector, 
                             small_angle=Quantity('5 deg'),
                             maxwell_garnett=False)[0]
    
    assert_array_almost_equal(refl4, refl5, decimal=3)
    
    # Same as previous test but with absorbing matrix
    # Non-core-shell
    radius6 = Quantity('120 nm')
    n_particle6 = Quantity(1.5+0.001j, '')
    n_matrix6 = Quantity(1.0+0.001j, '')
    
    species = Spheres(n_particle6, radius6, volume_fraction, pdi=0)
    system = Film(species, n_matrix6, n_medium, thickness, structure='glass')
    refl6 = model.reflection(system, source, detector, 
                             small_angle=Quantity('5 deg'),
                             maxwell_garnett=False)[0]

    
    # Core-shell
    n_particle7 = Quantity(np.array([1.5+0.001j, 1.5+0.001j]), '')
    radius7 = Quantity(np.array([110, 120]), 'nm')
    n_matrix7 = Quantity(1.0+0.001j, '')   
    
    species = Spheres(n_particle7, radius7, volume_fraction, pdi=0)
    system = Film(species, n_matrix7, n_medium, thickness, structure='glass')
    refl7 = model.reflection(system, source, detector, 
                             small_angle=Quantity('5 deg'),
                             maxwell_garnett=False)[0]
    
    assert_array_almost_equal(refl6, refl7, decimal=3)


def test_reflection_absorbing_particle():
    # test that the reflections with a real n_particle and with a complex
    # n_particle with a 0 imaginary component are the same 
    wavelength = Quantity(500, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120 nm')
    n_matrix = Quantity(1.0, '')
    n_medium = n_matrix
    n_particle_real = Quantity(1.5, '')
    n_particle_imag = Quantity(1.5 + 0j, '')

    source = Source(wavelength, polarization=None, incidence_angle=0)
    detector = DetectorSingleScat()
    
    # With Maxwell-Garnett
    species_real = Spheres(n_particle_real, radius, volume_fraction, pdi=0)
    system_real = Film(species_real, n_matrix, n_medium, thickness=np.inf, structure='glass')
    refl_mg1, _, _, g_mg1, lstar_mg1 = model.reflection(system_real, source, detector,
                                                        maxwell_garnett=True)
                             
    species_imag = Spheres(n_particle_imag, radius, volume_fraction, pdi=0)
    system_imag = Film(species_imag, n_matrix, n_medium, thickness=np.inf, structure='glass')
    refl_mg2, _, _, g_mg2, lstar_mg2 = model.reflection(system_imag, source, detector,
                                                        maxwell_garnett=True)                         
  
    assert_array_almost_equal(refl_mg1, refl_mg2)
    assert_array_almost_equal(g_mg1, g_mg2)
    assert_array_almost_equal(lstar_mg1, lstar_mg2)
    
    # Outputs before refactoring structcol
    refl_mg1_before = 0.2963964709617333
    refl_mg2_before = 0.29639647096173255
    g_mg1_before = -0.18774057969370997
    g_mg2_before = -0.18774057969370903
    lstar_mg1_before = 10810.069633192961
    lstar_mg2_before = 10810.069633193001
    
    assert_equal(refl_mg1_before, refl_mg1)
    assert_equal(refl_mg2_before, refl_mg2)
    assert_equal(g_mg1_before, g_mg1)
    assert_equal(g_mg2_before, g_mg2)
    assert_equal(lstar_mg1_before, lstar_mg1.magnitude)
    assert_equal(lstar_mg2_before, lstar_mg2.magnitude)
    
    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = model.reflection(system_real, source, detector,
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = model.reflection(system_imag, source, detector,
                                                        maxwell_garnett=False)                         

    assert_array_almost_equal(refl_bg1, refl_bg2)
    assert_array_almost_equal(g_bg1, g_bg2)
    assert_array_almost_equal(lstar_bg1, lstar_bg2)

    # Outputs before refactoring structcol
    refl_bg1_before = 0.2685710414987676
    refl_bg2_before = 0.2685710414987676
    g_bg1_before = -0.17681566915117486
    g_bg2_before = -0.17681566915117486
    lstar_bg1_before = 11593.280877304634
    lstar_bg2_before = 11593.280877304634

    assert_equal(refl_bg1_before, refl_bg1)
    assert_equal(refl_bg2_before, refl_bg2)
    assert_equal(g_bg1_before, g_bg1)
    assert_equal(g_bg2_before, g_bg2)
    assert_equal(lstar_bg1_before, lstar_bg1.magnitude)
    assert_equal(lstar_bg2_before, lstar_bg2.magnitude)
    
    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    n_particle_imag2 = Quantity(1.5+1e-8j, '')
    thickness = Quantity('100 um')
    
    # With Bruggeman
    species_imag2 = Spheres(n_particle_imag2, radius, volume_fraction, pdi=0)
    system_imag2 = Film(species_imag2, n_matrix, n_medium, thickness=thickness, structure='glass')
    refl_bg3, _, _, g_bg3, lstar_bg3 = model.reflection(system_imag2, source, detector,
                                                        maxwell_garnett=False)
                                                    
    assert_array_almost_equal(refl_bg1, refl_bg3, decimal=3)
    assert_array_almost_equal(g_bg1, g_bg3, decimal=3)
    assert_array_almost_equal(lstar_bg1.to('mm'), lstar_bg3.to('mm'), decimal=4)

                                                    
def test_calc_g():
    # test that the anisotropy factor for multilayer spheres are the same when
    # using calc_g from mie.py in pymie and using the model
    wavelength = Quantity(500, 'nm')
    
    # calculate g using the model
    n_particle = Quantity(np.array([1.5, 1.0]), '')
    radius = Quantity(np.array([120, 130]), 'nm')
    volume_fraction = Quantity(0.01, '')
    n_matrix = Quantity(1.0, '')
    n_medium = n_matrix
    
    source = Source(wavelength, polarization=None, incidence_angle=0)
    detector = DetectorSingleScat()
    
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness=np.inf, structure=None)
    _, _, _, g1, _ = model.reflection(system, source, detector,
                                      small_angle=Quantity('0.01 deg'), 
                                      num_angles=1000)
                                                        
    # calculate g using calc_g in pymie
    vf_array = np.empty(len(np.atleast_1d(radius)))
    r_array = np.array([0] + np.atleast_1d(radius).tolist()) 
    for r in np.arange(len(r_array)-1):
        vf_array[r] = ((r_array[r+1]**3-r_array[r]**3) / (r_array[-1:]**3) * 
                       volume_fraction.magnitude)
    
    n_sample = ri.n_eff(n_particle, n_matrix, vf_array)
    m = index_ratio(n_particle, n_sample).flatten()  
    x = size_parameter(wavelength, n_sample, radius).flatten()  
    qscat, qext, qback = mie.calc_efficiencies(m, x)
    g2 = mie.calc_g(m,x)   
    
    assert_array_almost_equal(g1, g2)
    
    # Outputs before refactoring structcol
    g1_before = 0.5064750277811477
    g2_before = 0.5064757158664487
    
    assert_equal(g1_before, g1)
    assert_equal(g2_before, g2)
    
    
def test_reflection_absorbing_matrix():
    # test that the reflections with a real n_matrix and with a complex
    # n_matrix with a 0 imaginary component are the same 
    wavelength = Quantity(500, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120 nm')
    n_matrix_real = Quantity(1.0, '')
    n_matrix_imag = Quantity(1.0 + 0j, '')
    n_medium = Quantity(1.0, '')
    n_particle = Quantity(1.5, '')
    
    source = Source(wavelength, polarization=None, incidence_angle=0)
    detector = DetectorSingleScat()
    species = Spheres(n_particle, radius, volume_fraction, pdi=0)
    
    # With Maxwell-Garnett
    system_real = Film(species, n_matrix_real, n_medium, thickness=np.inf, structure='glass')
    refl_mg1, _, _, g_mg1, lstar_mg1 = model.reflection(system_real, source, detector,
                                                        maxwell_garnett=True)
                                      
    system_imag = Film(species, n_matrix_imag, n_medium, thickness=np.inf, structure='glass')
    refl_mg2, _, _, g_mg2, lstar_mg2 = model.reflection(system_imag, source, detector,
                                                        maxwell_garnett=True)
    
    assert_array_almost_equal(refl_mg1, refl_mg2)
    assert_array_almost_equal(g_mg1, g_mg2)
    assert_array_almost_equal(lstar_mg1, lstar_mg2)
    
    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = model.reflection(system_real, source, detector,
                                                        maxwell_garnett=False)
    
    refl_bg2, _, _, g_bg2, lstar_bg2 = model.reflection(system_imag, source, detector,
                                                        maxwell_garnett=False)
    
    assert_array_almost_equal(refl_bg1, refl_bg2)
    assert_array_almost_equal(g_bg1, g_bg2)
    assert_array_almost_equal(lstar_bg1, lstar_bg2)
    
    # test that the reflectance is (almost) the same when using an
    # almost-non-absorbing index vs a non-absorbing index
    thickness = Quantity('100 um')
    n_matrix_imag2 = Quantity(1.0 + 1e-8j, '')
    system_imag2 = Film(species, n_matrix_imag2, n_medium, thickness=thickness, 
                        structure='glass')
    
    # With Bruggeman
    refl_bg3, _, _, g_bg3, lstar_bg3 = model.reflection(system_imag2, source, detector,
                                                        maxwell_garnett=False)
    
    assert_array_almost_equal(refl_bg1, refl_bg3, decimal=3)
    assert_array_almost_equal(g_bg1, g_bg3, decimal=3)
    assert_array_almost_equal(lstar_bg1.to('mm'), lstar_bg3.to('mm'), decimal=4)
    
    
def test_reflection_polydispersity():
    wavelength = Quantity(500, 'nm')
    radius1 = Quantity('120 nm')
    n_matrix = Quantity(1.0, '')
    n_medium = Quantity(1.0, '')
    n_particle = Quantity(1.5, '')
    radius2 = Quantity('120 nm')
    volume_fraction = Quantity(0.5, '')
    volume_fraction1 = Quantity(0.4, '')
    volume_fraction2 = Quantity(0.1, '')
    pdi = Quantity(1e-7, '')  # monodisperse limit
    
    source = Source(wavelength, polarization=None, incidence_angle=0)
    detector = DetectorSingleScat()
    
    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    species = Spheres(n_particle, radius1, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness=np.inf, 
                  structure=None)
    refl, _, _, g, lstar = model.reflection(system, source, detector)
                  
    species1 = Spheres(n_particle, radius1, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle, radius2, volume_fraction2, pdi=pdi)  
    system = Film([species1, species2], n_matrix, n_medium, thickness=np.inf, 
                   structure=None)            
    refl2, _, _, g2, lstar2 = model.reflection(system, source, detector)              
    
    assert_array_almost_equal(refl, refl2)
    assert_array_almost_equal(g, g2)
    assert_array_almost_equal(lstar.to('mm'), lstar2.to('mm'), decimal=4)
    
    # Outputs before refactoring structcol
    refl_before = 0.021202873774022364
    refl2_before = 0.02120287377402078
    g_before = 0.6149959692900278
    g2_before = 0.6149959692900626
    lstar_before = 0.0037795694345017063
    lstar2_before = 0.0037899271967178523
  
    assert_equal(refl_before, refl)
    assert_equal(refl2_before, refl2)
    assert_equal(g_before, g)
    assert_equal(g2_before, g2)
    assert_equal(lstar_before, lstar.to('mm').magnitude)
    assert_equal(lstar2_before, lstar2.to('mm').magnitude)
    
    # test that the reflectance using only the structure factor is the same 
    # using the polydisperse formula vs using Percus-Yevick in the limit of 
    # monodispersity
    system = Film(species, n_matrix, n_medium, thickness=np.inf, 
                  structure='glass')
    refl3, _, _, g3, lstar3 = model.reflection(system, source, detector, form=None)
                  
    system = Film([species1, species2], n_matrix, n_medium, thickness=np.inf, 
                   structure='glass')            
    refl4, _, _, g4, lstar4 = model.reflection(system, source, detector, form=None)   
    
    assert_array_almost_equal(refl3, refl4)
    assert_array_almost_equal(g3, g4)
    assert_array_almost_equal(lstar3.to('mm'), lstar4.to('mm'), decimal=4)

    # Outputs before refactoring structcol
    refl3_before= 0.6310965269823348
    refl4_before = 0.6310965259195878
    g3_before = -0.635630839621477
    g4_before = -0.6356308390717892
    lstar3_before = 0.0002005604473366244
    lstar4_before = 0.00020056044751316733
    
    assert_equal(refl3_before, refl3)
    assert_equal(refl4_before, refl4)
    assert_equal(g3_before, g3)
    assert_equal(g4_before, g4)
    assert_equal(lstar3_before, lstar3.to('mm').magnitude)
    assert_equal(lstar4_before, lstar4.to('mm').magnitude)
    
    # test that the reflectance using both the structure and form factors is 
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in 
    # the limit of monodispersity
    system = Film(species, n_matrix, n_medium, thickness=np.inf, 
                  structure='glass')
    refl5, _, _, g5, lstar5 = model.reflection(system, source, detector, form='sphere')
                  
    system = Film([species1, species2], n_matrix, n_medium, thickness=np.inf, 
                   structure='glass')            
    refl6, _, _, g6, lstar6 = model.reflection(system, source, detector, form='sphere') 
    
    assert_array_almost_equal(refl5, refl6)
    assert_array_almost_equal(g5, g6)
    assert_array_almost_equal(lstar5.to('mm'), lstar6.to('mm'), decimal=4)
    
    # Outputs before refactoring structcol
    refl5_before = 0.2685710414987676
    refl6_before = 0.2685710407296461
    g5_before = -0.17681566915117486
    g6_before = -0.1768156684026972
    lstar5_before = 0.011593280877304636
    lstar6_before = 0.011625051809100308
    
    assert_equal(refl5_before, refl5)
    assert_equal(refl6_before, refl6)
    assert_equal(g5_before, g5)
    assert_equal(g6_before, g6)
    assert_equal(lstar5_before, lstar5.to('mm').magnitude)
    assert_equal(lstar6_before, lstar6.to('mm').magnitude)
    
    # test that the reflectance is the same for a polydisperse monospecies
    # and a bispecies with equal types of particles
    pdi = Quantity(1e-1, '') 
    
    species = Spheres(n_particle, radius1, volume_fraction, pdi=pdi)
    system = Film(species, n_matrix, n_medium, thickness=np.inf, 
                  structure='glass')
    refl7, _, _, g7, lstar7 = model.reflection(system, source, detector, form='sphere')
                  
    species1 = Spheres(n_particle, radius1, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle, radius2, volume_fraction2, pdi=pdi)                   
    system = Film([species1, species2], n_matrix, n_medium, thickness=np.inf, 
                   structure='glass')            
    refl8, _, _, g8, lstar8 = model.reflection(system, source, detector, form='sphere') 
    
    assert_array_almost_equal(refl7, refl8)
    assert_array_almost_equal(g7, g8)
    assert_array_almost_equal(lstar7.to('mm'), lstar8.to('mm'))    
    
    # test that the reflectance is the same regardless of the order in which
    # the radii are specified
    radius3 = Quantity('90 nm')
    volume_fraction3 = Quantity(0.25, '')
    species1 = Spheres(n_particle, radius1, volume_fraction3, pdi=pdi)
    species2 = Spheres(n_particle, radius3, volume_fraction3, pdi=pdi)  
                 
    system1 = Film([species1, species2], n_matrix, n_medium, thickness=np.inf, 
                   structure='glass')        
    refl9, _, _, g9, lstar9 = model.reflection(system1, source, detector, form='sphere') 
    
    system2 = Film([species2, species1], n_matrix, n_medium, thickness=np.inf, 
                   structure='glass')  
    refl10, _, _, g10, lstar10 = model.reflection(system2, source, detector, form='sphere') 
    
    assert_array_almost_equal(refl9, refl10)
    assert_array_almost_equal(g9, g10)
    assert_array_almost_equal(lstar9.to('mm'), lstar10.to('mm'))   
    
    
def test_reflection_polydispersity_with_absorption():
    wavelength = Quantity(500, 'nm')
    volume_fraction = Quantity(0.5, '')
    volume_fraction1 = Quantity(0.4, '')
    volume_fraction2 = Quantity(0.1, '')
    radius1 = Quantity('120 nm')
    n_matrix = Quantity(1.0+0.0003j, '')
    n_medium = Quantity(1.0, '')
    n_particle = Quantity(1.5+0.0005j, '')
    radius2 = Quantity('120 nm')

    pdi = Quantity(1e-7, '')  # monodisperse limit
    thickness = Quantity('10 um')
    
    source = Source(wavelength, polarization=None, incidence_angle=0)
    detector = DetectorSingleScat()
    
    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    species = Spheres(n_particle, radius1, volume_fraction, pdi=0)
    system = Film(species, n_matrix, n_medium, thickness=thickness, 
                  structure=None)
    refl, _, _, g, lstar = model.reflection(system, source, detector)
                  
    species1 = Spheres(n_particle, radius1, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle, radius2, volume_fraction2, pdi=pdi)  
    system = Film([species1, species2], n_matrix, n_medium, thickness=thickness, 
                   structure=None)            
    refl2, _, _, g2, lstar2 = model.reflection(system, source, detector)  
    
    assert_array_almost_equal(refl, refl2, decimal=5)
    assert_array_almost_equal(g, g2, decimal=4)
    assert_array_almost_equal(lstar.to('mm'), lstar2.to('mm'), decimal=4)

    # Outputs before refactoring structcol
    refl_before = 0.020791487299024698
    refl2_before = 0.02079125872215926
    g_before = 0.61562921974002 #726274264.1349005
    g2_before = 0.6156292197400548 #726274264.1349416
    lstar_before = 0.0044717814146885779 #0.006279358811781641
    lstar2_before = 0.0044840361567639936 #0.006296567149019748
      
    assert_equal(refl_before, refl.magnitude)
    assert_equal(refl2_before, refl2.magnitude)
    assert_equal(g_before, g.magnitude)
    assert_equal(g2_before, g2.magnitude)
    assert_equal(lstar_before, lstar.to('mm').magnitude)
    assert_equal(lstar2_before, lstar2.to('mm').magnitude)
    
    # test that the reflectance using only the structure factor is the same 
    # using the polydisperse formula vs using Percus-Yevick in the limit of 
    # monodispersity
    system = Film(species, n_matrix, n_medium, thickness=thickness, 
                  structure='glass')
    refl3, _, _, g3, lstar3 = model.reflection(system, source, detector, form=None)
                  
    system = Film([species1, species2], n_matrix, n_medium, thickness=thickness, 
                   structure='glass')            
    refl4, _, _, g4, lstar4 = model.reflection(system, source, detector, form=None)   
    
    assert_array_almost_equal(refl3, refl4)
    assert_array_almost_equal(g3, g4, decimal=4)
    assert_array_almost_equal(lstar3.to('mm'), lstar4.to('mm'), decimal=4)

    # Outputs before refactoring structcol
    refl3_before = 0.6311022445010561
    refl4_before = 0.6311022434374303
    g3_before = -0.6356307606571816 #-27901.50120849103
    g4_before = -0.6356307601051542 #-27901.50118425936
    lstar3_before = 8.8037552221780592e-09 #1.4399291088853016e-08
    lstar4_before = 8.8037552299275471e-09 #1.4399291096668534e-08
  
    assert_equal(refl3_before, refl3.magnitude)
    assert_equal(refl4_before, refl4.magnitude)
    assert_equal(g3_before, g3.magnitude)
    assert_equal(g4_before, g4.magnitude)
    assert_equal(lstar3_before, lstar3.to('mm').magnitude)
    assert_equal(lstar4_before, lstar4.to('mm').magnitude)
    
    # test that the reflectance using both the structure and form factors is 
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in 
    # the limit of monodispersity
    system = Film(species, n_matrix, n_medium, thickness=thickness, 
                  structure='glass')
    refl5, _, _, g5, lstar5 = model.reflection(system, source, detector, form='sphere')
                  
    system = Film([species1, species2], n_matrix, n_medium, thickness=thickness, 
                   structure='glass')            
    refl6, _, _, g6, lstar6 = model.reflection(system, source, detector, form='sphere') 
    
    assert_array_almost_equal(refl5, refl6, decimal=3)
    assert_array_almost_equal(g5, g6)
    assert_array_almost_equal(lstar5.to('mm'), lstar6.to('mm'), decimal=4)
    
    # Outputs before refactoring structcol
    refl5_before = 0.11277597784758357
    refl6_before = 0.11259532698024184
    g5_before = -0.17376384100464944 #-209.15733480514967
    g6_before = -0.17376384019461683 #-209.1573338372998
    lstar5_before = 0.013809880819376879 #0.013405648948885825
    lstar6_before = 0.013847726256293521 #0.013442386605693767
    
    assert_equal(refl5_before, refl5.magnitude)
    assert_equal(refl6_before, refl6.magnitude)
    assert_equal(g5_before, g5.magnitude)
    assert_equal(g6_before, g6.magnitude)
    assert_equal(lstar5_before, lstar5.to('mm').magnitude)
    assert_equal(lstar6_before, lstar6.to('mm').magnitude)
    
    # test that the reflectances are (almost) the same when using an 
    # almost-non-absorbing vs an non-absorbing system
    n_matrix2 = Quantity(1.0+1e-8j, '')
    n_particle2 = Quantity(1.5+1e-8j, '')
    radius2 = Quantity('150 nm')
    
    species1 = Spheres(n_particle2.real, radius1, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle2.real, radius2, volume_fraction2, pdi=pdi)
    system = Film([species1, species2], n_matrix2.real, n_medium, thickness=thickness, 
                  structure='glass')
    refl7, _, _, g7, lstar7 = model.reflection(system, source, detector, form='sphere') 

    species1 = Spheres(n_particle2, radius1, volume_fraction1, pdi=pdi)
    species2 = Spheres(n_particle2, radius2, volume_fraction2, pdi=pdi)
    system = Film([species1, species2], n_matrix2, n_medium, thickness=thickness, 
                  structure='glass')
    refl8, _, _, g8, lstar8 = model.reflection(system, source, detector, form='sphere') 

    assert_array_almost_equal(refl7, refl8, decimal=3)
    assert_array_almost_equal(g7, g8, decimal=2)
    assert_array_almost_equal(lstar7.to('mm'), lstar8.to('mm'), decimal=4)


'''
These tests will no longer be relevant in the refactored version

def test_reflection_throws_valueerror_for_polydisperse_core_shells(): 
# test that a valueerror is raised when trying to run polydisperse core-shells                 
    with pytest.raises(ValueError):
        wavelength = Quantity(500, 'nm')
        volume_fraction = Quantity(0.5, '')
        radius = Quantity(np.array([110, 120]), 'nm')
        n_matrix = Quantity(1.0, '')
        n_medium = Quantity(1.0, '')
        n_particle = Quantity(np.array([1.5,1.5]), '')
        volume_fraction2 = Quantity(volume_fraction * (radius[1]**3 / radius[0]**3), '')
        thickness = Quantity('10 um')
        
        radius2 = Quantity('120 nm')
        concentration = Quantity(np.array([0.9,0.1]), '')
        pdi = Quantity(np.array([1e-7, 1e-7]), '') 

        # when running polydisperse core-shells, without absorption
        refl, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse')      
        refl2, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='glass',
                                            form_type='polydisperse')
        refl3, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type=None,
                                            form_type='polydisperse')
        refl4, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='sphere')
        refl5, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type=None)   
                                            
        # when running polydisperse core-shells, with absorption
        refl6, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse',
                                            thickness=thickness)      
        refl7, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='glass',
                                            form_type='polydisperse',
                                            thickness=thickness)
        refl8, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type=None,
                                            form_type='polydisperse',
                                            thickness=thickness)
        refl9, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='sphere', thickness=thickness)
        refl10, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            radius2 = radius2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type=None, thickness=thickness) 
                                            

def test_reflection_throws_valueerror_for_polydisperse_unspecified_parameters(): 
# test that a valueerror is raised when trying to run polydisperse core-shells                 
    with pytest.raises(ValueError):
        wavelength = Quantity(500, 'nm')
        volume_fraction = Quantity(0.5, '')
        radius = Quantity(np.array([110, 120]), 'nm')
        n_matrix = Quantity(1.0, '')
        n_medium = Quantity(1.0, '')
        n_particle = Quantity(np.array([1.5,1.5]), '')
        volume_fraction2 = Quantity(volume_fraction * (radius[1]**3 / radius[0]**3), '')
        
        concentration = Quantity(np.array([0.9,0.1]), '')
        pdi = Quantity(np.array([1e-7, 1e-7]), '') 

        # when running polydisperse core-shells, without absorption, 
        # and unspecified radius2
        refl, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse')     
                                            
        # when running polydisperse core-shells, with absorption, 
        # and unspecified radius2
        refl, _, _, _, _ = model.reflection(n_particle+0.01j, n_matrix+0.01j, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            concentration = concentration, 
                                            pdi = pdi, structure_type='polydisperse',
                                            form_type='polydisperse')  
'''