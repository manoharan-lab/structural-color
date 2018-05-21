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
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError

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
    
    # set theta_max to be slightly smaller than the theta corresponding to 
    # total internal reflection (calculated manually to be 2.61799388)
    theta_max = Quantity(2.617,'deg')  
    refl1, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                         wavelength, radius, volume_fraction,  
                                         theta_min=theta_min, 
                                         theta_max=theta_max, 
                                         structure_type=None)
    # try a different range of thetas (but keeping theta_max < total internal
    # reflection angle)
    theta_max = Quantity(2.,'deg')  
    refl2, _, _, _, _ = model.reflection(n_particle, n_matrix, n_medium, 
                                         wavelength, radius, volume_fraction,  
                                         theta_min=theta_min, 
                                         theta_max=theta_max, 
                                         structure_type=None)
    
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
    angles = Quantity(np.linspace(np.pi/2, np.pi, 200), 'rad')
    
    # Differential cross section for non-core-shells
    radius = Quantity('100 nm')    
    n_particle = Quantity(1.5, '')
    volume_fraction = Quantity(0.0001, '')              # IS VF TOO LOW?
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
    m = index_ratio(n_particle, n_sample)
    x = size_parameter(wavelen, n_sample, radius)  
    diff = model.differential_cross_section(m, x, angles, volume_fraction)
    
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
    diff_cs = model.differential_cross_section(m_cs, x_cs, angles, 
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

    refl1, _, _, g1, lstar1 = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction, 
                                            thickness = Quantity('15000.0 nm'), 
                                            theta_min = Quantity('90 deg'), 
                                            small_angle=Quantity('5 deg'),                    
                                            maxwell_garnett=True)
    
    # Non core-shell particles with Bruggeman effective index
    volume_fraction2 = Quantity(0.00001, '')
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction2, 
                                            thickness = Quantity('15000.0 nm'), 
                                            theta_min = Quantity('90 deg'), 
                                            small_angle=Quantity('5 deg'), 
                                            maxwell_garnett=False)
        
    # Core-shell particles of core diameter equal to non core shell particles, 
    # and shell index of air. With Bruggeman effective index
    n_particle3 = Quantity(np.array([1.5, 1.0]), '')
    radius3 = Quantity(np.array([120, 130]), 'nm')
    volume_fraction3 = volume_fraction2 * (radius3[1]**3 / radius3[0]**3)

    refl3, _, _, g3, lstar3 = model.reflection(n_particle3, n_matrix, n_medium, 
                                            wavelength, radius3, volume_fraction3, 
                                            thickness = Quantity('15000.0 nm'), 
                                            small_angle=Quantity('5 deg'), 
                                            theta_min = Quantity('90 deg'), 
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
    refl4 = model.reflection(n_particle4, n_matrix, n_medium, wavelength, 
                             radius4, volume_fraction, thickness=thickness)[0]
    
    # Absorbing core-shell
    n_particle5 = Quantity(np.array([1.5+0.001j, 1.5+0.001j]), '')
    radius5 = Quantity(np.array([110, 120]), 'nm')
    refl5 = model.reflection(n_particle5, n_matrix, n_medium, wavelength, 
                             radius5, volume_fraction, thickness=thickness)[0]
    
    assert_array_almost_equal(refl4, refl5, decimal=3)
    
    # Same as previous test but with absorbing matrix
    # Non-core-shell
    radius6 = Quantity('120 nm')
    n_particle6 = Quantity(1.5+0.001j, '')
    n_matrix6 = Quantity(1.0+0.001j, '')
    refl6 = model.reflection(n_particle6, n_matrix6, n_medium, wavelength, 
                             radius6, volume_fraction, thickness=thickness)[0]
    
    # Core-shell
    n_particle7 = Quantity(np.array([1.5+0.001j, 1.5+0.001j]), '')
    radius7 = Quantity(np.array([110, 120]), 'nm')
    n_matrix7 = Quantity(1.0+0.001j, '')    
    refl7 = model.reflection(n_particle7, n_matrix7, n_medium, wavelength, 
                             radius7, volume_fraction, thickness=thickness)[0]
    
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
    
    # With Maxwell-Garnett
    refl_mg1, _, _, g_mg1, lstar_mg1 = model.reflection(n_particle_real, n_matrix, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = model.reflection(n_particle_imag, n_matrix, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=True)
    
    assert_array_almost_equal(refl_mg1, refl_mg2)
    assert_array_almost_equal(g_mg1, g_mg2)
    assert_array_almost_equal(lstar_mg1, lstar_mg2)
    
    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = model.reflection(n_particle_real, n_matrix, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = model.reflection(n_particle_imag, n_matrix, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=False)
    
    assert_array_almost_equal(refl_bg1, refl_bg2)
    assert_array_almost_equal(g_bg1, g_bg2)
    assert_array_almost_equal(lstar_bg1, lstar_bg2)

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
    
    _, _, _, g1, _= model.reflection(n_particle, n_matrix, n_medium, 
                                     wavelength, radius, volume_fraction, 
                                     small_angle=Quantity('0.01 deg'), 
                                     num_angles=1000, structure_type=None)

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
    
    # With Maxwell-Garnett
    refl_mg1, _, _, g_mg1, lstar_mg1 = model.reflection(n_particle, n_matrix_real, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=True)
    refl_mg2, _, _, g_mg2, lstar_mg2 = model.reflection(n_particle, n_matrix_imag, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=True)
    
    assert_array_almost_equal(refl_mg1, refl_mg2)
    assert_array_almost_equal(g_mg1, g_mg2)
    assert_array_almost_equal(lstar_mg1, lstar_mg2)
    
    # With Bruggeman
    refl_bg1, _, _, g_bg1, lstar_bg1 = model.reflection(n_particle, n_matrix_real, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=False)
    refl_bg2, _, _, g_bg2, lstar_bg2 = model.reflection(n_particle, n_matrix_imag, 
                                                        n_medium, wavelength, 
                                                        radius, volume_fraction, 
                                                        maxwell_garnett=False)
    
    assert_array_almost_equal(refl_bg1, refl_bg2)
    assert_array_almost_equal(g_bg1, g_bg2)
    assert_array_almost_equal(lstar_bg1, lstar_bg2)
    
def test_reflection_polydispersity():
    wavelength = Quantity(500, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120 nm')
    n_matrix = Quantity(1.0, '')
    n_medium = Quantity(1.0, '')
    n_particle = Quantity(1.5, '')
    radius2 = Quantity('120 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit

    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    refl, _, _, g, lstar = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction,
                                            structure_type=None,
                                            form_type='sphere')
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix, 
                                                  n_medium, wavelength, radius, 
                                                  volume_fraction, 
                                                  radius2 = radius2, 
                                                  concentration = concentration, 
                                                  pdi = pdi, structure_type=None,
                                                  form_type='polydisperse')
    
    assert_array_almost_equal(refl, refl2)
    assert_array_almost_equal(g, g2)
    assert_array_almost_equal(lstar.to('mm'), lstar2.to('mm'), decimal=4)
  
    # test that the reflectance using only the structure factor is the same 
    # using the polydisperse formula vs using Percus-Yevick in the limit of 
    # monodispersity
    refl3, _, _, g3, lstar3 = model.reflection(n_particle, n_matrix, n_medium, 
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type=None)
    refl4, _, _, g4, lstar4 = model.reflection(n_particle, n_matrix, 
                                               n_medium, wavelength, radius, 
                                               volume_fraction, 
                                               radius2 = radius2, 
                                               concentration = concentration, 
                                               pdi = pdi, 
                                               structure_type='polydisperse',
                                               form_type=None)
    
    assert_array_almost_equal(refl3, refl4)
    assert_array_almost_equal(g3, g4)
    assert_array_almost_equal(lstar3.to('mm'), lstar4.to('mm'), decimal=4)
    
    # test that the reflectance using both the structure and form factors is 
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in 
    # the limit of monodispersity
    refl5, _, _, g5, lstar5 = model.reflection(n_particle, n_matrix, n_medium, 
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type='sphere')
    refl6, _, _, g6, lstar6 = model.reflection(n_particle, n_matrix, 
                                               n_medium, wavelength, radius, 
                                               volume_fraction, 
                                               radius2 = radius2, 
                                               concentration = concentration, 
                                               pdi = pdi, 
                                               structure_type='polydisperse',
                                               form_type='polydisperse')
    
    assert_array_almost_equal(refl5, refl6)
    assert_array_almost_equal(g5, g6)
    assert_array_almost_equal(lstar5.to('mm'), lstar6.to('mm'), decimal=4)
    
def test_reflection_polydispersity_with_absorption():
    wavelength = Quantity(500, 'nm')
    volume_fraction = Quantity(0.5, '')
    radius = Quantity('120 nm')
    n_matrix = Quantity(1.0+0.0003j, '')
    n_medium = Quantity(1.0, '')
    n_particle = Quantity(1.5+0.0005j, '')
    radius2 = Quantity('120 nm')
    concentration = Quantity(np.array([0.9,0.1]), '')
    pdi = Quantity(np.array([1e-7, 1e-7]), '')  # monodisperse limit
    thickness = Quantity('10 um')
    # test that the reflectance using only the form factor is the same using
    # the polydisperse formula vs using Mie in the limit of monodispersity
    refl, _, _, g, lstar = model.reflection(n_particle, n_matrix, n_medium, 
                                            wavelength, radius, volume_fraction,
                                            structure_type=None,
                                            form_type='sphere', 
                                            thickness=thickness)
    refl2, _, _, g2, lstar2 = model.reflection(n_particle, n_matrix, 
                                                  n_medium, wavelength, radius, 
                                                  volume_fraction, 
                                                  radius2 = radius2, 
                                                  concentration = concentration, 
                                                  pdi = pdi, structure_type=None,
                                                  form_type='polydisperse',
                                                  thickness=thickness)
    
    assert_array_almost_equal(refl, refl2, decimal=5)
    assert_array_almost_equal(g, g2, decimal=4)
    assert_array_almost_equal(lstar.to('mm'), lstar2.to('mm'), decimal=4)
  
    # test that the reflectance using only the structure factor is the same 
    # using the polydisperse formula vs using Percus-Yevick in the limit of 
    # monodispersity
    refl3, _, _, g3, lstar3 = model.reflection(n_particle, n_matrix, n_medium, 
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type=None, 
                                               thickness=thickness)
    refl4, _, _, g4, lstar4 = model.reflection(n_particle, n_matrix, 
                                               n_medium, wavelength, radius, 
                                               volume_fraction, 
                                               radius2 = radius2, 
                                               concentration = concentration, 
                                               pdi = pdi, 
                                               structure_type='polydisperse',
                                               form_type=None, 
                                               thickness=thickness)
    
    assert_array_almost_equal(refl3, refl4)
    assert_array_almost_equal(g3, g4, decimal=4)
    assert_array_almost_equal(lstar3.to('mm'), lstar4.to('mm'), decimal=4)
    
    # test that the reflectance using both the structure and form factors is 
    # the same using the polydisperse formula vs using Mie and Percus-Yevick in 
    # the limit of monodispersity
    refl5, _, _, g5, lstar5 = model.reflection(n_particle, n_matrix, n_medium, 
                                               wavelength, radius, volume_fraction,
                                               structure_type='glass',
                                               form_type='sphere', 
                                               thickness=thickness)
    refl6, _, _, g6, lstar6 = model.reflection(n_particle, n_matrix, 
                                               n_medium, wavelength, radius, 
                                               volume_fraction, 
                                               radius2 = radius2, 
                                               concentration = concentration, 
                                               pdi = pdi, 
                                               structure_type='polydisperse',
                                               form_type='polydisperse', 
                                               thickness=thickness)
    
    assert_array_almost_equal(refl5, refl6, decimal=3)
    assert_array_almost_equal(g5/1e6, g6/1e6)
    assert_array_almost_equal(lstar5.to('mm'), lstar6.to('mm'), decimal=4)
