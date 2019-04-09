# Copyright 2016, Vinothan N. Manoharan, Sofia Makgiriadou
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
Calculates optical properties of an amorphous colloidal suspension ("a photonic
glass") using a calculated structure factor, form factor, and Fresnel
coefficients, based on the single-scattering model in reference [1]_

References
----------
[1] Magkiriadou, S., Park, J.-G., Kim, Y.-S., and Manoharan, V. N. “Absence of
Red Structural Color in Photonic Glasses, Bird Feathers, and Certain Beetles”
Physical Review E 90, no. 6 (2014): 62302. doi:10.1103/PhysRevE.90.062302

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>
"""

import numpy as np
from . import refractive_index as ri
from . import ureg, Quantity
from . import structure
import pymie as pm
from pymie import mie, size_parameter, index_ratio
from pymie import multilayer_sphere_lib as msl

@ureg.check('[]', '[]', '[]', '[length]', '[length]', '[]')
def reflection(n_particle, n_matrix, n_medium, wavelen, radius, volume_fraction,
               thickness=None,
               theta_min=Quantity('90 deg'),
               theta_max=Quantity('180 deg'),
               phi_min=Quantity('0 deg'),
               phi_max=Quantity('360 deg'),
               incident_angle=Quantity('0 deg'),
               num_angles=200,
               small_angle=Quantity('1 deg'),
               structure_type='glass',
               form_type='sphere',
               maxwell_garnett=False):
    """
    Calculate fraction of light reflected from an amorphous colloidal
    suspension (a "photonic glass").

    Parameters
    ----------
    n_particle: array of structcol.Quantity [dimensionless]
        refractive index of particles or voids at wavelength=wavelen. In case 
        of core-shell particles, define indices from the innermost to the 
        outermost layer. 
    n_matrix: structcol.Quantity [dimensionless]
        refractive index of the matrix surrounding the particles (at wavelen)
    n_medium: structcol.Quantity [dimensionless]
        refractive index of the medium surrounding the sample.  This is
        usually air or vacuum
    wavelen: structcol.Quantity [length]
        wavelength of light in the medium (which is usually air or vacuum)
    radius: array of structcol.Quantity [length]
        radii of particles or voids. In case of core-shell particles, define
        radii from the innermost to the outermost layer. 
    volume_fraction: array of structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix. If it's a core-shell 
        particle, must be the volume fraction of the entire core-shell particle 
        in the matrix.
    thickness: structcol.Quantity [length] (optional)
        thickness of photonic glass.  If unspecified, assumed to be infinite
    theta_min: structcol.Quantity [dimensionless] (optional)
    theta_max: structcol.Quantity [dimensionless] (optional)
        along with theta_min, specifies the angular range over which to
        integrate the scattered signal. The angles are the scattering angles
        (polar angle, measured from the incident light direction) after the
        light exits into the medium. The function will correct for refraction
        at the interface to map this range of exit angles onto the range of
        scattering angles from the particles. If theta_min and theta_max are
        unspecified, the integral is carried out over the entire backscattering
        hemisphere (90 to 180 degrees). Usually one would set theta_min to
        correspond to the numerical aperture of the detector. Setting theta_max
        to a value less than 180 degrees corresponds to dark-field detection.
        Both theta_min and theta_max can carry explicit units of radians or
        degrees.
    phi_min: structcol.Quantity [dimensionless] (optional)
    phi_max: structcol.Quantity [dimensionless] (optional)
        along with phi_min, specifies the azimuthal angular range over which to
        integrate the scattered signal. The angles are the azimuthal angles
        (measured from the incident light direction) after the
        light exits into the medium. The function will correct for refraction
        at the interface to map this range of exit angles onto the range of
        scattering angles from the particles. If phi_min and phi_max are
        unspecified, the integral is carried out over the entire backscattering
        hemisphere (0 to 360 degrees). 
    incident_angle: structcol.Quantity [dimensionless] (optional)
        incident angle, measured from the normal (specify degrees or radians by
        using the appropriate units in Quantity())
    num_angles: integer
        number of angles to use in evaluation of the cross-section, which is
        done by numerical integration (fixed quadrature). The default value
        (200) seems to do OK for 280-nm-diameter spheres, but could use more
        testing.
    small_angle: structcol.Quantity [dimensionless] (optional)
        If the analytic formula is used, S(q=0) returns nan. This doesn't
        matter when calculating the scattering cross section because sin(0) =
        0, so the contribution of the differential scattering cross section at
        theta = 0 to the total cross section is zero. To prevent any errors or
        warnings, set small_angle equal to some minimum angle
        at which to calculate the structure factor (and, by extension, the
        total cross-section).  The default value is chosen to give good
        agreement with Mie theory for a single sphere, but it may not be
        reasonable for all calculations.
    structure_type: string, dictionary, or None (optional)
        Can be string specifying structure type. Current options are "glass" or
        "paracrystal". Can also be dictionary specifying structure type and
        parameters for structures that require them. Expects keys of 
        'name': 'paracrystal', and 'sigma': int or float. Can also set to None
        in order to only visualize effect of form factor on reflectance 
        spectrum.
    form_type: string or None (optional)
        String specifying form factor type. Currently, 'sphere' is only shape 
        option. Can also set to None in order to only visualize the effect of 
        structure factor on reflectance spectrum. 
    maxwell_garnett: boolean
        If true, the model uses Maxwell-Garnett's effective index for the 
        sample. In that case, the user must specify one refractive index for 
        the particle and one for the matrix. If false, the model uses 
        Bruggeman's formula, which can be used for multilayer particles. 
        
    Returns
    -------
    float (5-tuple):
        fraction of light reflected from sample for unpolarized light, parallel
        polarization, and perpendicular polarization;
        asymmetry parameter and transport length for unpolarized light

    Notes
    -----
    Uses eqs. 5 and 6 from [1]_. If the system has absorption (either in the 
    particle or the matrix), the model uses adapted versions of eqs. 5 and 6. 
    The function uses an effective refractive index of the sample (computed 
    with either the Bruggeman or the Maxwell-Garnett formulas) to calculate the 
    scattering from individual spheres within the matrix. It also uses the 
    effective refractive index to calculate the Fresnel coefficients at the 
    boundary between the medium (referred to as "air" in the reference) and 
    the sample.

    References
    ----------
    [1] Magkiriadou, S., Park, J.-G., Kim, Y.-S., and Manoharan, V. N. “Absence
    of Red Structural Color in Photonic Glasses, Bird Feathers, and Certain
    Beetles” Physical Review E 90, no. 6 (2014): 62302.
    doi:10.1103/PhysRevE.90.062302
    """
    
    # check that the number of indices and radii is the same
    if len(np.atleast_1d(n_particle)) != len(np.atleast_1d(radius)):
       raise ValueError('Arrays of indices and radii must be the same length')
        
    # calculate array of volume fractions of each layer in the particle. If 
    # particle is not core-shell, volume fraction remains the same
    vf_array = np.empty(len(np.atleast_1d(radius)))
    r_array = np.array([0] + np.atleast_1d(radius).tolist()) 
    for r in np.arange(len(r_array)-1):
        vf_array[r] = (r_array[r+1]**3-r_array[r]**3) / (r_array[-1:]**3) * volume_fraction.magnitude
    if len(vf_array) == 1:
        vf_array = float(vf_array)

    # use Bruggeman formula to calculate effective index of
    # particle-matrix composite
    n_sample = ri.n_eff(n_particle, n_matrix, vf_array, 
                        maxwell_garnett=maxwell_garnett)
                    
    if len(np.atleast_1d(radius)) > 1:
        m = index_ratio(n_particle, n_sample).flatten()  
        x = size_parameter(wavelen, n_sample, radius).flatten()
    else:
        m = index_ratio(n_particle, n_sample)
        x = size_parameter(wavelen, n_sample, radius)

    k = 2*np.pi*n_sample/wavelen  

    # calculate transmission and reflection coefficients at first interface
    # between medium and sample
    # (TODO: include correction for reflection off the back interface of the
    # sample)
    t_medium_sample = fresnel_transmission(n_medium, n_sample, incident_angle)
    r_medium_sample = fresnel_reflection(n_medium, n_sample, incident_angle)
    
    theta_min = theta_min.to('rad').magnitude
    theta_max = theta_max.to('rad').magnitude
    phi_min = phi_min.to('rad').magnitude
    phi_max = phi_max.to('rad').magnitude
    small_angle = small_angle.to('rad').magnitude
    # calculate the min theta, taking into account refraction at the interface
    # between the medium and the sample. This is the scattering angle at which
    # light exits into the medium at (180-theta_min) degrees from the normal.
    # (Snell's law: n_medium sin(alpha_medium) = n_sample sin(alpha_sample)
    # where alpha = pi - theta)
    sin_alpha_sample_theta_min = np.sin(np.pi-theta_min) * n_medium/np.abs(n_sample)   # TODO: use n_sample.real or abs(n_sample)?
    sin_alpha_sample_theta_max = np.sin(np.pi-theta_max) * n_medium/np.abs(n_sample)

    if sin_alpha_sample_theta_min >= 1:
        # in this case, theta_min and the ratio of n_medium/n_sample are
        # sufficiently large so that all the scattering from 90-180 degrees
        # exits into the range of angles captured by the detector (assuming
        # that the theta_max is set to pi)
        theta_min_refracted = np.pi/2.0
    else:
        theta_min_refracted = np.pi - np.arcsin(sin_alpha_sample_theta_min)
    
    if sin_alpha_sample_theta_max >= 1:
        # in this case, theta_max and the ratio of n_medium/n_sample are such 
        # that all of the scattering from 90-180 degrees exits into angles
        # that are outside of the range of theta_min to theta_max. Thus, the
        # reflectance will be ~0 (only fresnel will contribute to reflectance)
        theta_max_refracted = np.pi/2.0
    else:
        theta_max_refracted = np.pi - np.arcsin(sin_alpha_sample_theta_max)

    # integrate form_factor*structure_factor*transmission
    # coefficient*sin(theta) over angles to get sigma_detected (eq 5)
    angles = Quantity(np.linspace(theta_min_refracted, theta_max_refracted, 
                                  num_angles), 'rad')                          
    azi_angle_range = Quantity(phi_max-phi_min,'rad')

    transmission = fresnel_transmission(n_sample, n_medium, np.pi-angles)
                             
    # if n_sample is complex, then we must calculate the cross sections 
    # with the exact Mie solutions that account for absorption
    if np.abs(n_sample.imag.magnitude) > 0. and form_type == 'sphere':   
        if thickness is None:
            raise ValueError('Thickness must be specified when there is absorption')
        
        # calculate only the structure factor in the detected range of angles
        struct_factor_det = differential_cross_section(m, x, angles, volume_fraction,
                                                       form_type=None) 
        # calculate the form factor from the exact Mie solutions                                              
        form_factor_scat = mie.diff_scat_intensity_complex_medium(m, x, angles, 
                                                                  k*thickness)
        diff_cs_par = form_factor_scat[0] * struct_factor_det[0] 
        diff_cs_perp = form_factor_scat[1] * struct_factor_det[1]     
        # integrate the differential cross section over the reflection angles        
        cscat = mie.integrate_intensity_complex_medium(diff_cs_par*transmission[0], 
                                                       diff_cs_perp*transmission[1], 
                                                       thickness, angles, k,
                                                       phi_min=Quantity(phi_min, 'rad'), 
                                                       phi_max=Quantity(phi_max, 'rad'))      
        cscat_detected = cscat[0]
        cscat_detected_par = cscat[1]
        cscat_detected_perp = cscat[2]                                                            
        
        # calculate total scattering cross section
        angles = Quantity(np.linspace(0.0+small_angle, np.pi, num_angles), 'rad')
        struct_factor_tot = differential_cross_section(m, x, angles, volume_fraction,
                                                       form_type=None) 
        form_factor_tot = mie.diff_scat_intensity_complex_medium(m, x, angles, 
                                                                 k*thickness)
        diff_scat_tot_par = form_factor_tot[0] * struct_factor_tot[0]
        diff_scat_tot_perp = form_factor_tot[1] * struct_factor_tot[1]
        cscat_total = mie.integrate_intensity_complex_medium(diff_scat_tot_par, 
                                                             diff_scat_tot_perp, 
                                                             thickness, angles, k)[0]  
        
        # calculate total absorption cross section 
        nstop = mie._nstop(np.array(x).max())
        # if the index ratio m is an array with more than 1 element, it's a 
        # multilayer particle
        if len(np.atleast_1d(m)) > 1:
            coeffs = msl.scatcoeffs_multi(m, x)
            cabs_total = mie._cross_sections_complex_medium_sudiarta(coeffs[0], coeffs[1], x, radius)[1]
            if cabs_total.magnitude < 0.0:
                cabs_total = 0.0 * cabs_total.units
        else:
            coeffs = mie._scatcoeffs(m, x, nstop)   
            internal_coeffs = mie._internal_coeffs(m, x, nstop)
            x_scat = size_parameter(wavelen, n_particle, radius)
            cabs_total = mie._cross_sections_complex_medium_fu(coeffs[0], coeffs[1], internal_coeffs[0], 
                                                               internal_coeffs[1], radius, n_particle, 
                                                               n_sample, x_scat, x, wavelen)[1]                                                      
        cext_total = cscat_total + cabs_total
        
    else:    
        diff_cs = differential_cross_section(m, x, angles, volume_fraction,
                                             structure_type, form_type)                                   
        cscat_detected_par = _integrate_cross_section(diff_cs[0],
                                                  transmission[0]/np.abs(k)**2, 
                                                  angles, azi_angle_range)
        cscat_detected_perp = _integrate_cross_section(diff_cs[1],
                                                   transmission[1]/np.abs(k)**2, 
                                                   angles, azi_angle_range)
        cscat_detected = (cscat_detected_par + cscat_detected_perp)/2.0
    
        # now integrate from 0 to 180 degrees to get total cross-section.
        angles = Quantity(np.linspace(0.0+small_angle, np.pi, num_angles), 'rad')
        azi_angle_range = Quantity(2*np.pi,'rad')        
        # Fresnel coefficients do not appear in this integral since we're using 
        # the total cross-section to account for the attenuation in intensity 
        # as light propagates through the sample
        diff_cs = differential_cross_section(m, x, angles, volume_fraction,
                                             structure_type, form_type)
        cscat_total_par = _integrate_cross_section(diff_cs[0], 1.0/np.abs(k)**2,  
                                               angles, azi_angle_range)
        cscat_total_perp = _integrate_cross_section(diff_cs[1], 1.0/np.abs(k)**2, 
                                                angles, azi_angle_range)
        cscat_total = (cscat_total_par + cscat_total_perp)/2.0
        
        cext_total = cscat_total               
        
    # to calculate asymmetry parameter, use the far-field Mie solutions because
    # the phase function does not change whether there is absorption or not 
    # calculate asymmetry parameter using integral from 0 to 180 degrees
    angles = Quantity(np.linspace(0.0+small_angle, np.pi, num_angles), 'rad')
    azi_angle_range = Quantity(2*np.pi,'rad')    
    diff_cs = differential_cross_section(m, x, angles, volume_fraction,
                                        structure_type, form_type)
    asymmetry_par = _integrate_cross_section(diff_cs[0], np.cos(angles)*1.0/np.abs(k)**2,
                                             angles, azi_angle_range)
    asymmetry_perp = _integrate_cross_section(diff_cs[1], np.cos(angles)*1.0/np.abs(k)**2,
                                              angles, azi_angle_range)
    # calculate for unpolarized light
    asymmetry_parameter = (asymmetry_par + asymmetry_perp)/cscat_total/2.0

    # now eq. 6 for the total reflection
    rho = _number_density(volume_fraction, radius.max())
            
    if thickness is None:
        # assume semi-infinite sample
        factor = 1.0
    else:
        # use Beer-Lambert law to account for attenuation
        factor = (1.0 - np.exp(-rho*cext_total*thickness)) * cscat_total/cext_total               

    # one critical difference from Sofia's original code is that this code
    # calculates the reflected intensity in each polarization channel
    # separately, then averages them. The original code averaged the
    # transmission coefficients for the two polarization channels before
    # integrating. However, we do average the total cross section to normalize
    # the reflection cross-sections (that is, we use sigma_total rather than
    # sigma_total_par or sigma_total_perp).
    reflected_par = t_medium_sample[0] * cscat_detected_par/cext_total * \
                        factor + r_medium_sample[0]  
    reflected_perp = t_medium_sample[1] * cscat_detected_perp/cext_total * \
                         factor + r_medium_sample[1]             
    reflectance = (reflected_par + reflected_perp)/2
    
    # and the transport length for unpolarized light
    # (see eq. 5 of Kaplan, Dinsmore, Yodh, Pine, PRE 50(6): 4827, 1994)
    transport_length = 1/(1.0-asymmetry_parameter)/rho/cscat_total  # TODO is this cscat or cext_tot?

    return reflectance, reflected_par, reflected_perp, asymmetry_parameter, \
           transport_length
    
@ureg.check('[]', '[]', '[]', '[]')
def differential_cross_section(m, x, angles, volume_fraction, 
                               structure_type = 'glass', 
                               form_type = 'sphere',
                               coordinate_system = 'scattering plane',
                               phis = None, kd=1):
    """
    Calculate dimensionless differential scattering cross-section for a sphere,
    including contributions from the structure factor. Need to multiply by k**2
    to get the dimensional differential cross section.
    
    Parameters
    ----------
    m: float 
        complex particle relative refractive index, n_particle/n_sample
    x: float
        size parameter 
    angles: ndarray(structcol.Quantity [dimensionless])
        array of angles. Must be entered as a Quantity to allow specifying
        units (degrees or radians) explicitly
    volume_fraction: float
        volume fraction of the particles. If core-shell, should be volume 
        fraction of the entire core-shell particle.
    structure_type: str or None
        type of structure to calculate the structure factor. Can be 'glass', 
        'paracrystal', or None. 
    form_type: str or None
        type of particle geometry to calculate the form factor. Can be 'sphere'
        or None.
    coordinate_system: string
        default value 'scattering plane' means scattering calculations will be 
        carried out in the basis defined by basis vectors parallel and 
        perpendicular to scattering plane. Variable also accepts value 
        'cartesian' which scattering calculations will be carried out in the 
        basis defined by basis vectors x and y in the lab frame, with z 
        as the direction of propagation.
    phis: None or ndarray
        azimuthal angles
    kd: k * distance, where k = 2*np.pi*n_matrix/wavelen, and distance is the
        distance away from the center of the particle. The far-field solution
        is obtained when distance >> radius. (Quantity, dimensionless)
    
    Returns
    -------
    float (2-tuple):
        parallel and perpendicular components of the differential scattering
        cross section.

    """    
    # calculate form factor    
    if form_type == 'sphere':  
        if coordinate_system == 'scattering plane':
            f_par, f_perp = mie.calc_ang_dist(m, x, angles)
        if coordinate_system == 'cartesian':
            thetas = angles
            f_par, f_perp = mie.diff_scat_intensity_complex_medium(m, x, 
                                                thetas, kd, 
                                                coordinate_system='cartesian',
                                                phis=phis)
    elif form_type is None:
        f_par = 1
        f_perp = 1
    else:
        raise ValueError('form factor type not recognized!')

    # calculate structure factor
    qd = 4*np.array(np.abs(x)).max()*np.sin(angles/2)  #TODO: should it be x.real or x.abs?

    if isinstance(structure_type, dict):
        if structure_type['name'] == 'paracrystal':
            s = structure.factor_para(qd, volume_fraction, 
                                      sigma = structure_type['sigma'])
        else:
            raise ValueError('structure factor type not recognized!')
            
    elif isinstance(structure_type, str):
        if structure_type == 'glass':    
            s = structure.factor_py(qd, volume_fraction)
        elif structure_type == 'paracrystal':
            s = structure.factor_para(qd)
        else: 
            raise ValueError('structure factor type not recognized!')
            
    elif structure_type is None:
        s = 1
    else:
        raise ValueError('structure factor type not recognized!')
        
    scat_par = s * f_par
    scat_perp = s * f_perp

    return scat_par, scat_perp

def _integrate_cross_section(cross_section, factor, angles, 
                             azi_angle_range = 2*np.pi):
    """
    Integrate differential cross-section (multiplied by factor) over angles
    using trapezoid rule
    """    
    # integrand
    integrand = cross_section * factor * np.sin(angles)
    
    # np.trapz does not preserve units, so need to state explicitly that we are
    # in the same units as the integrand
    integral = np.trapz(integrand, x=angles) * integrand.units
    
    # multiply by 2*pi to account for integral over phi
    sigma = azi_angle_range * integral
    #sigma = 2 * np.pi * integral

    return sigma


@ureg.check('[]', '[]', '[]')
def fresnel_reflection(n1, n2, incident_angle):
    """
    Calculates Fresnel coefficients for the reflected intensity of parallel
    (p) and perpendicular (s) polarized light incident on a boundary between
    two dielectric, nonmagnetic materials.

    Parameters
    ----------
    n1: structcol.Quantity [dimensionless]
        refractive index of the first medium along the direction of propagation
    n2: structcol.Quantity [dimensionless]
        refractive index of the second medium along the direction of propagation
    incident_angle: structcol.Quantity [dimensionless] or ndarray of such
        incident angle, measured from the normal (specify degrees or radians by
        using the appropriate units in Quantity())

    Returns
    -------
    (float, float) or ndarray(float, float):
        Parallel (p) and perpendicular (s) reflection coefficients for the
        intensity
    """
    theta = np.atleast_1d(incident_angle.to('rad').magnitude)
    if np.any(theta > np.pi/2.0):
        raise ValueError('Unphysical angle of incidence.  Angle must be \n'+
                         'less than or equal to 90 degrees with respect to' +
                         'the normal.')
    else:
        r_par = np.zeros(theta.size)
        r_perp = np.zeros(theta.size)
        ms = (n2/n1)

        if ms < 1:
            # handle case of total internal reflection; this code is written
            # using index arrays so that theta can be input as an array
            tir_vals = theta >= np.arcsin(ms)
            good_vals = ~tir_vals   # ~ means logical negation
            r_perp[tir_vals] = 1
            r_par[tir_vals] = 1
        else:
            good_vals = np.ones(theta.size, dtype=bool)

        # see, e.g., http://www.ece.rutgers.edu/~orfanidi/ewa/ch07.pdf
        # equations 7.4.2 for fresnel coefficients in terms of the incident
        # angle only
        # take the absolute value inside the square root because sometimes
        # this value is very close to 0 and negative due to numerical precision
        root = np.sqrt(np.abs(n2**2 - (n1 * np.sin(theta[good_vals]))**2))
        costheta = np.cos(theta[good_vals])
        r_par[good_vals] = (np.abs((n1*root - n2**2 * costheta)/ \
                                   (n1*root + n2**2 * costheta)))**2
        r_perp[good_vals] = (np.abs((n1*costheta - root) / \
                                    (n1*costheta + root)))**2
        
    return np.squeeze(r_par), np.squeeze(r_perp)

@ureg.check('[]', '[]', '[]')
def fresnel_transmission(index1, index2, incident_angle):
    """
    Calculates Fresnel coefficients for the transmitted intensity of parallel
    (p) and perpendicular (s) polarized light incident on a boundary between
    two dielectric, nonmagnetic materials.

    Parameters
    ----------
    n1: structcol.Quantity [dimensionless]
        refractive index of the first medium along the direction of propagation
    n2: structcol.Quantity [dimensionless]
        refractive index of the second medium along the direction of propagation
    incident_angle: structcol.Quantity [dimensionless] or ndarray of such
        incident angle, measured from the normal (specify degrees or radians by
        using the appropriate units in Quantity())

    Returns
    -------
    (float, float) or ndarray(float, float):
        Parallel (p) and perpendicular (s) transmission coefficients for the
        intensity
    """
    r_par, r_perp = fresnel_reflection(index1, index2, incident_angle)
    return 1.0-r_par, 1.0-r_perp

def _number_density(volume_fraction, radius):
    return 3.0 * volume_fraction / (4.0 * np.pi * radius**3)

