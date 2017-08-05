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
from . import mie, structure, index_ratio, size_parameter

@ureg.check('[]', '[]', '[]', '[length]', '[length]', '[]')
def reflection(n_particle, n_matrix, n_medium, wavelen, radius, volume_fraction,
               thickness=None,
               theta_min=Quantity('90 deg'),
               theta_max=Quantity('180 deg'),
               phi_min=Quantity('0 deg'),
               phi_max=Quantity('360 deg'),
               incident_angle=Quantity('0 deg'),
               num_angles = 200,
               small_angle=Quantity('5 deg'),
               structure_type='glass',
               form_type = 'sphere'):
    """
    Calculate fraction of light reflected from an amorphous colloidal
    suspension (a "photonic glass").

    Parameters
    ----------
    n_particle: structcol.Quantity [dimensionless]
        refractive index of particles or voids at wavelength=wavelen
    n_matrix: structcol.Quantity [dimensionless]
        refractive index of the matrix surrounding the particles (at wavelen)
    n_medium: structcol.Quantity [dimensionless]
        refractive index of the medium surrounding the sample.  This is
        usually air or vacuum
    wavelen: structcol.Quantity [length]
        wavelength of light in the medium (which is usually air or vacuum)
    radius: structcol.Quantity [length]
        radius of particles or voids
    volume_fraction: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix
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
    Returns
    -------
    float (5-tuple):
        fraction of light reflected from sample for unpolarized light, parallel
        polarization, and perpendicular polarization;
        asymmetry parameter and transport length for unpolarized light

    Notes
    -----
    Uses eqs. 5 and 6 from [1]_. As described in the reference, the function
    uses the Maxwell-Garnett effective refractive index of the sample to
    calculate the scattering from individual spheres within the matrix. It also
    uses the effective refractive index to calculate the Fresnel coefficients
    at the boundary between the medium (referred to as "air" in the reference)
    and the sample.

    References
    ----------
    [1] Magkiriadou, S., Park, J.-G., Kim, Y.-S., and Manoharan, V. N. “Absence
    of Red Structural Color in Photonic Glasses, Bird Feathers, and Certain
    Beetles” Physical Review E 90, no. 6 (2014): 62302.
    doi:10.1103/PhysRevE.90.062302
    """

    # use Maxwell-Garnett formula to calculate effective index of
    # particle-matrix composite
    n_sample = ri.n_eff(n_particle, n_matrix, volume_fraction)
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
    sin_alpha_sample = np.sin(np.pi - theta_min) * n_medium/n_sample
    if sin_alpha_sample >= 1:
        # in this case, theta_min and the ratio of n_medium/n_sample are
        # sufficiently large so that all the scattering from 90-180 degrees
        # exits into the range of angles captured by the detector
        theta_min_refracted = np.pi/2.0
    else:
        theta_min_refracted = np.pi - np.arcsin(sin_alpha_sample)

    # integrate form_factor*structure_factor*transmission
    # coefficient*sin(theta) over angles to get sigma_detected (eq 5)
    angles = Quantity(np.linspace(theta_min_refracted, theta_max, num_angles),
                      'rad')
    azi_angle_range = Quantity(phi_max-phi_min,'rad')
    diff_cs = differential_cross_section(m, x, angles, volume_fraction, 
                                         structure_type, form_type)
    transmission = fresnel_transmission(n_sample, n_medium, np.pi-angles)
    sigma_detected_par = _integrate_cross_section(diff_cs[0],
                                                  transmission[0]/k**2, angles, azi_angle_range)
    sigma_detected_perp = _integrate_cross_section(diff_cs[1],
                                                  transmission[1]/k**2, angles, azi_angle_range)
    sigma_detected = (sigma_detected_par + sigma_detected_perp)/2.0

    # now integrate from 0 to 180 degrees to get total cross-section.
    angles = Quantity(np.linspace(0.0+small_angle, np.pi, num_angles), 'rad')
    # Fresnel coefficients do not appear in this integral since we're using the
    # total cross-section to account for the attenuation in intensity as light
    # propagates through the sample
    diff_cs = differential_cross_section(m, x, angles, volume_fraction, 
                                         structure_type, form_type)
    sigma_total_par = _integrate_cross_section(diff_cs[0], 1.0/k**2, angles, azi_angle_range)
    sigma_total_perp = _integrate_cross_section(diff_cs[1], 1.0/k**2, angles, azi_angle_range)
    sigma_total = (sigma_total_par + sigma_total_perp)/2.0

    # calculate asymmetry parameter using integral from 0 to 180 degrees
    asymmetry_par = _integrate_cross_section(diff_cs[0], np.cos(angles)*1.0/k**2,
                                             angles, azi_angle_range)
    asymmetry_perp = _integrate_cross_section(diff_cs[1], np.cos(angles)*1.0/k**2,
                                              angles, azi_angle_range)
    # calculate for unpolarized light
    asymmetry_parameter = (asymmetry_par + asymmetry_perp)/sigma_total/2.0

    # now eq. 6 for the total reflection
    rho = _number_density(volume_fraction, radius)
    if thickness is None:
        # assume semi-infinite sample
        factor = 1.0
    else:
        # use Beer-Lambert law to account for attenuation
        factor = 1.0-np.exp(-rho*sigma_total*thickness)

    # one critical difference from Sofia's original code is that this code
    # calculates the reflected intensity in each polarization channel
    # separately, then averages them. The original code averaged the
    # transmission coefficients for the two polarization channels before
    # integrating. However, we do average the total cross section to normalize
    # the reflection cross-sections (that is, we use sigma_total rather than
    # sigma_total_par or sigma_total_perp).
    reflected_par = t_medium_sample[0] * sigma_detected_par/sigma_total * \
                    factor + r_medium_sample[0]
    reflected_perp = t_medium_sample[1] * sigma_detected_perp/sigma_total * \
                     factor + r_medium_sample[1]

    # and the transport length for unpolarized light
    # (see eq. 5 of Kaplan, Dinsmore, Yodh, Pine, PRE 50(6): 4827, 1994)
    transport_length = 1/(1.0-asymmetry_parameter)/rho/sigma_total

    return (reflected_par + reflected_perp)/2.0, \
        reflected_par, reflected_perp, \
        asymmetry_parameter, transport_length

@ureg.check('[]', '[]', '[]', '[]')
def differential_cross_section(m, x, angles, volume_fraction,
                               structure_type = 'glass', 
                               form_type = 'sphere'):
    """
    Calculate dimensionless differential scattering cross-section for a sphere,
    including contributions from the structure factor. Need to multiply by k**2
    to get the dimensional differential cross section.
    """
    if form_type == 'sphere':   
        form_factor = mie.calc_ang_dist(m, x, angles)
        f_par = form_factor[0]
        f_perp = form_factor[1]
    elif form_type is None:
        f_par = 1
        f_perp = 1
    else:
        raise ValueError('form factor type not recognized!')
        

    qd = 4*x*np.sin(angles/2)
    
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
        root = np.sqrt(n2**2 - (n1 * np.sin(theta[good_vals]))**2)
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

