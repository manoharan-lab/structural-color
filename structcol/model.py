# Copyright 2016, Vinothan N. Manoharan, Sofia Makgiriadou, Victoria Hwang
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
.. moduleauthor :: Victoria Hwang <vhwang@g.harvard.edu>
"""

import numpy as np
from pymie import mie
from scipy.special import factorial
from scipy.integrate import trapezoid
import xarray as xr
import structcol as sc

from . import Quantity
from . import ureg

class Model:
    """Base class for different types of single-scattering models.

    A Model object specifies the arrangement of components with different
    refractive indices and how to calculate the differential scattering
    cross-section of this structure. The Model class isn't used on its own but
    instead functions as a base class that can be subclassed to describe
    different types of arrangments, such as bicontinuous structures or glassy
    arrangments of spheres.

    Attributes
    ----------
    index_medium : `sc.Index` object
        index of refraction of medium around structure

    """
    def __init__(self, index_medium):
        self.index_medium = index_medium

    def differential_cross_section(self, wavelen, angles, distance=None):
        """Calculates differential scattering cross-section as a function of
        wavelength and angle. This method, which depends on the structure, must
        be implemented in derived classes that specify a structure.

        """
        raise NotImplementedError


class FormStructureModel(Model):
    """Class for defining a model in which scattering is calculated from the
    product of form factor and structure factor.

    Attributes
    ----------
    form_factor : Function (or None)
        Function or method used to calculate the form factor as a function of
        wavelength, angles, and the index external to the particle. If None,
        form factor will be set to unity when calculating the differential
        scattering cross section.
    structure_factor : `sc.structure.StructureFactor` object
        Structure factor used in the calculation
    index_external : `sc.Index` object
        Refractive index of the material outside the particles, which is
        needed to calculate the form factor.  Can be an effective index
        (`sc.EffectiveIndex` object).


    """
    def __init__(self, form_factor, structure_factor, index_external,
                 index_medium):
        self.form_factor = form_factor
        self.structure_factor = structure_factor
        self.index_external = index_external
        super().__init__(index_medium)

    def differential_cross_section(self, wavelen, angles,
                                   lengthscale,
                                   kd=None,
                                   cartesian=False,
                                   incident_vector=None,
                                   phis=None):
        """Calculate dimensionless differential scattering cross-section,
        including contributions from the structure factor. Need to multiply by
        1/k**2 to get the dimensional differential cross section.

        Parameters
        ----------
        wavelen : float (structcol.Quantity [length])
            Wavelength of light in vacuum.
        angles : ndarray(structcol.Quantity [dimensionless])
            array of scattering angles. Must be entered as a Quantity to allow
            specifying units (degrees or radians) explicitly
        lengthscale : float (structcol.Quantity [length])
            Length scale to use to calculate the size parameter.  Since the
            FormStructureModel is particle-agnostic, we don't assume this is
            equal to the radius (though it should be set to such for a sphere)
        kd : float
            distance (nondimensionalized by k) at which to integrate the
            differential cross section to get the total cross section. Needed
            only if n_external is complex. Ignored otherwise.
        cartesian : boolean (default False)
            If set to True, calculation will be done in the basis defined by
            basis vectors x and y in the lab frame, with z as the direction of
            propagation. If False (default), calculation will be carried out in
            the basis defined by basis vectors parallel and perpendicular to
            scattering plane.
        incident_vector : tuple (optional, default None)
            vector describing the incident electric field. It is multiplied by
            the amplitude scattering matrix to find the vector scattering
            amplitude. Unless `cartesian` is set, this vector should be in the
            scattering plane basis, where the first element is the parallel
            component and the second element is the perpendicular component. If
            `cartesian` is set to True, this vector should be in the Cartesian
            basis, where the first element is the x-component and the second
            element is the y-component. Note that the vector for unpolarized
            light is the same in either basis, since either way it should be an
            equal mix between the two othogonal polarizations: (1,1). Note that
            if indicent_vector is None, the function assigns a value based on
            the coordinate system. For scattering plane coordinates, the
            assigned value is (1,1) because most scattering plane calculations
            we're interested in involve unpolarized light. For Cartesian
            coordinates, the assigned value is (1,0) because if we are going to
            the trouble to use the cartesian coordinate system, it is usually
            because we want to do calculations using polarization, and these
            calculations are much easier to convert to measured quantities when
            in the cartesian coordinate system.
        phis : ndarray (optional, default None)
            Azimuthal angles. If `cartesian` is set to True, the scattering
            matrix depends on phi, so an `phis` should be provided. In this
            case both `angles` and `phis` should be 2D, as output from
            `np.meshgrid`. In the default scattering plane coordinates
            (`cartesian=False`), `phis` is ignored, since the the scattering
            matrix does not depend on phi.

        Returns
        -------
        float (2-tuple):
            parallel and perpendicular components of the differential
            scattering cross section.

        """
        # calculate form factor
        if self.form_factor is not None:
            f_par, f_perp = self.form_factor(wavelen, angles,
                                             self.index_external, kd=kd,
                                             cartesian=cartesian,
                                             incident_vector=incident_vector,
                                             phis=phis)
        else:
            f_par = 1
            f_perp = 1

        # calculate structure factor
        n_ext = self.index_external(wavelen)
        ql = sc.ql(n_ext, lengthscale, angles)
        s = self.structure_factor(ql).to_numpy()

        scat_par = s * f_par
        scat_perp = s * f_perp

        return scat_par, scat_perp


class HardSpheres(FormStructureModel):
    """Model of scattering from a hard-sphere liquid or glass.

    Models scattering from a hard-sphere liquid or glass using the product of
    the Mie form factor and the Percus-Yevick structure factor for hard
    spheres.

    Attributes
    ----------
    sphere : `sc.Sphere` object
        Particles that make up the structure
    volume_fraction : float
        volume fraction of spheres that make up the structure
    index_matrix : `sc.Index` object
        Index of matrix material between the spheres.  Should be the actual
        index, not an effective index
    maxwell_garnett: boolean (optional, default False)
        If True, the model uses the Maxwell-Garnett formula to calculate the
        effective index. If False (default), the model uses the Bruggeman
        formula, which can be used for multilayer particles.
    ql_cutoff : float (optional)
        ql below which to use approximate solution to structure factor

    """
    def __init__(self, sphere, volume_fraction, index_matrix, index_medium,
                 maxwell_garnett=False, ql_cutoff=None):
        self.sphere = sphere
        if isinstance(volume_fraction, sc.Quantity):
            volume_fraction = volume_fraction.magnitude
        self.volume_fraction = volume_fraction
        self.index_matrix = index_matrix
        self.maxwell_garnett = maxwell_garnett

        # calculate array of volume fractions of each layer in the particle. If
        # particle is not core-shell, volume fraction remains the same
        vf_array = self.sphere.volume_fraction(self.volume_fraction)
        index_list = self.sphere.index_list(self.index_matrix)

        # Calculate effective index of particle-matrix composite
        index_external = sc.EffectiveIndex(index_list, vf_array,
                                           maxwell_garnett =
                                           self.maxwell_garnett)

        if ql_cutoff is None:
            structure_factor = sc.structure.PercusYevick(volume_fraction)
        else:
            structure_factor = sc.structure.PercusYevick(volume_fraction,
                                                         ql_cutoff = ql_cutoff)

        form_factor = self.sphere.form_factor
        super().__init__(form_factor, structure_factor, index_external,
                         index_medium)

    def differential_cross_section(self, wavelen, angles,
                                   kd=None,
                                   cartesian=False,
                                   incident_vector=None,
                                   phis=None):
        """Calculate dimensionless differential scattering cross-section,
        including contributions from the structure factor. Need to multiply by
        1/k**2 to get the dimensional differential cross section.

        """
        # for a sphere we use the radius to calculate size parameter x
        lengthscale = self.sphere.radius_q

        return super().differential_cross_section(wavelen, angles,
                                                  lengthscale,
                                                  kd=kd,
                                                  cartesian=cartesian,
                                                  incident_vector =
                                                  incident_vector,
                                                  phis=None)

class PolydisperseHardSpheres(FormStructureModel):
    """Model for scattering from a polydisperse hard-sphere liquid or glass.

    Can handle one or two species. If two species, each species can have its
    own polydispersity index. Models scattering using the Percus-Yevick
    structure factor for polydisperse hard spheres and Mie theory for the
    scattering amplitude.

    TODO: currently makes Rayleigh-Gans-Debye approximation implicitly.  Should
    be reimplemented to make use of partial structure factors rather than
    measurable structure factors.

    Attributes
    ----------
    sphere_dist : `sc.SphereDistribution` object
        Distribution of spherical particles that make up the structure.
    volume_fraction : float
        total volume fraction of all sphere species
    index_matrix : `sc.Index` object
        Index of matrix material between the spheres.  Should be the actual
        index, not an effective index
    """
    def __init__(self, sphere_dist, volume_fraction, index_matrix,
                 index_medium):
        self.sphere_dist = sphere_dist
        self.volume_fraction = volume_fraction
        self.index_matrix = index_matrix

        # calculate array of volume fractions, assuming that the sphere indices
        # are the same
        sphere = self.sphere_dist.spheres[-1]
        vf_array = sphere.volume_fraction(self.volume_fraction)
        index_list = sphere.index_list(self.index_matrix)

        # Calculate effective index of particle-matrix composite
        index_external = sc.EffectiveIndex(index_list, vf_array,
                                           maxwell_garnett=False)

        structure_factor = sc.structure.Polydisperse(self.volume_fraction,
                                                     self.sphere_dist)
        form_factor = self.sphere_dist.form_factor
        super().__init__(form_factor, structure_factor, index_external,
                         index_medium)

    def differential_cross_section(self, wavelen, angles,
                                   kd=None,
                                   cartesian=False,
                                   incident_vector=None,
                                   phis=None):
        """Calculate the dimensionless differential scattering cross-section
        for polydisperse systems.

        """
        # for a polydisperse system we use the first mean diameter (of the
        # bispecies system) to calculate size parameter x.  This is just a
        # convention.
        lengthscale = self.sphere_dist.spheres[0].radius_q

        return super().differential_cross_section(wavelen, angles,
                                                  lengthscale,
                                                  kd=kd,
                                                  cartesian=cartesian,
                                                  incident_vector =
                                                  incident_vector,
                                                  phis=None)


class Detector:
    """Class to describe far-field detector used in single-scattering
    calculations.

    Attributes
    ----------
    theta_min, theta_max : structcol.Quantity [angle]
        Specifies the angular range over which to integrate the scattered
        signal. The angles are the scattering angles (polar angle, measured
        from the incident light direction) after the light exits into the
        medium. The model will correct for refraction at the interface to
        map this range of exit angles onto the range of scattering angles from
        the particles.
    phi_min, phi_max : structcol.Quantity [angle] (optional)
        Specifies the azimuthal angular range over which to integrate the
        scattered signal. The angles are the azimuthal angles (measured from
        the incident light direction) after the light exits into the medium.
        The model will correct for refraction at the interface to map this
        range of exit angles onto the range of scattering angles from the
        particles.

    """
    @ureg.check(None, "[]", "[]", "[]", "[]")
    def __init__(self,
                 theta_min=sc.Quantity(np.pi/2, 'rad'),
                 theta_max=sc.Quantity(np.pi, 'rad'),
                 phi_min=sc.Quantity(0, 'rad'),
                 phi_max=sc.Quantity(2*np.pi, 'rad')):
        # store in radians and discard units
        self.theta_min = theta_min.to('rad').magnitude
        self.theta_max = theta_max.to('rad').magnitude
        self.phi_min = phi_min.to('rad').magnitude
        self.phi_max = phi_max.to('rad').magnitude


class HemisphericalReflectanceDetector(Detector):
    """A Detector that captures all light scattered into the reflection
    hemisphere.

    Attributes
    ----------
    None

    """
    def __init__(self):
        super().__init__(theta_min=Quantity('90.0 deg'),
                         theta_max=Quantity('180.0 deg'),
                         phi_min=Quantity('0.0 deg'),
                         phi_max=Quantity('360.0 deg'))


@ureg.check(None, None, None, '[length]', '[length]', '[]', None, None, None,
            None, None, None, None, None, None, None, None, None, None)
def reflection(index_particle, index_matrix, index_medium, wavelen, radius,
               volume_fraction,
               radius2=None,
               concentration=None,
               pdi=None,
               thickness=None,
               detector=HemisphericalReflectanceDetector(),
               incident_angle=Quantity('0.0 deg'),
               num_angles=200,
               small_angle=Quantity('1.0 deg'),
               structure_type='glass',
               form_type='sphere',
               maxwell_garnett=False,
               structure_s_data=None,
               structure_qd_data=None):
    """
    Calculate fraction of light reflected from an amorphous colloidal
    suspension (a "photonic glass").

    Parameters
    ----------
    index_particle: `sc.Index` object or list of such
        refractive index of particles or voids. In case of core-shell
        particles, define indices from the innermost to the outermost layer.
    index_matrix: `sc.Index` object
        refractive index of the matrix surrounding the particles
    index_medium: `sc.Index` object
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
    radius2 : float (structcol.Quantity [length])
        Mean radius of secondary scatterer. Specify only if the system is
        binary, meaning that there are two mean particle radii (for example,
        one small and one large).
    concentration : 2-element array (structcol.Quantity [dimensionless])
        Concentration of each scatterer if the system is binary. For
        polydisperse monospecies systems, specify the concentration as
        [0., 1.]. The concentrations must add up to 1.
    pdi : 2-element array (structcol.Quantity [dimensionless])
        Polydispersity index of each scatterer if the system is polydisperse.
        For polydisperse monospecies systems, specify the pdi as a 2-element
        array with repeating values (for example, [0.01, 0.01]).
    thickness: structcol.Quantity [length] (optional)
        thickness of photonic glass.  If unspecified, assumed to be infinite
    detector : `sc.model.Detector` object (optional)
        Specifies the angles at which the reflectance is measured.  If
        unspecified, a hemispherical reflectance detector is used
    incident_angle: structcol.Quantity [angle] (optional)
        incident angle, measured from the normal
    num_angles: integer
        number of angles to use in evaluation of the cross-section, which is
        done by numerical integration (fixed quadrature). The default value
        (200) seems to do OK for 280-nm-diameter spheres, but could use more
        testing.
    small_angle: structcol.Quantity [dimensionless] (optional)
        Because of numerical instabilities, some structure factor calculations
        may return nan when evaluated at q=0. This doesn't matter when
        calculating the scattering cross section because sin(0) = 0, so the
        contribution of the differential scattering cross section at theta = 0
        to the total cross section is zero. However, errors or warnings may
        appear due to the nan calculation. To prevent such errors or warnings,
        set small_angle equal to some minimum angle at which to calculate the
        structure factor (and, by extension, the total cross-section). The
        default value is chosen to give good agreement with Mie theory for a
        single sphere, but it may not be reasonable for all calculations.
    structure_type: string, dictionary, or None (optional)
        Can be string specifying structure type. Current options are "glass",
        "polydisperse", "data", or None. Can also set to None in order to only
        visualize effect of form factor on reflectance spectrum. If set to
        'data', you must also provide structure_s_data and structure_qd_data.
    form_type: string or None (optional)
        String specifying form factor type. Currently, 'sphere' or
        'polydisperse' are the options. Can also set to None in order to only
        visualize the effect of structure factor on reflectance spectrum.
    maxwell_garnett: boolean
        If true, the model uses Maxwell-Garnett's effective index for the
        sample. In that case, the user must specify one refractive index for
        the particle and one for the matrix. If false, the model uses
        Bruggeman's formula, which can be used for multilayer particles.
    structure_s_data: None or 1d array
        if structure_type is 'data', the structure factor data must be provided
        here in the form of a one dimensional array
    structure_qd_data: None or 1d array
        if structure_type is 'data', the qd data must be provided here in the
        form of a one dimensional array

    Returns
    -------
    float (5-tuple):
        fraction of light reflected from sample for unpolarized light, parallel
        polarization, and perpendicular polarization; asymmetry parameter and
        transport length for unpolarized light

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

    # make sure we're working in the same units
    wavelen = wavelen.to_preferred()
    radius = radius.to_preferred()

    particle = sc.Sphere(index_particle, radius)
    n_particle = particle.n(wavelen)
    n_matrix = index_matrix(wavelen)
    n_medium = index_medium(wavelen)

    # radius and radius2 should be in the same units (for polydisperse samples)
    if radius2 is not None:
        radius2 = radius2.to(radius.units)

    # construct Sphere or SphereDistribution objects
    # TODO: remove after further refactoring.
    if isinstance(concentration, sc.Quantity):
        concentration = concentration.magnitude
    if pdi is not None:
        if (np.ndim(radius) != 0) or (np.ndim(radius2) != 0):
            raise ValueError("cannot handle polydispersity for "
                             "layered spheres")
        if radius2 is not None:
            sphere1 = sc.Sphere(index_particle, radius)
            sphere2 = sc.Sphere(index_particle, radius2)
            dist = sc.SphereDistribution([sphere1, sphere2], concentration,
                                         pdi)
        else:
            radius2 = radius
            sphere1 = sc.Sphere(index_particle, radius)
            dist = sc.SphereDistribution(sphere1, concentration, pdi)
        # calculate number density
        rho = dist.number_density(volume_fraction)
    else:
        radius2 = radius
        rho = particle.number_density(volume_fraction)
        print(rho)

    # define the mean diameters in case the system is polydisperse
    mean_diameters = Quantity(np.hstack([2*radius.magnitude,
                                        2*radius2.magnitude]),
                                    radius.units)

    # check that the number of indices and radii is the same
    if len(np.atleast_1d(n_particle)) != len(np.atleast_1d(radius)):
        raise ValueError('Arrays of indices and radii must be the same length')

    # calculate array of volume fractions of each layer in the particle. If
    # particle is not core-shell, volume fraction remains the same
    vf_array = particle.volume_fraction(volume_fraction)
    index_list = particle.index_list(index_matrix)

    # use Bruggeman formula to calculate effective index of
    # particle-matrix composite
    n_sample = sc.index.effective_index(index_list, vf_array, wavelen,
                                        maxwell_garnett=maxwell_garnett)

    if len(np.atleast_1d(radius)) > 1:
        # particle is multilayer
        m = sc.index.ratio(n_particle, n_sample).flatten()
        x = sc.size_parameter(n_sample, radius).to_numpy().flatten()
    else:
        m = sc.index.ratio(n_particle, n_sample)
        x = sc.size_parameter(n_sample, radius)

    k = sc.wavevector(n_sample)
    # calculate transmission and reflection coefficients at first interface
    # between medium and sample
    # (TODO: include correction for reflection off the back interface of the
    # sample)
    t_medium_sample = fresnel_transmission(n_medium.to_numpy().squeeze(),
                                           n_sample.to_numpy().squeeze(),
                                           incident_angle)
    r_medium_sample = fresnel_reflection(n_medium.to_numpy().squeeze(),
                                         n_sample.to_numpy().squeeze(),
                                         incident_angle)

    theta_min = detector.theta_min
    theta_max = detector.theta_max
    phi_min = detector.phi_min
    phi_max = detector.phi_max
    small_angle = small_angle.to('rad').magnitude
    # calculate the min theta, taking into account refraction at the interface
    # between the medium and the sample. This is the scattering angle at which
    # light exits into the medium at (180-theta_min) degrees from the normal.
    # (Snell's law: n_medium sin(alpha_medium) = n_sample sin(alpha_sample)
    # where alpha = pi - theta)
    # TODO: use n_sample.real or abs(n_sample)?
    sin_alpha_sample_theta_min = (np.sin(np.pi-theta_min)
                                  * n_medium.to_numpy().squeeze()
                                  / np.abs(n_sample.to_numpy().squeeze()))
    sin_alpha_sample_theta_max = (np.sin(np.pi-theta_max)
                                  * n_medium.to_numpy().squeeze()
                                  / np.abs(n_sample.to_numpy().squeeze()))

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
    angles_tot = Quantity(np.linspace(0.0 + small_angle, np.pi, num_angles),
                          'rad')
    azi_angle_range = Quantity(phi_max - phi_min,'rad')
    azi_angle_range_tot = Quantity(2 * np.pi, 'rad')

    transmission = fresnel_transmission(n_sample.to_numpy().squeeze(),
                                        n_medium.to_numpy().squeeze(),
                                        np.pi-angles)

    # calculate the absorption cross section
    if np.abs(n_sample.imag) > 0.0:
        # The absorption coefficient can be calculated from the imaginary
        # component of the samples's refractive index
        mu_abs = 4 * np.pi * n_sample.imag.to_numpy().squeeze() / wavelen
        cabs_total = mu_abs / rho
    else:
        cross_sections = mie.calc_cross_sections(m, x,
                            (wavelen/(n_sample.to_numpy().squeeze())))
        cabs_total = cross_sections[2]

    # calculate the differential cross section in the detected range of angles
    # and in the total angles. We calculate it at a distance = radius when
    # there is absorption in the system, making sure that near_fields are False
    # (which is the default). Including near-fields leads to strange effects
    # when the calculation is done not over all angles but only a subset.
    # When there isn't absorption, the distance does not enter the calculation.
    if form_type == 'polydisperse':
        distance = mean_diameters / 2
    else:
        distance = mean_diameters.max() / 2

    if (form_type == "sphere") and (structure_type == "glass"):
        kd = (k*distance).to('')
        model = HardSpheres(particle, volume_fraction, index_matrix,
                            index_medium, maxwell_garnett=maxwell_garnett)
        diff_cs_detected = model.differential_cross_section(wavelen, angles,
                                                            kd=kd)
        diff_cs_total = model.differential_cross_section(wavelen, angles_tot,
                                                         kd=kd)
    elif (form_type == "polydisperse") and (structure_type == "polydisperse"):
        kd = (k*distance).to('')
        if len(mean_diameters) == 2:
            sphere1 = sc.Sphere(index_particle, mean_diameters[0]/2)
            sphere2 = sc.Sphere(index_particle, mean_diameters[1]/2)
            dist = sc.SphereDistribution([sphere1, sphere2], concentration,
                                         pdi)
        else:
            sphere1 = sc.Sphere(index_particle, mean_diameters[0]/2)
            dist = sc.SphereDistribution(sphere1, concentration, pdi)
        model = PolydisperseHardSpheres(dist, volume_fraction, index_matrix,
                                        index_medium)
        diff_cs_detected = model.differential_cross_section(wavelen, angles,
                                                            kd=kd)
        diff_cs_total = model.differential_cross_section(wavelen, angles_tot,
                                                         kd=kd)
    else:
        diff_cs_detected = differential_cross_section(m, x, angles,
                                        volume_fraction,
                                        structure_type=structure_type,
                                        form_type=form_type,
                                        diameters=mean_diameters,
                                        concentration=concentration,
                                        pdi=pdi, wavelen=wavelen,
                                        n_matrix=n_sample, k=k,
                                        distance=distance,
                                        structure_s_data=structure_s_data,
                                        structure_qd_data=structure_qd_data)

        diff_cs_total = differential_cross_section(m, x, angles_tot,
                                        volume_fraction,
                                        structure_type=structure_type,
                                        form_type=form_type,
                                        diameters=mean_diameters,
                                        concentration=concentration,
                                        pdi=pdi, wavelen=wavelen,
                                        n_matrix=n_sample, k=k,
                                        distance=distance,
                                        structure_s_data=structure_s_data,
                                        structure_qd_data=structure_qd_data)

    # integrate the differential cross sections to get the total cross section
    if np.abs(n_sample.imag) > 0.:
        if form_type == 'polydisperse' and len(concentration) > 1:
            # When the system is binary and absorbing, we integrate the
            # polydisperse differential cross section at the surface of each
            # component (meaning at a distance of each mean radius). Then we
            # do a number average the total cross sections.
            cscat1 = mie.integrate_intensity_complex_medium(
                                        diff_cs_detected[0] * transmission[0],
                                        diff_cs_detected[1] * transmission[1],
                                        distance[0], angles, k,
                                        phi_min=Quantity(phi_min, 'rad'),
                                        phi_max=Quantity(phi_max, 'rad'))
            cscat2 = mie.integrate_intensity_complex_medium(
                                        diff_cs_detected[0] * transmission[0],
                                        diff_cs_detected[1] * transmission[1],
                                        distance[1], angles, k,
                                        phi_min=Quantity(phi_min, 'rad'),
                                        phi_max=Quantity(phi_max, 'rad'))
            cscat_detected1 = cscat1[0]
            cscat_detected_par1 = cscat1[1]
            cscat_detected_perp1 = cscat1[2]
            cscat_detected2 = cscat2[0]
            cscat_detected_par2 = cscat2[1]
            cscat_detected_perp2 = cscat2[2]

            cscat_detected = (cscat_detected1 * concentration[0]
                              + cscat_detected2 * concentration[1])
            cscat_detected_par = (cscat_detected_par1 * concentration[0]
                                  + cscat_detected_par2 * concentration[1])
            cscat_detected_perp = (cscat_detected_perp1 * concentration[0]
                                   + cscat_detected_perp2 * concentration[1])

            cscat_total1 = mie.integrate_intensity_complex_medium(
                                                        diff_cs_total[0],
                                                        diff_cs_total[1],
                                                        distance[0],
                                                        angles_tot, k)[0]
            cscat_total2 = mie.integrate_intensity_complex_medium(
                                                        diff_cs_total[0],
                                                        diff_cs_total[1],
                                                        distance[1],
                                                        angles_tot, k)[0]
            cscat_total = (cscat_total1 * concentration[0] +
                           cscat_total2 * concentration[1])

            # Similarly, we calculate the asymmetry parameter integrating at
            # the surface of each mean component of the binary mixture and
            # then average
            factor = np.cos(angles_tot)
            asymmetry_unpolarized1 = mie.integrate_intensity_complex_medium(
                                            diff_cs_total[0] * factor,
                                            diff_cs_total[1] * factor,
                                            distance[0],
                                            angles_tot, k)[0]
            asymmetry_unpolarized2 = mie.integrate_intensity_complex_medium(
                                            diff_cs_total[0] * factor,
                                            diff_cs_total[1] * factor,
                                            distance[1],
                                            angles_tot, k)[0]
            asymmetry_unpolarized = (asymmetry_unpolarized1 * concentration[0]
                                     + asymmetry_unpolarized2
                                       * concentration[1])

        else:
            # We calculate the detected and total cross sections using the full
            # Mie solutions with the asymptotic form of the spherical Hankel
            # functions (see mie.diff_scat_intensity_complex_medium()). By
            # doing so, we ignore near-field effects but still include the
            # complex k into the Mie solutions. Since the cross sections then
            # decay over distance, we integrate them at the surface of the
            # particle. The decay through the sample is accounted for later
            # with Beer- Lambert's law.
            cscat = mie.integrate_intensity_complex_medium(
                                        diff_cs_detected[0]*transmission[0],
                                        diff_cs_detected[1]*transmission[1],
                                        distance, angles, k,
                                        phi_min=Quantity(phi_min, 'rad'),
                                        phi_max=Quantity(phi_max, 'rad'))
            cscat_detected = cscat[0]
            cscat_detected_par = cscat[1]
            cscat_detected_perp = cscat[2]

            cscat_total = mie.integrate_intensity_complex_medium(
                                        diff_cs_total[0],
                                        diff_cs_total[1],
                                        distance, angles_tot, k)[0]
            asym_factor = np.cos(angles_tot)
            asymmetry_unpolarized = mie.integrate_intensity_complex_medium(
                                        diff_cs_total[0]*asym_factor,
                                        diff_cs_total[1]*asym_factor,
                                        distance,
                                        angles_tot, k)[0]

        asymmetry_parameter = asymmetry_unpolarized/cscat_total

        # Calculate the transport length for unpolarized light (see eq. 5 of
        # Kaplan, Dinsmore, Yodh, Pine, PRE 50(6): 4827, 1994)
        # TODO is this cscat or cext_tot?
        transport_length = 1/(1.0-asymmetry_parameter)/rho/cscat_total

    # if there is no absorption in the system
    else:
        cscat_detected_par = _integrate_cross_section(diff_cs_detected[0],
                                                transmission[0]/np.abs(k)**2,
                                                angles, azi_angle_range)
        cscat_detected_perp = _integrate_cross_section(diff_cs_detected[1],
                                                transmission[1]/np.abs(k)**2,
                                                angles, azi_angle_range)
        cscat_detected = (cscat_detected_par + cscat_detected_perp)/2.0

        cscat_total_par = _integrate_cross_section(diff_cs_total[0],
                                                   1.0/np.abs(k)**2,
                                                   angles_tot,
                                                   azi_angle_range_tot)
        cscat_total_perp = _integrate_cross_section(diff_cs_total[1],
                                                    1.0/np.abs(k)**2,
                                                    angles_tot,
                                                    azi_angle_range_tot)
        cscat_total = (cscat_total_par + cscat_total_perp)/2.0

        asymmetry_par = _integrate_cross_section(diff_cs_total[0],
                                        np.cos(angles_tot)*1.0/np.abs(k)**2,
                                        angles_tot,
                                        azi_angle_range_tot)
        asymmetry_perp = _integrate_cross_section(diff_cs_total[1],
                                        np.cos(angles_tot)*1.0/np.abs(k)**2,
                                        angles_tot,
                                        azi_angle_range_tot)
        asymmetry_parameter = (asymmetry_par + asymmetry_perp)/cscat_total/2.0

        # calculate transport cscat
        # not currently returned, but could be useful in the future
        transport_cscat_par = _integrate_cross_section(
                                    diff_cs_total[0],
                                    (1-np.cos(angles_tot))*1.0/np.abs(k)**2,
                                    angles_tot, azi_angle_range_tot)
        transport_cscat_perp = _integrate_cross_section(
                                    diff_cs_total[1],
                                    (1-np.cos(angles_tot))*1.0/np.abs(k)**2,
                                    angles_tot, azi_angle_range_tot)
        transport_cscat = (transport_cscat_par + transport_cscat_perp)/2

        # Calculate the transport length for unpolarized light (see eq. 5 of
        # Kaplan, Dinsmore, Yodh, Pine, PRE 50(6): 4827, 1994)
        # TODO is this cscat or cext_tot?
        transport_length = 1/(1.0-asymmetry_parameter)/rho/cscat_total

    cext_total = cscat_total.to('um**2') + cabs_total.to('um**2')

    # now eq. 6 for the total reflection
    if thickness is None:
        # assume semi-infinite sample
        factor = 1.0
    else:
        # use Beer-Lambert law to account for attenuation
        factor = ((1.0 - np.exp(-rho*cext_total*thickness))
                  * cscat_total/cext_total).to('')

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

    return reflectance, reflected_par, reflected_perp, asymmetry_parameter, \
           transport_length


@ureg.check(None, None, '[]', '[]', None, None, None, None, None,None, None,
            None, None, None, None, None, None, None)
def differential_cross_section(m, x, angles, volume_fraction,
                               structure_type = 'glass',
                               form_type = 'sphere',
                               diameters=None,
                               concentration=None,
                               pdi=None,
                               wavelen=None,
                               n_matrix=None,
                               k=None,
                               distance=None,
                               coordinate_system='scattering plane',
                               incident_vector=None,
                               phis=None,
                               structure_s_data=None,
                               structure_qd_data=None):
    """
    Calculate dimensionless differential scattering cross-section for a sphere,
    including contributions from the structure factor. Need to multiply by
    1/k**2 to get the dimensional differential cross section.

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
        'polydisperse', 'data', or None. If structure_type=='data', you must
        also provide structure_s_data and structure_qd_data.
    form_type: str or None
        type of particle geometry to calculate the form factor. Can be 'sphere'
        or None.
    diameters: ndarray(structcol.Quantity [length])
        Only for polydisperse systems. Mean diameters of each species of
        particles (can be one for a monospecies or two for bispecies).
    concentration : 2-element array (structcol.Quantity [dimensionless])
        'Number' concentration of each scatterer if the system is binary. For
        ex, a system composed of 90 A particles and 10 B particles would have
        c = [0.9, 0.1]. For polydisperse monospecies systems, specify the
        concentration as [1.0, 0.0].
    pdi : 2-element array (structcol.Quantity [dimensionless])
        Polydispersity index of each scatterer if the system is polydisperse.
        For polydisperse monospecies systems, specify the pdi as a 2-element
        array with repeating values (for example, [0.01, 0.01]).
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    n_matrix : float (structcol.Quantity [dimensionless])
        Refractive index of the matrix (will be the index of the sample when
        running the model).
    k: float (sc.Quantity [1/length])
        k vector. k = 2*pi*n_sample / wavelength
    distance: float (sc.Quantity [length])
        distance at which we perform the integration of the differential cross
        section to get the total cross section.
    coordinate_system: string
        default value 'scattering plane' means scattering calculations will be
        carried out in the basis defined by basis vectors parallel and
        perpendicular to scattering plane. Variable also accepts value
        'cartesian' which scattering calculations will be carried out in the
        basis defined by basis vectors x and y in the lab frame, with z
        as the direction of propagation.
    incident_vector: None or tuple
        vector describing the incident electric field. It is multiplied by the
        amplitude scattering matrix to find the vector scattering amplitude. If
        coordinate_system is 'scattering plane', then this vector should be in
        the 'scattering plane' basis, where the first element is the parallel
        component and the second element is the perpendicular component. If
        coordinate_system is 'cartesian', then this vector should be in the
        'cartesian' basis, where the first element is the x-component and the
        second element is the y-component. Note that the vector for unpolarized
        light is the same in either basis, since either way it should be an
        equal mix between the two othogonal polarizations: (1,1). Note that if
        indicent_vector is None, the function assigns a value based on the
        coordinate system. For 'scattering plane', the assigned value is (1,1)
        because most scattering plane calculations we're interested in involve
        unpolarized light. For 'cartesian', the assigned value is (1,0) because
        if we are going to the trouble to use the cartesian coordinate system,
        it is usually because we want to do calculations using polarization,
        and these calculations are much easier to convert to measured
        quantities when in the cartesian coordinate system.
    phis: None or ndarray
        azimuthal angles
    structure_s_data: None or 1d array
        if structure_type is 'data', the structure factor data must be provided
        here in the form of a one dimensional array
    structure_qd_data: None or 1d array
        if structure_type is 'data', the qd data must be provided here in the
        form of a one dimensional array

    Returns
    -------
    float (2-tuple):
        parallel and perpendicular components of the differential scattering
        cross section.
    """
    if isinstance(k, Quantity):
        k = k.to_preferred()
    if isinstance(distance, Quantity):
        distance = distance.to_preferred()
    if isinstance(volume_fraction, Quantity):
        volume_fraction = volume_fraction.magnitude


    if (form_type == "polydisperse") or (structure_type == "polydisperse"):
        # construct a SphereDistribution object and use its form_factor and/or
        # structure_factor methods. (this is some temporary scaffolding to
        # ensure that the SphereDistribution.formfactor() method is tested by
        # existing tests that call model.reflection(); will be removed after
        # further refactoring).
        if isinstance(concentration, sc.Quantity):
            concentration = concentration.magnitude
        n_particle = (m * n_matrix.to_numpy())
        if len(diameters) > 2:
            raise ValueError("cannot handle polydispersity for "
                             "layered spheres")
        if np.ndim(n_particle) >= 1:
            index_particle = sc.Index.constant(n_particle[0])
        else:
            index_particle = sc.Index.constant(n_particle)
        index_external = sc.Index.constant(n_matrix.to_numpy())
        if len(diameters) >= 2:
            sphere1 = sc.Sphere(index_particle, diameters[0]/2)
            sphere2 = sc.Sphere(index_particle, diameters[1]/2)
            dist = sc.SphereDistribution([sphere1, sphere2], concentration,
                                         pdi)
        else:
            sphere1 = sc.Sphere(index_particle, diameters[0]/2)
            dist = sc.SphereDistribution(sphere1, concentration, pdi)

    if isinstance(x, xr.DataArray):
        x = x.to_numpy()

    # calculate form factor
    if form_type == 'sphere':
        if k is not None and (np.abs(k.imag.magnitude) > 0.
                              or coordinate_system == 'cartesian'):
            if distance is None:
                raise ValueError('must specify distance for absorbing systems')
            form_factor = mie.diff_scat_intensity_complex_medium(m, x, angles,
                                        k*distance,
                                        coordinate_system=coordinate_system,
                                        incident_vector=incident_vector,
                                        phis=phis)
        else:
            form_factor = mie.calc_ang_dist(m, x, angles)
            #if k is not None:
            #    form_factor = form_factor/k**2

        f_par = form_factor[0]
        f_perp = form_factor[1]

    elif form_type == 'polydisperse':
        if (diameters is None or concentration is None
            or pdi is None or wavelen is None or n_matrix is None):
            raise ValueError('must specify diameters, concentration, pdi, '
                             'wavelength, and n_matrix for polydisperse '
                             'systems')

        if coordinate_system == 'cartesian':
            cartesian = True
        else:
            cartesian = False

        if k is not None:
            kd = k*distance
        else:
            kd = None

        form_factor = dist.form_factor(wavelen, angles, index_external,
                                       kd=kd, cartesian=cartesian,
                                       incident_vector=incident_vector,
                                       phis=phis)

        f_par = form_factor[0]
        f_perp = form_factor[1]

    elif form_type is None:
        f_par = 1
        f_perp = 1
    else:
        raise ValueError('form factor type not recognized!')

    # calculate structure factor
    # TODO: should it be x.real or x.abs?
    qd = 4*np.array(np.abs(x)).max()*np.sin(angles/2)
    if isinstance(qd, Quantity):
        qd = qd.magnitude

    if isinstance(structure_type, str):
        if structure_type == 'glass':
            structure_factor = sc.structure.PercusYevick(volume_fraction)
            if len(qd.shape) == 2:
                s = structure_factor(qd[:,0]).to_numpy()
            else:
                s = structure_factor(qd).to_numpy()

        elif structure_type == 'data':
            structure_factor = sc.structure.Interpolated(structure_s_data,
                                                         structure_qd_data)
            s = structure_factor(qd)

        elif structure_type == 'polydisperse':
            if diameters is None or concentration is None or pdi is None:
                raise ValueError('must specify diameters, concentration, '
                                 'and pdi for polydisperse systems')
            if len(np.atleast_1d(m)) > 1:
                raise ValueError('cannot handle polydispersity in '
                                 'core-shell particles')

            structure_factor = sc.structure.Polydisperse(volume_fraction, dist)
            s = structure_factor(qd).to_numpy()
        else:
            raise ValueError('structure factor type not recognized!')

    elif structure_type is None:
        s = 1
    else:
        raise ValueError('structure factor type not recognized!')

    if len(qd.shape) == 2:
        scat_par = s[:, np.newaxis] * f_par
        scat_perp = s[:, np.newaxis] * f_perp
    else:
        scat_par = s * f_par
        scat_perp = s * f_perp

    return scat_par, scat_perp

def absorption_cross_section(form_type, m, diameters, n_matrix, x,
                             wavelen, n_particle, concentration=None,
                             pdi=None):         # pragma: no cover
    """
    Calculate the absorption cross section.
    Note: this function is currently NOT used anywhere in this package.

    Parameters
    ----------
    form_type: str or None
        type of particle geometry to calculate the form factor. Can be 'sphere'
        or None.
    m: float
        complex particle relative refractive index, n_particle/n_sample
    diameters: ndarray(structcol.Quantity [length])
        Only for polydisperse systems. Mean diameters of each species of
        particles (can be one for a monospecies or two for bispecies).
    n_matrix : float (structcol.Quantity [dimensionless])
        Refractive index of the matrix (will be the index of the sample when
        running the model).
    x: float
        size parameter
    wavelen : float (structcol.Quantity [length])
        Wavelength of light in vacuum.
    n_particle: float (structcol.Quantity [dimensionless])
        Refractive index of the scatterer.
    concentration : 2-element array (structcol.Quantity [dimensionless])
        'Number' concentration of each scatterer if the system is binary. For
        ex, a system composed of 90 A particles and 10 B particles would have
        c = [0.9, 0.1]. For polydisperse monospecies systems, specify the
        concentration as [1.0, 0.0].
    pdi : 2-element array (structcol.Quantity [dimensionless])
        Polydispersity index of each scatterer if the system is polydisperse.
        For polydisperse monospecies systems, specify the pdi as a 2-element
        array with repeating values (for example, [0.01, 0.01]).

    Returns
    -------
    float (2-tuple):
        absorption cross section.
    """

    if np.abs(n_matrix.imag) == 0.:
        cabs_total = Quantity(0.0, 'um^2')

    if form_type == 'polydisperse':
        if concentration is None or pdi is None:
            raise ValueError('must specify concentration and pdi '
                             'for absorbing polydisperse systems')
        if len(np.atleast_1d(m)) > 1:
            raise ValueError('cannot handle polydispersity in '
                             'core-shell particles')

        # if the pdi is zero, assume it's very small (we get the same results)
        # because otherwise we get a divide by zero error
        pdi = Quantity(np.atleast_1d(pdi).astype(float), pdi.units)
        np.atleast_1d(pdi)[np.atleast_1d(pdi) < 1e-5] = 1e-5

        # t is a measure of the width of the Schulz distribution, and
        # pdi is the polydispersity index
        t = np.abs(1/(pdi**2)) - 1

        # define the range of diameters of the size distribution
        three_std_dev = 3*diameters/np.sqrt(t+1)
        min_diameter = diameters - three_std_dev
        min_diameter[min_diameter.magnitude < 0] = Quantity(0.0,
                                                            diameters.units)
        max_diameter = diameters + three_std_dev

        cabs_poly = np.empty(len(np.atleast_1d(diameters)))

        # for each mean diameter, calculate the Schulz distribution and
        # the size parameter x_poly
        for d in np.arange(len(np.atleast_1d(diameters))):
            # the diameter range is the range between the min diameter and
            # the max diameter of the Schulz distribution
            diameter_range = np.linspace(np.atleast_1d(min_diameter)[d],
                                         np.atleast_1d(max_diameter)[d], 50)
            distr = size_distribution(diameter_range,
                                      np.atleast_1d(diameters)[d],
                                      np.atleast_1d(t)[d])
            x_poly = sc.size_parameter(n_matrix,
                                       Quantity(diameter_range/2,
                                                diameters.units))

            # for polydisperse mu_abs calculation
            x_scat = sc.size_parameter(n_particle, diameters[d]/2)

            cabs_magn = np.empty(len(diameter_range))

            # for each diameter in the distribution, calculate the detected
            # and the total form factors for absorbing systems
            for s in np.arange(len(diameter_range)):
                # if the system has absorption, use the absorption formula from
                # Mie for polydisperse mu_abs calculation
                nstop = mie._nstop(np.array(x_poly[s]).max())
                coeffs = mie._scatcoeffs(m, x_poly[s], nstop)
                internal_coeffs = mie._internal_coeffs(m, x_poly[s], nstop)

                cabs = mie._cross_sections_complex_medium_fu(coeffs[0],
                                                coeffs[1],
                                                internal_coeffs[0],
                                                internal_coeffs[1],
                                                Quantity(diameter_range[s]/2,
                                                         diameters.units),
                                                n_particle,
                                                n_matrix, x_scat,
                                                x_poly[s],
                                                wavelen)[1]
                cabs_magn[s] = cabs.magnitude
            # integrate and multiply the mu_abs by the concentrations to get
            # the polydisperse mu_abs
            cabs_poly[d] = (trapezoid(cabs_magn*distr, x=diameter_range)
                            * np.atleast_1d(concentration)[d])
        cabs_total = Quantity(np.sum(cabs_poly), cabs.units)

    if form_type is None:
        cabs_total = Quantity(0.0, 'um^2')

    if form_type == 'sphere':
        radius = diameters[0]/2
        # calculate total absorption cross section
        nstop = mie._nstop(np.array(x).max())
        # if the index ratio m is an array with more than 1 element, it's a
        # multilayer particle
        if len(np.atleast_1d(m)) > 1:
            coeffs = mie._scatcoeffs_multi(m, x)
            cabs_total = mie._cross_sections_complex_medium_sudiarta(coeffs[0],
                                                                     coeffs[1],
                                                                     x,
                                                                     radius)[1]
            if cabs_total.magnitude < 0.0:
                cabs_total = 0.0 * cabs_total.units
        else:
            coeffs = mie._scatcoeffs(m, x, nstop)
            internal_coeffs = mie._internal_coeffs(m, x, nstop)
            x_scat = sc.size_parameter(n_particle, radius)

            cabs_total = mie._cross_sections_complex_medium_fu(
                                                        coeffs[0], coeffs[1],
                                                        internal_coeffs[0],
                                                        internal_coeffs[1],
                                                        radius, n_particle,
                                                        n_matrix, x_scat,
                                                        x, wavelen)[1]
    return(cabs_total)


def size_distribution(diameter_range, mean, t):
    """
    Calculate the Schulz distribution for polydisperse systems. When the
    polydispersity is small, the Schulz distribution tends to a Gaussian.

    Parameters
    ----------
    diameter_range: array
        Range of diameters of the distribution.
    mean: 1-element array
        Mean diameter of the distribution.
    t: 1-element array
        'Width' of the distribution. t = (1 - p**2) / p**2, where p is the
        polydispersity index.

    Returns
    -------
    Schulz distribution.

    """
    if isinstance(diameter_range, Quantity):
        diameter_range = diameter_range.magnitude
    if isinstance(diameter_range, Quantity):
        diameter_range = diameter_range.magnitude
    if isinstance(mean, Quantity):
        mean = mean.magnitude
    if isinstance(t, Quantity):
        t = t.magnitude
    if isinstance(t, Quantity):
        t = t.magnitude

    if t <= 100:
        schulz = (((t+1)/mean)**(t+1) * diameter_range**t / factorial(t)
                  * np.exp(-diameter_range/mean*(t+1)))
        norm = trapezoid(schulz, x=diameter_range)
        distr = schulz / norm
    else:
        std_dev = diameter_range / np.sqrt(t+1)
        distr = (np.exp(-(diameter_range - mean)**2 / (2 * std_dev**2))
                 / np.sqrt(2*np.pi*std_dev**2))
        norm = trapezoid(distr, x=diameter_range)
        distr = distr/norm
    return(distr)

def _integrate_cross_section(cross_section, factor, angles,
                             azi_angle_range = 2*np.pi):
    """
    Integrate differential cross-section (multiplied by factor) over angles
    using trapezoid rule
    """
    # integrand
    integrand = cross_section * factor * np.sin(angles)
    # pint does not yet preserve units for scipy.integrate.trapezoid, so we
    # need to state explicitly that we are in the same units as the integrand.
    if isinstance(integrand, Quantity):
        integral = (trapezoid(integrand.magnitude, x=angles.magnitude)
                    * integrand.units)
    else:
        integral = trapezoid(integrand, x=angles.magnitude)
    # multiply by azimuthal angular range to account for integral over phi
    sigma = azi_angle_range * integral

    return sigma


@ureg.check(None, None, '[]')
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
        refractive index of the second medium along the direction of
        propagation
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
    if isinstance(theta, Quantity):
        theta = theta.magnitude
    if isinstance(n1, Quantity):
        n1 = n1.magnitude
    if isinstance(n2, Quantity):
        n2 = n2.magnitude

    if np.any(theta > np.pi/2.0):
        raise ValueError('Unphysical angle of incidence.  Angle must be \n'+
                         'less than or equal to 90 degrees with respect to' +
                         'the normal.')
    else:
        r_par = np.zeros(theta.size)
        r_perp = np.zeros(theta.size)
        ms = (n2/n1)

        if ms.real < 1:
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

@ureg.check(None, None, '[]')
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
        refractive index of the second medium along the direction of
        propagation
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
