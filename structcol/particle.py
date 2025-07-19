# Copyright 2016, Vinothan N. Manoharan, Sofia Makgiriadou, Victoria Hwang,
# Anna B. Stephenson
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
Classes for specifying geometry and optical properties of particles, as well as
methods for calculating form factors from such particles.
.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from pymie import mie
import structcol as sc


def _make_coords(wavelen, angles, cartesian, phis=None):
    """Convenience function to make DataArray coordinates for outputs from
    scattering methods (e.g. form_factor()).

    """
    if cartesian:
        coords = {sc.Coord.POL: ["x", "y"]}
    else:
        coords = {sc.Coord.POL: ["par", "perp"]}

    # set up coords for DataArray, avoiding scalar dimension for wavelen
    coords[sc.Coord.WAVELEN] = np.atleast_1d(wavelen.magnitude)
    if angles.ndim == 2:
        coords[sc.Coord.THETA] = angles[:, 0].magnitude
        coords[sc.Coord.PHI] = angles[0, :].magnitude
    else:
        coords[sc.Coord.THETA] = angles.magnitude

    return coords


class Particle:
    """Base class for specifying geometry and optical properties of a particle.

    Attributes
    ----------
    index : array-like[Index]
        Index object that returns index of particle at specified wavelengths.
        If array-like, specifies multiple refractive indices for different
        parts of the particle.
    size : `xr.DataArray`
        Size of particle. Specified as a `sc.Quantity` object and stored as a
        DataArray. The elements of the DataArray specify length scales
        corresponding to the structure of the particle. For example, for a
        multilayer sphere, the sizes are the radii of the different layers. For
        a (hypothetical) ellipsoid, the sizes might be the major and minor
        axes. DataArray should name these dimensions accordingly (this is
        handled by derived classes). Attribute `sc.Attr.LENGTH_UNIT` of
        DataArray gives the units of length.
    coords : dict (default None)
        Coordinate names and ranges to be used for the size DataArray.

    """
    @sc.ureg.check(None, None, '[length]', None)
    def __init__(self, index, size, coords=None):
        self.index = index
        # store sizes in internal units and strip units after saving them
        self.original_units = size.units
        self.size = xr.DataArray(size.to_preferred().magnitude, coords=coords)
        self.current_units = size.to_preferred().units
        self.size.attrs[sc.Attr.LENGTH_UNIT] = self.current_units

    @property
    def size_q(self):
        """Dimensions of particle, reported with units"""
        return sc.Quantity(self.size.to_numpy(),
                           self.size.attrs[sc.Attr.LENGTH_UNIT])

    @sc.ureg.check(None, '[length]')
    def n(self, wavelen):
        """Calculate index as a function of vacuum wavelength

        Parameters
        ----------
        wavelen : array-like [sc.Quantity]
            wavelengths at which to calculate index of refraction

        Returns
        -------
        xr.DataArray
            Index of refraction at each specified wavelength
        """
        return self.index(wavelen)

    def index_list(self, index_matrix=None):
        """List of refractive indices that the particle comprises.

        If `index_matrix` is specified, this Index is appended to the list.
        This method is used along with `Particle.volume_fraction()` to
        calculate effective indices.

        Parameters
        ----------
        index_matrix: `sc.Index` object
            Index of the matrix surrounding the particle

        """
        if np.ndim(self.index) != 0:
            # this will unpack the index object irrespective of whether it is
            # an array or list, then repack it into a list
            index_list = [*self.index]
        else:
            index_list = [self.index]
        if index_matrix is not None:
            index_list.extend([index_matrix])
        return index_list

    def volume_fraction(self, total_volume_fraction=None):
        """Volume fractions of each material in the particle.  Must be
        implemented in derived classes (depends on the geometry of the
        particle).
        """
        raise NotImplementedError

    def form_factor(self, wavelen, angles, index_external, distance=None):
        """Calculates form factor of the particle in a matrix with index of
        refraction `n_external`. Because the form factor depends on the
        particle, this method must be implemented in derived classes.

        """
        raise NotImplementedError


class Sphere(Particle):
    """Spherical homogeneous or layered particle.

    Notes
    -----
    Radii will be reported when you query the size of the particle.  If you
    want diameters, use the `Sphere.diameter` property.

    Attributes
    ----------
    index: list of Index objects
        Refractive index of particles or voids.  If specified as an array-like
        object, the refractive indices correspond to each layer in a multilayer
        sphere, from the innermost to the outermost layer.
    radii: list of sc.Quantity objects
        Radii of particles or voids, defined from the innermost to the
        outermost layer.

    """
    @sc.ureg.check(None, None, '[length]')
    def __init__(self, index, radius):
        index = np.atleast_1d(index)
        size = np.atleast_1d(radius)
        num_layers = len(index)

        if num_layers > 1:
            self.layered = True

            # check to make sure radii are sorted
            if not np.all(size[:-1] < size[1:]):
                raise ValueError("For a multilayer sphere, radii must be "
                                 "specified from smallest to largest.")
            # check to make sure we have the same number of elements in both
            # arrays
            if size.shape != index.shape:
                raise ValueError("Must specify index for each layer; got "
                                 f"{index.shape} indexes, {size.shape} radii")
            coords = {sc.Coord.LAYER: np.arange(num_layers)}
        else:
            self.layered = False
            # use scalars instead of arrays
            index = index.item()
            size = size.item()
            coords = None

        # store internal structure of sphere
        super().__init__(index, size, coords=coords)

    @property
    def radius(self):
        return self.size

    @property
    def diameter(self):
        return self.size*2

    @property
    def outer_radius(self):
        """Outer radius of the particle.  Used for calculating, for example,
        concentration of a layered sphere species"""
        if self.layered:
            return self.radius[-1]
        else:
            return self.radius

    @property
    def outer_diameter(self):
        """Outer diameter of the particle."""
        return self.outer_radius * 2

    @property
    def outer_radius_q(self):
        """Outer radius with dimensions"""
        return self.outer_radius.to_numpy() * self.current_units

    @property
    def radius_q(self):
        """Radius with units"""
        return self.size_q

    @property
    def diameter_q(self):
        """Diameter with units"""
        return self.size_q * 2

    @property
    def layers(self):
        """Number of layers in sphere"""
        return len(np.atleast_1d(self.index))

    def volume_fraction(self, total_volume_fraction=None):
        """Volume fraction of each material in the sphere.

        By default, returns volume fraction of each layer relative to the total
        volume of the sphere. If `total_volume_fraction` is specified, the
        volume fractions are multipled by this value, so that a different
        reference basis can be applied. Then a final value of
        (1-total_volume_fraction) is appended to the output array, so that the
        computed volume fractions sum to 1. This method is useful for
        calculating effective indices.

        Parameters
        ----------
        total_volume_fraction : float (optional)
            if specified, volume fractions are multiplied by this value, and
            the output is augmented to give the volume fraction of the medium
            around the spheres as as well.

        Returns
        -------
        `xr.DataArray` :
            Volume fraction of each layer

        """
        # must add zero to the list of radii for the volume calculation to work
        radii = np.insert(np.atleast_1d(self.radius), 0, 0)
        # volume fractions relative to sphere volume
        vf = (radii[1:]**3 - radii[:-1]**3) / radii[-1]**3
        if total_volume_fraction is not None:
            vf = np.append(vf * total_volume_fraction,
                           1 - total_volume_fraction)
            materials = self.layers + 1
        else:
            materials = self.layers
        return xr.DataArray(vf, coords = {sc.Coord.MAT : range(materials)})

    def n(self, wavelen):
        """Calculate index as a function of vacuum wavelength

        Parameters
        ----------
        wavelen : array-like [sc.Quantity]
            wavelengths at which to calculate index of refraction

        Returns
        -------
        xr.DataArray
            Index of refraction at each specified wavelength for each layer in
            the particle (if layered)
        """
        if self.layered:
            return sc.index._indexes_from_list(self.index, wavelen)
        else:
            return super().n(wavelen)

    def number_density(self, volume_fraction):
        """Calculate number density of spheres

        """
        if self.layers > 1:
            radius = self.radius_q.max()
        else:
            radius = self.radius_q
        return 3.0 * volume_fraction / (4.0 * np.pi * radius**3)

    def form_factor(self, wavelen, angles, index_external, kd=None,
                    cartesian=False, incident_vector=None, phis=None):
        """Calculate form factor from Mie theory.

        Parameters
        ----------
        wavelen : array-like [sc.Quantity]
            wavelengths at which to calculate form factor
        angles : array-like
            scattering angles at which to calculate form factor.  Specified in
            radians.
        index_external : `sc.Index` object
            Index of refraction of the medium around the particle.  Can be an
            effective index.
        kd : float (optional)
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
            (`cartesian=False`), `phis` is ignored, since the scattering
            matrix does not depend on phi.

        Returns
        -------
        float (2-tuple):
            Form factor for parallel and perpendicular polarizations as a
            function of scattering angle.

        """
        wavelen = wavelen.to_preferred()
        angles = angles.to("rad")
        if phis is not None:
            phis = phis.to("rad")

        n_ext = index_external(wavelen)
        n_particle = self.n(wavelen)

        m = sc.index.ratio(n_particle, n_ext)
        x = sc.size_parameter(n_ext, self.radius_q).to_numpy()

        if cartesian:
            coordinate_system = "cartesian"
        else:
            coordinate_system = "scattering plane"

        form_factor = self._form_factor(m, x, angles, kd=kd,
                                        coordinate_system=coordinate_system,
                                        incident_vector=incident_vector,
                                        phis=phis)

        coords = _make_coords(wavelen, angles, cartesian, phis=phis)

        # convert tuple to array, adding a dimension with size 1 if the
        # wavelength is a scalar
        form_factor = np.array([*form_factor])
        if len(np.atleast_1d(wavelen)) == 1:
            form_factor = np.expand_dims(form_factor, axis=1)

        form_factor = xr.DataArray(form_factor, coords=coords)
        form_factor.attrs[sc.Attr.LENGTH_UNIT] = wavelen.units

        return form_factor

    def _form_factor(self, m, x, angles, kd=None, coordinate_system=None,
                     incident_vector=None, phis=None):
        """Thin wrapper around pymie form-factor routines. Called internally by
        form_factor() methods in cases where speed is important.

        """
        if np.any(x.imag > 0) or (coordinate_system=='cartesian'):
            if kd is None:
                raise ValueError("must specify distance for absorbing systems")
            form_factor = mie.diff_scat_intensity_complex_medium(
                            m, x, angles, kd,
                            coordinate_system=coordinate_system,
                            incident_vector=incident_vector,
                            phis=phis)
        else:
            form_factor = mie.calc_ang_dist(m, x, angles)

        return form_factor


class SphereDistribution:
    """Class to describe a continuous size distribution of spheres.

    Can handle one or two species. If two species, each species can have its
    own polydispersity index.

    Attributes
    ----------
    spheres : array-like of sc.Sphere objects
        Species of spherical particles that make up the structure.  Each size
        represents the mean size of a distribution.  If monospecies, should
        contain only one element.
    concentrations : 2-element array (structcol.Quantity [dimensionless])
        Number fractions of each species. For example, a system composed of 90
        A particles and 10 B particles would have c = [0.9, 0.1]. For
        polydisperse monospecies systems, specify the concentration as [1.0,
        0.0].
    polydispersities : 2-element array (structcol.Quantity [dimensionless])
        Polydispersity index of each species if the system is polydisperse.
        For polydisperse monospecies systems, specify the pdi as a 2-element
        array with repeating values (for example, [0.01, 0.01]).
    polydispersity_bound : float (default 1e-5)
        Lower bound on the polydispersity (polydispersity of zero will lead to
        errors)
    """
    def __init__(self, spheres, concentrations, polydispersities,
                 polydispersity_bound = 1e-5):
        spheres = list(np.atleast_1d(spheres))
        if len(spheres) > 2:
            raise ValueError("Can only handle one or two species")
        self.spheres = spheres
        self.num_components = len(spheres)

        self.diameters = []
        for sphere in spheres:
            self.diameters = self.diameters + [sphere.outer_diameter]
        self.diameters = np.array(self.diameters)

        if isinstance(concentrations, sc.Quantity):
            concentrations = concentrations.to('').magnitude
        concentrations = np.atleast_1d(concentrations)
        if np.sum(concentrations) != 1.0:
            raise ValueError("Concentrations must sum to 1")
        self.concentrations = concentrations

        self.polydispersity_bound = polydispersity_bound
        if isinstance(polydispersities, sc.Quantity):
            polydispersities = polydispersities.to('').magnitude
        if not np.isscalar(polydispersities):
            if len(self.diameters) == 1 and len(polydispersities) !=1:
                raise ValueError("polydispersity overspecified; only one "
                                 "value is needed for a single species")
        # if the pdi is zero, assume it's very small (we get the same results)
        # because otherwise we get a divide by zero error
        pdi = np.atleast_1d(polydispersities).astype(float)
        pdi[pdi < self.polydispersity_bound] = self.polydispersity_bound

        self.pdi = pdi

    @property
    def outer_radii(self):
        """Returns outer radii as a DataArray with coordinate
        `sc.Coord.COMPONENT`
        """
        coords={sc.Coord.COMPONENT: np.arange(self.num_components)}
        outer_r_da = xr.DataArray(self.diameters, coords=coords)/2
        outer_r_da.attrs[sc.Attr.LENGTH_UNIT] = self.spheres[0].current_units

        # drop component coord if only monospecies
        return outer_r_da.squeeze(drop=True)

    @property
    def has_layered(self):
        """Returns True if the distribution contains a layered sphere
        """
        return any(sphere.layered for sphere in self.spheres)

    @property
    def diameters_q(self):
        """Returns mean diameters with units

        """
        return self.diameters * self.spheres[0].current_units

    def number_density(self, volume_fraction):
        """General number density formula for binary systems; converges to
        monospecies formula when the concentration of either particle is zero.

        """
        radius = self.diameters_q/2.0
        if np.any(self.concentrations == 0):
            rho = self.spheres[0].number_density(volume_fraction)
        else:
            term1 = 1 / (radius[0] ** 3 + radius[1] ** 3
                         * self.concentrations[1]/self.concentrations[0])
            term2 = 1 / (radius[1] ** 3 + radius[0] ** 3
                         * self.concentrations[0]/self.concentrations[1])
            rho = 3.0 * volume_fraction / (4.0 * np.pi) * (term1 + term2)
        return rho

    def form_factor(self, wavelen, angles, index_external, kd=None,
                    cartesian=False, incident_vector=None, phis=None):
        """
        Calculate the form factor for polydisperse systems.

        Parameters
        ----------
        wavelen : array-like [sc.Quantity]
            wavelengths at which to calculate form factor
        angles : array-like
            scattering angles at which to calculate form factor.  Specified in
            radians.
        index_external : `sc.Index` object
            Index of refraction of the medium around the particle.  Can be an
            effective index.
        kd : float (optional)
            distance (nondimensionalized by k) at which to calculate the
            differential cross section. Needed only if n_external is complex.
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
            (`cartesian=False`), `phis` is ignored, since the scattering
            matrix does not depend on phi.

        Returns
        -------
        float (2-tuple):
            polydisperse form factor for parallel and perpendicular
            polarizations as a function of scattering angle.
        """
        wavelen = wavelen.to_preferred()
        angles = angles.to('rad')
        n_ext = index_external(wavelen)
        if cartesian:
            coordinate_system = 'cartesian'
        else:
            coordinate_system = 'scattering plane'

        if self.has_layered:
            raise ValueError("Cannot handle polydispersity in core-shell ",
                             "particles")

        if len(self.spheres) == 2:
            if self.spheres[0].index != self.spheres[1].index:
                raise ValueError("Currently can handle only species with the "
                                 "same refractive index.")
        index_particle = self.spheres[0].index
        n_particle = index_particle(wavelen)

        m = sc.index.ratio(n_particle, n_ext)


        # t is a measure of the width of the Schulz distribution, and
        # pdi is the polydispersity index
        t = np.abs(1/(self.pdi**2)) - 1

        # define the range of diameters of the size distribution
        # (all of the below will be in preferred units)
        three_std_dev = 3*self.diameters/np.sqrt(t+1)
        min_diameter = self.diameters - three_std_dev
        min_diameter[min_diameter < 0] = 0.000001
        max_diameter = self.diameters + three_std_dev

        if ((np.abs(n_ext.imag) > 0. or cartesian)
            and (kd is not None)):
            kd = np.resize(kd, len(self.diameters))

        F = {}
        for pol in ('par', 'perp'):
            if cartesian:
                F[pol] = np.empty([len(self.spheres), angles.shape[0],
                                   angles.shape[1]])
            else:
                F[pol] = np.empty([len(self.spheres), len(angles)])

        # for each mean diameter, calculate the Schulz distribution and
        # the size parameter x_poly
        for d in np.arange(len(self.diameters)):
            # the diameter range is the range between the min diameter and
            # the max diameter of the Schulz distribution
            diameter_range = np.linspace(np.atleast_1d(min_diameter)[d],
                                         np.atleast_1d(max_diameter)[d], 50)
            distr = sc.model.size_distribution(diameter_range,
                                               self.diameters[d],
                                               np.atleast_1d(t)[d])
            if cartesian:
                distr_array = np.tile(distr,
                                      [angles.shape[0], angles.shape[1], 1])
            else:
                distr_array = np.tile(distr, [len(angles), 1])
            angles_array = np.tile(angles, [len(diameter_range), 1])

            # size parameter will be a 2D array [1, num_diameters]. Because
            # this would be interpreted as a layered particle by pymie, we
            # convert to a 1D array before looping
            x_poly = sc.size_parameter(n_ext,
                                       (diameter_range/2 *
                                        self.spheres[0].current_units))
            x_poly = x_poly.to_numpy()[0]

            form_factor = {}
            integrand = {}
            for pol in ('par', 'perp'):
                if cartesian:
                    form_factor[pol] = np.empty([angles.shape[0],
                                                 angles.shape[1],
                                                 len(diameter_range)])
                    integrand[pol] = np.empty([angles.shape[0],
                                               angles.shape[1],
                                               len(diameter_range)])
                else:
                    form_factor[pol] = np.empty([len(angles),
                                                 len(diameter_range)])
                    integrand[pol] = np.empty([len(angles),
                                               len(diameter_range)])

            # for each diameter in the distribution, calculate the detected
            # and the total form factors for absorbing systems
            for s in np.arange(len(diameter_range)):
                sphere = sc.Sphere(index_particle,
                                   sc.Quantity(diameter_range[s]/2,
                                               self.diameters_q.units))
                if kd is not None:
                    kd_new = kd[d]
                else:
                    kd_new = None

                # sphere.form_factor() has too much overhead for a loop (the
                # overhead is related to all the unit checking, xarray
                # wrapping, and calculating quantities like m, x, and indices.
                # Since most of the calculations are constant, we use the
                # underlying faster sphere_form_factor() to avoid the overhead.
                ff = sphere._form_factor(m, x_poly[s], angles_array[s],
                                         kd=kd_new,
                                         coordinate_system=coordinate_system,
                                         incident_vector=incident_vector,
                                         phis=phis)

                    # it might seem reasonable to calculate the form factor of
                    # each individual radius in the Schulz distribution
                    # (meaning that we could use diameter_range[s] instead of
                    # distance_array[d]), but this doesn't lead to reasonable
                    # results because we later integrate the diff cross section
                    # at the mean radii, not at each of the radii of the
                    # distribution. So we need to be consistent with the
                    # distances we use for the integrand and the integral. For
                    # now, we use the mean radii.

                if cartesian:
                    form_factor['par'][:, :, s] = ff[0]
                    form_factor['perp'][:, :, s] = ff[1]
                else:
                    form_factor['par'][:, s] = ff[0]
                    form_factor['perp'][:, s] = ff[1]

            # integrate and multiply by the concentration of the mean
            # diameter to get the polydisperse form factor
            for pol in ('par', 'perp'):
                # multiply the form factors by the Schulz distribution
                integrand[pol] = form_factor[pol] * distr_array

                if cartesian:
                    axis_int = 2
                else:
                    axis_int = 1
                integral = (trapezoid(integrand[pol], x=diameter_range,
                                      axis=axis_int)
                                 * self.concentrations[d])

                if cartesian:
                    F[pol][d, :, :] = integral
                else:
                    F[pol][d, :] = integral

        # the final polydisperse form factor as a function of angle is
        # calculated as the average of each mean diameter's form factor
        f_par = np.sum(F['par'], axis=0)
        f_perp = np.sum(F['perp'], axis=0)

        return(f_par, f_perp)
