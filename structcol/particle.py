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
from pymie import mie
import structcol as sc

from . import ureg

class Particle:
    """Base class for specifying geometry and optical properties of a particle.

    Attributes
    ----------
    index : array-like[Index]
        Index object that returns index of particle at specified wavelengths.
        If array-like, specifies multiple refractive indices for different
        parts of the particle.
    size : array-like[sc.Quantity]
        Size of particle. If array-like, specifies length scales corresponding
        to structure of particle. For example, for a multilayer sphere, the
        sizes are the radii of the different layers. For a (hypothetical)
        ellipsoid,the sizes might be the major and minor axes.

    """
    @ureg.check(None, None, '[length]')
    def __init__(self, index, size):
        self.index = index
        # store sizes in internal units and strip units after saving them
        self.original_units = size.units
        self.size = size.to_preferred().magnitude
        self.current_units = size.to_preferred().units

    @property
    def size_q(self):
        """Dimensions of particle, reported with units"""
        return self.size * self.current_units

    @ureg.check(None, '[length]')
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
    @ureg.check(None, None, '[length]')
    def __init__(self, index, radius):
        index = np.atleast_1d(index)
        size = np.atleast_1d(radius)

        if len(index) > 1:
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
        else:
            self.layered = False
            # use scalars instead of arrays
            index = index.item()
            size = size.item()

        # store internal structure of sphere
        super().__init__(index, size)

    @property
    def radius(self):
        return self.size

    @property
    def diameter(self):
        return self.size*2

    @property
    def outer_diameter(self):
        """Outer diameter of the particle.  Used for calculating, for example,
        concentration of a layered sphere species"""
        if self.layered:
            return self.radius[-1]*2
        else:
            return self.radius*2

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
        incident_vector: tuple (optional, default None)
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
            Form factor for parallel and perpendicular polarizations as a
            function of scattering angle.

        """
        wavelen = wavelen.to_preferred()

        n_ext = index_external(wavelen)
        n_particle = self.n(wavelen)

        m = sc.index.ratio(n_particle, n_ext)
        x = sc.size_parameter(n_ext, self.radius_q)

        if np.any(n_ext.imag > 0) or (cartesian is True):
            if kd is None:
                raise ValueError("must specify distance for absorbing systems")
            if cartesian:
                coordinate_system = 'cartesian'
            else:
                coordinate_system = 'scattering plane'
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
        errors
    """
    def __init__(self, spheres, concentrations, polydispersities,
                 polydispersity_bound = 1e-5):
        spheres = list(np.atleast_1d(spheres))
        if len(spheres) > 2:
            raise ValueError("Can only handle one or two species")
        self.spheres = spheres
        self.diameters = []
        for sphere in spheres:
            self.diameters = self.diameters + [sphere.outer_diameter]
        self.diameters = np.array(self.diameters)

        if np.isscalar(concentrations):
            concentrations = np.array([concentrations, 0.0])
        if len(spheres) == 1:
            if not np.array_equal(concentrations, np.array([1.0, 0])):
                raise ValueError("When only one species is specified, "
                                 "concentrations must be set to [1.0, 0]")
        if np.sum(concentrations) != 1.0:
            raise ValueError("Concentrations must sum to 1")
        self.concentrations = concentrations

        self.polydispersity_bound = 1e-5
        if isinstance(polydispersities, sc.Quantity):
            polydispersities = polydispersities.to('').magnitude
        if np.isscalar(polydispersities):
            polydispersities = np.array([polydispersities, 0.0])
        # if the pdi is zero, assume it's very small (we get the same results)
        # because otherwise we get a divide by zero error
        pdi = np.atleast_1d(polydispersities).astype(float)
        pdi[pdi < self.polydispersity_bound] = self.polydispersity_bound

        self.pdi = pdi

    @property
    def has_layered(self):
        """Returns True if the distribution contains a layered sphere
        """
        return any(sphere.layered for sphere in self.spheres)

