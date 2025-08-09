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
The structural-color (structcol) python package includes theoretical models for
predicting the structural color from disordered colloidal samples (also known
as "photonic glasses").


Notes
-----
Based on work by Sofia Magkiriadou in the Manoharan Lab at Harvard University
[1]_

Requires pint:
PyPI: https://pypi.python.org/pypi/Pint/
Github: https://github.com/hgrecco/pint
Docs: https://pint.readthedocs.io/en/latest/

References
----------
[1] Magkiriadou, S., Park, J.-G., Kim, Y.-S., and Manoharan, V. N. “Absence of
Red Structural Color in Photonic Glasses, Bird Feathers, and Certain Beetles”
Physical Review E 90, no. 6 (2014): 62302. doi:10.1103/PhysRevE.90.062302

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>.
"""

# Load the default unit registry from pint and use it everywhere.
# Using the unit registry (and wrapping all functions) ensures that we don't
# make unit mistakes.
# Also load commonly used functions from pymie package
from pymie import Quantity, ureg, q, np, mie
from . import refractive_index as index
from .refractive_index import Index, EffectiveIndex
from .particle import Particle, Sphere, SphereDistribution
from . import structure, model
import xarray as xr

# make sure attributes are preserved during arithmetic operations
xr.set_options(keep_attrs=True)

# Global variable speed of light
# get this from Pint in a somewhat indirect way:
LIGHT_SPEED_VACUUM = Quantity(1.0, 'speed_of_light').to('m/s')

class Coord():
    """Simple class to standardize dimension/coordinate names that we use in
    xarray objects.

    """

    WAVELEN = "wavelength"
    VOLFRAC = "volume_fraction"
    # both LAYER and MAT map to the same name, so that we can describe the
    # components of a multilayer sphere as layers and the components of a
    # multimaterial matrix as materials, but we can calculate an effective
    # index for both
    LAYER = "material"
    MAT = "material"
    # polar and azimuthal angles for scattering
    THETA = "theta"
    PHI = "phi"
    # incident angle for Fresnel calculations
    INCIDENT = "incident_angle"
    # species index for multispecies systems
    SPECIES = "species"
    # polarization for scattering calculations (should take on values "x", "y"
    # for cartesian basis or "par", "perp" for scattering-plane basis)
    POL = "polarization"
    # Fresnel coefficient (should be either "r" or "t")
    FRESNEL = "fresnel"
    # generic coord to refer to a MultiIndex describing stacked coordinates
    # (such as wavelength/angle stacks)
    VALUE = "values"

class Attr():
    """Simple class to standardize metadata (attributes) used in xarray
    objects.

    """
    LENGTH_UNIT = "length unit"


# Preferred unit for length. Because the package allows calculations as a
# function of wavelength and radius, it's not always clear what length scale to
# use for nondimensionalization. We specify a preferred length scale here for
# nondimensionalizing length scales internally. All dimensional quantities
# (specified using pint) are converted to the same units as the preferred and
# are then nondimensionalized. We choose micrometers because all the dispersion
# relations are expressed in terms of micrometers.
LENGTH_UNIT = ureg.micrometer
ureg.default_preferred_units = [LENGTH_UNIT]


def _make_input_coords(wavelen, thetas, phis=None):
    """Convenience function to generate DataArray coordinates to be used as
    inputs to differential_cross_section() and form_factor() methods.  Called
    by `sc.model.FormStructureModel.make_input_coords()` but can also be called
    by test functions.

    Takes `sc.Quantity` objects for wavelen, thetas, and (optionally) phis, and
    returns an `xr.Coordinates` object.  See
    `sc.model.FormStructureModel.make_input_coords() for information on
    parameters and return.

    """
    if isinstance(wavelen, Quantity):
        wavelen = wavelen.to_preferred().magnitude
    if isinstance(thetas, Quantity):
        thetas = thetas.to("rad").magnitude
    if phis is not None:
        if isinstance(phis, Quantity):
            phis = phis.to("rad").magnitude

    coords = xr.Coordinates()

    # set up coords for DataArray, avoiding scalar dimension for wavelen
    coords[Coord.WAVELEN] = np.atleast_1d(wavelen)
    coords[Coord.THETA] = thetas
    if phis is not None:
        coords[Coord.PHI] = phis

    return coords


def refraction(angles, n_before, n_after):
    '''
    Returns angles after refracting through an interface

    Parameters
    ----------
    angles: float or array of floats
        angles relative to normal before the interface
    n_before: float
        Refractive index of the medium light is coming from
    n_after: float
        Refractive index of the medium light is going to

    '''
    snell = n_before / n_after * np.sin(angles)
    snell[abs(snell) > 1] = np.nan  # this avoids a warning
    return np.arcsin(snell)


def normalize(x, y, z, return_nan=True):
    '''
    normalize a vector

    Parameters
    ----------
    x: float or array
        1st component of vector
    y: float or array
        2nd component of vector
    z: float or array
        3rd component of vector

    Returns
    -------
    array of normalized vector(s) components
    '''
    magnitude = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2)

    # we ignore divide by zero error here because we do not want an error
    # in the case where we try to normalize a null vector <0,0,0>
    with np.errstate(divide='ignore', invalid='ignore'):
        if (not return_nan) and magnitude.all() == 0:
            magnitude[magnitude == 0] = 1
        return np.array([x / magnitude, y / magnitude, z / magnitude])


def select_events(inarray, events):
    '''
    Selects the items of inarray according to event coordinates

    Parameters
    ----------
    inarray: 2D or 3D array
        Should have axes corresponding to events, trajectories
        or coordinates, events, trajectories
    events: 1D array
        Should have length corresponding to ntrajectories.
        Non-zero entries correspond to the event of interest

    Returns
    -------
    1D array: contains only the elements of inarray corresponding to non-zero
              events values.

    '''
    # make inarray a numpy array if not already
    if isinstance(inarray, Quantity):
        inarray = inarray.magnitude
    inarray = np.array(inarray)

    # there is no 0th event, so disregard a 0 (or less) in the events array
    valid_events = (events > 0)

    # The 0th element in arrays such as direction refer to the 1st event
    # so subtract 1 from all the valid events to correct for array indexing
    ev = events[valid_events].astype(int) - 1

    # find the trajectories where there are valid events
    tr = np.where(valid_events)[0]

    # want output of the same form as events, so create variable
    # for object type
    dtype = type(np.ndarray.flatten(inarray)[0])

    # get an output array with elements corresponding to the input events
    if len(inarray.shape) == 2:
        outarray = np.zeros(len(events), dtype=dtype)
        outarray[valid_events] = inarray[ev, tr]

    if len(inarray.shape) == 3:
        outarray = np.zeros((inarray.shape[0], len(events)), dtype=dtype)
        outarray[:, valid_events] = inarray[:, ev, tr]

    if isinstance(inarray, Quantity):
        outarray = Quantity(outarray, inarray.units)
    return outarray


def size_parameter(n_medium, radius):
    """
    Calculates the size parameter x=k_medium*a needed for Mie calculations.

    This function expects n_medium to be a DataArray returned by an Index
    object, which will consist of index of refraction at various wavelengths.

    Notes
    -----
    Since the size parameter is nondimensional, this function strips units,
    returning a pure DataArray (not a Quantity object)

    Parameters
    ----------
    n_medium : `xr.DataArray`
        refractive index of medium at various wavelengths, as calculated by an
        `sc.Index` object.
    radius: array-like of structcol.Quantity [length] or `xr.DataArray`
        radius of particle.  If specified as DataArray, must be in preferred
        units

    Returns
    -------
    `xr.DataArray` (complex or float):
        DataArray of size parameters with dimensions WAVELEN and LAYER.  If
        volume fraction is one of the coordinates of n_medium (which should be
        the case if n_medium is an effective index), also returns VOLFRAC
        dimension.

    """

    if not isinstance(n_medium, xr.DataArray):
        raise ValueError("Index of medium must be a DataArray. "
                         "Ensure that you are using the output from an Index "
                         "object as input to this function.")

    wavelen = n_medium.coords[Coord.WAVELEN]
    if isinstance(radius, Quantity):
        radius = np.atleast_1d(radius.to_preferred().magnitude)
        radius = xr.DataArray(radius, coords={Coord.LAYER: range(len(radius))})
    elif isinstance(radius, xr.DataArray):
        if Coord.LAYER not in radius.coords:
            radius = radius.expand_dims({Coord.LAYER: [0]}, axis=-1)
    else:
        raise ValueError("radius must be specified as either sc.Quantity "
                         f"or DataArray, not {type(radius)}")

    sp = (2 * np.pi * n_medium / wavelen * radius)

    return sp


def wavevector(n_medium, d=None):
    """
    Calculates the wavevector in medium for Mie calculations.

    This function expects n_medium to be a DataArray returned by an Index
    object, which will consist of index of refraction at various wavelengths.

    Parameters
    ----------
    n_medium : `xr.DataArray`
        refractive index of medium at various wavelengths, as calculated by an
        `sc.Index` object.  Wavelengths (and possibly volume fractions, if an
        effective index was used to generate n_medium) are given in the
        coordinates.
    d : `sc.Quantity`
        length scale to nondimensionalize the wavevector.  If provided, the
        wavevector will be multiplied by this scale.

    Returns
    -------
    `xr.DataArray` :
        DataArray [float or complex] of wavevectors as a function of the
        coordinates in n_medium

    """

    if not isinstance(n_medium, xr.DataArray):
        raise ValueError("Index of medium must be a DataArray. "
                         "Ensure that you are using the output from an Index "
                         "object as input to this function.")

    wavelen = n_medium.coords[Coord.WAVELEN]
    units = n_medium.attrs[Attr.LENGTH_UNIT]

    k = 2 * np.pi * n_medium/wavelen
    if d is not None:
        k = k * d.to(units).magnitude

    return k


@ureg.check(None, "[length]", None)
def ql(n_medium, lengthscale, angles):
    """Calculates the nondimensional scattering wavevector in medium for
    structure factor calculations.

    This function expects n_medium to be a DataArray returned by an Index
    object, which will consist of index of refraction at various wavelengths.

    Parameters
    ----------
    n_medium : `xr.DataArray`
        refractive index of medium at various wavelengths, as calculated by an
        `sc.Index` object.  Wavelengths are given in the coordinates.
    lengthscale : sc.Quantity [length]
        lengthscale to use to calculate the size parameter.  For a sphere,
        this is the radius (will lead to diameter being used to
        nondimensionalize q)
    angles : `xr.DataArray`
        array of scattering angles. Must be specified in radians.

    Returns
    -------
    ndarray :
        `xr.DataArray` with dimensions wavelength, angle

    """
    # Use outer radius for multilayer particles (LAYER=-1)
    x = size_parameter(n_medium, lengthscale).isel({Coord.LAYER: -1},
                                                   drop=True)

    # this should automatically broadcast since angles is a DataArray
    # TODO: should it be x.real or x.abs?
    ql = 4*np.abs(x)*np.sin(angles/2)

    return ql


# Create a module-wide random number generator object that will be used by
# default in any functions that do random sampling. Users can override the
# default by passing their own rng to such functions. A user-specified rng is
# needed for testing and may be useful for parallel computation.
rng = np.random.default_rng()
