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
from pymie import Quantity, ureg, q, index_ratio, size_parameter, np, mie

# Global variable speed of light
# get this from Pint in a somewhat indirect way:
LIGHT_SPEED_VACUUM = Quantity(1.0, 'speed_of_light').to('m/s')

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
    1D array: contains only the elements of inarray corresponding to non-zero events values.

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

# Create a global random number generator object
rng = np.random.default_rng()

def set_seed(seed):
    global rng

    # Create a random number generator that reproduces old sampling behavior
    # Note that seed is in the form of a list. This is to reproduce the previous values which are hardcoded in the tests. 
    # Setting the seed without the list brackets yeilds a different set of random numbers. 
    rng = np.random.RandomState([seed])

    # Also set the seed for the random module 
    # in case any random numbers are generated without the Generator object.
    np.random.seed([seed])
