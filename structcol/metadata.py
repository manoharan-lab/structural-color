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
Coordinate names and Attribute names to label DataArrays and Datasets.

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

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
    # coords THETA and PHI refer to the actual values of the angles.  THETAIDX
    # and PHIIDX refer to integer indexes of the angles
    THETAIDX = "theta_index"
    PHIIDX = "phi_index"
    # incident angle for Fresnel calculations
    INCIDENT = "incident_angle"
    # species index for multispecies systems
    SPECIES = "species"
    # polarization for scattering calculations (should take on values "x", "y"
    # for cartesian basis or "par", "perp" for scattering-plane basis)
    POL = "polarization"
    # Fresnel coefficient (should be either "r" or "t")
    FRESNEL = "fresnel"


class Attr():
    """Simple class to standardize metadata (attributes) used in xarray
    objects.

    """
    LENGTH_UNIT = "length unit"

