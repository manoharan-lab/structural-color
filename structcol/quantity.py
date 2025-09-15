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
Subclass of pint.Quantity

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

import pint
from typing import TypeAlias

class MyQuantity(pint.UnitRegistry.Quantity):
    """Child class of pint.Quantity, created to fix bugs and add methods to
    convert to/from xarray

    """
    # patch pint's to_preferred(), which is now broken
    def to_preferred(self):
        # we really only have three cases to handle (wavelengths, cross
        # sections, and wavevectors), so we don't need to handle the general
        # case of length units mixed with other units
        if self.check("[length]"):
            new_q = self.to(LENGTH_UNIT)
        elif self.check("[length]^2"):
            new_q = self.to(LENGTH_UNIT**2)
        elif self.check("[length]^-1"):
            new_q = self.to(1/LENGTH_UNIT)
        else:
            raise ValueError(f"Quantity {self} does not have expected units")
        return new_q

    def to_dataarray(self):
        """Convert pint Quantity to DataArray. This works only for quantities
        with dimension of length. Stores length unit as attribute.

        """
        pass


# subclass registry so that we can use new MyQuantity class, following
# https://pint.readthedocs.io/en/latest/advanced/custom-registry-class.html
class MyRegistry(pint.UnitRegistry):
    Quantity: TypeAlias = MyQuantity
    Unit: TypeAlias = pint.Unit


ureg = MyRegistry()
Quantity = ureg.Quantity

# setting the application registry allows pymie to use the same registry, with
# the new Quantity object
pint.set_application_registry(ureg)

# Preferred unit for length. Because the package allows calculations as a
# function of wavelength and radius, it's not always clear what length scale to
# use for nondimensionalization. We specify a preferred length scale here for
# nondimensionalizing length scales internally. All dimensional quantities
# (specified using pint) are converted to the same units as the preferred and
# are then nondimensionalized. We choose micrometers because all the dispersion
# relations are expressed in terms of micrometers.
LENGTH_UNIT = ureg.micrometer
