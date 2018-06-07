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
