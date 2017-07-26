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
Functions for calculating refractive index as a function of wavelength for
various materials.

Notes
-----
Most of this data is from refractiveindex.info [1]_. According to
http://refractiveindex.info/download.php,
"refractiveindex.info database is in public domain. Copyright and related
rights were waived by Mikhail Polyanskiy through the CC0 1.0 Universal Public
Domain Dedication. You can copy, modify and distribute refractiveindex.info
database, even for commercial purposes, all without asking permission."

References
----------
[1] Dispersion formulas from M. N. Polyanskiy. "Refractive index database,"
http://refractiveindex.info (accessed August 14, 2016).

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>.

"""

import numpy as np
from . import ureg, Quantity  # unit registry and Quantity constructor from pint

# dictionary of refractive index dispersion formulas. This is used by the 'n'
# function below; it's outside the function definition so that it doesn't have
# to be initialized on every function call (see stackoverflow 60208).
#
# NOTE: If you add a material to the dictionary, you need to add a test
# function to structcol/tests/test_refractive_index.py that will test to make
# sure the dispersion relation returns the proper values of the refractive
# index at two or more points.
#
# np.power doesn't seem to be supported by pint -- hence the w*w... or
# /w/w/w/w... syntax
n_dict = {
    # water data from M. Daimon and A. Masumura. Measurement of the refractive
    # index of distilled water from the near-infrared region to the ultraviolet
    # region, Appl. Opt. 46, 3811-3820 (2007).
    # Fit of the experimental data with the Sellmeier dispersion formula:
    # refractiveindex.info
    # data for high performance liquid chromatography (HPLC) distilled water at
    # 20.0 °C
    'water': lambda w: np.sqrt(5.684027565e-1*w*w/
                                    (w*w - Quantity('5.101829712e-3 um^2')) +
                                    1.726177391e-1*w*w/
                                    (w*w - Quantity('1.821153936e-2 um^2')) +
                                    2.086189578e-2*w*w/
                                    (w*w - Quantity('2.620722293e-2 um^2')) +
                                    1.130748688e-1*w*w/
                                    (w*w - Quantity('1.069792721e1 um^2')) + 1),


    # polystyrene data from N. Sultanova, S. Kasarova and I. Nikolov. Dispersion
    # properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009).
    # Fit of the experimental data with the Sellmeier dispersion formula:
    # refractiveindex.info
    # data for 20 degrees C, 0.4368-1.052 micrometers
    'polystyrene': lambda w: np.sqrt(1.4435*w*w/
                                     (w*w-Quantity("0.020216 um^2"))+1),

    # pmma data from G. Beadie, M. Brindza, R. A. Flynn, A. Rosenberg, and J.
    # S. Shirk. Refractive index measurements of poly(methyl methacrylate)
    # (PMMA) from 0.4-1.6 micrometers, Appl. Opt. 54, F139-F143 (2015)
    # refractiveindex.info
    # data for 20.1 degrees C, 0.42-1.62 micrometers
    'pmma': lambda w: np.sqrt(2.1778 + Quantity('6.1209e-3 um^-2')*w*w -
                              Quantity('1.5004e-3 um^-4')*w*w*w*w +
                              Quantity('2.3678e-2 um^2')/w/w -
                              Quantity('4.2137e-3 um^4')/w/w/w/w +
                              Quantity('7.3417e-4 um^6')/w/w/w/w/w/w -
                              Quantity('4.5042e-5 um^8')/w/w/w/w/w/w/w/w),

    # rutile TiO2 from J. R. Devore. Refractive Indices of Rutile and
    # Sphalerite, J. Opt. Soc. Am. 41, 416-419 (1951)
    # refractiveindex.info
    # data for rutile TiO2, ordinary ray, 0.43-1.53 micrometers
    'rutile': lambda w: np.sqrt(5.913 +
                                Quantity('0.2441 um^2')/
                                (w*w - Quantity('0.0803 um^2'))),

    # fused silica (amorphous quartz) data from I. H. Malitson. Interspecimen
    # Comparison of the Refractive Index of Fused Silica, J. Opt. Soc. Am. 55,
    # 1205-1208 (1965)
    # refractiveindex.info
    # data for "room temperature", 0.21-3.71 micrometers
    'fused silica': lambda w: np.sqrt(1 + 0.6961663*w*w/
                                      (w*w - Quantity('0.0684043**2 um^2')) +
                                      0.4079426*w*w/
                                      (w*w - Quantity('0.1162414**2 um^2')) +
                                      0.8974794*w*w/
                                      (w*w - Quantity('9.896161**2 um^2'))),
    # soda lime glass data from M. Rubin. Optical properties of soda lime 
    # silica glasses, Solar Energy Materials 12, 275-288 (1985)
    # refractiveindex.info
    # data for "room temperature", 0.31-4.6 micrometers
    'soda lime glass': lambda w: 1.5130 - Quantity('0.003169 um^-2')*w*w + 
                                                  Quantity('0.003962 um^2')/(w*w),
                                      
    # the w/w is a crude hack to make the function output an array when the
    # input is an array
    'vacuum': lambda w: Quantity('1.0')*w/w
}

@ureg.check(None, '[length]')   # ensures wavelen has units of length
def n(material, wavelen):
    """Refractive index of various materials

    Parameters
    ----------
    material: string
        material type; if not found in dictionary, assumes vacuum
    w : structcol.Quantity [length]
        Wavelength in vacuum.

    Returns
    -------
    structcol.Quantity (dimensionless)
        refractive index

    Dispersion formulas from M. N. Polyanskiy. "Refractive index database,"
    http://refractiveindex.info (accessed August 14, 2016).
    """

    try:
        return n_dict[material](wavelen)
    except KeyError:
        print("Material \""+material+"\" not implemented.  Perhaps a typo?")
        raise

# for the rest of these materials, need to find dispersion relations and
# implement the functions in the dictionary.
def n_air(w):
    return 1.0

def n_silica_colloidal(w):
    return 1.40

def n_water(w):
    return 1.33

def n_cargille(i,series,wavelength):
    # where i is the cardinal number of the liquid (starting with 0) and w is the wavelength, in nm (converted to um)
    return 1.45

def n_keratin(w):
    return 1.532

def n_ptbma(w):
    # from http://www.sigmaaldrich.com/catalog/product/aldrich/181587?lang=en&region=US
    return 1.46

def n_eff(n_inclusion, n_matrix, volume_fraction):
    """
    Calculates Maxwell-Garnett effective refractive index for a composite of
    two dielectric media.

    Parameters
    ----------
    n_inclusion: float or structcol.Quantity (dimensionless)
        refractive index of inclusions (particles or voids)
    n_matrix : float or structcol.Quantity (dimensionless)
        refractive index of matrix phase
    volume_fraction: float
        volume fraction of inclusions

    Returns
    -------
    structcol.Quantity (dimensionless)
        refractive index
    """
    ni = n_inclusion
    nm = n_matrix
    phi = volume_fraction
    neff =  nm * np.sqrt((2*nm**2 + ni**2 + 2*phi*((ni**2)-(nm**2))) /
                         (2*nm**2 + ni**2 - phi*((ni**2)-(nm**2))))
    return Quantity(neff)
