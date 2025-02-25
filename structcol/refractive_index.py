# -*- coding: utf-8 -*-
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
.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>
.. moduleauthor :: Victoria Hwang <vhwang@g.harvard.edu>.
"""

import numpy as np
import xarray as xr
# unit registry and Quantity constructor from pint
from . import ureg, Quantity
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from functools import partial

@ureg.check(None, '[length]')
def _constant_index(index, wavelen):
    """
    Returns a constant index irrespective of the wavelength

    Parameters
    ----------
    wavelen : array-like of structcol.Quantity[length]
        Wavelengths (in vacuum) at which to calculate index
    index : float or complex
        Index to return

    Returns
    -------
    float or complex
        Index of refraction (same at all wavelengths)
    """
    return np.ones_like(wavelen.magnitude) * index

class Index:
    """Class describing index of refraction as a function of wavelength.

    An instance of this class will have a function `index_func` that can
    calculate the (possibly complex) index as a function of wavelength. The
    wavelength must be a `sc.Quantity` object (can be a Quantity-wrapped
    ndarray) with units of length. The function should be called by calling the
    class instance (see Examples), which does unit checking, rather than by
    calling the function itself, which may not do any unit checking.

    Attributes
    ----------
    index_func : Function
        Returns index as a function of wavelength in vacuum and other
        parameters
    kwargs : dict
        Keyword arguments to pass to `index_func`.

    Examples
    --------
    >>> def index_func(wavelen):
    >>>     return 1.4 + sc.Quantity(500, 'nm^2')/(wavelen*wavelen)
    >>> my_index = Index(index_func)
    >>> my_index(sc.Quantity('400 nm'))
    1.403125

    """
    def __init__(self, index_func, **kwargs):
        self._n = partial(index_func, **kwargs)
        self._n_args = kwargs

    @ureg.check(None, '[length]')
    def __call__(self, wavelen):
        """Calculate index of refraction over a set of wavelengths.

        This method enables the user to call "myindex(wavelen)" where myindex
        is an Index object and wavelen is the array over which to calculate the
        index.  It does some unit checking to ensure that the `_n()` function
        returns the correct index for any units of length.

        Notes
        -----
        The returned index is *not* returned as an pint Quantity object so that
        other functions and classes that use this class do not need to strip
        units.

        Parameters
        ----------
        wavelen : array-like of structcol.Quantity[length]
            Wavelengths (in vacuum) at which to calculate index

        Returns
        -------
        xr.DataArray [float/complex]
            Refractive indices (possibly complex) at specified wavelengths.
            The wavelength is stored as a coordinate in the DataArray, with its
            units stored as the attribute "wavelength.unit"

        """
        index = self._n(wavelen)
        if isinstance(index, Quantity):
            if index.to_base_units().units != '':
                raise ValueError("Dispersion formula returned index with "
                                 f"units {index.units}.  Check that "
                                 "coefficients of the formula have the "
                                 "correct units.")
            # the "to_base_units()" converts units like nm/m to dimensionless
            index = index.to_base_units().magnitude

        if np.isscalar(index):
            return index
        else:
            # set up DataArray to return
            coords={"wavelength": wavelen.to_preferred().magnitude}
            index_array = xr.DataArray(index, coords=coords)
            index_array.attrs["wavelength unit"] = wavelen.to_preferred().units

            return index_array.squeeze()

    @classmethod
    def constant(cls, index):
        """Makes an Index object with a constant index of refraction for all
        wavelengths.

        Parameters
        ----------
        index : float or complex
            Index of refraction

        """
        if isinstance(index, Quantity):
            if index.to_base_units().units != '':
                raise ValueError("Specified constant index has units "
                                 f"{index.units}.  Should be dimensionless "
                                 "or not a Quantity object")
            index = index.to_base_units().magnitude

        return cls(partial(_constant_index, index))

    @classmethod
    @ureg.check(None, '[length]', None, None)
    def from_data(cls, wavelength_data, index_data, kind=None):
        """Make an Index object that interpolates from data to calculate
        indices of refraction.

        Parameters
        ----------
        index_data : array_like (float or complex)
            Refractive index data from literature or experiment. The index data
            can be real or complex.
        wavelength_data : array_like (sc.Quantity[length])
            Wavelength data corresponding to index_data. Must be specified as
            pint Quantity so that units are .
        kind : string (optional)
            Type of interpolation. Options are given in the documentation for
            `scipy.interpolate.interp1d()`. The default is 'linear'.

        """
        if kind is None:
            kind = 'linear'

        if wavelength_data.shape != index_data.shape:
            raise ValueError("Lengths of the wavelength data and index data "
                             "arrays must be the same.")
        if isinstance(index_data, Quantity):
            if index_data.to_base_units().units != '':
                raise ValueError("Index data must be specified in "
                                 "nondimensional units or as plain array. "
                                 f"Given units were {index_data.units}.")
            index_data = index_data.to_base_units().magnitude

        # convert wavelength to default units so that interpolation
        # coefficients are standardized
        wavelen = wavelength_data.to_preferred().magnitude
        fit = interp1d(wavelen, index_data, kind=kind)
        def index_func(wavelen):
            return fit(wavelen.to_preferred().magnitude)
        return cls(index_func)

# Refractive index dispersion formulas and Index objects.
#
# NOTE: If you add a material to this module, add a test function to
# structcol/tests/test_refractive_index.py that will test to make sure the
# dispersion relation returns the proper values of the refractive index at two
# or more points.

vacuum = Index.constant(1.0)

def _water_sellmeier(wavelen):
    # water data from M. Daimon and A. Masumura. Measurement of the refractive
    # index of distilled water from the near-infrared region to the ultraviolet
    # region, Appl. Opt. 46, 3811-3820 (2007).
    # Fit of the experimental data with the Sellmeier dispersion formula:
    # refractiveindex.info
    # data for high performance liquid chromatography (HPLC) distilled water at
    # 20.0 °C

    index = np.sqrt(5.684027565e-1 * wavelen**2
                    / (wavelen**2 - Quantity('5.101829712e-3 um^2'))
                    + 1.726177391e-1 * wavelen**2
                    / (wavelen**2 - Quantity('1.821153936e-2 um^2'))
                    + 2.086189578e-2 * wavelen**2
                    / (wavelen**2 - Quantity('2.620722293e-2 um^2'))
                    + 1.130748688e-1 * wavelen**2
                    / (wavelen**2 - Quantity('1.069792721e1 um^2'))
                    + 1)
    return index

water = Index(_water_sellmeier)

def _polystyrene_sellmeier(wavelen):
    # polystyrene data from N. Sultanova, S. Kasarova and I. Nikolov.
    # Dispersion properties of optical polymers, Acta Physica Polonica A 116,
    # 585-587 (2009).
    # Fit of the experimental data with the Sellmeier dispersion formula:
    # refractiveindex.info
    # data for 20 degrees C, 0.4368-1.052 micrometers
    index =  np.sqrt(1.4435 * wavelen**2
                     / (wavelen**2 - Quantity("0.020216 um^2"))
                     + 1)
    return index

polystyrene = Index(_polystyrene_sellmeier)

def _pmma_sellmeier(wavelen):
    # pmma data from G. Beadie, M. Brindza, R. A. Flynn, A. Rosenberg, and J.
    # S. Shirk. Refractive index measurements of poly(methyl methacrylate)
    # (PMMA) from 0.4-1.6 micrometers, Appl. Opt. 54, F139-F143 (2015)
    # refractiveindex.info
    # data for 20.1 degrees C, 0.42-1.62 micrometers
    index = np.sqrt(2.1778
                    + Quantity('6.1209e-3 um^-2') * wavelen**2
                    - Quantity('1.5004e-3 um^-4') * wavelen**4
                    + Quantity('2.3678e-2 um^2') / wavelen**2
                    - Quantity('4.2137e-3 um^4') / wavelen**4
                    + Quantity('7.3417e-4 um^6') / wavelen**6
                    - Quantity('4.5042e-5 um^8') / wavelen**8)

    return index

pmma = Index(_pmma_sellmeier)

def _rutile_sellmeier(wavelen):
    # rutile TiO2 from J. R. Devore. Refractive Indices of Rutile and
    # Sphalerite, J. Opt. Soc. Am. 41, 416-419 (1951)
    # refractiveindex.info
    # data for rutile TiO2, ordinary ray, 0.43-1.53 micrometers
    index = np.sqrt(5.913
                    + Quantity('0.2441 um^2')
                    / (wavelen**2 - Quantity('0.0803 um^2')))
    return index

rutile = Index(_rutile_sellmeier)

def _brookite_sellmeier(wavelen):
    # brookite TiO2 from Radhakrishnan. "The Optical Properties of titanium
    # dioxide". Proceedings of the Indian Academy of Sciences-Mathematical
    # Sciences March 1982, 35:117. Note that this is for n_alpha. However,
    # n_alpha is almost identical to n_beta, which in turn is very similar to
    # rutile. However n_gamma is a bit different, but is not considered
    # data for rutile TiO2, ordinary ray, 0.43-0.71 micrometers
    index = np.sqrt(2.9858 + 2.1036 * wavelen**2
                    / (wavelen**2 - Quantity('0.287**2 um^2'))
                    - Quantity('0.18 um^-2') * wavelen**2
                    +1.)
    return index

brookite = Index(_brookite_sellmeier)

def _anatase_cauchy(wavelen):
    # anatase TiO2 from Wang et al. Think Solid Films. 405, 2002, 50-54
    # measured from 500-1700 nm
    index = (2.1526 + Quantity('4.1155e-2 um^2') / (wavelen**2)
             + Quantity('2.1798e-3 um^4') / (wavelen**4))
    return index

anatase = Index(_anatase_cauchy)

def _fused_silica_sellmeier(wavelen):
    # fused silica (amorphous quartz) data from I. H. Malitson. Interspecimen
    # Comparison of the Refractive Index of Fused Silica, J. Opt. Soc. Am. 55,
    # 1205-1208 (1965)
    # refractiveindex.info
    # data for "room temperature", 0.21-3.71 micrometers
    index = np.sqrt(1 + 0.6961663 * wavelen**2
                    / (wavelen**2 - Quantity('0.0684043**2 um^2'))
                    + 0.4079426 * wavelen**2
                    / (wavelen**2 - Quantity('0.1162414**2 um^2'))
                    + 0.8974794 * wavelen**2
                    / (wavelen**2 - Quantity('9.896161**2 um^2')))

    return index

fused_silica = Index(_fused_silica_sellmeier)

def _soda_lime_glass_cauchy(wavelen):
    # soda lime glass data from M. Rubin. Optical properties of soda lime
    # silica glasses, Solar Energy Materials 12, 275-288 (1985)
    # refractiveindex.info
    # data for "room temperature", 0.31-4.6 micrometers
    index = (1.5130 - Quantity('0.003169 um^-2') * wavelen**2
             + Quantity('0.003962 um^2')/(wavelen**2))

    return index

soda_lime_glass = Index(_soda_lime_glass_cauchy)

def _zirconia_sellmeier(wavelen):
    # zirconia (ZrO2) data from I. Bodurov, I. Vlaeva, A. Viraneva,
    # T. Yovcheva, S. Sainov. Modified design of a laser refractometer,
    # Nanoscience & Nanotechnology 16, 31-33 (2016).
    # data for 24 degrees C, 0.405 - 0.635 micrometers
    index = np.sqrt(1 + 3.3037 * wavelen**2
                    / (wavelen**2 - Quantity('0.1987971**2 um**2')))
    return index

zirconia = Index(_zirconia_sellmeier)

def _ethanol_cauchy(wavelen):
    # ethanol data from J. Rheims, J Köser and T Wriedt. Refractive-index
    # measurements in the near-IR using an Abbe refractometer,
    # Meas. Sci. Technol. 8, 601-605 (1997)
    # refractiveindex.info
    index = (1.35265 + Quantity('0.00306 um^2') / (wavelen**2)
             + Quantity('0.00002 um^4') / (wavelen**4))
    return index

ethanol = Index(_ethanol_cauchy)

#------------------------------------------------------------------------------
# OTHER MATERIALS
# for the rest of these materials, need to find dispersion relations and
# implement the functions in the dictionary.

silica_colloidal = Index.constant(1.40)

keratin = Index.constant(1.532)

# from
# http://www.sigmaaldrich.com/catalog/product/aldrich/181587?lang=en&region=US
ptmba = Index.constant(1.46)


#------------------------------------------------------------------------------
# CARGILLE OILS

@ureg.check('[length]', None, None)
def n_cargille(wavelen, i, series):
    """Refractive index of cargille index-matching oils from
    http://www.cargille.com/refractivestandards.shtml

    Parameters
    ----------
    i: int
        The cardinal number of the liquid (starting with 0
        valid cardinal numbers:
        AAA: 0-19
        AA: 0-29
        A: 0-90
        B: 0-29
        E: 0-28
        acrylic: 0
    series: string
        the series of the cargille index matching liquid. Can be A, AA, AAA, B,
        E, or acrylic
    w : structcol.Quantity [length]
        Wavelength in vacuum.

    Returns
    -------
    structcol.Quantity (dimensionless)
        refractive index

    Examples
    --------
    To create an Index object for a particular oil (for example, series AA
    number 1) use
    >>> cargille = sc.Index(sc.index.n_cargille, i=1, series="AA")

    """
    cs = {}
    ds = {}
    es = {}

    # convert wavelength to micrometers to make units compatible for given oil
    # coefficients

    wavelen = wavelen.to('um')

    ## Series AAA ##

    cs['AAA'] = np.array([1.295542, 1.30031, 1.305078, 1.309845, 1.314614,
                    1.319379, 1.324146, 1.328914, 1.333685, 1.338451, 1.343219,
                    1.347986, 1.352753, 1.357522, 1.362290, 1.367058, 1.371824,
                    1.376592, 1.38136, 1.386127])

    ds['AAA'] = np.array([148828.2, 157595.9, 166363.5, 175352.1, 184119.7,
                    193034.7, 201949.6, 210643.5, 219558.5, 228252.4, 237020,
                    245861.3, 254850, 263470.2, 272237.8, 281079.1, 290067.7,
                    298835.3, 307603, 316444.2]) * 10**(-8)

    es['AAA'] = np.array([2.05E+11, 1.85E+11, 1.56E+11, 1.29E+11, 1.01E+11,
                    7.42E+10, 4.94E+10, 2.22E+10, -2.47E+09, -3.21E+10,
                    -5.44E+10, -8.16E+10, -1.06E+11, -1.33E+11, -1.58E+11,
                    -1.83E+11, -2.15E+11, -2.40E+11, -2.65E+11,
                    -2.89E+11]) * 10**(-16)

    ## Series AA ##

    cs['AA'] = np.array([1.387868, 1.389882, 1.391889, 1.393901, 1.395912,
                    1.397926, 1.399937, 1.401949, 1.403958, 1.40597, 1.407981,
                    1.409992, 1.412004, 1.414014, 1.416028, 1.418037, 1.420049,
                    1.422058, 1.42407, 1.426082, 1.428093, 1.430105, 1.432116,
                    1.434128, 1.43614, 1.438149, 1.440161, 1.442173, 1.444184,
                    1.446195])

    ds['AA'] = np.array([434180.6, 432928.1, 431970.3, 430644.1, 429465.3,
                    428360.1, 427033.9, 425928.8, 424971, 423571.1, 422465.9,
                    421287.1, 419960.9, 418929.4, 417455.9, 416571.7, 415392.9,
                    414214, 412961.5, 412003.7, 410530.2, 409572.4, 408393.5,
                    406993.7, 405814.8, 404783.4, 403604.5, 402425.7, 401173.2,
                    400141.7]) * 10**(-8)

    es['AA'] = np.array([-4.47E+11, -4.18E+11, -3.93E+11, -3.66E+11, -3.39E+11,
                    -3.14E+11, -2.77E+11, -2.57E+11, -2.27E+11, -2.03E+11,
                    -1.78E+11, -1.51E+11, -1.16E+11, -8.65E+10, -6.67E+10,
                    -3.71E+10, -1.24E+10, 1.48E+10, 4.45E+10, 7.17E+10,
                    9.64E+10, 1.24E+11, 1.51E+11, 1.80E+11, 2.08E+11, 2.35E+11,
                    2.60E+11, 2.87E+11, 3.19E+11, 3.44E+11]) * 10**(-16)

    ## Series A ##

    cs['A'] = np.array([1.447924, 1.449697, 1.451466, 1.453239, 1.45501,
                    1.456781, 1.458555, 1.460322, 1.462094, 1.463866, 1.465634,
                    1.467407, 1.469178, 1.470951, 1.472723, 1.474492, 1.476265,
                    1.478035, 1.479808, 1.481575, 1.483347, 1.485118, 1.486889,
                    1.488659, 1.490433, 1.492205, 1.493976, 1.495745, 1.497516,
                    1.499288, 1.501059, 1.502832, 1.504600, 1.506372, 1.508142,
                    1.509916, 1.511686, 1.513456, 1.515231, 1.517000, 1.518769,
                    1.520540, 1.522312, 1.524080, 1.525856, 1.527626, 1.529397,
                    1.531168, 1.532941, 1.534712, 1.536480, 1.538251, 1.540023,
                    1.541794, 1.543566, 1.545338, 1.546669, 1.548351, 1.550034,
                    1.551717, 1.553395, 1.555081, 1.556763, 1.558443, 1.560123,
                    1.561807, 1.563488, 1.565175, 1.566852, 1.568536, 1.570218,
                    1.571900, 1.573582, 1.575262, 1.576944, 1.578628, 1.580308,
                    1.581992, 1.583672, 1.585352, 1.587038, 1.588717, 1.590402,
                    1.592081, 1.593764, 1.595445, 1.597127, 1.598809, 1.600493,
                    1.602174, 1.603857])

    ds['A'] = np.array([407435.7, 413993, 420624, 426960.2, 433591.2, 440148.5,
                    446558.4, 453336.7, 459967.7, 466451.3, 473303.3, 479860.6,
                    486196.8, 492975.2, 499237.7, 505942.4, 512647.0, 519130.6,
                    525835.3, 532539.9, 538728.8, 545433.4, 551917.0, 558695.3,
                    565179.0, 571736.2, 578219.8, 584998.1, 591481.8, 598039.0,
                    604670.0, 611080.0, 618005.6, 624489.2, 630972.8, 637603.8,
                    644234.8, 650718.3, 657128.3, 663759.2, 670316.5, 677094.8,
                    683725.8, 690283.1, 696840.4, 703250.3, 709955.0, 716438.5,
                    722995.8, 729626.8, 736331.4, 742962.4, 749372.3, 756003.3,
                    762560.6, 769191.5, 785400.5, 792252.5, 799546.6, 806251.2,
                    813471.6, 820544.6, 827470.3, 834322.3, 841542.6, 848321.0,
                    855394.0, 862246.0, 869466.4, 876465.7, 883391.4, 890243.4,
                    897390.1, 904242.1, 911241.4, 918314.4, 925240.1, 932239.5,
                    939312.5, 946238.2, 953090.2, 960384.2, 967162.6, 974088.2,
                    981308.6, 988234.2, 995159.9, 1002380.0, 1009085.0,
                    1016084.0, 1023305.0]) * 10**(-8)

    es['A'] = np.array([4.15E+11, 4.62E+11, 5.12E+11, 5.61E+11, 6.08E+11,
                    6.50E+11, 7.05E+11, 7.49E+11, 7.96E+11, 8.45E+11, 8.90E+11,
                    9.42E+11, 9.89E+11, 1.04E+12, 1.08E+12, 1.13E+12, 1.18E+12,
                    1.23E+12, 1.27E+12, 1.32E+12, 1.37E+12, 1.41E+12, 1.46E+12,
                    1.51E+12, 1.56E+12, 1.60E+12, 1.65E+12, 1.70E+12, 1.75E+12,
                    1.80E+12, 1.84E+12, 1.89E+12, 1.94E+12, 1.99E+12, 2.04E+12,
                    2.08E+12, 2.13E+12, 2.18E+12, 2.23E+12, 2.27E+12, 2.32E+12,
                    2.37E+12, 2.42E+12, 2.45E+12, 2.51E+12, 2.56E+12, 2.61E+12,
                    2.66E+12, 2.70E+12, 2.75E+12, 2.80E+12, 2.84E+12, 2.89E+12,
                    2.94E+12, 2.99E+12, 3.03E+12, 3.27E+12, 3.41E+12, 3.55E+12,
                    3.70E+12, 3.83E+12, 3.97E+12, 4.11E+12, 4.26E+12, 4.40E+12,
                    4.54E+12, 4.68E+12, 4.82E+12, 4.96E+12, 5.10E+12, 5.24E+12,
                    5.38E+12, 5.53E+12, 5.66E+12, 5.80E+12, 5.95E+12, 6.08E+12,
                    6.23E+12, 6.36E+12, 6.51E+12, 6.65E+12, 6.79E+12, 6.93E+12,
                    7.07E+12, 7.21E+12, 7.35E+12, 7.49E+12, 7.64E+12, 7.78E+12,
                    7.92E+12, 8.05E+12]) * 10**(-16)

    ## Series B ##

    cs['B'] = np.array([1.605535, 1.607222, 1.608903, 1.610586, 1.612267,
                    1.613949, 1.61563, 1.617315, 1.617298, 1.619131, 1.62096,
                    1.622794, 1.624623, 1.626456, 1.628283, 1.630115, 1.631944,
                    1.633776, 1.635606, 1.637437, 1.639267, 1.641097, 1.642928,
                    1.644759, 1.646585, 1.64842, 1.650249, 1.652081, 1.653913,
                    1.655743])

    ds['B'] = np.array([1030304, 1037009, 1043861, 1050934, 1058154, 1065080,
                    1072005, 1079078, 1192615, 1193941, 1195415, 1196225,
                    1197404, 1198657, 1199983, 1201162, 1202341, 1203298,
                    1204551, 1205951, 1207130, 1208382, 1209561, 1210519,
                    1212066, 1212950, 1214129, 1215529, 1216708,
                    1217813]) * 10**(-8)

    es['B'] = np.array([8.19E+12, 8.34E+12, 8.48E+12, 8.62E+12, 8.76E+12,
                    8.90E+12, 9.04E+12, 9.18E+12, 7.66E+12, 7.82E+12, 7.99E+12,
                    8.15E+12, 8.32E+12, 8.48E+12, 8.64E+12, 8.80E+12, 8.97E+12,
                    9.13E+12, 9.29E+12, 9.45E+12, 9.61E+12, 9.78E+12, 9.95E+12,
                    1.01E+13, 1.03E+13, 1.04E+13, 1.06E+13, 1.08E+13, 1.09E+13,
                    1.11E+13]) * 10**(-16)

    ## Series E ##

    cs['E'] = np.array([1.47825, 1.482607, 1.486951, 1.491295, 1.495639,
                    1.499986, 1.504328, 1.508673, 1.51302, 1.517363, 1.521711,
                    1.526055, 1.531458, 1.538248, 1.545039, 1.549192, 1.553395,
                    1.5576302, 1.561807, 1.566011, 1.570218, 1.574422,
                    1.578628, 1.582834, 1.587038, 1.591241, 1.595445, 1.599651,
                    1.603857])

    ds['E'] = np.array([575420.1, 583524.6, 592071.2, 600470.4, 608648.6,
                    617121.5, 625299.6, 633698.9, 641950.8, 650350, 658601.8,
                    666780, 690062, 732500.2, 775085.7, 796083.8, 813471.6,
                    830859.5, 848321, 865782.5, 883391.4, 900852.9, 918314.4,
                    935849.7, 953090.2, 970551.7, 988234.2, 1005622,
                    1023157]) * 10**(-8)

    es['E'] = np.array([6.23E+12, 6.74E+12, 7.24E+12, 7.74E+12, 8.24E+12,
                    8.74E+12, 9.24E+12, 9.75E+12, 1.02E+13, 1.07E+13, 1.12E+13,
                    1.18E+13, 1.05E+13, 6.82E+12, 3.18E+12, 3.49E+12, 3.83E+12,
                    4.18E+12, 4.54E+12, 4.89E+12, 5.24E+12, 5.59E+12, 5.95E+12,
                    6.30E+12, 6.65E+12, 7.00E+12, 7.35E+12, 7.71E+12,
                    8.06E+12]) * 10**(-16)

    ## Acrylic-matching liquid, Code 5032

    cs['acrylic'] = np.array([1.478419])

    ds['acrylic'] = np.array([463182.1]) * 10**(-8)

    es['acrylic'] = np.array([-8.637338E+10]) * 10**(-16)

    try:
        n = cs[str(series)][i] + (ds[str(series)][i] / wavelen.magnitude**2) \
            + (es[str(series)][i] / wavelen.magnitude**4)
    except IndexError:
        raise ValueError("""An oil with this cardinal number was not found.
            Check your cardinal number and make sure it is valid for the
            selected series """)
    except KeyError:
        raise ValueError("""An oil of this series was not found.
            Check your series and make sure it is valid. """)

    return n

#------------------------------------------------------------------------------
# EFFECTIVE INDEX CALCULATION

def n_eff(n_particle, n_matrix, volume_fraction, maxwell_garnett=False):
    """
    Calculates Bruggeman effective refractive index for a composite of n
    dielectric media. If maxwell_garnett is set to true and there are two
    media, calculates the effective index using the Maxwell-Garnett
    formulation. Both Maxwell-Garnett and Bruggeman formulas can handle complex
    refractive indices.

    Parameters
    ----------
    n_particle: float
        refractive indices of the inclusion. If it's a core-shell particle,
        must be an array of indices from innermost to outermost layer.
    n_matrix: float
        refractive index of the matrix.
    volume_fraction: float or structcol.Quantity (dimensionless)
        volume fraction of inclusion. If it's a core-shell particle,
        must be an array of volume fractions from innermost to outermost layer.
    maxwell_garnett: boolean
        If true, the model uses Maxwell-Garnett's effective index for the
        sample. In that case, the user must specify one refractive index for
        the particle and one for the matrix. If false, the model uses
        Bruggeman's formula, which can be used for multilayer particles.

    Returns
    -------
    structcol.Quantity (dimensionless)
        refractive index

    References
    ----------
    [1] Markel, V. A. "Introduction to the Maxwell Garnett approximation:
        tutorial". Vol. 33, No. 7, Journal of the Optical Society of America A
        (2016).
        Bruggeman's equation in Eq. 29.
        Maxwell-Garnett relation in Eq. 18.

    """
    if isinstance(volume_fraction, Quantity):
        volume_fraction = volume_fraction.magnitude

    if maxwell_garnett:
        # check that the particle and matrix indices have the same length
        if (len(np.array([n_particle]).flatten())
            != len(np.array([n_matrix]).flatten())):
            raise ValueError('Maxwell-Garnett requires particle and '
                             'matrix index arrays to have the same length')
        ni = n_particle
        nm = n_matrix
        phi = volume_fraction
        neff =  nm * np.sqrt((2*nm**2 + ni**2 + 2*phi*((ni**2)-(nm**2))) /
                         (2*nm**2 + ni**2 - phi*((ni**2)-(nm**2))))

        if neff.imag == 0:
            return neff.real
        else:
            return neff

    else:
        # convert the particle index and volume fractions into 1D arrays
        n_particle = np.array([n_particle]).flatten()
        volume_fraction = np.array([volume_fraction]).flatten()

        # check that the number of volume fractions and of indices is the same
        if len(n_particle) != len(volume_fraction):
            raise ValueError('Arrays of indices and volume fractions '
                             'must be the same length')

        volume_fraction_matrix = Quantity(1 - np.sum(volume_fraction), '')

        # create arrays combining the particle and the matrix' indices and vf
        n_particle_list = n_particle.tolist()
        n_particle_list.append(n_matrix)
        n_array = np.array(n_particle_list)

        vf_list = volume_fraction.tolist()
        vf_list.append(volume_fraction_matrix.magnitude)
        vf_array = np.array(vf_list)

        # define a function for Bruggeman's equation
        def sum_bg(n_bg, vf, n_array):
            N = len(n_array.flatten())
            a, b = n_bg
            S = sum((vf[n]*(n_array[n]**2 - (a+b*1j)**2)
                     /(n_array[n]**2 + 2*(a+b*1j)**2))
                    for n in np.arange(0, N))
            return (S.real, S.imag)

        # set an initial guess and solve for Bruggeman's refractive index of
        # the composite
        # most refractive indices range between 1 and 3
        initial_guess = [1.5, 0]
        n_bg_real, n_bg_imag = fsolve(sum_bg, initial_guess, args=(vf_array,
                                                                   n_array))

        if n_bg_imag == 0:
            return n_bg_real
        elif n_bg_imag < 0:
            raise ValueError('Cannot find positive imaginary root for the '
                             'effective index')
        else:
            return n_bg_real + n_bg_imag*1j
