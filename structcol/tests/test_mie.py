# Copyright 2016, Vinothan N. Manoharan
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
Tests for the mie module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, index_ratio, size_parameter, np, mie
from pytest import raises
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError

def test_cross_sections():
    # Test cross sections against values calculated from BHMIE code (originally
    # calculated for testing fortran-based Mie code in holopy)

    # test case is PS sphere in water
    wavelen = Quantity('658.0 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.33, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    qscat, qext, qback = mie.calc_efficiencies(m, x)
    g = mie.calc_g(m,x)   # asymmetry parameter

    qscat_std, qext_std, g_std = 3.6647, 3.6677, 0.92701
    assert_almost_equal(qscat, qscat_std, decimal=4)
    assert_almost_equal(qext, qext_std, decimal=4)
    assert_almost_equal(g, g_std, decimal=4)

    # test to make sure calc_cross_sections returns the same values as
    # calc_efficiencies and calc_g
    cscat = qscat * np.pi * radius**2
    cext = qext * np.pi * radius**2
    cback  = qback * np.pi * radius**2
    cscat2, cext2, _, cback2, g2 = mie.calc_cross_sections(m, x, wavelen/n_matrix)
    assert_almost_equal(cscat.to('m^2').magnitude, cscat2.to('m^2').magnitude)
    assert_almost_equal(cext.to('m^2').magnitude, cext2.to('m^2').magnitude)
    assert_almost_equal(cback.to('m^2').magnitude, cback2.to('m^2').magnitude)
    assert_almost_equal(g, g2.magnitude)

    # test that calc_cross_sections throws an exception when given an argument
    # with the wrong dimensions
    raises(DimensionalityError, mie.calc_cross_sections,
                  m, x, Quantity('0.25 J'))
    raises(DimensionalityError, mie.calc_cross_sections,
                  m, x, Quantity('0.25'))

def test_form_factor():
    wavelen = Quantity('658.0 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)

    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # these values are calculated from MiePlot
    # (http://www.philiplaven.com/mieplot.htm), which uses BHMIE
    iperp_bhmie = np.array([2046.60203864487, 1282.28646423634, 299.631502275208,
                            7.35748912156671, 47.4215270799552, 51.2437259188946,
                            1.48683515673452, 32.7216414263307, 1.4640166361956,
                            10.1634538431238, 4.13729254895905, 0.287316587318158,
                            5.1922111829055, 5.26386476102605, 1.72503962851391,
                            7.26013963969779, 0.918926070270738, 31.5250813730405,
                            93.5508557840006])
    ipar_bhmie = np.array([2046.60203864487, 1100.18673543798, 183.162880455348,
                           13.5427093640281, 57.6244243689505, 35.4490544770251,
                           41.0597781235887, 14.8954859951121, 34.7035437764261,
                           5.94544441735711, 22.1248452485893, 3.75590232882822,
                           10.6385606309297, 0.881297551245856, 16.2259629218812,
                           7.24176462105438, 76.2910238480798, 54.1983836607738,
                           93.5508557840006])

    ipar, iperp = mie.calc_ang_dist(m, x, angles)
    assert_array_almost_equal(ipar, ipar_bhmie)
    assert_array_almost_equal(iperp, iperp_bhmie)

def test_efficiencies():
    x = np.array([0.01, 0.01778279, 0.03162278, 0.05623413, 0.1, 0.17782794,
                  0.31622777, 0.56234133, 1, 1.77827941, 3.16227766, 5.62341325,
                  10, 17.7827941, 31.6227766, 56.23413252, 100, 177.827941,
                  316.22776602, 562.34132519, 1000])
    # these values are calculated from MiePlot
    # (http://www.philiplaven.com/mieplot.htm), which uses BHMIE
    qext_bhmie = np.array([1.86E-06, 3.34E-06, 6.19E-06, 1.35E-05, 4.91E-05,
                           3.39E-04, 3.14E-03, 3.15E-02, 0.2972833954,
                           1.9411047797, 4.0883764682, 2.4192037463, 2.5962875796,
                           2.097410246, 2.1947770304, 2.1470056626, 2.1527225028,
                           2.0380806126, 2.0334715395, 2.0308028599, 2.0248011731])
    qsca_bhmie = np.array([3.04E-09, 3.04E-08, 3.04E-07, 3.04E-06, 3.04E-05,
                           3.05E-04, 3.08E-03, 3.13E-02, 0.2969918262,
                           1.9401873562, 4.0865768252, 2.4153820014,
                           2.5912825599, 2.0891233123, 2.1818510296,
                           2.1221614258, 2.1131226379, 1.9736114111,
                           1.922984002, 1.8490112847, 1.7303694187])
    qback_bhmie = np.array([3.62498741762823E-10, 3.62471372652178E-09,
                            3.623847844672E-08, 3.62110791613906E-07,
                            3.61242786911475E-06, 3.58482008581018E-05,
                            3.49577114878315E-04, 3.19256234186963E-03,
                            0.019955229811329, 1.22543944129328E-02,
                            0.114985907473273, 0.587724020116958,
                            0.780839362788633, 0.17952369257935,
                            0.068204471161473, 0.314128510891842,
                            0.256455963161882, 3.84713481428992E-02,
                            1.02022022710453, 0.51835427781473,
                            0.331000402174976])

    wavelen = Quantity('658.0 nm')
    n_matrix = Quantity(1.00, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)

    effs = [mie.calc_efficiencies(m, x) for x in x]
    q_arr = np.asarray(effs)
    qsca = q_arr[:,0]
    qext = q_arr[:,1]
    qback = q_arr[:,2]
    # use two decimal places for the small size parameters because MiePlot
    # doesn't report sufficient precision
    assert_array_almost_equal(qsca[0:9], qsca_bhmie[0:9], decimal=2)
    assert_array_almost_equal(qext[0:9], qext_bhmie[0:9], decimal=2)
    # there is some disagreement at 4 decimal places in the cross
    # sections at large x.  Not sure if this points to a bug in the algorithm
    # or improved precision over the bhmie results.  Should be investigated
    # more.
    assert_array_almost_equal(qsca[9:], qsca_bhmie[9:], decimal=3)
    assert_array_almost_equal(qext[9:], qext_bhmie[9:], decimal=3)

    # test backscattering efficiencies (still some discrepancies at 3rd decimal
    # point for large size parameters)
    assert_array_almost_equal(qback, qback_bhmie, decimal=2)

def test_absorbing_materials():
    # test calculations for gold, which has a high imaginary refractive index
    wavelen = Quantity('658.0 nm')
    n_matrix = Quantity(1.00, '')
    n_particle = Quantity(0.1425812 + 3.6813284 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)
    x = 10.0

    angles = Quantity(np.linspace(0, 90., 10), 'deg')
    # these values are calculated from MiePlot
    # (http://www.philiplaven.com/mieplot.htm), which uses BHMIE
    iperp_bhmie = np.array([4830.51401095968, 2002.39671236719,
                            73.6230330613015, 118.676685975947,
                            38.348829860926, 46.0044258298926,
                            31.3142368857685, 31.3709239005213,
                            27.8720309121251, 27.1204995833711])
    ipar_bhmie = np.array([4830.51401095968, 1225.28102200945,
                           216.265206462472, 17.0794942389782,
                           91.4145998381414, 39.0790253214751,
                           24.9801217735053, 53.2319915708624,
                           8.26505988320951, 47.4736966179677])

    ipar, iperp = mie.calc_ang_dist(m, x, angles)
    assert_array_almost_equal(ipar, ipar_bhmie)
    assert_array_almost_equal(iperp, iperp_bhmie)

