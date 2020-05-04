# Copyright 2016, Vinothan N. Manoharan, Victoria Hwang
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
Tests for the refractive_index module of structcol

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
"""

from .. import refractive_index as ri
from .. import Quantity
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_warns
from pint.errors import DimensionalityError
import numpy as np

def test_n():
    # make sure that a material not in the dictionary raises a KeyError
    assert_raises(KeyError, ri.n, 'badkey', Quantity('0.5 um'))

    # make sure that specifying no units throws an exception
    assert_raises(AttributeError, ri.n, 'polystyrene', 0.5)
    # and specifying the wrong units, too
    assert_raises(DimensionalityError, ri.n, 'polystyrene', Quantity('0.5 J'))

# the next few tests make sure that the various dispersion formulas give values
# of n close to those listed by refractiveindex.info (or other source) at the
# boundaries of the visible spectrum.  This is mostly to make sure that the
# coefficients of the dispersion formulas are entered properly

def test_water():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('water', Quantity('0.40930 um')),
                        Quantity('1.3427061376724'))
    assert_almost_equal(ri.n('water', Quantity('0.80700 um')),
                        Quantity('1.3284883366632'))

def test_npmma():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('pmma', Quantity('0.42 um')),
                        Quantity('1.5049521933717'))
    assert_almost_equal(ri.n('pmma', Quantity('0.804 um')),
                        Quantity('1.4866523830528'))

def test_nps():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('polystyrene', Quantity('0.4491 um')),
                        Quantity('1.6137854760669'))
    assert_almost_equal(ri.n('polystyrene', Quantity('0.7998 um')),
                        Quantity('1.5781660671827'))

def test_rutile():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('rutile', Quantity('0.4300 um')),
                        Quantity('2.8716984534676'))
    assert_almost_equal(ri.n('rutile', Quantity('0.8040 um')),
                        Quantity('2.5187663081355'))

def test_fused_silica():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('fused silica', Quantity('0.3850 um')),
                        Quantity('1.4718556531995'))
    assert_almost_equal(ri.n('fused silica', Quantity('0.8050 um')),
                        Quantity('1.4532313266004'))
def test_zirconia():
    # values from refractiveindex.info
    assert_almost_equal(ri.n('zirconia', Quantity('.405 um')),
                       Quantity('2.3135169070958'))
    assert_almost_equal(ri.n('zirconia', Quantity('.6350 um')),
                        Quantity('2.1593242574339'))


def test_vacuum():
    assert_almost_equal(ri.n('vacuum', Quantity('0.400 um')), Quantity('1.0'))
    assert_almost_equal(ri.n('vacuum', Quantity('0.800 um')), Quantity('1.0'))
    
def test_cargille():
    assert_almost_equal(ri.n_cargille(1,'AAA',Quantity('0.400 um')),
                        Quantity('1.3101597437500001'))
    assert_almost_equal(ri.n_cargille(1,'AAA',Quantity('0.700 um')),
                        Quantity('1.303526242857143'))
    assert_almost_equal(ri.n_cargille(1,'AA',Quantity('0.400 um')),
                        Quantity('1.4169400062500002'))
    assert_almost_equal(ri.n_cargille(1,'AA',Quantity('0.700 um')),
                        Quantity('1.3987172673469388'))
    assert_almost_equal(ri.n_cargille(1,'A',Quantity('0.400 um')),
                        Quantity('1.4755715625000001'))
    assert_almost_equal(ri.n_cargille(1,'A',Quantity('0.700 um')),
                        Quantity('1.458145836734694'))
    assert_almost_equal(ri.n_cargille(1,'B',Quantity('0.400 um')),
                        Quantity('1.6720350625'))
    assert_almost_equal(ri.n_cargille(1,'B',Quantity('0.700 um')),
                        Quantity('1.6283854489795917'))
    assert_almost_equal(ri.n_cargille(1,'E',Quantity('0.400 um')),
                        Quantity('1.5190772875'))
    assert_almost_equal(ri.n_cargille(1,'E',Quantity('0.700 um')),
                        Quantity('1.4945156653061225'))
    assert_almost_equal(ri.n_cargille(0,'acrylic',Quantity('0.400 um')),
                        Quantity('1.50736788125'))
    assert_almost_equal(ri.n_cargille(0,'acrylic',Quantity('0.700 um')),
                        Quantity('1.4878716959183673'))
    
def test_neff():
    # test that at low volume fractions, Maxwell-Garnett and Bruggeman roughly
    # match for a non-core-shell particle
    n_particle = Quantity(2.7, '')
    n_matrix = Quantity(2.2, '')
    vf = Quantity(0.001, '')
    
    neff_mg = ri.n_eff(n_particle, n_matrix, vf, maxwell_garnett=True)
    neff_bg = ri.n_eff(n_particle, n_matrix, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg, neff_bg)
    
    # test that the non-core-shell particle with Maxwell-Garnett matches with 
    # the core-shell of shell index of air with Bruggeman at low volume fractions
    n_particle2 = Quantity(np.array([2.7, 2.2]), '')
    vf2 = Quantity(np.array([0.001, 0.1]), '')
    neff_bg2 = ri.n_eff(n_particle2, n_matrix, vf2, maxwell_garnett=False)
    
    assert_almost_equal(neff_mg, neff_bg2)
    assert_almost_equal(neff_bg, neff_bg2)
    
    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    vf3 = Quantity(0.5, '')
    neff_bg3 = ri.n_eff(n_particle, n_matrix, vf3, maxwell_garnett=False)
    
    vf3_cs = Quantity(np.array([0.5, 0.1]), '')
    neff_bg3_cs = ri.n_eff(n_particle2, n_matrix, vf3_cs, maxwell_garnett=False)
    
    assert_almost_equal(neff_bg3, neff_bg3_cs)
    
    # repeat the tests using complex indices    
    n_particle_complex = Quantity(2.7+0.001j, '')
    n_matrix_complex = Quantity(2.2+0.001j, '')
    
    neff_mg_complex = ri.n_eff(n_particle_complex, n_matrix_complex, vf, maxwell_garnett=True)
    neff_bg_complex = ri.n_eff(n_particle_complex, n_matrix_complex, vf, maxwell_garnett=False)

    assert_almost_equal(neff_mg_complex, neff_bg_complex)
    
    # test that the non-core-shell particle with Maxwell-Garnett matches with 
    # the core-shell of shell index of air with Bruggeman at low volume fractions
    n_particle2_complex = Quantity(np.array([2.7+0.001j, 2.2+0.001j]), '')
    neff_bg2_complex = ri.n_eff(n_particle2_complex, n_matrix_complex, vf2, maxwell_garnett=False)
    
    assert_almost_equal(neff_mg_complex, neff_bg2_complex)
    assert_almost_equal(neff_bg_complex, neff_bg2_complex)
    
    # test that the effective indices for a non-core-shell and a core-shell of
    # shell index of air match using Bruggeman at intermediate volume fractions
    neff_bg3_complex = ri.n_eff(n_particle_complex, n_matrix_complex, vf3, maxwell_garnett=False)
    
    neff_bg3_cs_complex = ri.n_eff(n_particle2_complex, n_matrix_complex, vf3_cs, maxwell_garnett=False)
    
    assert_almost_equal(neff_bg3_complex, neff_bg3_cs_complex)
    
def test_data():
    # Test that we can input data for refractive index
    wavelength = Quantity(np.array([400,500,600]), 'nm')
    data = np.array([1.5,1.55,1.6])
    assert_equal(ri.n('data', wavelength, index=data).all(), data.all())
    
    # Test that it also works for complex values
    data_complex = np.array([1.5+0.01j,1.55+0.02j,1.6+0.03j])
    assert_equal(ri.n('data', wavelength, index=data_complex).all(), data_complex.all())
    
    # Test that keyerror is raised when no index is specified for 'data'
    assert_raises(KeyError, ri.n, 'data', Quantity('0.5 um'), index=None)

    # Test warning message when user specifies index for a material other than 'data'
    assert_warns(Warning, ri.n, 'water', Quantity('0.5 um'), index=data)