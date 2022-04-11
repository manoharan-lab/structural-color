# Copyright 2016, Sofia Makgiriadou, Vinothan N. Manoharan, Victoria Hwang, 
# Annie Stephenson
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
Routines for simulating and calculating structures, structural parameters, and
structure factors

.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>
.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Victoria Hwang <vhwang@g.harvard.edu>
.. moduleauthor :: Annie Stephenson <stephenson@g.harvard.edu>
"""

import numpy as np
from . import ureg, Quantity  # unit registry and Quantity constructor from pint
import scipy as sp
import pandas as pd
import scipy
import os

@ureg.check('[]','[]')    # inputs should be dimensionless
def factor_py(qd, phi):
    """
    Calculate structure factor of hard spheres using the Ornstein-Zernike equation
    and Percus-Yevick approximation [1]_ [2]_.
    
    Parameters:
    ----------
    qd: 1D numpy array
        dimensionless quantity that represents the frequency space value that 
        the structure factor depends on        
    phi: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix       
    
    Returns:
    -------
    1D numpy array:
        The structure factor as a function of qd.

    Notes
    -----
    Might not be accurate for volume fractions above 0.5 (need to check against
    some simulated structure factors).

    This code is fully vectorized, so you can feed it orthogonal vectors for
    both qd and phi and it will produce a 2D output:
        qd = np.arange(0.1, 20, 0.01)
        phi = np.array([0.15, 0.3, 0.45])
        s = structure.factor_py(qd.reshape(-1,1), phi.reshape(1,-1))

    References
    ----------
    [1] Boualem Hammouda, "Probing Nanoscale Structures -- The SANS Toolbox"
    http://www.ncnr.nist.gov/staff/hammouda/the_SANS_toolbox.pdf Chapter 32
    (http://www.ncnr.nist.gov/staff/hammouda/distance_learning/chapter_32.pdf)

    [2] Boualem Hammouda, "A Tutorial on Small-Angle Neutron Scattering from
    Polymers", http://www.ncnr.nist.gov/programs/sans/pdf/polymer_tut.pdf, pp.
    51--53.
    """

    # constants in the direct correlation function
    lambda1 = (1 + 2*phi)**2 / (1 - phi)**4
    lambda2 = -(1 + phi/2.)**2 /  (1 - phi)**4
    # Fourier transform of the direct correlation function (eq X.33 of [2]_)
    c = -24*phi*(lambda1 * (np.sin(qd) - qd*np.cos(qd)) / qd**3  -
                 6*phi*lambda2 * (qd**2 * np.cos(qd) - 2*qd*np.sin(qd) -
                                  2*np.cos(qd)+2.0) / qd**4 -
                 (phi*lambda1/2.) * (qd**4 * np.cos(qd) - 4*qd**3 * np.sin(qd) -
                                     12 * qd**2 * np.cos(qd) + 24*qd * np.sin(qd) +
                                     24 * np.cos(qd) - 24.0) / qd**6)
    # Structure factor at qd (eq X.34 of [2]_)
    return 1.0/(1-c)

def factor_para(qd, phi, sigma = .15):
    """
    Calculate structure factor of a structure characterized by disorder of the 
    second kind as defined in Guinier [1]. This type of structure is referred to as
    paracrystalline by Hoseman [2]. See also [3] for concise description.
    
    Parameters:
    ----------
    qd: 1D numpy array
        dimensionless quantity that represents the frequency space value that 
        the structure factor depends on        
    phi: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix       
    sigma: float
        The standard deviation of a Gaussian representing the distribution of 
        particle/void spacings in the structure. Sigma has implied units of 
        particle diamter squared. A larger sigma will give more broad peaks,
        and a smaller sigma more sharp peaks. 
    
    Returns:
    -------
    1D numpy array:
        The structure factor as a function of qd.

    Notes
    -----
    This code is fully vectorized, so you can feed it orthogonal vectors for
    both qd and phi and it will produce a 2D output:
        qd = np.arange(0.1, 20, 0.01)
        phi = np.array([0.15, 0.3, 0.45])
        s = structure.factor_py(qd.reshape(-1,1), phi.reshape(1,-1))

    References
    ----------
    [1] Guinier, A (1963). X-Ray Diffraction. San Francisco and London: WH Freeman.

    [2] Lindenmeyer, PH; Hosemann, R (1963). "Application of the Theory of 
    Paracrystals to the Crystal Structure Analysis of Polyacrylonitrile". 
    J. Applied Physics. 34: 42
    
    [3] https://en.wikipedia.org/wiki/Structure_factor#Disorder_of_the_second_kind
    """
    r = np.exp(-(qd*phi**(-1/3)*sigma)**2/2)
    return (1-r**2)/(1+r**2-2*r*np.cos(qd*phi**(-1/3)))

def factor_poly(q, phi, diameters, c, pdi):
    """
    Calculate polydisperse structure factor for a monospecies (one mean 
    particle size) or a bispecies (two different mean particle sizes) system, 
    each with its own polydispersity. The size distribution is assumed to be 
    the Schulz distribution, which tends to Gaussian when the polydispersity 
    goes to zero, and skews to larger sizes when the polydispersity becomes
    large. 
    
    Parameters
    ----------
    qd: 1D numpy array
        dimensionless quantity that represents the frequency space value that 
        the structure factor depends on 
    phi: structcol.Quantity [dimensionless]
        volume fraction of all the particles or voids in matrix
    diameters: array of structcol.Quantity [length]
        mean diameters of each species of particles (can be one for a 
        monospecies or two for bispecies). 
    c:  array of structcol.Quantity [dimensionless]
        'number' concentration of each species. For ex, a system composed of 90 
        A particles and 10 B particles would have c = [0.9, 0.1].
    pdi: array of float 
        polydispersity index of each species. 
        
    Returns
    -------
    1D numpy array: The structure factor as a function of qd.
    
    References
    ----------
    M. Ginoza and M. Yasutomi, "Measurable Structure Factor of a Multi-Species 
    Polydisperse Percus-Yevick Fluid with Schulz Distributed Diameters", 
    J. Phys. Soc. Japan, 68, 7, 2292-2297 (1999).
    """
    
    def fm(x, t, tm, m):
        if isinstance(x, Quantity):
            x = x.to('').magnitude
        if isinstance(t, Quantity):
            t = t.to('').magnitude
        if isinstance(tm, Quantity):
            tm = tm.to('').magnitude
        t = np.reshape(t, (len(np.atleast_1d(t)),1))
        tm = np.reshape(tm, (len(tm),1))
        return(tm * (1 + x/(t+1))**(-(t+1+m)))
    
    def tm(m, Dsigma, t):
        t = np.reshape(t, (len(np.atleast_1d(t)),1))
        num_array = np.arange(m, 0, -1) + t
        prod = np.prod(num_array, axis=1).reshape((len(t), 1))
        return(prod / (t + 1)**m)      

    # if the pdi is zero, assume it's very small (we get the same results)
    # because otherwise we get a divide by zero error
    #pdi = Quantity(np.atleast_1d(pdi).astype(float), pdi.units)
    pdi = np.atleast_1d(pdi).astype(float).magnitude
    np.atleast_1d(pdi)[np.atleast_1d(pdi) < 1e-5] = 1e-5  

    Dsigma = pdi**2    
    Delta = 1 - phi
    t = np.abs(1/Dsigma) - 1
    t0 = tm(0, Dsigma, t)
    t1 = tm(1, Dsigma, t)
    t2 = Dsigma + 1
    t3 = (Dsigma + 1) * (2*Dsigma + 1)
    
    # if monospecies, no need to calculate individual species parameters
    if len(np.atleast_1d(c)) == 1:
        rho = 6*phi/(t3*np.pi*diameters**3)
    else:
        phi_ratio = 1 / (c[0]/c[1] * (diameters[0]/diameters[1])**3 * 
                        t3[0]/t3[1] + 1)
        phi_tau1 = phi_ratio * phi
        phi_tau0 = phi - phi_tau1

        rho_tau0 = 6*phi_tau0/(t3[0]*np.pi*diameters[0]**3)
        rho_tau1 = 6*phi_tau1/(t3[1]*np.pi*diameters[1]**3)
        rho = rho_tau0 + rho_tau1

    # this is the "effective" mean interparticle spacing
    sigma0 = (6*phi / (np.pi*rho))**(1/3)
   
    #q = qd / sigma0

    t2 = np.reshape(t2, (len(np.atleast_1d(t2)),1))
    c = np.reshape(c, (len(np.atleast_1d(c)),1))
    diameters = np.reshape(diameters, (len(np.atleast_1d(diameters)),1))
    
    q_shape = q.shape
    if len(q_shape)==2:
        q = Quantity(np.ndarray.flatten(q.magnitude), q.units) # added
    s = 1j*q
    x = s*diameters
    F0 = rho 
    zeta2 = rho * sigma0**2 
    
    f0 = fm(x,t,t0,0)
    f1 = fm(x,t,t1,1)
    f2 = fm(x,t,t2,2)
    f0_inv = fm(-x,t,t0,0)
    f1_inv = fm(-x,t,t1,1)
    f2_inv = fm(-x,t,t2,2)
  
    fa = 1/x**3 * (1 - x/2 - f0 - x/2 * f1)
    fb = 1/x**3 * (1 - x/2 * t2 - f1 - x/2 * f2)
    fc = 1/x**2 * (1 - x - f0)
    fd = 1/x**2 * (1 - x*t2 - f1)

    Ialpha1 = 24/s**3 * np.sum(c * F0 * (-1/2*(1-f0) + x/4 * (1 + f1)), axis=0)
    Ialpha2 = 24/s**3 * np.sum(c * F0 * (-diameters/2 * (1-f1) + 
                               s*diameters**2/4 * (t2 + f2)), axis=0)

    Iw1 = 2*np.pi*rho/(Delta*s**3) * (Ialpha1 + s/2*Ialpha2)
    Iw2 = (np.pi*rho/(Delta*s**2) * (1 + np.pi*zeta2/(Delta*s))*Ialpha1 + 
           np.pi**2*zeta2*rho/(2*Delta**2*s**2) * Ialpha2)

    F11 = np.sum(c*2*np.pi*rho*diameters**3/Delta * fa, axis=0)  
    F12 = np.sum(c/diameters * ((np.pi/Delta)**2 *rho*zeta2*diameters**4*fa + 
                 np.pi*rho*diameters**3/Delta * fc), axis=0)
    F21 = np.sum(c * diameters * 2*np.pi*rho*diameters**3/Delta * fb, axis=0)
    F22 = np.sum(c * ((np.pi/Delta)**2 *rho*zeta2*diameters**4*fb + 
                 np.pi*rho*diameters**3/Delta * fd), axis=0)
                
    FF11 = 1 - F11
    FF12 = -F12
    FF21 = -F21
    FF22 = 1 - F22
    
    G11 = FF22 / (FF11 * FF22 - FF12 * FF21)
    G12 = -FF12 / (FF11 * FF22 - FF12 * FF21)
    G21 = -FF21 / (FF11 * FF22 - FF12 * FF21)
    G22 = FF11 / (FF11 * FF22 - FF12 * FF21)
    
    I0 = -9/2*(2/s)**6 * np.sum(c * F0**2 * (1-1/2*(f0_inv + f0) + 
                                x/2 *(f1_inv - f1) - 
                                (s**2*diameters**2)/8 * (f2_inv + f2 + 2*t2)), 
                                axis=0)
    
    term1 = Iw1 * G11 * Ialpha1 / I0
    term2 = Iw1 * G12 * Ialpha2 / I0
    term3 = Iw2 * G21 * Ialpha1 / I0
    term4 = Iw2 * G22 * Ialpha2 / I0

    h2 = (term1 + term2 + term3 + term4).real
    
    SM = 1 - 2*h2
    SM[SM<0] = 0
    if len(q_shape)==2:
        SM = np.reshape(SM,q_shape)
    return(SM)
    
def factor_data(qd, s_data, qd_data):
    """
    Calculate an interpolated structure factor using data
    
    Parameters:
    ----------
    qd: 1D numpy array
        dimensionless quantity that represents the frequency space value that 
        the structure factor depends on     
    s_data: 1D numpy array
        structure factor values from data
    qd_data: 1D numpy array
        qd values from data
    
    Returns:
    -------
    1D numpy array:
        The structure factor as a function of qd.
    """
    s_func = sp.interpolate.interp1d(qd_data, s_data, kind = 'linear',
                                     bounds_error=False, fill_value=s_data[0])
    
    return s_func(qd)
    
def field_phase_data(qd, filename='spf.dat'):
    s_file = os.path.join(os.getcwd(),filename)
    s_phase_data=np.loadtxt(s_file)
    qd_data = s_phase_data[:,0]
    s_phase = s_phase_data[:,1]
    s_phase_func = sp.interpolate.interp1d(qd_data, s_phase, kind = 'linear',
                                           bounds_error=False, fill_value=s_phase_data[0,1])
    return s_phase_func(qd)
    
def phase_factor(qd, phi, n=1000):
    # define r/d
    r_d = np.linspace(0,10, n)
    
    # calculate g
    g = radial_dist_py(phi, x = r_d)
    integral = np.zeros(qd.shape)
    rho = 3.0 * phi / (4.0 * np.pi) # dimensionless rho*sigma**3
    
    # calculate the integral for each qd
    for i in range(qd.shape[0]):
        for j in range(qd.shape[1]):
            bessel = rho*4*np.pi*r_d**2*np.pi*scipy.special.jv(0, qd[i,j]*r_d)
            integral[i,j] = np.trapz(bessel*g, x=r_d)
        
    return integral

    
def field_phase_py(qd, phi, n=10000, r_d=np.arange(1,5,0.005)):
    '''
    Calcualte the phase shift contribution based on the radial distribution
    function calculated using the Percus-Yevick approximation
    
    Parameters:
    ----------
    qd: 1D numpy array
        dimensionless quantity q times diameter        
    phi: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix  
    n: float  
        number of samples of g(r)
    r_d: 1D numpy array 
        range of radial positions normalized by particle diameter. 
    
    Returns:
    --------
    field_s: 1D numpy array
        phase shift contributions based on the structure
    '''
    # calculate radial distribution function up to r/R= 5
    #g_file = os.path.join(os.getcwd(),'g_4.csv')
    #df=pd.read_csv(g_file, sep=',',header=None)
    #r_d = np.array(df[0])
    #g = np.array(df[1])
    g = radial_dist_py(phi, x = r_d)
    
    # sample the g of r probability distribution
    r_samp = np.random.choice(r_d, n, p = g/np.sum(g))
    
    # calculate the field term
    field_s = np.zeros(qd.shape, dtype='complex')
    for i in range(qd.shape[0]):
        for j in range(qd.shape[1]):
            field_s[i,j] = 1/n*np.sum(np.exp(1j*qd[i,j]*r_samp))
            
    return field_s
    
def radial_dist_py(phi, x=np.arange(1,5,0.005)):
    '''
    Calcualte the radial distribution function for hard spheres using the Percus-Yevick approximation.
    
    This function and its helper functions is based on the code found here:
    https://github.com/FTurci/hard-spheres-utilities/blob/master/Percus-Yevick.py
    This method for calculating g(r) is described in the SI of: 
    J. W. E. Drewitt, F. Turci, B. J. Heinen, S. G. Macleod, F. Qin, A. K. Kleppe, and O. T. Lord. Phys. Rev. Lett. 124
    
    Parameters:
    -----------
    phi: structcol.Quantity [dimensionless]
        volume fraction of particles or voids in matrix   
    x: 1D numpy array
       dimensionless value defined as position over particle diameter (r/d)
    
    Returns:
    --------
    g_fcn(x): 1D numpy array
            The radial distribution function calculated at the specified x values.
    '''
    # number density
    if isinstance(phi,Quantity):
        phi = phi.magnitude
    rho=6./np.pi*phi
    
    # get the direct correlation function c(r) from the analytic Percus-Yevick solution
    # vectorizing the function
    c=np.vectorize(cc)
    
    # space discretization
    dr=0.005
    r=np.arange(1,1024*2+1,1 )*dr
    
    # reciprocal space discretization (highest available frequency)
    dk=1/r[-1]
    k=np.arange(1,1024*2+1,1 )*dk
    
    # direct correlation function c(r)
    c_direct=c(r,phi)
    
    # calculate the Fourier transform
    ft_c_direct=spherical_FT(c_direct, k,r,dr)
    
    # using the Ornstein-Zernike equation, calculate the structure factor
    ft_h=ft_c_direct/(1.-rho*ft_c_direct)
    
    # inverse Fourier transform
    h=inverse_spherical_FT(ft_h, k,r,dk)
    
    # radial distribution function
    gg=h+1
    
    # clean the r<1 region
    g=np.zeros(len(gg))
    g[r>=1]=gg[r>=1]
    
    # make g function from interpolation
    g_fcn=sp.interpolate.InterpolatedUnivariateSpline(r, g)
    
    return g_fcn(x)

def spherical_FT(f,k,r,dr):
    '''
    Spherical Fourier Transform (using the liquid isotropicity)
    '''
    ft=np.zeros(len(k))
    for i in range(len(k)):
        ft[i]=4.*np.pi*np.sum(r*np.sin(k[i]*r)*f*dr)/k[i]
    return ft

def inverse_spherical_FT(ff,k,r,dk):
    '''
    Inverse spherical Fourier Transform (using the liquid isotropicity)
    '''
    ift=np.zeros(len(r))
    for i in range(len(r)):
        ift[i]=np.sum(k*np.sin(k*r[i])*ff*dk)/r[i]/(2*np.pi**2)
    return ift

# functions to calcualte direct correlation function
# from Percus-Yevick. See D. Henderson "Condensed Matter Physics" 2009, Vol. 12, No. 2, pp. 127-135
# or M. S Wertheim "Exact Solutions of the Percus-Yevick Integral for Hard Spheres" PRL. Vol. 10, No. 8, 1963
def c0(eta):
    return -(1.+2.*eta)**2/(1.-eta)**4
def c1(eta):
    return 6.*eta*(1.+eta*0.5)**2/(1.-eta)**4
def c3(eta):
    return eta*0.5*c0(eta)
def cc(r,eta):
    if r>1:
        return 0
    else:
        return c0(eta)+c1(eta)*r +c3(eta)*r**3