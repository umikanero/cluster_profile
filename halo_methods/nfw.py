## @file nfw.py
#
#  NFW FUNCTIONS 
#
#  Functions for calculating NFW parameters.
#
#  REFERENCES:
#  1) Lokas and Mamon, Properties of spherical galaxies and clusters
#  with an NFW density profile, 2001, MNRAS 321, 155. (LM2001)
#  2) Mamon, Biviano and Boue, MAMPOSSt: Modelling Anisotropy and Mass
#  Profiles of Observed Spherical Systems - I. Gaussian 3D velocities,
#  2013, MNRAS, 429, 3079. (MBB2013)
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np

##
#  Function calculates the NFW profile
#  projected surface density.
#
#  Equations 42 and 43 from LM2001.
#
#  @param[in] t: t = R / r_s.
#
def nfw_proj_sd(t):

    if np.around(t, 8) == 0.0:
        sd = 0.0
        
    elif 1.0 - np.around(t, 8) == 0.0:
        sd = 1.0 / 3.0
        
    else:
        cm1 = 1.0
        if t < 1:
                cm1 = np.arccosh(1.0 / t)
        elif t > 1:
                cm1 = np.arccos(1.0 / t)
        sd = (1.0 - cm1 / np.sqrt(np.abs(t ** 2 - 1.0))) / (t ** 2 - 1.0)
    
    return sd / (2.0 * np.log(2.0) - 1.0)

##
#  Function calculates the NFW profile
#  projected mass along a cylinder in
#  units of M(r-2).
#
#  Equation B1 from MBB2013.
#
#  @param[in] t: t = R / r_s.
#
def nfw_proj_mass(t):

    if np.around(t, 8) == 0.0:
        mp = 0.0

    elif 1.0 - np.around(t, 8) == 0.0:
        mp = 1.0 - np.log(2.0)
        
    else:
        capc = 1.0
        if t > 1:
            capc = np.arccos(1.0 / t)
        elif t < 1:
            capc = np.arccosh(1.0 / t)
        mp = capc / np.sqrt(np.abs(t ** 2 - 1.0)) + np.log(t / 2.0)

    return mp / (np.log(2.0) - 0.5)

##
#  NFW number density for given projected
#  radius, scale radius and background
#  density.
#
#  @param[in] R_proj: Projected radius.
#  @param[in] r_scale: Scale radius.
#  @param[in] bg_density: Background density.
#
def nfw_num_density(R_proj, r_scale, bg_density):

    t_up = max(R_proj) / r_scale
    t_low = min(R_proj) / r_scale

    nfw_num = len(R_proj) - np.pi * r_scale ** 2 * (t_up ** 2 - t_low ** 2) * bg_density
    nfw_den = np.pi * r_scale ** 2 * (nfw_proj_mass(t_up) - nfw_proj_mass(t_low))

    if nfw_num <= 0:
        nfw_num = 0.1

    return nfw_num / nfw_den

##
#  Maximum liklihood for projected NFW
#  with background.
#
#  @param[in] r_scale_bg: Scale radius
#  and background density.
#  @param[in] R_proj: Projected radius.
#  @param[in] weights: Optional weights.
#
def nfw_proj_maxlik_bg(r_scale_bg, R_proj, weights = None):
    
    nfw_sd = np.array(map(nfw_proj_sd, R_proj / r_scale_bg[0]))

    nfw_nden = nfw_num_density(R_proj, r_scale_bg[0], r_scale_bg[1])
    
    prob = nfw_nden * nfw_sd + r_scale_bg[1]

    if weights is None:
        return np.sum(-np.log(prob))

    else:
        return np.sum(-np.log(prob) * weights)


############ 3D NFW Profile ############
from colossus.cosmology import cosmology

cosmo = cosmology.setCosmology('planck15');
# Cosmology "planck15"
#     flat = True, Om0 = 0.3089, Ode0 = 0.6910, Ob0 = 0.0486, H0 = 67.74, sigma8 = 0.8159, ns = 0.9667
#     de_model = lambda, relspecies = True, Tcmb0 = 2.7255, Neff = 3.0460, powerlaw = False

Omega_m0 = cosmo.Om0
Omega_Lambda0 = cosmo.Ode0
Omega_gamma0 = 0
Omega_k0=  0

light_speed = 2.99792458E6 # speed of light (km/s)
gravit_const_SI = 6.67428E-11 #gravitational constant (m^3.s^-2.kg^-1)
heliocentric_gravit_const = 1.3271244E20 #heliocentric gravitational constant (m^3.s^-2)
solar_mass = heliocentric_gravit_const/gravit_const_SI #solar mass (kg)
kiloparsec = 3.08567758149E19 #kiloparsec (m)
gravit_const = gravit_const_SI/kiloparsec*solar_mass*1.0E-6 #gravitational constant (kpc.km^2.s^-2.M_\odot^-1)
hubble_rate0 = 1.0E-1 #presente day Hubble rate (h.km.s^-1.kpc-1)
crit_density0 = (3.0*hubble_rate0**2.0)/(8.0*np.pi*gravit_const) #critical density (M_\odot.h^2.kpc^-3)
hubble_distance = light_speed/hubble_rate0 #hubble distante (kpc/h)

#mass density of the universe at redshift z (M_\odot.h^2.kpc^-3)
def rho_m(z):
    return crit_density0*Omega_m0*(1.0+z)**3.0

#density of the given overdensity (M_\odot.h^2.kpc^-3)
def rho_Delta(mass_def, z):
    Delta = float(mass_def[:-1])
    rho_Delta = Delta*rho_m(z)
    return rho_Delta

#radius of the given overdensity (kpc/h)
def r_Delta(M_Delta, rho_Delta):
    return ((3.0*M_Delta)/(4.0*np.pi*rho_Delta))**(1.0/3.0)


# (inner) NFW density profile fundamental parameters
def parametersNFW(rho_Delta, r_Delta, c):

    r_s = r_Delta/c
    rho_s = (rho_Delta*c**3.0)/(3.0*(np.log(1.0+c)-c/(1.0+c)))

    return r_s, rho_s

# (inner) NFW density profile
def rhoNFW(r_s, rho_s, r):

    x = r/r_s

    return rho_s/(x*(1.0+x)**2.0)

# NFW mass in units of M(r_s).
# Equation 35 and 36 [Mamon et al, 2013]
def massNFW(r_s, r):

    x = r/r_s
    M_hat = np.log(x+1.0) + x/(x+1)

    return M_hat/(np.log(2.0) - 0.5)

# Maximum liklihood for 3D NFW with background.
def NFW_3D_maxlik_bg(r_scale_bg, R_proj, weights=None):
    nfw_sd = np.array(map(rhoNFW, r_scale_bg[0], rho_s, r))

    nfw_nden = nfw_num_density(r, r_scale_bg[0], r_scale_bg[1])

    prob = nfw_nden * nfw_sd + r_scale_bg[1]

    if weights is None:
        return np.sum(-np.log(prob))

    else:
        return np.sum(-np.log(prob) * weights)