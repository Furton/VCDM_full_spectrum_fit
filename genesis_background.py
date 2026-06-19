"""Canonical Genesis-VCDM background and perturbation functions.

Defined once here and imported by vcdm_R_evolution.ipynb and fit_curve_pytorch.ipynb.

Conventions (main.tex, "Full numerical solution"): units tau_B = -1, so the
time variable is x = tau/tau_B = -tau and y = 1 + x. `kappa` is the
*dimensionless* wavenumber kappa = -k*tau_B (not the physical k); `H` is the
conformal Hubble rate (tilde H); `alpha0` is the dimensionless tilde alpha_0.

Domain: x > 0 (eta diverges at x = 0, where eps -> 0) and kappa < 1 (so that
kappa**d < 1 keeps nu real in L0_logC). Both hold throughout the scan priors.
"""
import numpy as np
from scipy.special import gammaln


def background(x, d, alpha0):
    y = 1.0 + np.asarray(x)
    return {
        'H':     (1.0 + d) * y ** (-1.0 - d),
        'a':     np.exp(((1.0 + d) / d) * y ** (-d)),
        'alpha': alpha0 * y ** (2.0 * d),
        'eps':   1.0 - y ** d,
        'eta':   -d * y ** (2.0 * d) / ((1.0 + d) * (y ** d - 1.0)),
        'beta':  -(2.0 * d / (1.0 + d)) * y ** d,
    }


def z_squared(x, d, alpha0, kappa):
    b = background(x, d, alpha0)
    num = 2.0 * b['a']**2 * b['alpha'] * (2.0 * kappa**2 + 3.0 * b['alpha'] * b['H']**2)
    den = 4.0 * kappa**2 + b['H']**2 * b['alpha'] * (6.0 + b['alpha'] - 2.0 * b['eps'])
    return num / den


def z_of_x(x, d, alpha0, kappa):
    return np.sqrt(z_squared(x, d, alpha0, kappa))


def cR_squared(x, d, alpha0, kappa):
    b = background(x, d, alpha0)
    alp, eps, eta, beta, H = b['alpha'], b['eps'], b['eta'], b['beta'], b['H']
    B1 = 2.0 * (alp * (-4.0*beta + 2.0*eps + alp)
                + 4.0 * eps * (6.0 + beta - 2.0*eps + eta))
    B2 = alp * (-(alp - 2.0*eps)**2 + 36.0*eps
                + 6.0 * (alp*(-1.0 - beta + eps)
                         + 2.0*eps*(1.0 - eps + eta)))
    A1 = 2.0 * alp * (12.0 + alp - 2.0*eps)
    A2 = 3.0 * alp**2 * (6.0 + alp - 2.0*eps)
    num = 8.0*kappa**4 + H**2 * B1 * kappa**2 + H**4 * B2
    den = 8.0*kappa**4 + H**2 * A1 * kappa**2 + H**4 * A2
    return num / den


def zddz(x, d, alpha0, kappa):
    x = np.asarray(x); y = 1.0 + x
    alpha    = alpha0 * y**(2.0*d)
    alpha_p  = alpha * (2.0*d/y);  alpha_pp = alpha * (2.0*d*(2.0*d-1.0)/y**2)
    eps      = 1.0 - y**d
    eps_p    = -d*y**(d-1.0);     eps_pp   = -d*(d-1.0)*y**(d-2.0)
    Ht       = (1.0+d)*y**(-1.0-d)
    Ht_p     = -((1.0+d)/y)*Ht;   Ht_pp    = ((1.0+d)*(2.0+d)/y**2)*Ht
    H2       = Ht**2
    H2_p     = 2.0*Ht*Ht_p;       H2_pp    = 2.0*(Ht_p**2 + Ht*Ht_pp)
    N        = 2.0*kappa**2 + 3.0*alpha*H2
    N_p      = 3.0*(alpha_p*H2 + alpha*H2_p)
    N_pp     = 3.0*(alpha_pp*H2 + 2.0*alpha_p*H2_p + alpha*H2_pp)
    S        = 6.0 + alpha - 2.0*eps
    S_p      = alpha_p - 2.0*eps_p;  S_pp = alpha_pp - 2.0*eps_pp
    Q        = H2*alpha*S
    Q_p      = H2_p*alpha*S + H2*alpha_p*S + H2*alpha*S_p
    Q_pp     = (H2_pp*alpha*S + H2*alpha_pp*S + H2*alpha*S_pp
                + 2.0*(H2_p*alpha_p*S + H2_p*alpha*S_p + H2*alpha_p*S_p))
    D        = 4.0*kappa**2 + Q;  D_p = Q_p;  D_pp = Q_pp
    LA_p     = -2.0*Ht + 2.0*d/y
    LA_pp    = -2.0*Ht_p - 2.0*d/y**2
    LN_p     = N_p/N;  LN_pp = N_pp/N - (N_p/N)**2
    LD_p     = D_p/D;  LD_pp = D_pp/D - (D_p/D)**2
    L_p      = LA_p + LN_p - LD_p
    L_pp     = LA_pp + LN_pp - LD_pp
    return 0.5*L_pp + 0.25*L_p**2


def omega2(x, d, alpha0, kappa):
    return cR_squared(x, d, alpha0, kappa) * kappa**2 - zddz(x, d, alpha0, kappa)


def L0_logC(d, alpha0, kappa):
    A = 3.0*(1+d)**2; s = kappa**d
    nu = np.sqrt(2.25 + A*s*(1-s)); u = A*(1-s)/(nu + 1.5); rA = np.sqrt(A)
    J = 3 + rA*np.arctan(u/rA) + 1.5*np.log(A/(A + u**2)) - A*(3+u)/(A + u**2)
    L0 = (-0.5*np.log(2.0) - 0.5*np.log(alpha0) - 1.5*np.log(kappa)
          - (1+d)/d + J/d)
    return L0, (2*nu - 3)*np.log(2.0) + 2*(gammaln(nu) - gammaln(1.5))
