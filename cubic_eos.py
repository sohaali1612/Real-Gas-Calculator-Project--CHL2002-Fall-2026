#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cubic_eos.py -- PVT behaviour, enthalpy, entropy and fugacity of pure real gases
=================================================================================
Generic cubic equation of state (Smith, Van Ness, Abbott & Swihart, "Introduction
to Chemical Engineering Thermodynamics", 8th ed., Eq. 3.41):

        P = R T / (V - b)  -  a(T) / [ (V + eps*b) (V + sigma*b) ]

with the four members of Table 3.1: van der Waals (vdW), Redlich/Kwong (RK),
Soave/Redlich/Kwong (SRK) and Peng/Robinson (PR).

Working equations (equation numbers refer to the textbook)
  beta = b P/(R T)        = Omega * Pr / Tr                          (3.46), (3.50)
  q    = a(T)/(b R T)     = Psi * alpha(Tr) / (Omega * Tr)           (3.47), (3.51)
  Z    = 1 + beta - q*beta*(Z - beta)/((Z + eps*beta)(Z + sigma*beta))        (3.48)
  I    = 1/(sigma - eps) * ln[(Z + sigma*beta)/(Z + eps*beta)]  (vdW: I = beta/Z)  (13.72)
  G^R/RT = ln(phi) = Z - 1 - ln(Z - beta) - q*I                      (13.74), (13.85)
  H^R/RT = Z - 1 + [dln(alpha)/dln(Tr) - 1] * q * I                  (13.75)
  S^R/R  = ln(Z - beta) + [dln(alpha)/dln(Tr)] * q * I               (13.76)
  H = H_ig(T) + H^R ,   S = S_ig(T, P) + S^R                          (6.50), (6.51)
  f = phi * P                                                         (10.34)

Units used everywhere in this module
  T [K]   P [bar]   V [cm^3/mol]   H [J/mol]   S [J/(mol K)]   f [bar]

Only NumPy is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# 1. Constants and reference state
# ----------------------------------------------------------------------------
R_J = 8.314462618        # J/(mol K)
R_V = 10.0 * R_J         # bar cm^3/(mol K)   (1 bar cm^3 = 0.1 J)
T_REF = 298.15           # K    ideal gas at (T_REF, P_REF) is the zero of H and S
P_REF = 1.0              # bar
P_STD_ATM = 1.01325      # bar  (1 atm), used for normal-boiling-point checks


# ----------------------------------------------------------------------------
# 2. Pure-species data: critical constants / acentric factor (Table B.1) and
#    ideal-gas heat capacity  Cp_ig/R = A + B*T + C*T^2 + D/T^2   (Table C.1)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Species:
    name: str
    Tc: float                      # K
    Pc: float                      # bar
    omega: float                   # acentric factor
    cp: Optional[Tuple[float, float, float, float]] = None   # (A, B, C, D), T in K
    Tmax: Optional[float] = None   # upper temperature limit of the Cp correlation, K

    @property
    def has_cp(self) -> bool:
        return self.cp is not None


# name, omega, Tc/K, Pc/bar, (A, 10^3 B, 10^6 C, 10^-5 D) or None, Tmax/K
_DATA = [
    ("methane",           0.012, 190.6,  45.99, (1.702,   9.081,  -2.164,  0.0),    1500),
    ("ethane",            0.100, 305.3,  48.72, (1.131,  19.225,  -5.561,  0.0),    1500),
    ("propane",           0.152, 369.8,  42.48, (1.213,  28.785,  -8.824,  0.0),    1500),
    ("n-butane",          0.200, 425.1,  37.96, (1.935,  36.915, -11.402,  0.0),    1500),
    ("isobutane",         0.181, 408.1,  36.48, (1.677,  37.853, -11.945,  0.0),    1500),
    ("n-pentane",         0.252, 469.7,  33.70, (2.464,  45.351, -14.111,  0.0),    1500),
    ("n-hexane",          0.301, 507.6,  30.25, (3.025,  53.722, -16.791,  0.0),    1500),
    ("n-heptane",         0.350, 540.2,  27.40, (3.570,  62.127, -19.486,  0.0),    1500),
    ("n-octane",          0.400, 568.7,  24.90, (4.108,  70.567, -22.208,  0.0),    1500),
    ("ethylene",          0.087, 282.3,  50.40, (1.424,  14.394,  -4.392,  0.0),    1500),
    ("propylene",         0.140, 365.6,  46.65, (1.637,  22.706,  -6.915,  0.0),    1500),
    ("1-butene",          0.191, 420.0,  40.43, (1.967,  31.630,  -9.873,  0.0),    1500),
    ("1,3-butadiene",     0.190, 425.2,  42.77, (2.734,  26.786,  -8.882,  0.0),    1500),
    ("acetylene",         0.187, 308.3,  61.39, (6.132,   1.952,   0.0,   -1.299),  1500),
    ("benzene",           0.210, 562.2,  48.98, (-0.206, 39.064, -13.301,  0.0),    1500),
    ("toluene",           0.262, 591.8,  41.06, (0.290,  47.052, -15.716,  0.0),    1500),
    ("cyclohexane",       0.210, 553.6,  40.73, (-3.876, 63.249, -20.928,  0.0),    1500),
    ("methanol",          0.564, 512.6,  80.97, (2.211,  12.216,  -3.450,  0.0),    1500),
    ("ethanol",           0.645, 513.9,  61.48, (3.518,  20.001,  -6.002,  0.0),    1500),
    ("ammonia",           0.253, 405.7, 112.80, (3.578,   3.020,   0.0,   -0.186),  1800),
    ("carbon monoxide",   0.048, 132.9,  34.99, (3.376,   0.557,   0.0,   -0.031),  2500),
    ("carbon dioxide",    0.224, 304.2,  73.83, (5.457,   1.045,   0.0,   -1.157),  2000),
    ("hydrogen",         -0.216,  33.19, 13.13, (3.249,   0.422,   0.0,    0.083),  3000),
    ("hydrogen sulfide",  0.094, 373.5,  89.63, (3.931,   1.490,   0.0,   -0.232),  2300),
    ("nitrogen",          0.038, 126.2,  34.00, (3.280,   0.593,   0.0,    0.040),  2000),
    ("nitrous oxide",     0.141, 309.6,  72.45, (5.328,   1.214,   0.0,   -0.928),  2000),
    ("nitric oxide",      0.583, 180.2,  64.80, (3.387,   0.629,   0.0,    0.014),  2000),
    ("oxygen",            0.022, 154.6,  50.43, (3.639,   0.506,   0.0,   -0.227),  2000),
    ("sulfur dioxide",    0.245, 430.8,  78.84, (5.699,   0.801,   0.0,   -1.015),  2000),
    ("water",             0.345, 647.1, 220.55, (3.470,   1.450,   0.0,    0.121),  2000),
    ("chlorine",          0.069, 417.2,  77.10, (4.442,   0.089,   0.0,   -0.344),  3000),
    ("air",               0.035, 132.2,  37.45, (3.355,   0.575,   0.0,   -0.016),  2000),
    ("hydrogen chloride", 0.132, 324.7,  83.10, (3.156,   0.623,   0.0,    0.151),  2000),
    ("hydrogen cyanide",  0.410, 456.7,  53.90, (4.736,   1.359,   0.0,   -0.725),  2500),
    ("carbon disulfide",  0.111, 552.0,  79.00, (6.311,   0.805,   0.0,   -0.906),  1800),
    # Monatomic gas: Cp_ig = 5R/2 exactly (kinetic theory); not tabulated in Table C.1
    ("argon",             0.000, 150.9,  48.98, (2.5,     0.0,     0.0,    0.0),    None),
]

SPECIES: Dict[str, Species] = {}
for _n, _w, _tc, _pc, _cp, _tmax in _DATA:
    SPECIES[_n] = Species(_n, _tc, _pc, _w,
                          (_cp[0], _cp[1] * 1e-3, _cp[2] * 1e-6, _cp[3] * 1e5), _tmax)

ALIASES = {
    "butane": "n-butane", "c4h10": "n-butane", "ch4": "methane", "c2h6": "ethane",
    "c3h8": "propane", "c2h4": "ethylene", "c3h6": "propylene", "c6h6": "benzene",
    "co2": "carbon dioxide", "co": "carbon monoxide", "h2": "hydrogen",
    "h2s": "hydrogen sulfide", "n2": "nitrogen", "n2o": "nitrous oxide",
    "no": "nitric oxide", "o2": "oxygen", "so2": "sulfur dioxide", "h2o": "water",
    "steam": "water", "cl2": "chlorine", "hcl": "hydrogen chloride",
    "hcn": "hydrogen cyanide", "cs2": "carbon disulfide", "nh3": "ammonia",
    "ar": "argon", "pentane": "n-pentane", "hexane": "n-hexane",
    "heptane": "n-heptane", "octane": "n-octane", "butadiene": "1,3-butadiene",
    "iso-butane": "isobutane", "i-butane": "isobutane",
}


def get_species(name: str) -> Species:
    """Look a species up by name, formula or unique prefix (case-insensitive)."""
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key in SPECIES:
        return SPECIES[key]
    hits = [k for k in SPECIES if k.startswith(key)]
    if len(hits) == 1:
        return SPECIES[hits[0]]
    raise KeyError(f"Species '{name}' not found. Available: {', '.join(sorted(SPECIES))}")


# ----------------------------------------------------------------------------
# 3. The four equations of state of Table 3.1
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EOSModel:
    name: str
    sigma: float
    epsilon: float
    Omega: float
    Psi: float
    Zc: float                                         # critical Z implied by the EOS
    alpha: Callable[[float, float], float]            # alpha(Tr, omega)
    dln_alpha: Callable[[float, float], float]        # d ln(alpha) / d ln(Tr)


def _const_alpha(Tr, w):      return 1.0
def _const_dlna(Tr, w):       return 0.0
def _rk_alpha(Tr, w):         return Tr ** -0.5
def _rk_dlna(Tr, w):          return -0.5


def _kappa_srk(w):            return 0.480 + 1.574 * w - 0.176 * w * w
def _kappa_pr(w):             return 0.37464 + 1.54226 * w - 0.26992 * w * w


def _make_soave_type(kappa: Callable[[float], float]):
    """alpha = [1 + kappa (1 - sqrt(Tr))]^2  ->  dln(alpha)/dln(Tr) = -kappa sqrt(Tr)/[1 + kappa(1 - sqrt(Tr))]"""
    def alpha(Tr, w):
        return (1.0 + kappa(w) * (1.0 - math.sqrt(Tr))) ** 2

    def dln_alpha(Tr, w):
        k, s = kappa(w), math.sqrt(Tr)
        return -k * s / (1.0 + k * (1.0 - s))
    return alpha, dln_alpha


_srk_alpha, _srk_dlna = _make_soave_type(_kappa_srk)
_pr_alpha, _pr_dlna = _make_soave_type(_kappa_pr)

MODELS: Dict[str, EOSModel] = {
    "VDW": EOSModel("vdW", 0.0, 0.0, 1.0 / 8.0, 27.0 / 64.0, 3.0 / 8.0, _const_alpha, _const_dlna),
    "RK":  EOSModel("RK", 1.0, 0.0, 0.08664, 0.42748, 1.0 / 3.0, _rk_alpha, _rk_dlna),
    "SRK": EOSModel("SRK", 1.0, 0.0, 0.08664, 0.42748, 1.0 / 3.0, _srk_alpha, _srk_dlna),
    "PR":  EOSModel("PR", 1.0 + math.sqrt(2.0), 1.0 - math.sqrt(2.0), 0.07780, 0.45724, 0.30740,
                    _pr_alpha, _pr_dlna),
}


def get_model(name: str) -> EOSModel:
    try:
        return MODELS[name.strip().upper()]
    except KeyError:
        raise KeyError(f"Unknown EOS '{name}'. Choose from: vdW, RK, SRK, PR") from None


# ----------------------------------------------------------------------------
# 4. Ideal-gas integrals (the textbook's ICPH, ICPS, MCPH, MCPS functions)
# ----------------------------------------------------------------------------
def icph(T0, T, A, B=0.0, C=0.0, D=0.0):
    """Integral of Cp_ig/R dT from T0 to T [K]."""
    return (A * (T - T0) + B / 2.0 * (T ** 2 - T0 ** 2) + C / 3.0 * (T ** 3 - T0 ** 3)
            - D * (1.0 / T - 1.0 / T0))


def icps(T0, T, A, B=0.0, C=0.0, D=0.0):
    """Integral of (Cp_ig/R) dT/T from T0 to T [K]."""
    return (A * math.log(T / T0) + B * (T - T0) + C / 2.0 * (T ** 2 - T0 ** 2)
            - D / 2.0 * (1.0 / T ** 2 - 1.0 / T0 ** 2))


def mcph(T0, T, A, B=0.0, C=0.0, D=0.0):
    """Mean heat capacity <Cp_ig>_H / R between T0 and T."""
    return icph(T0, T, A, B, C, D) / (T - T0)


def mcps(T0, T, A, B=0.0, C=0.0, D=0.0):
    """Mean heat capacity <Cp_ig>_S / R between T0 and T."""
    return icps(T0, T, A, B, C, D) / math.log(T / T0)


# ----------------------------------------------------------------------------
# 5. Result containers
# ----------------------------------------------------------------------------
@dataclass
class State:
    """Everything the EOS says about one pure fluid at (T, P) for one root."""
    species: str
    eos: str
    T: float
    P: float
    Tr: float
    Pr: float
    beta: float
    q: float
    I: float
    Z: float
    V: float                 # cm^3/mol
    phase: str               # 'vapor', 'liquid' or 'supercritical'
    roots: List[float]       # all physical real Z roots (ascending)
    GR_RT: float             # = ln(phi)
    HR_RT: float
    SR_R: float
    HR: float                # J/mol
    SR: float                # J/(mol K)
    GR: float                # J/mol
    lnphi: float
    phi: float
    f: float                 # bar
    note: str = ""


@dataclass
class Props:
    """Total H and S (relative to the ideal-gas reference state) plus the pieces."""
    H: float                 # J/mol
    S: float                 # J/(mol K)
    H_ig: float
    S_ig: float
    state: State
    T0: float
    P0: float
    warning: str = ""


@dataclass
class Saturation:
    T: float
    P: float                 # bar
    liquid: State
    vapor: State
    dHvap: float             # J/mol   = H^R(vapor) - H^R(liquid)
    dSvap: float             # J/(mol K)


# ----------------------------------------------------------------------------
# 6. Cubic-equation solver for Z
# ----------------------------------------------------------------------------
def _solve_cubic_Z(beta: float, q: float, eps: float, sig: float) -> List[float]:
    """Physical real roots (Z > beta, i.e. V > b) of Eq. (3.48) written as a cubic polynomial:

        Z^3 + [(eps+sig-1) beta - 1] Z^2 + [q beta + eps sig beta^2 - (eps+sig) beta (beta+1)] Z
            - [eps sig beta^2 (beta+1) + q beta^2] = 0
    """
    c2 = (eps + sig - 1.0) * beta - 1.0
    c1 = q * beta + eps * sig * beta ** 2 - (eps + sig) * beta * (beta + 1.0)
    c0 = -(eps * sig * beta ** 2 * (beta + 1.0) + q * beta ** 2)
    raw = np.roots([1.0, c2, c1, c0])

    roots: List[float] = []
    for r in raw:
        if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
            z = float(r.real)
            for _ in range(30):                       # Newton polishing
                f = ((z + c2) * z + c1) * z + c0
                df = (3.0 * z + 2.0 * c2) * z + c1
                if df == 0.0:
                    break
                dz = f / df
                z -= dz
                if abs(dz) <= 1e-15 * max(1.0, abs(z)):
                    break
            roots.append(z)

    roots = sorted(z for z in roots if z > beta * (1.0 + 1e-10))
    unique: List[float] = []
    for z in roots:                                   # merge (near-)double roots
        if not unique or abs(z - unique[-1]) > 1e-9 * max(1.0, z):
            unique.append(z)
    return unique


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-13, maxit: int = 300) -> float:
    """Plain bisection; f(lo) and f(hi) must have opposite signs."""
    flo = f(lo)
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
        if abs(hi - lo) <= tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# 7. The EOS object
# ----------------------------------------------------------------------------
class CubicEOS:
    """Generic cubic EOS for one pure species.  CubicEOS(get_species('n-butane'), 'PR')"""

    def __init__(self, species: Species, model: str = "PR"):
        self.sp = species
        self.m = get_model(model)
        self.b = self.m.Omega * R_V * species.Tc / species.Pc            # cm^3/mol, Eq. (3.44)

    # ---- parameters -------------------------------------------------------
    def alpha(self, T: float) -> float:
        return self.m.alpha(T / self.sp.Tc, self.sp.omega)

    def a(self, T: float) -> float:
        """a(T) in bar cm^6/mol^2, Eq. (3.45)."""
        sp = self.sp
        return self.m.Psi * self.alpha(T) * (R_V * sp.Tc) ** 2 / sp.Pc

    def beta_q(self, T: float, P: float) -> Tuple[float, float]:
        sp, m = self.sp, self.m
        Tr, Pr = T / sp.Tc, P / sp.Pc
        beta = m.Omega * Pr / Tr                                        # (3.50)
        q = m.Psi * m.alpha(Tr, sp.omega) / (m.Omega * Tr)              # (3.51)
        return beta, q

    @staticmethod
    def _check(T=None, P=None):
        if T is not None and T <= 0.0:
            raise ValueError("Temperature must be positive (K).")
        if P is not None and P <= 0.0:
            raise ValueError("Pressure must be positive (bar).")

    # ---- PVT --------------------------------------------------------------
    def pressure(self, T: float, V: float) -> float:
        """P [bar] from T [K] and molar volume V [cm^3/mol]  (Eq. 3.41)."""
        self._check(T=T)
        if V <= self.b:
            raise ValueError(f"V must exceed b = {self.b:.3f} cm3/mol.")
        m, b = self.m, self.b
        return R_V * T / (V - b) - self.a(T) / ((V + m.epsilon * b) * (V + m.sigma * b))

    def Z_roots(self, T: float, P: float) -> List[float]:
        """All physical real roots of the EOS for Z at (T, P), ascending."""
        self._check(T, P)
        beta, q = self.beta_q(T, P)
        return _solve_cubic_Z(beta, q, self.m.epsilon, self.m.sigma)

    def Z_iterative(self, T: float, P: float, phase: str = "vapor", tol: float = 1e-13,
                    maxit: int = 100000) -> float:
        """Textbook successive substitution: Eq. (3.48) from Z=1 (vapor root),
        Eq. (3.49) from Z=beta (liquid root).  Used to cross-check the polynomial solver."""
        beta, q = self.beta_q(T, P)
        e, s = self.m.epsilon, self.m.sigma
        if phase == "vapor":
            Z = 1.0
            step = lambda Z: 1.0 + beta - q * beta * (Z - beta) / ((Z + e * beta) * (Z + s * beta))
        elif phase == "liquid":
            Z = beta
            step = lambda Z: beta + (Z + e * beta) * (Z + s * beta) * (1.0 + beta - Z) / (q * beta)
        else:
            raise ValueError("phase must be 'vapor' or 'liquid'")
        for _ in range(maxit):
            Zn = step(Z)
            if abs(Zn - Z) < tol:
                return Zn
            Z = Zn
        raise RuntimeError("Successive substitution did not converge")

    def volume(self, T: float, P: float, phase: str = "auto") -> float:
        """Molar volume [cm^3/mol] at (T, P)."""
        return self.state(T, P, phase).V

    def temperature(self, P: float, V: float) -> float:
        """T [K] such that the EOS gives pressure P [bar] at molar volume V [cm^3/mol]."""
        self._check(P=P)
        if V <= self.b:
            raise ValueError(f"V must exceed b = {self.b:.3f} cm3/mol.")
        f = lambda T: self.pressure(T, V) - P
        lo, hi = 1e-3 * self.sp.Tc, 2.0 * self.sp.Tc
        for _ in range(60):                                  # expand upper bracket
            if f(hi) > 0.0:
                break
            hi *= 2.0
        if f(hi) <= 0.0 or f(lo) >= 0.0:
            raise ValueError("No temperature reproduces this (P, V) with this EOS.")
        return _bisect(f, lo, hi)

    # ---- residual properties ---------------------------------------------
    def _terms(self, Z: float, beta: float, q: float, dlna: float) -> Tuple[float, float, float, float]:
        """Return (I, G^R/RT, H^R/RT, S^R/R) for one root  -- Eqs. (13.72)-(13.76)."""
        e, s = self.m.epsilon, self.m.sigma
        if abs(s - e) < 1e-12:                     # vdW (case II of Eq. 13.72)
            I = beta / (Z + e * beta)
        else:                                      # case I
            I = math.log((Z + s * beta) / (Z + e * beta)) / (s - e)
        lnZb = math.log(Z - beta)
        GR = Z - 1.0 - lnZb - q * I
        HR = Z - 1.0 + (dlna - 1.0) * q * I
        SR = lnZb + dlna * q * I
        return I, GR, HR, SR

    def _label(self, T: float, V: float) -> str:
        sp = self.sp
        if T >= sp.Tc:
            return "supercritical"
        Vc_eos = self.m.Zc * R_V * sp.Tc / sp.Pc
        return "vapor" if V > Vc_eos else "liquid"

    def state(self, T: float, P: float, phase: str = "auto") -> State:
        """Z, V, residual H and S, ln(phi), phi, f at (T, P).

        phase = 'vapor'  -> largest root,  'liquid' -> smallest root,
                'auto'   -> root of lowest Gibbs energy (lowest ln phi), i.e. the stable phase.
        """
        self._check(T, P)
        sp, m = self.sp, self.m
        Tr, Pr = T / sp.Tc, P / sp.Pc
        beta, q = self.beta_q(T, P)
        dlna = m.dln_alpha(Tr, sp.omega)
        roots = _solve_cubic_Z(beta, q, m.epsilon, m.sigma)
        if not roots:
            raise RuntimeError("EOS has no physical root at this state.")
        terms = [self._terms(Z, beta, q, dlna) for Z in roots]

        ph = phase.strip().lower()
        if ph == "vapor":
            k = len(roots) - 1
        elif ph == "liquid":
            k = 0
        elif ph in ("auto", "stable"):
            k = min(range(len(roots)), key=lambda i: terms[i][1])
        else:
            raise ValueError("phase must be 'auto', 'vapor' or 'liquid'")

        note = ""
        if len(roots) == 1 and ph in ("vapor", "liquid") and T < sp.Tc:
            note = "Only one real root exists: it is returned for either phase request."
        elif len(roots) >= 2:
            lnphi_l, lnphi_v = terms[0][1], terms[-1][1]
            if ph == "vapor" and lnphi_v > lnphi_l + 1e-9:
                note = "Metastable (supersaturated) vapor: the liquid root has the lower Gibbs energy."
            if ph == "liquid" and lnphi_l > lnphi_v + 1e-9:
                note = "Metastable (superheated) liquid: the vapor root has the lower Gibbs energy."

        Z = roots[k]
        I, GR_RT, HR_RT, SR_R = terms[k]
        V = Z * R_V * T / P
        return State(
            species=sp.name, eos=m.name, T=T, P=P, Tr=Tr, Pr=Pr, beta=beta, q=q, I=I, Z=Z, V=V,
            phase=self._label(T, V), roots=roots,
            GR_RT=GR_RT, HR_RT=HR_RT, SR_R=SR_R,
            HR=R_J * T * HR_RT, SR=R_J * SR_R, GR=R_J * T * GR_RT,
            lnphi=GR_RT, phi=math.exp(GR_RT), f=math.exp(GR_RT) * P, note=note)

    # ---- fugacity ---------------------------------------------------------
    def fugacity(self, T: float, P: float, phase: str = "auto") -> Tuple[float, float]:
        """Return (f [bar], phi)."""
        st = self.state(T, P, phase)
        return st.f, st.phi

    # ---- enthalpy and entropy ----------------------------------------------
    def _cp(self):
        if not self.sp.has_cp:
            raise ValueError(f"No ideal-gas heat-capacity constants for '{self.sp.name}'. "
                             "Residual properties are still available via state().")
        return self.sp.cp

    def H_ig(self, T: float, T0: float = T_REF) -> float:
        """Ideal-gas enthalpy relative to T0 [J/mol]  = R * ICPH."""
        return R_J * icph(T0, T, *self._cp())

    def S_ig(self, T: float, P: float, T0: float = T_REF, P0: float = P_REF) -> float:
        """Ideal-gas entropy relative to (T0, P0) [J/(mol K)] = R*ICPS - R ln(P/P0)."""
        return R_J * icps(T0, T, *self._cp()) - R_J * math.log(P / P0)

    def properties(self, T: float, P: float, phase: str = "auto",
                   T0: float = T_REF, P0: float = P_REF) -> Props:
        """H = H_ig + H^R and S = S_ig + S^R  (Eqs. 6.50 and 6.51); H = S = 0 for the
        ideal gas at (T0, P0)."""
        st = self.state(T, P, phase)
        Hig, Sig = self.H_ig(T, T0), self.S_ig(T, P, T0, P0)
        warn = ""
        sp = self.sp
        if sp.Tmax is not None and not (298.0 <= T <= sp.Tmax):
            warn = (f"T = {T:.1f} K is outside the 298-{sp.Tmax:.0f} K range of the Cp correlation "
                    "(Table C.1): ideal-gas part is extrapolated.")
        return Props(H=Hig + st.HR, S=Sig + st.SR, H_ig=Hig, S_ig=Sig, state=st, T0=T0, P0=P0, warning=warn)

    def delta(self, T1: float, P1: float, T2: float, P2: float,
              phase1: str = "auto", phase2: str = "auto") -> Dict[str, float]:
        """Property change state 1 -> state 2.  The reference state cancels."""
        p1, p2 = self.properties(T1, P1, phase1), self.properties(T2, P2, phase2)
        return {
            "dH": p2.H - p1.H, "dS": p2.S - p1.S,
            "dH_ig": p2.H_ig - p1.H_ig, "dS_ig": p2.S_ig - p1.S_ig,
            "dHR": p2.state.HR - p1.state.HR, "dSR": p2.state.SR - p1.state.SR,
            "state1": p1, "state2": p2,
        }

    # ---- vapor/liquid equilibrium of the pure species ---------------------------
    def spinodal_volumes(self, T: float) -> Tuple[float, float]:
        """Volumes of the minimum and maximum of the subcritical isotherm (dP/dV = 0)."""
        m, b, a, RT = self.m, self.b, self.a(T), R_V * T
        s, p = m.epsilon + m.sigma, m.epsilon * m.sigma
        # RT (V^2 + s b V + p b^2)^2 = a (2V + s b)(V - b)^2   expanded as a quartic in V
        coeff = [RT,
                 2 * s * b * RT - 2 * a,
                 RT * b * b * (s * s + 2 * p) - a * (s - 4) * b,
                 2 * s * p * b ** 3 * RT - a * (2 - 2 * s) * b * b,
                 RT * p * p * b ** 4 - a * s * b ** 3]
        r = np.roots(coeff)
        Vs = sorted(x.real for x in r if abs(x.imag) <= 1e-9 * abs(x.real) and x.real > b * (1 + 1e-9))
        if len(Vs) < 2:
            raise RuntimeError("No van der Waals loop at this temperature (is T >= Tc?).")
        return Vs[0], Vs[-1]

    def saturation(self, T: float) -> Saturation:
        """Vapor pressure from equal fugacity of saturated liquid and vapor,
        ln(phi_l) = ln(phi_v)  (Eq. 13.78), plus latent heat  dHvap = H^R_v - H^R_l.

        Because d(ln phi_l - ln phi_v)/dP = (V_l - V_v)/RT < 0, the function is monotonic in P, so
        bisection on ln P between the two spinodal pressures is guaranteed to converge."""
        sp = self.sp
        self._check(T=T)
        if T >= sp.Tc:
            raise ValueError(f"T = {T} K is not below Tc = {sp.Tc} K: no vapor/liquid equilibrium.")
        V1, V2 = self.spinodal_volumes(T)
        P_hi = self.pressure(T, V2)
        P_lo = max(self.pressure(T, V1), 1e-9 * sp.Pc)
        if P_hi <= P_lo:
            raise RuntimeError("Spinodal pressures are not ordered; T too close to Tc.")

        def F(lnP: float) -> float:                  # ln phi_l - ln phi_v
            P = math.exp(lnP)
            beta, q = self.beta_q(T, P)
            roots = _solve_cubic_Z(beta, q, self.m.epsilon, self.m.sigma)
            if len(roots) < 2:
                return float("nan")
            dlna = self.m.dln_alpha(T / sp.Tc, sp.omega)
            return self._terms(roots[0], beta, q, dlna)[1] - self._terms(roots[-1], beta, q, dlna)[1]

        lo, hi = math.log(P_lo * (1 + 1e-9)), math.log(P_hi * (1 - 1e-9))
        flo, fhi = F(lo), F(hi)
        if not (flo > 0.0 > fhi):
            raise ValueError("Saturation pressure is outside the range where the EOS has three roots "
                             "(T far below Tc: vapor pressure below 1e-9*Pc).")
        lnP = _bisect(F, lo, hi, tol=1e-14)
        P = math.exp(lnP)
        liq, vap = self.state(T, P, "liquid"), self.state(T, P, "vapor")
        dH = vap.HR - liq.HR
        return Saturation(T=T, P=P, liquid=liq, vapor=vap, dHvap=dH, dSvap=dH / T)


# ----------------------------------------------------------------------------
# 8. Convenience helpers
# ----------------------------------------------------------------------------
def compare_models(species: Species, T: float, P: float, phase: str = "vapor") -> Dict[str, State]:
    """State of the same species at (T, P) from vdW, RK, SRK and PR."""
    return {m.name: CubicEOS(species, m.name).state(T, P, phase) for m in MODELS.values()}


def custom_species(name: str, Tc: float, Pc: float, omega: float,
                   cp: Optional[Tuple[float, float, float, float]] = None,
                   Tmax: Optional[float] = None) -> Species:
    """Build a species that is not in the database.  cp = (A, B, C, D) with Cp_ig/R = A + B T + C T^2 + D/T^2."""
    return Species(name, Tc, Pc, omega, cp, Tmax)


if __name__ == "__main__":
    # Smoke test: n-butane at 500 K and 50 bar (textbook Table 13.7)
    for name, st in compare_models(get_species("n-butane"), 500.0, 50.0).items():
        print(f"{name:4s} Z = {st.Z:.4f}   HR = {st.HR:8.0f} J/mol   SR = {st.SR:7.3f} J/mol/K   phi = {st.phi:.4f}")
