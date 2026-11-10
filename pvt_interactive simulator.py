import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

from IPython.display import display

# In Jupyter, plots are displayed inline automatically; no IPython magic is needed in a .py file.

# ----------------------------------------------------------------------------
# 1. Constants & Reference State
# ----------------------------------------------------------------------------
R_J = 8.314462618        # J/(mol K)
R_V = 10.0 * R_J         # bar cm^3/(mol K)
T_REF = 298.15           # K
P_REF = 1.0              # bar


# ----------------------------------------------------------------------------
# 2. 36-Species Thermodynamic Database
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Species:
    name: str
    Tc: float                      # K
    Pc: float                      # bar
    omega: float                   # acentric factor
    cp: Optional[Tuple[float, float, float, float]] = None
    Tmax: Optional[float] = None

    @property
    def has_cp(self) -> bool:
        return self.cp is not None


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
    ("argon",             0.000, 150.9,  48.98, (2.5,     0.0,     0.0,    0.0),    None),
]

SPECIES: Dict[str, Species] = {
    _n.title(): Species(_n.title(), _tc, _pc, _w, (_cp[0], _cp[1] * 1e-3, _cp[2] * 1e-6, _cp[3] * 1e5), _tmax)
    for _n, _w, _tc, _pc, _cp, _tmax in _DATA
}


# ----------------------------------------------------------------------------
# 3. EOS Models & Thermodynamic Equations
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EOSModel:
    name: str; sigma: float; epsilon: float; Omega: float; Psi: float; Zc: float
    alpha: Callable[[float, float], float]; dln_alpha: Callable[[float, float], float]

def _const_alpha(Tr, w): return 1.0
def _const_dlna(Tr, w): return 0.0
def _rk_alpha(Tr, w): return Tr ** -0.5
def _rk_dlna(Tr, w): return -0.5
def _kappa_srk(w): return 0.480 + 1.574 * w - 0.176 * w * w
def _kappa_pr(w): return 0.37464 + 1.54226 * w - 0.26992 * w * w

def _make_soave_type(kappa):
    def alpha(Tr, w): return (1.0 + kappa(w) * (1.0 - math.sqrt(Tr))) ** 2
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
    "PR":  EOSModel("PR", 1.0 + math.sqrt(2.0), 1.0 - math.sqrt(2.0), 0.07780, 0.45724, 0.30740, _pr_alpha, _pr_dlna),
}

def icph(T0, T, A, B=0.0, C=0.0, D=0.0):
    return A * (T - T0) + B / 2.0 * (T**2 - T0**2) + C / 3.0 * (T**3 - T0**3) - D * (1.0 / T - 1.0 / T0)

def icps(T0, T, A, B=0.0, C=0.0, D=0.0):
    return A * math.log(T / T0) + B * (T - T0) + C / 2.0 * (T**2 - T0**2) - D / 2.0 * (1.0 / T**2 - 1.0 / T0**2)

@dataclass
class State:
    species: str; eos: str; T: float; P: float; Z: float; V: float
    phase: str; roots: List[float]; HR: float; SR: float; GR: float
    lnphi: float; phi: float; f: float

def _solve_cubic_Z(beta: float, q: float, eps: float, sig: float) -> List[float]:
    c2 = (eps + sig - 1.0) * beta - 1.0
    c1 = q * beta + eps * sig * beta**2 - (eps + sig) * beta * (beta + 1.0)
    c0 = -(eps * sig * beta**2 * (beta + 1.0) + q * beta**2)
    raw = np.roots([1.0, c2, c1, c0])
    roots = [float(r.real) for r in raw if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real))]
    return sorted([z for z in set(roots) if z > beta * (1.0 + 1e-10)])

class CubicEOS:
    def __init__(self, species: Species, model: str = "PR"):
        self.sp = species
        self.m = MODELS[model.upper()]
        self.b = self.m.Omega * R_V * species.Tc / species.Pc

    def alpha(self, T: float) -> float: return self.m.alpha(T / self.sp.Tc, self.sp.omega)
    def a(self, T: float) -> float: return self.m.Psi * self.alpha(T) * (R_V * self.sp.Tc)**2 / self.sp.Pc
    def beta_q(self, T: float, P: float) -> Tuple[float, float]:
        Tr, Pr = T / self.sp.Tc, P / self.sp.Pc
        return self.m.Omega * Pr / Tr, self.m.Psi * self.m.alpha(Tr, self.sp.omega) / (self.m.Omega * Tr)

    def pressure(self, T: float, V: float) -> float:
        return R_V * T / (V - self.b) - self.a(T) / ((V + self.m.epsilon * self.b) * (V + self.m.sigma * self.b))

    def H_ig(self, T: float) -> float: return R_J * icph(T_REF, T, *self.sp.cp)
    def S_ig(self, T: float, P: float) -> float: return R_J * icps(T_REF, T, *self.sp.cp) - R_J * math.log(P / P_REF)

    def state(self, T: float, P: float) -> State:
        beta, q = self.beta_q(T, P)
        dlna = self.m.dln_alpha(T / self.sp.Tc, self.sp.omega)
        roots = _solve_cubic_Z(beta, q, self.m.epsilon, self.m.sigma)
        
        e, s = self.m.epsilon, self.m.sigma
        terms = []
        for z in roots:
            I = beta / (z + e * beta) if abs(s - e) < 1e-12 else math.log((z + s * beta) / (z + e * beta)) / (s - e)
            GR_RT = z - 1.0 - math.log(z - beta) - q * I
            HR_RT = z - 1.0 + (dlna - 1.0) * q * I
            SR_R = math.log(z - beta) + dlna * q * I
            terms.append((I, GR_RT, HR_RT, SR_R))
            
        k = min(range(len(roots)), key=lambda i: terms[i][1])
        Z = roots[k]
        GR_RT, HR_RT, SR_R = terms[k][1], terms[k][2], terms[k][3]
        V = Z * R_V * T / P
        phase = "Supercritical" if T >= self.sp.Tc else ("Vapor" if V > (self.m.Zc * R_V * self.sp.Tc / self.sp.Pc) else "Liquid")
        return State(self.sp.name, self.m.name, T, P, Z, V, phase, roots, R_J * T * HR_RT, R_J * SR_R, R_J * T * GR_RT, GR_RT, math.exp(GR_RT), math.exp(GR_RT) * P)


# ----------------------------------------------------------------------------
# 4. Interactive Widget Callback
# ----------------------------------------------------------------------------
def update_pvt_dashboard(gas_name, temp_k, press_bar, pmax_bar):
    fluid = SPECIES[gas_name]
    
    # 1. Print Fluid Critical Properties Summary
    print(f"=== {fluid.name.upper()} THERMODYNAMIC STATE ===")
    print(f"Critical Constants: Tc = {fluid.Tc} K | Pc = {fluid.Pc} bar | Acentric Factor (ω) = {fluid.omega}")
    print(f"Current Condition:  T  = {temp_k} K | P  = {press_bar} bar\n")

    # 2. Compute Model Table
    table_rows = []
    pr_eos = CubicEOS(fluid, "PR")

    for m in ["IDEAL", "VDW", "RK", "SRK", "PR"]:
        if m == "IDEAL":
            z_val, v_molar, rho_mol, ln_phi, phi, f_bar = 1.0, R_V * temp_k / press_bar, 1000.0 / (R_V * temp_k / press_bar), 0.0, 1.0, press_bar
            gr, hr, sr = 0.0, 0.0, 0.0
            h_tot, s_tot = pr_eos.H_ig(temp_k), pr_eos.S_ig(temp_k, press_bar)
            roots_str, phase = "1.0000", "Ideal Gas"
        else:
            solver = CubicEOS(fluid, m)
            st_res = solver.state(temp_k, press_bar)
            z_val, v_molar, rho_mol = st_res.Z, st_res.V, 1000.0 / st_res.V
            ln_phi, phi, f_bar = st_res.lnphi, st_res.phi, st_res.f
            gr, hr, sr = st_res.GR, st_res.HR, st_res.SR
            h_tot, s_tot = pr_eos.H_ig(temp_k) + hr, pr_eos.S_ig(temp_k, press_bar) + sr
            roots_str, phase = ", ".join(f"{r:.4f}" for r in st_res.roots), st_res.phase

        table_rows.append({
            "EOS Model": m,
            "Z (-)": round(z_val, 5),
            "Vm (cm³/mol)": round(v_molar, 2),
            "ρ (mol/L)": round(rho_mol, 4),
            "ln(φ)": round(ln_phi, 4),
            "φ (-)": round(phi, 4),
            "f (bar)": round(f_bar, 3),
            "G^R (kJ/mol)": round(gr/1000, 4),
            "H^R (kJ/mol)": round(hr/1000, 4),
            "S^R (J/mol·K)": round(sr, 4),
            "H_tot (kJ/mol)": round(h_tot/1000, 4),
            "S_tot (J/mol·K)": round(s_tot, 4),
            "Z Roots": roots_str,
            "Phase": phase
        })

    df = pd.DataFrame(table_rows)
    display(df)

    # 3. Generate Matplotlib Charts
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=120)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    # P-V Isotherms
    v_grid = np.logspace(np.log10(pr_eos.b * 1.08), np.log10(50.0 * pr_eos.b), 450)
    for mult in [0.90, 0.95, 1.00, 1.05, 1.15]:
        t_iso = mult * fluid.Tc
        p_iso_bar = [pr_eos.pressure(t_iso, v) for v in v_grid]
        ax1.plot(v_grid, p_iso_bar, lw=1.8 if mult == 1.0 else 1.2, label=f"T={t_iso:.1f} K")

    ax1.axhline(fluid.Pc, color="black", linestyle="--", alpha=0.5, label=f"Pc={fluid.Pc:.1f} bar")
    ax1.set_xscale("log"); ax1.set_ylim(0, fluid.Pc * 2.5)
    ax1.set_xlabel("Molar Volume Vm (cm³/mol)"); ax1.set_ylabel("Pressure P (bar)")
    ax1.set_title(f"{fluid.name}: P-V Isotherms (PR)", fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6); ax1.legend(fontsize=7)

    # Property curves vs P
    p_grid_bar = np.linspace(1.0, pmax_bar, 130)
    for m in ["IDEAL", "VDW", "RK", "SRK", "PR"]:
        if m == "IDEAL":
            z_vals, phi_vals, f_vals, hr_vals, sr_vals = [1.0]*130, [1.0]*130, p_grid_bar.tolist(), [0.0]*130, [0.0]*130
        else:
            solver = CubicEOS(fluid, m)
            states = [solver.state(temp_k, p) for p in p_grid_bar]
            z_vals = [s.Z for s in states]; phi_vals = [s.phi for s in states]
            f_vals = [s.f for s in states]; hr_vals = [s.HR/1000 for s in states]; sr_vals = [s.SR for s in states]

        style = "--" if m == "IDEAL" else "-"
        ax2.plot(p_grid_bar, z_vals, linestyle=style, label=m)
        ax3.plot(p_grid_bar, phi_vals, linestyle=style, label=m)
        ax4.plot(p_grid_bar, f_vals, linestyle=style, label=m)
        ax5.plot(p_grid_bar, hr_vals, linestyle=style, label=m)
        ax6.plot(p_grid_bar, sr_vals, linestyle=style, label=m)

    specs = [
        (ax2, "Z (-)", f"Compressibility Z (T={temp_k} K)"),
        (ax3, "Fugacity Coeff φ (-)", f"Fugacity Coefficient φ (T={temp_k} K)"),
        (ax4, "Fugacity f (bar)", f"Pure Fugacity f (T={temp_k} K)"),
        (ax5, "H^R (kJ/mol)", f"Residual Enthalpy H^R (T={temp_k} K)"),
        (ax6, "S^R (J/mol·K)", f"Residual Entropy S^R (T={temp_k} K)"),
    ]
    for ax, ylabel, title in specs:
        ax.set_xlabel("Pressure P (bar)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=7)

    fig.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# 5. Render Interactive Controls
# ----------------------------------------------------------------------------
if widgets is not None:
    w_gas = widgets.Dropdown(options=list(SPECIES.keys()), value="Carbon Dioxide", description="Fluid:")
    w_temp = widgets.FloatSlider(value=310.0, min=50.0, max=1000.0, step=5.0, description="T (K):")
    w_press = widgets.FloatSlider(value=80.0, min=1.0, max=500.0, step=5.0, description="P (bar):")
    w_pmax = widgets.FloatSlider(value=250.0, min=10.0, max=1000.0, step=10.0, description="P_max (bar):")

    interactive_ui = widgets.interactive(
        update_pvt_dashboard,
        gas_name=w_gas,
        temp_k=w_temp,
        press_bar=w_press,
        pmax_bar=w_pmax
    )
    display(interactive_ui)
else:
    print("ipywidgets is not installed. The calculation function is available, but interactive controls are disabled.")
    print("Install with: pip install ipywidgets")
