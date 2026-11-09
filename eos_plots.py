#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eos_plots.py -- plotting helpers (needs matplotlib; the core library does not).

Every function draws on a matplotlib Axes that you pass in, so the same code builds
both the interactive plots of eos_app.py and the report figure of validate_and_plot.py.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from cubic_eos import CubicEOS, MODELS, R_V, Species

STYLE = {                                   # one look for each EOS, used in every figure
    "vdW": dict(color="#7f7f7f", ls=":",  lw=1.6),
    "RK":  dict(color="#1f77b4", ls="--", lw=1.4),
    "SRK": dict(color="#2ca02c", ls="-.", lw=1.4),
    "PR":  dict(color="#d62728", ls="-",  lw=1.6),
}


def saturation_dome(eos: CubicEOS, n: int = 60, Tr_min: float = 0.55, Tr_max: float = 0.997):
    """Saturated-liquid and saturated-vapor curves in the P-V plane (from equal fugacity)."""
    P, Vl, Vv = [], [], []
    for Tr in np.linspace(Tr_min, Tr_max, n):
        try:
            s = eos.saturation(Tr * eos.sp.Tc)
        except (ValueError, RuntimeError):
            continue
        P.append(s.P)
        Vl.append(s.liquid.V)
        Vv.append(s.vapor.V)
    return np.array(P), np.array(Vl), np.array(Vv)


def plot_pv(ax, eos: CubicEOS, Tr_list: Sequence[float] = (0.80, 0.90, 1.00, 1.10, 1.30),
            exp_points: Optional[Iterable[tuple]] = None, pmax_factor: float = 1.5):
    """P-V isotherms with the vapor/liquid dome and the equal-fugacity (Maxwell) tie lines."""
    sp, b = eos.sp, eos.b
    Vc = eos.m.Zc * R_V * sp.Tc / sp.Pc
    V = np.geomspace(1.02 * b, 40.0 * Vc, 900)
    Pd, Vl, Vv = saturation_dome(eos)
    ax.plot(np.r_[Vl, Vv[::-1]], np.r_[Pd, Pd[::-1]], color="k", lw=1.0)
    colors = ["#4575b4", "#74add1", "#1a9850", "#f46d43", "#d73027"]
    for Tr, c in zip(Tr_list, colors):
        T = Tr * sp.Tc
        P = np.array([eos.pressure(T, v) for v in V])
        ax.plot(V, P, color=c, lw=1.3, label=f"$T_r$ = {Tr:.2f}")
        if Tr < 1.0:
            try:
                s = eos.saturation(T)
                ax.hlines(s.P, s.liquid.V, s.vapor.V, color=c, ls="--", lw=0.9)
            except (ValueError, RuntimeError):
                pass
    ax.plot([Vc], [sp.Pc], "k*", ms=7)
    if exp_points:
        for (v, p, lab) in exp_points:
            ax.plot([v], [p], "ko", ms=4, mfc="white", mew=1.2, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(0.9 * b, 40.0 * Vc)
    ax.set_ylim(0.0, pmax_factor * sp.Pc)
    ax.set_xlabel("V  (cm$^3$/mol)")
    ax.set_ylabel("P  (bar)")


def _sweep(species: Species, T: float, P_arr: np.ndarray, models: Iterable[str], phase: str) -> Dict[str, list]:
    out = {}
    for m in models:
        eos = CubicEOS(species, m)
        out[m] = [eos.state(T, float(p), phase) for p in P_arr]
    return out


def plot_Z_vs_P(ax, species: Species, T: float, P_max: float, models=("vdW", "RK", "SRK", "PR"),
                phase: str = "vapor", ref_points: Optional[Iterable[tuple]] = None, n: int = 120):
    P = np.linspace(P_max / n, P_max, n)
    for m, sts in _sweep(species, T, P, models, phase).items():
        ax.plot(P, [s.Z for s in sts], label=m, **STYLE[m])
    ax.axhline(1.0, color="k", lw=0.6)
    if ref_points:
        for (p, z, lab) in ref_points:
            ax.plot([p], [z], "ko", ms=5, mfc="white", mew=1.3, label=lab)
    ax.set_xlabel("P  (bar)")
    ax.set_ylabel("Z = PV/RT")


def plot_residual_vs_P(ax, species: Species, T: float, P_max: float, which: str,
                       models=("vdW", "RK", "SRK", "PR"), phase: str = "vapor",
                       ref_points: Optional[Iterable[tuple]] = None, n: int = 120):
    """which = 'H' (H^R in J/mol) or 'S' (S^R in J/mol/K)."""
    P = np.linspace(P_max / n, P_max, n)
    for m, sts in _sweep(species, T, P, models, phase).items():
        ax.plot(P, [s.HR if which == "H" else s.SR for s in sts], label=m, **STYLE[m])
    ax.axhline(0.0, color="k", lw=0.6)
    if ref_points:
        for (p, y, lab) in ref_points:
            ax.plot([p], [y], "ko", ms=5, mfc="white", mew=1.3, label=lab)
    ax.set_xlabel("P  (bar)")
    ax.set_ylabel("$H^R$  (J/mol)" if which == "H" else "$S^R$  (J/mol/K)")


def plot_phi_vs_P(ax, species: Species, T: float, P_max: float, models=("vdW", "RK", "SRK", "PR"),
                  phase: str = "vapor", data: Optional[tuple] = None, n: int = 120, what: str = "phi"):
    """Fugacity coefficient (what='phi') or fugacity (what='f') versus pressure.
    data = (P_array, value_array, label) adds measured/steam-table points."""
    P = np.linspace(P_max / n, P_max, n)
    for m, sts in _sweep(species, T, P, models, phase).items():
        ax.plot(P, [s.phi if what == "phi" else s.f for s in sts], label=m, **STYLE[m])
    if what == "f":
        ax.plot(P, P, color="k", lw=0.8, ls=(0, (1, 2)), label="ideal gas")
    else:
        ax.axhline(1.0, color="k", lw=0.6)
    if data is not None:
        ax.plot(data[0], data[1], "ko", ms=4, mfc="white", mew=1.2, label=data[2])
    ax.set_xlabel("P  (bar)")
    ax.set_ylabel(r"fugacity coefficient  $\phi$" if what == "phi" else "fugacity  $f$  (bar)")


def plot_psat(ax, species: Species, models=("vdW", "RK", "SRK", "PR"), Tr_min: float = 0.60,
              Tr_max: float = 0.995, n: int = 40, exp_points: Optional[Iterable[tuple]] = None):
    """Vapor-pressure curve ln(P^sat) vs 1/T obtained from equal fugacity."""
    for m in models:
        eos = CubicEOS(species, m)
        Ts, Ps = [], []
        for Tr in np.linspace(Tr_min, Tr_max, n):
            try:
                s = eos.saturation(Tr * species.Tc)
            except (ValueError, RuntimeError):
                continue
            Ts.append(Tr * species.Tc)
            Ps.append(s.P)
        ax.plot(1000.0 / np.array(Ts), Ps, label=m, **STYLE[m])
    if exp_points:
        for (T, p, lab) in exp_points:
            ax.plot([1000.0 / T], [p], "ko", ms=5, mfc="white", mew=1.3, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("1000 / T  (1/K)")
    ax.set_ylabel(r"$P^{sat}$  (bar)")
