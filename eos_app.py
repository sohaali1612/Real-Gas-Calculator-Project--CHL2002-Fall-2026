#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eos_app.py -- REAL-GAS PROPERTY CALCULATOR  (cubic equations of state)
=======================================================================
PVT behaviour, enthalpy, entropy and pure-species fugacity of real gases from the
van der Waals, Redlich/Kwong, Soave/Redlich/Kwong or Peng/Robinson equation.

Run the menu program:           python eos_app.py
Run a single state and exit:    python eos_app.py --species n-butane --eos PR --T 500 --P 50
Quick demonstration:            python eos_app.py --demo

Inputs accept units:  T: 500, 500K, 25C, 77F      P: 50, 5MPa, 500kPa, 2atm, 100psi
                      V: 1200, 1200cm3/mol, 0.0012m3/mol, 1.2L/mol        (defaults: K, bar, cm3/mol)
"""
from __future__ import annotations

import argparse
import math
import re
import sys

from cubic_eos import (MODELS, R_J, R_V, SPECIES, T_REF, P_REF, CubicEOS, State, Species,
                       compare_models, custom_species, get_species)

# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _split(text: str):
    m = re.fullmatch(rf"\s*({_NUM})\s*([A-Za-z°/0-9\.]*)\s*", text)
    if not m:
        raise ValueError(f"cannot read '{text}' as a number with optional unit")
    return float(m.group(1)), m.group(2).lower().replace("°", "")


def parse_T(text: str) -> float:
    x, u = _split(text)
    conv = {"": x, "k": x, "c": x + 273.15, "f": (x - 32.0) / 1.8 + 273.15, "r": x / 1.8}
    if u not in conv:
        raise ValueError(f"unknown temperature unit '{u}' (use K, C, F, R)")
    T = conv[u]
    if T <= 0:
        raise ValueError("temperature must be above absolute zero")
    return T


def parse_P(text: str) -> float:
    x, u = _split(text)
    f = {"": 1.0, "bar": 1.0, "kpa": 0.01, "mpa": 10.0, "pa": 1e-5, "atm": 1.01325,
         "psi": 0.0689476, "psia": 0.0689476, "torr": 0.00133322, "mmhg": 0.00133322}
    if u not in f:
        raise ValueError(f"unknown pressure unit '{u}' (use bar, kPa, MPa, Pa, atm, psi, torr)")
    if x <= 0:
        raise ValueError("pressure must be positive")
    return x * f[u]


def parse_V(text: str) -> float:
    x, u = _split(text)
    f = {"": 1.0, "cm3/mol": 1.0, "m3/mol": 1e6, "l/mol": 1e3, "m3/kmol": 1e3, "ft3/lbmol": 62.428}
    if u not in f:
        raise ValueError(f"unknown volume unit '{u}' (use cm3/mol, m3/mol, L/mol, m3/kmol, ft3/lbmol)")
    if x <= 0:
        raise ValueError("volume must be positive")
    return x * f[u]


def ask(prompt: str, parser=str, default=None):
    while True:
        txt = input(f"  {prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if txt == "" and default is not None:
            txt = str(default)
        try:
            return parser(txt)
        except (ValueError, KeyError) as exc:
            print(f"   ! {exc}")


def parse_phase(text: str) -> str:
    t = text.strip().lower()
    t = {"a": "auto", "v": "vapor", "l": "liquid", "g": "vapor", "gas": "vapor"}.get(t, t)
    if t not in ("auto", "vapor", "liquid"):
        raise ValueError("enter auto, vapor or liquid")
    return t


def line(ch: str = "-", n: int = 74):
    print(ch * n)


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def header(st: State):
    line("=")
    print(f" {st.species}  |  {st.eos} EOS  |  T = {st.T:.2f} K   P = {st.P:.4f} bar"
          f"   (Tr = {st.Tr:.4f}, Pr = {st.Pr:.4f})")
    line()


def show_pvt(st: State):
    header(st)
    roots = ", ".join(f"{z:.5f}" for z in st.roots)
    print(f" beta = {st.beta:.6f}     q = {st.q:.5f}")
    print(f" Physical real roots of the cubic (Z > beta): {roots}")
    print(f" Selected root -> phase: {st.phase}")
    print(f"   Z = PV/RT     = {st.Z:.5f}")
    print(f"   V             = {st.V:.3f} cm3/mol        (ideal gas: {R_V * st.T / st.P:.3f} cm3/mol)")
    print(f"   density       = {1e3 / st.V:.4f} mol/L")
    if st.note:
        print(f"   NOTE: {st.note}")
    line("=")


def show_hs(eos: CubicEOS, T: float, P: float, phase: str, T0: float, P0: float):
    st = eos.state(T, P, phase)
    header(st)
    print(f" Phase: {st.phase}      Z = {st.Z:.5f}      V = {st.V:.3f} cm3/mol")
    print(f" Residual properties (Eqs. 13.75, 13.76):")
    print(f"   H^R / RT = {st.HR_RT:.5f}          H^R = {st.HR:12.2f} J/mol")
    print(f"   S^R / R  = {st.SR_R:.5f}          S^R = {st.SR:12.4f} J/(mol K)")
    if eos.sp.has_cp:
        p = eos.properties(T, P, phase, T0, P0)
        print(f" Ideal-gas part, reference state: ideal gas at {T0:.2f} K, {P0:g} bar  (H = S = 0)")
        print(f"   H^ig = {p.H_ig:12.2f} J/mol        S^ig = {p.S_ig:12.4f} J/(mol K)")
        print(f" Total (Eqs. 6.50, 6.51):")
        print(f"   H    = {p.H:12.2f} J/mol        S    = {p.S:12.4f} J/(mol K)")
        if p.warning:
            print(f"   WARNING: {p.warning}")
    else:
        print(" (No ideal-gas Cp data for this species -> only residual properties are available.)")
    if st.note:
        print(f"   NOTE: {st.note}")
    line("=")


def show_fugacity(st: State):
    header(st)
    print(f" Phase: {st.phase}      Z = {st.Z:.5f}")
    print(f"   G^R / RT = ln(phi) = {st.lnphi:.6f}   (Eq. 13.74)")
    print(f"   phi = f/P          = {st.phi:.6f}")
    print(f"   f                  = {st.f:.5f} bar   ({st.f * 100:.3f} kPa)")
    print(f"   (ideal-gas value: f = P = {st.P:.5f} bar, phi = 1)")
    if st.note:
        print(f"   NOTE: {st.note}")
    line("=")


# ----------------------------------------------------------------------------
# Menu actions
# ----------------------------------------------------------------------------
def act_TP(eos, mode):
    T = ask("Temperature T", parse_T)
    P = ask("Pressure P", parse_P)
    phase = ask("Phase to report (auto/vapor/liquid)", parse_phase, "auto")
    st = eos.state(T, P, phase)
    if mode == "pvt":
        show_pvt(st)
    elif mode == "fug":
        show_fugacity(st)
        if input("  Print a table of Z, phi and f versus pressure at this T? (y/N): ").strip().lower() == "y":
            P1 = ask("Lowest pressure", parse_P, "1")
            P2 = ask("Highest pressure", parse_P, f"{P:g}")
            n = int(ask("Number of points", float, 10))
            print(f"\n  {'P/bar':>10s} {'Z':>9s} {'phi':>10s} {'f/bar':>11s}")
            for i in range(n + 1):
                Pi = P1 + (P2 - P1) * i / n
                s = eos.state(T, Pi, phase)
                print(f"  {Pi:10.4f} {s.Z:9.5f} {s.phi:10.6f} {s.f:11.5f}")


def act_hs(eos):
    if not eos.sp.has_cp:
        print("  This species has no Cp data; only residual properties will be printed.")
    T = ask("Temperature T", parse_T)
    P = ask("Pressure P", parse_P)
    phase = ask("Phase to report (auto/vapor/liquid)", parse_phase, "auto")
    T0, P0 = T_REF, P_REF
    if eos.sp.has_cp and input("  Change reference state (298.15 K, 1 bar)? (y/N): ").strip().lower() == "y":
        T0 = ask("Reference T0", parse_T, "298.15")
        P0 = ask("Reference P0", parse_P, "1")
    show_hs(eos, T, P, phase, T0, P0)


def act_P_from_TV(eos):
    T = ask("Temperature T", parse_T)
    V = ask("Molar volume V", parse_V)
    if V <= eos.b:
        print(f"  ! V must exceed b = {eos.b:.3f} cm3/mol"); return
    P = eos.pressure(T, V)
    line("=")
    print(f" {eos.sp.name} | {eos.m.name} EOS | T = {T:.2f} K | V = {V:.3f} cm3/mol")
    print(f"   P (EOS)       = {P:.4f} bar")
    print(f"   P (ideal gas) = {R_V * T / V:.4f} bar")
    print(f"   Z = PV/RT     = {P * V / (R_V * T):.5f}")
    if P <= 0:
        print("   NOTE: negative pressure -> this V lies inside the mechanically unstable loop / liquid tension region.")
    line("=")


def act_T_from_PV(eos):
    P = ask("Pressure P", parse_P)
    V = ask("Molar volume V", parse_V)
    try:
        T = eos.temperature(P, V)
    except ValueError as exc:
        print(f"  ! {exc}"); return
    line("=")
    print(f" {eos.sp.name} | {eos.m.name} EOS | P = {P:.4f} bar | V = {V:.3f} cm3/mol")
    print(f"   T (EOS)       = {T:.3f} K   ({T - 273.15:.2f} C)")
    print(f"   T (ideal gas) = {P * V / R_V:.3f} K")
    line("=")


def act_delta(eos):
    if not eos.sp.has_cp:
        print("  ! This species has no ideal-gas Cp data, so total changes cannot be computed."); return
    print("  State 1:")
    T1 = ask("T1", parse_T); P1 = ask("P1", parse_P)
    ph1 = ask("phase 1 (auto/vapor/liquid)", parse_phase, "auto")
    print("  State 2:")
    T2 = ask("T2", parse_T); P2 = ask("P2", parse_P)
    ph2 = ask("phase 2 (auto/vapor/liquid)", parse_phase, "auto")
    r = eos.delta(T1, P1, T2, P2, ph1, ph2)
    line("=")
    print(f" {eos.sp.name} | {eos.m.name} EOS     ({T1:.2f} K, {P1:g} bar) -> ({T2:.2f} K, {P2:g} bar)")
    print(f"   state 1: {r['state1'].state.phase:12s} Z = {r['state1'].state.Z:.5f}   state 2: {r['state2'].state.phase:12s} Z = {r['state2'].state.Z:.5f}")
    print(f"   ideal-gas part   : dH_ig = {r['dH_ig']:12.2f} J/mol    dS_ig = {r['dS_ig']:10.4f} J/(mol K)")
    print(f"   residual part    : dH^R  = {r['dHR']:12.2f} J/mol    dS^R  = {r['dSR']:10.4f} J/(mol K)")
    print(f"   TOTAL            : dH    = {r['dH']:12.2f} J/mol    dS    = {r['dS']:10.4f} J/(mol K)")
    for w in (r['state1'].warning, r['state2'].warning):
        if w:
            print(f"   WARNING: {w}")
    line("=")


def act_sat(eos):
    T = ask("Temperature T (below Tc)", parse_T)
    try:
        s = eos.saturation(T)
    except (ValueError, RuntimeError) as exc:
        print(f"  ! {exc}"); return
    line("=")
    print(f" {eos.sp.name} | {eos.m.name} EOS | T = {T:.2f} K  (Tr = {T / eos.sp.Tc:.4f})")
    print(f"   Saturation pressure Psat = {s.P:.5f} bar   (equal-fugacity condition, Eq. 13.78)")
    print(f"   {'':16s}{'liquid':>14s}{'vapor':>14s}")
    print(f"   {'Z':16s}{s.liquid.Z:14.5f}{s.vapor.Z:14.5f}")
    print(f"   {'V (cm3/mol)':16s}{s.liquid.V:14.3f}{s.vapor.V:14.3f}")
    print(f"   {'ln(phi)':16s}{s.liquid.lnphi:14.8f}{s.vapor.lnphi:14.8f}")
    print(f"   {'phi':16s}{s.liquid.phi:14.6f}{s.vapor.phi:14.6f}")
    print(f"   {'f (bar)':16s}{s.liquid.f:14.5f}{s.vapor.f:14.5f}")
    print(f"   {'H^R (J/mol)':16s}{s.liquid.HR:14.1f}{s.vapor.HR:14.1f}")
    print(f"   Latent heat  dHvap = H^R(v) - H^R(l) = {s.dHvap:.1f} J/mol   (dSvap = {s.dSvap:.3f} J/(mol K))")
    line("=")


def act_compare(sp: Species):
    T = ask("Temperature T", parse_T)
    P = ask("Pressure P", parse_P)
    phase = ask("Phase to report (auto/vapor/liquid)", parse_phase, "vapor")
    res = compare_models(sp, T, P, phase)
    ideal_V = R_V * T / P
    line("=")
    print(f" {sp.name}   T = {T:.2f} K   P = {P:.4f} bar   (ideal-gas V = {ideal_V:.2f} cm3/mol)")
    line()
    print(f" {'EOS':5s}{'Z':>9s}{'V cm3/mol':>12s}{'H^R J/mol':>12s}{'S^R J/mol/K':>13s}{'phi':>10s}{'f/bar':>11s}  phase")
    for name, st in res.items():
        print(f" {name:5s}{st.Z:9.4f}{st.V:12.2f}{st.HR:12.1f}{st.SR:13.3f}{st.phi:10.4f}{st.f:11.4f}  {st.phase}")
    line("=")


def act_plot(eos: CubicEOS):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from eos_plots import plot_pv, plot_Z_vs_P, plot_phi_vs_P
    except ImportError:
        print("  ! matplotlib is not installed:  pip install matplotlib"); return
    sp = eos.sp
    T = ask("Temperature for Z / phi / f plots", parse_T, f"{1.1 * sp.Tc:.1f}")
    Pmax = ask("Maximum pressure for those plots", parse_P, f"{2.0 * sp.Pc:.1f}")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    plot_pv(ax[0, 0], eos)
    ax[0, 0].set_title(f"{sp.name}: P-V isotherms ({eos.m.name})")
    ax[0, 0].legend(fontsize=8)
    plot_Z_vs_P(ax[0, 1], sp, T, Pmax, models=(eos.m.name,), phase="auto")
    ax[0, 1].set_title(f"Compressibility factor at {T:.1f} K")
    plot_phi_vs_P(ax[1, 0], sp, T, Pmax, models=(eos.m.name,), phase="auto", what="phi")
    ax[1, 0].set_title("Fugacity coefficient")
    plot_phi_vs_P(ax[1, 1], sp, T, Pmax, models=(eos.m.name,), phase="auto", what="f")
    ax[1, 1].set_title("Fugacity")
    ax[1, 1].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"plots_{sp.name.replace(' ', '_')}_{eos.m.name}.png"
    fig.savefig(fname, dpi=200)
    print(f"  Figure saved as {fname}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


# ----------------------------------------------------------------------------
# Species / EOS selection
# ----------------------------------------------------------------------------
def choose_species() -> Species:
    names = sorted(SPECIES)
    print("\n  Species in the database (Tables B.1 and C.1):")
    for i in range(0, len(names), 3):
        print("   " + "".join(f"{n:26s}" for n in names[i:i + 3]))
    print("   (type a name or formula such as CO2 / n-butane / H2O, or 'custom' to enter your own)")
    while True:
        txt = input("  Species: ").strip()
        if txt.lower() == "custom":
            name = input("   name: ").strip() or "custom"
            Tc = ask("Tc (K)", float); Pc = ask("Pc (bar)", float); w = ask("acentric factor", float)
            cp_txt = input("   Cp_ig/R = A + B*T + C*T^2 + D/T^2 -> enter 'A B C D' (blank = skip H,S): ").strip()
            cp = tuple(float(x) for x in cp_txt.split()) if cp_txt else None
            if cp is not None and len(cp) != 4:
                print("   ! need four numbers"); continue
            return custom_species(name, Tc, Pc, w, cp)
        try:
            return get_species(txt)
        except KeyError as exc:
            print(f"   ! {exc}")


def choose_eos(default: str = "PR") -> str:
    def p(t):
        t = t.strip().upper()
        if t not in MODELS:
            raise ValueError("choose vdW, RK, SRK or PR")
        return MODELS[t].name
    return ask("Equation of state (vdW / RK / SRK / PR)", p, default)


MENU = """
 ------------------------------ MAIN MENU ------------------------------
  1  PVT : molar volume V and Z from T and P
  2  PVT : pressure P from T and V
  3  PVT : temperature T from P and V
  4  Enthalpy and entropy (H, S, H^R, S^R) at T, P
  5  Pure-component fugacity f and fugacity coefficient phi at T, P
  6  Change in H and S between two states
  7  Saturation (vapor) pressure and latent heat at T
  8  Compare vdW / RK / SRK / PR at one state
  9  Plots (P-V isotherms, Z, phi, f)
  s  Change species      e  Change equation of state      q  Quit
 -----------------------------------------------------------------------"""


def interactive():
    print("=" * 74)
    print("  REAL-GAS PROPERTY CALCULATOR  --  cubic equations of state")
    print("  (notation and data: Smith, Van Ness, Abbott & Swihart, 8th ed.)")
    print("=" * 74)
    sp = choose_species()
    eos_name = choose_eos()
    eos = CubicEOS(sp, eos_name)
    while True:
        print(MENU)
        print(f"  Current: {sp.name}  |  {eos.m.name}  (Tc = {sp.Tc} K, Pc = {sp.Pc} bar, omega = {sp.omega})")
        try:
            c = input("  Select: ").strip().lower()
            if c in ("q", "0", "quit", "exit"):
                break
            elif c == "1": act_TP(eos, "pvt")
            elif c == "2": act_P_from_TV(eos)
            elif c == "3": act_T_from_PV(eos)
            elif c == "4": act_hs(eos)
            elif c == "5": act_TP(eos, "fug")
            elif c == "6": act_delta(eos)
            elif c == "7": act_sat(eos)
            elif c == "8": act_compare(sp)
            elif c == "9": act_plot(eos)
            elif c == "s": sp = choose_species(); eos = CubicEOS(sp, eos.m.name)
            elif c == "e": eos = CubicEOS(sp, choose_eos(eos.m.name))
            else: print("  ! unknown option")
        except (ValueError, RuntimeError) as exc:
            print(f"  ! {exc}")
        except (EOFError, KeyboardInterrupt):
            print(); break
    print("  Goodbye.")


# ----------------------------------------------------------------------------
# One-shot mode and demo
# ----------------------------------------------------------------------------
def one_shot(args):
    sp = get_species(args.species)
    eos = CubicEOS(sp, args.eos)
    T, P = parse_T(str(args.T)), parse_P(str(args.P))
    st = eos.state(T, P, args.phase)
    show_pvt(st)
    show_hs(eos, T, P, args.phase, T_REF, P_REF)
    show_fugacity(st)


def demo():
    sp = get_species("n-butane")
    print("\nDEMO 1 - textbook Table 13.7: n-butane at 500 K and 50 bar")
    res = compare_models(sp, 500.0, 50.0, "vapor")
    print(f" {'EOS':5s}{'Z':>9s}{'H^R J/mol':>12s}{'S^R J/mol/K':>13s}{'phi':>10s}{'f/bar':>10s}")
    for n, s in res.items():
        print(f" {n:5s}{s.Z:9.4f}{s.HR:12.0f}{s.SR:13.3f}{s.phi:10.4f}{s.f:10.3f}")
    print("   book: vdW 0.6608 -3937 -5.424 | RK 0.6850 -4505 -6.546 | SRK 0.7222 -4824 -7.413 | PR 0.6907 -4988 -7.426")
    print("\nDEMO 2 - textbook Example 3.9: n-butane at 350 K and 9.4573 bar (saturation)")
    print(f" {'EOS':5s}{'Vvapor':>10s}{'Vliquid':>10s}      book: vdW 2667/191.0  RK 2555/133.3  SRK 2520/127.8  PR 2486/112.6")
    for m in MODELS.values():
        e = CubicEOS(sp, m.name)
        print(f" {m.name:5s}{e.state(350, 9.4573, 'vapor').V:10.1f}{e.state(350, 9.4573, 'liquid').V:10.1f}")
    print("\nDEMO 3 - saturation pressure from equal fugacity, PR, n-butane at 350 K (experiment: 9.4573 bar)")
    s = CubicEOS(sp, "PR").saturation(350.0)
    print(f"   Psat = {s.P:.4f} bar,  ln(phi_v) = {s.vapor.lnphi:.8f},  ln(phi_l) = {s.liquid.lnphi:.8f}")


def main():
    ap = argparse.ArgumentParser(description="Real-gas PVT, H, S and fugacity from cubic equations of state")
    ap.add_argument("--species", help="species name or formula, e.g. n-butane, CO2")
    ap.add_argument("--eos", default="PR", help="vdW, RK, SRK or PR (default PR)")
    ap.add_argument("--T", help="temperature, e.g. 500 or 25C")
    ap.add_argument("--P", help="pressure, e.g. 50 or 5MPa")
    ap.add_argument("--phase", default="auto", help="auto, vapor or liquid")
    ap.add_argument("--demo", action="store_true", help="reproduce a few textbook examples and exit")
    args = ap.parse_args()
    if args.demo:
        demo()
    elif args.species and args.T and args.P:
        try:
            one_shot(args)
        except (KeyError, ValueError, RuntimeError) as exc:
            sys.exit(f"Error: {exc}")
    else:
        interactive()


if __name__ == "__main__":
    main()
