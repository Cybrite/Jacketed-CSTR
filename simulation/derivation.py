"""Symbolic derivations for the jacketed exothermic CSTR.

This module prints the governing nonlinear balances, the steady-state equations,
and the first-order Taylor linearization around an operating point. The goal is
not to replace the numerical model, but to document the process-control
mathematics clearly and in a form that can be printed from ``main.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from models.parameters import CSTRParameters


@dataclass(frozen=True)
class DerivationReport:
    """Text report and symbolic expressions for the CSTR model."""

    governing_equations: str
    steady_state_equations: str
    linearized_equations: str
    partial_derivatives: str
    transfer_function: str


def build_derivation_report(parameters: CSTRParameters, operating_point: Sequence[float], Fss: float) -> DerivationReport:
    """Build a human-readable symbolic derivation report."""

    Ca, T, F, Ca0, T0, Tc = sp.symbols("C_A T F C_A0 T_0 T_c", real=True)
    Css, Tss = sp.symbols("C_A_ss T_ss", real=True)
    V, deltaH, rho, Cp, UA, k0, E, R = sp.symbols("V DeltaH rho Cp UA k0 E R", positive=True, real=True)
    s = sp.symbols("s", complex=True)

    rA = k0 * sp.exp(-E / (R * T)) * Ca
    dCa_dt = F / V * (Ca0 - Ca) - rA
    dT_dt = F / V * (T0 - T) + (-deltaH) / (rho * Cp) * rA - UA / (rho * Cp * V) * (T - Tc)

    kss = k0 * sp.exp(-E / (R * Tss))
    dr_dCa = kss
    dr_dT = kss * Css * E / (R * Tss**2)

    a11 = -Fss / V - dr_dCa
    a12 = -dr_dT
    a21 = (-deltaH) / (rho * Cp) * dr_dCa
    a22 = -Fss / V + (-deltaH) / (rho * Cp) * dr_dT - UA / (rho * Cp * V)
    b1 = (Ca0 - Css) / V
    b2 = (T0 - Tss) / V

    transfer_function = sp.simplify((b2 * (s - a11) - a12 * b1) / ((s - a11) * (s - a22) - a12 * a21))

    op_ca, op_t = float(operating_point[0]), float(operating_point[1])
    governing = (
        "Material balance:\n"
        f"  dC_A/dt = F/V (C_A0 - C_A) - k0 exp(-E/(RT)) C_A\n"
        "  accumulation = in - out - consumption\n\n"
        "Energy balance:\n"
        f"  rho Cp V dT/dt = F rho Cp (T0 - T) + (-DeltaH) V r_A - UA (T - T_c)\n"
        "  accumulation = in - out + heat generated - heat removed\n\n"
        f"Reaction rate:\n  r_A = k0 exp(-E/(RT)) C_A\n\n"
        f"Inputs and outputs:\n  manipulated input u = F\n  disturbances = C_A0, T0, T_c\n  controlled output y = T\n"
    )

    steady_state = (
        "Steady-state equations (dC_A/dt = 0, dT/dt = 0):\n"
        f"  0 = F_ss/V (C_A0 - C_Ass) - k0 exp(-E/(R T_ss)) C_Ass\n"
        f"  0 = F_ss/V (T0 - T_ss) + (-DeltaH)/(rho Cp) r_A,ss - UA/(rho Cp V) (T_ss - T_c)\n\n"
        f"Operating point used for linearization:\n  C_Ass = {op_ca:.6f}\n  T_ss = {op_t:.6f} K\n  F_ss = {Fss:.6f}\n"
    )

    linearized = (
        "Deviation variables:\n"
        "  C_A' = C_A - C_Ass\n"
        "  T'   = T - T_ss\n"
        "  F'   = F - F_ss\n\n"
        "Linearized model:\n"
        "  dC_A'/dt = a11 C_A' + a12 T' + b1 F'\n"
        "  dT'/dt   = a21 C_A' + a22 T' + b2 F'\n\n"
        f"  a11 = {sp.simplify(a11)}\n"
        f"  a12 = {sp.simplify(a12)}\n"
        f"  a21 = {sp.simplify(a21)}\n"
        f"  a22 = {sp.simplify(a22)}\n"
        f"  b1  = {sp.simplify(b1)}\n"
        f"  b2  = {sp.simplify(b2)}\n"
    )

    partials = (
        "Partial derivatives used in the Taylor expansion:\n"
        f"  dr/dC_A = {sp.simplify(dr_dCa)}\n"
        f"  dr/dT   = {sp.simplify(dr_dT)}\n"
        f"  d(dC_A/dt)/dF = (C_A0 - C_Ass)/V\n"
        f"  d(dT/dt)/dF   = (T0 - T_ss)/V\n"
    )

    transfer = (
        "Transfer function from feed-flow deviation to temperature deviation:\n"
        "  G(s) = Y(s)/U(s) = T'(s)/F'(s)\n"
        f"  G(s) = {sp.simplify(transfer_function)}\n"
    )

    return DerivationReport(
        governing_equations=governing,
        steady_state_equations=steady_state,
        linearized_equations=linearized,
        partial_derivatives=partials,
        transfer_function=transfer,
    )
