"""Symbolic derivations and report generation for the jacketed exothermic CSTR.

The report is written in markdown so it can be copied directly into a notebook
or exported as a project appendix. The content follows classical process control
methodology: nonlinear balances, steady-state analysis, deviation variables,
linearization, state-space form, transfer-function derivation, open-loop
analysis, and PI control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sympy as sp

from models.parameters import CSTRParameters


@dataclass(frozen=True)
class DerivationReport:
    """Markdown report and the main text sections used by the project."""

    markdown: str
    nomenclature: str
    balances: str
    linearization: str
    transfer_function: str
    control_analysis: str


def _matrix_to_latex(matrix: sp.Matrix) -> str:
    return sp.latex(matrix)


def build_derivation_report(
    parameters: CSTRParameters,
    operating_point: Sequence[float],
    Fss: float,
    Fcss: float,
) -> DerivationReport:
    """Build a full markdown derivation report for the jacketed CSTR."""

    Ca, TR, TJ, F, Fc, Ca0, T0, Tcin = sp.symbols("C_A T_R T_J F F_C C_A0 T_0 T_Cin", real=True)
    V, Vj, deltaH, rho, Cp, rhoj, Cpj, UA, k0, E, R = sp.symbols("V V_j DeltaH rho Cp rho_j Cp_j UA k0 E R", positive=True, real=True)
    s = sp.symbols("s", complex=True)

    rA = k0 * sp.exp(-E / (R * TR)) * Ca
    dCa_dt = F / V * (Ca0 - Ca) - rA
    dTR_dt = F / V * (T0 - TR) + (-deltaH) / (rho * Cp) * rA - UA / (rho * Cp * V) * (TR - TJ)
    dTJ_dt = Fc / Vj * (Tcin - TJ) + UA / (rhoj * Cpj * Vj) * (TR - TJ)

    dr_dCa = sp.diff(rA, Ca)
    dr_dTR = sp.diff(rA, TR)

    A = sp.Matrix(
        [
            [-F / V - dr_dCa, -dr_dTR, 0],
            [(-deltaH) / (rho * Cp) * dr_dCa, -F / V + (-deltaH) / (rho * Cp) * dr_dTR - UA / (rho * Cp * V), UA / (rho * Cp * V)],
            [0, UA / (rhoj * Cpj * Vj), -Fc / Vj - UA / (rhoj * Cpj * Vj)],
        ]
    )
    B = sp.Matrix([0, 0, (Tcin - TJ) / Vj])
    C = sp.Matrix([[0, 1, 0]])
    transfer = sp.simplify((C * (s * sp.eye(3) - A).adjugate() * B)[0] / sp.factor((s * sp.eye(3) - A).det()))

    op_ca, op_tr, op_tj = float(operating_point[0]), float(operating_point[1]), float(operating_point[2])
    nomenclature = (
        "| Symbol | Meaning | Units |\n"
        "| --- | --- | --- |\n"
        "| V | reactor volume | L |\n"
        "| V_j | jacket volume | L |\n"
        "| F | feed flow rate | L/min |\n"
        "| F_C | coolant flow rate | L/min |\n"
        "| C_A | reactant concentration in reactor | mol/L |\n"
        "| C_A0 | feed concentration | mol/L |\n"
        "| T_R | reactor temperature | K |\n"
        "| T_J | jacket temperature | K |\n"
        "| T_0 | feed temperature | K |\n"
        "| T_C,in | coolant inlet temperature | K |\n"
        "| U A | heat-transfer coefficient times area | cal/min-K |\n"
        "| \u0394H | reaction enthalpy | cal/mol |\n"
        "| \u03c1 | reactor density | g/L |\n"
        "| C_p | reactor heat capacity | cal/g-K |\n"
        "| \u03c1_j | jacket density | g/L |\n"
        "| C_{p,j} | jacket heat capacity | cal/g-K |\n"
        "| k_0 | pre-exponential factor | 1/min |\n"
        "| E | activation energy | cal/mol |\n"
        "| R | gas constant | cal/mol-K |\n"
    )

    balances = rf"""## 3. Nonlinear material balance

Start with accumulation = in - out - consumption:

$$V\frac{{dC_A}}{{dt}} = F(C_{{A0}} - C_A) - V r_A$$

with

$$r_A = k_0 e^{{-E/(R T_R)}} C_A$$

Therefore

$$\frac{{dC_A}}{{dt}} = \frac{{F}}{{V}}(C_{{A0}} - C_A) - k_0 e^{{-E/(R T_R)}} C_A$$

The first term is dilution by the feed; the second is reaction consumption.

## 4. Reactor energy balance

$$\rho C_p V\frac{{dT_R}}{{dt}} = F\rho C_p(T_0 - T_R) + (-\Delta H) V r_A - U A(T_R - T_J)$$

The three contributions are sensible enthalpy flow, heat generation by reaction, and heat removed to the jacket.

## 5. Jacket energy balance

$$\rho_j C_{{p,j}} V_j\frac{{dT_J}}{{dt}} = F_C\rho_j C_{{p,j}}(T_{{C,in}} - T_J) + U A(T_R - T_J)$$

The coolant flow rate appears because the coolant stream carries enthalpy into and out of the jacket. The jacket adds a third dynamic state because it stores thermal energy independently of the reactor.
"""

    steady_state = rf"""## 6. Steady-state relations

Set the three derivatives to zero. The concentration equation gives an explicit result:

$$C_{{A,s}} = \frac{{F C_{{A0}}}}{{F + V k_s}}, \qquad k_s = k_0 e^{{-E/(R T_{{R,s}})}}$$

The jacket balance gives a closed-form coolant-flow relation:

$$F_{{C,s}} = -\frac{{U A(T_{{R,s}} - T_{{J,s}})}}{{\rho_j C_{{p,j}}(T_{{C,in}} - T_{{J,s}})}}$$

The reactor temperature remains implicitly coupled through the Arrhenius term and the jacket temperature. Numerically, the operating point used here is:

$$C_{{A,s}} = {op_ca:.4f}, \quad T_{{R,s}} = {op_tr:.2f}\,K, \quad T_{{J,s}} = {op_tj:.2f}\,K, \quad F_{{C,s}} = {Fcss:.2f}$$
"""

    deviation_and_linearization = rf"""## 7. Deviation variables

Define deviations from the steady state:

$$C_A' = C_A - C_{{A,s}},\; T_R' = T_R - T_{{R,s}},\; T_J' = T_J - T_{{J,s}},\; F_C' = F_C - F_{{C,s}}$$

Deviations simplify linear analysis because the equilibrium is shifted to the origin.

## 8. Taylor linearization

Let $f_1, f_2, f_3$ be the right-hand sides of the three differential equations. Then

$$\Delta \dot{{x}} \approx J_x \Delta x + J_u \Delta u$$

where the Jacobian matrices are evaluated at the steady state. The partial derivatives are:

$$\frac{{\partial r_A}}{{\partial C_A}} = {sp.latex(dr_dCa)} , \qquad \frac{{\partial r_A}}{{\partial T_R}} = {sp.latex(dr_dTR)}$$

The linearized model is:

$$\begin{{bmatrix}} C_A' \\ T_R' \\ T_J' \end{{bmatrix}}^{{\cdot}} = { _matrix_to_latex(A) }\begin{{bmatrix}} C_A' \\ T_R' \\ T_J' \end{{bmatrix}} + { _matrix_to_latex(B) } F_C'$$

with output equation

$$T_R' = { _matrix_to_latex(C) }\begin{{bmatrix}} C_A' \\ T_R' \\ T_J' \end{{bmatrix}}$$
"""

    transfer_function = rf"""## 9. State-space and transfer function

The state-space model is

$$\dot{{x}} = Ax + Bu, \qquad y = Cx + Du$$

with $x = [C_A', T_R', T_J']^T$, $u = F_C'$, and $y = T_R'$.

Eliminating the states in the Laplace domain gives a third-order plant:

$$G_p(s) = \frac{{T_R'(s)}}{{F_C'(s)}} = \frac{{a_{{23}}b_3(s-a_{{11}})}}{{(s-a_{{22}})(s-a_{{11}})(s-a_{{33}}) - a_{{21}}a_{{12}}(s-a_{{33}}) - a_{{23}}a_{{32}}(s-a_{{11}})}}$$

with the coefficients taken from the Jacobian. After substitution, the symbolic plant transfer function is:

$$G_p(s) = {sp.latex(transfer)}$$

Because the model has concentration, reactor-temperature, and jacket-temperature states, it is fundamentally third order. In many operating regions the concentration state is faster than the thermal states, so the dominant behavior often looks approximately second order.
"""

    control_analysis = (
        "## 10. Open-loop analysis\n\n"
        "The linearized plant is analyzed through poles, zeros, a unit step response, and a Bode plot. A stable operating point should have all poles in the left half-plane.\n\n"
        "## 11. PI controller\n\n"
        "A classical PI controller is\n\n"
        "$$G_c(s) = K_c \left(1 + \frac{1}{\tau_I s}\right)$$\n\n"
        "Proportional action reacts immediately to error, integral action removes offset, and the controller output manipulates coolant flow.\n\n"
        "## 12. Closed loop\n\n"
        "The standard feedback form is\n\n"
        "$$\frac{T_R(s)}{R(s)} = \frac{G_c(s)G_p(s)}{1 + G_c(s)G_p(s)}$$\n\n"
        "This project evaluates both Ziegler-Nichols and IMC PI tuning. IMC is usually more robust for exothermic reactors because it intentionally trades bandwidth for stability margin.\n\n"
        "## 13. Performance metrics\n\n"
        "The project compares rise time, settling time, overshoot, steady-state error, IAE, ISE, and ITAE for open loop, ZN-PI, and IMC-PI.\n"
    )

    markdown = (
        "# Jacketed Exothermic CSTR Control Report\n\n"
        "## 1. Process description\n\n"
        "A jacketed exothermic continuous stirred-tank reactor is regulated by manipulating coolant flow through the jacket. The controlled variable is reactor temperature $T_R$, the manipulated variable is coolant flow $F_C$, and the jacket temperature $T_J$ mediates the thermal interaction.\n\n"
        "## 2. Nomenclature\n\n"
        f"{nomenclature}\n"
        f"{balances}\n"
        f"{steady_state}\n"
        f"{deviation_and_linearization}\n"
        f"{transfer_function}\n"
        f"{control_analysis}\n"
        "## 14. Python implementation\n\n"
        "The repository organizes the classical workflow into `models/cstr.py`, `controllers/pi.py`, `simulation/analysis.py`, `simulation/derivation.py`, and `plots/visualization.py`. The top-level script `main.py` generates figures, metrics, and this report automatically.\n"
    )

    return DerivationReport(
        markdown=markdown,
        nomenclature=nomenclature,
        balances=balances,
        linearization=deviation_and_linearization,
        transfer_function=transfer_function,
        control_analysis=control_analysis,
    )
