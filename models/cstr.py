"""Nonlinear jacketed exothermic CSTR model.

The plant is modeled with the classical three-state balances used in chemical
process control texts:

    x = [C_A, T_R, T_J]^T

The manipulated variable is the coolant flow rate F_C. Feed flow F, feed
concentration C_A0, feed temperature T_0, and coolant inlet temperature
T_C,in are treated as disturbances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from models.parameters import CSTRParameters


@dataclass(frozen=True)
class SteadyStateSolution:
    """Steady-state solution and local stability information."""

    Ca: float
    TR: float
    TJ: float
    Fc: float
    eigenvalues: np.ndarray


class CSTRModel:
    """Nonlinear jacketed CSTR model and linearization utilities."""

    def __init__(self, params: CSTRParameters):
        self.params = params

    def reaction_rate(self, Ca: float, TR: float) -> float:
        """Arrhenius reaction rate r_A = k(T_R) C_A."""

        p = self.params
        return p.k0 * np.exp(-p.E / (p.R * TR)) * Ca

    def dynamics(
        self,
        t: float,
        state: np.ndarray,
        Fc: float,
        F: Optional[float] = None,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
        Tcin: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the nonlinear state derivatives."""

        p = self.params
        Ca, TR, TJ = (float(state[0]), float(state[1]), float(state[2]))
        flow_rate = p.F if F is None else float(F)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)
        coolant_flow = float(Fc)

        reaction = self.reaction_rate(Ca, TR)
        feed_flow_term = flow_rate / p.V
        jacket_flow_term = coolant_flow / p.Vj
        heat_release = (-p.deltaH) / (p.rho * p.Cp)
        heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
        heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)

        dCa_dt = feed_flow_term * (Ca_feed - Ca) - reaction
        dTR_dt = feed_flow_term * (T_feed - TR) + heat_release * reaction - heat_transfer_reactor * (TR - TJ)
        dTJ_dt = jacket_flow_term * (Tcin_value - TJ) + heat_transfer_jacket * (TR - TJ)

        return np.array([dCa_dt, dTR_dt, dTJ_dt], dtype=float)

    def simulate_open_loop(
        self,
        time_span: Tuple[float, float],
        initial_state: np.ndarray,
        Fc_profile: Callable[[float], float],
        F_profile: Optional[Callable[[float], float]] = None,
        Tcin_profile: Optional[Callable[[float], float]] = None,
        Ca0_profile: Optional[Callable[[float], float]] = None,
        T0_profile: Optional[Callable[[float], float]] = None,
        time_eval: Optional[np.ndarray] = None,
    ) -> solve_ivp:
        """Simulate the nonlinear reactor with solve_ivp."""

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.dynamics(
                t=t,
                state=x,
                Fc=Fc_profile(t),
                F=None if F_profile is None else F_profile(t),
                Ca0=None if Ca0_profile is None else Ca0_profile(t),
                T0=None if T0_profile is None else T0_profile(t),
                Tcin=None if Tcin_profile is None else Tcin_profile(t),
            )

        return solve_ivp(rhs, time_span, np.asarray(initial_state, dtype=float), t_eval=time_eval, method="RK45")

    def steady_state(
        self,
        F: Optional[float] = None,
        Fc: Optional[float] = None,
        Tcin: Optional[float] = None,
        guess: Optional[Iterable[float]] = None,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the nonlinear steady-state equations for a given operating point."""

        p = self.params
        flow_rate = p.F if F is None else float(F)
        coolant_flow = p.Fc if Fc is None else float(Fc)
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        x0 = np.array([0.5, 350.0, 305.0], dtype=float) if guess is None else np.asarray(list(guess), dtype=float)

        def residual(x: np.ndarray) -> np.ndarray:
            Ca, TR, TJ = (float(x[0]), float(x[1]), float(x[2]))
            reaction = self.reaction_rate(Ca, TR)
            feed_flow_term = flow_rate / p.V
            jacket_flow_term = coolant_flow / p.Vj
            heat_release = (-p.deltaH) / (p.rho * p.Cp)
            heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
            heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)
            return np.array(
                [
                    feed_flow_term * (Ca_feed - Ca) - reaction,
                    feed_flow_term * (T_feed - TR) + heat_release * reaction - heat_transfer_reactor * (TR - TJ),
                    jacket_flow_term * (Tcin_value - TJ) + heat_transfer_jacket * (TR - TJ),
                ],
                dtype=float,
            )

        solution = root(residual, x0, method="hybr")
        if not solution.success:
            raise RuntimeError(f"Steady-state solver failed: {solution.message}")
        state = np.asarray(solution.x, dtype=float)
        A, _, _, _ = self.linearize(state, flow_rate, coolant_flow, Ca0=Ca_feed, T0=T_feed, Tcin=Tcin_value)
        eigenvalues = np.linalg.eigvals(A)
        return state, eigenvalues

    def find_steady_state_candidates(
        self,
        F: Optional[float] = None,
        Fc: Optional[float] = None,
        Tcin: Optional[float] = None,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
    ) -> List[SteadyStateSolution]:
        """Search for multiple steady states from a grid of initial guesses."""

        p = self.params
        flow_rate = p.F if F is None else float(F)
        coolant_flow = p.Fc if Fc is None else float(Fc)
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        candidates: List[SteadyStateSolution] = []
        guesses = [
            np.array([ca_guess, tr_guess, tj_guess], dtype=float)
            for ca_guess in (0.05, 0.2, 0.5, 0.8, 1.0)
            for tr_guess in (320.0, 340.0, 360.0, 380.0, 400.0, 420.0)
            for tj_guess in (295.0, 300.0, 305.0, 310.0, 320.0)
        ]

        for guess in guesses:
            try:
                state, eigenvalues = self._steady_state_from_guess(guess, flow_rate, coolant_flow, Tcin_value, Ca_feed, T_feed)
            except RuntimeError:
                continue
            if any(np.linalg.norm(state - np.array([candidate.Ca, candidate.TR, candidate.TJ])) < 1e-4 for candidate in candidates):
                continue
            candidates.append(
                SteadyStateSolution(
                    Ca=float(state[0]),
                    TR=float(state[1]),
                    TJ=float(state[2]),
                    Fc=coolant_flow,
                    eigenvalues=eigenvalues,
                )
            )
        candidates.sort(key=lambda item: item.TR)
        return candidates

    def _steady_state_from_guess(
        self,
        guess: np.ndarray,
        F: float,
        Fc: float,
        Tcin: float,
        Ca0: float,
        T0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params

        def residual(x: np.ndarray) -> np.ndarray:
            Ca, TR, TJ = (float(x[0]), float(x[1]), float(x[2]))
            reaction = self.reaction_rate(Ca, TR)
            feed_flow_term = F / p.V
            jacket_flow_term = Fc / p.Vj
            heat_release = (-p.deltaH) / (p.rho * p.Cp)
            heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
            heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)
            return np.array(
                [
                    feed_flow_term * (Ca0 - Ca) - reaction,
                    feed_flow_term * (T0 - TR) + heat_release * reaction - heat_transfer_reactor * (TR - TJ),
                    jacket_flow_term * (Tcin - TJ) + heat_transfer_jacket * (TR - TJ),
                ],
                dtype=float,
            )

        solution = root(residual, guess, method="hybr")
        if not solution.success:
            raise RuntimeError(solution.message)
        state = np.asarray(solution.x, dtype=float)
        A, _, _, _ = self.linearize(state, F, Fc, Ca0=Ca0, T0=T0, Tcin=Tcin)
        eigenvalues = np.linalg.eigvals(A)
        return state, eigenvalues

    def coolant_flow_for_steady_state(self, state: np.ndarray, Tcin: Optional[float] = None) -> float:
        """Compute the coolant flow needed to hold a steady jacket temperature."""

        p = self.params
        _, TR, TJ = (float(state[0]), float(state[1]), float(state[2]))
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)
        numerator = p.UA * (TR - TJ)
        denominator = p.rho_j * p.Cp_j * (Tcin_value - TJ)
        if abs(denominator) < 1e-12:
            raise ValueError("Cannot compute steady-state coolant flow when Tcin equals TJ.")
        return float(-numerator / denominator)

    def linearize(
        self,
        state: np.ndarray,
        F: float,
        Fc: float,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
        Tcin: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the continuous-time state-space matrices A, B, C, D."""

        p = self.params
        Ca, TR, TJ = (float(state[0]), float(state[1]), float(state[2]))
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)

        reaction = self.reaction_rate(Ca, TR)
        exp_term = np.exp(-p.E / (p.R * TR))
        dr_dCa = p.k0 * exp_term
        dr_dTR = reaction * (p.E / (p.R * TR**2))

        feed_flow_term = float(F) / p.V
        jacket_flow_term = float(Fc) / p.Vj
        heat_release = (-p.deltaH) / (p.rho * p.Cp)
        heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
        heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)

        A = np.array(
            [
                [-feed_flow_term - dr_dCa, -dr_dTR, 0.0],
                [heat_release * dr_dCa, -feed_flow_term + heat_release * dr_dTR - heat_transfer_reactor, heat_transfer_reactor],
                [0.0, heat_transfer_jacket, -jacket_flow_term - heat_transfer_jacket],
            ],
            dtype=float,
        )
        B = np.array([[0.0], [0.0], [(Tcin_value - TJ) / p.Vj]], dtype=float)
        C = np.array([[0.0, 1.0, 0.0]], dtype=float)
        D = np.array([[0.0]], dtype=float)
        return A, B, C, D

    def linearized_transfer_function(self, state: np.ndarray, F: float, Fc: float, Tcin: Optional[float] = None):
        """Return the transfer function from coolant-flow deviation to reactor temperature."""

        import control

        A, B, C, D = self.linearize(state, F, Fc, Tcin=Tcin)
        state_space = control.ss(A, B, C, D)
        return control.ss2tf(state_space)

    def step_response_linear(self, state: np.ndarray, F: float, Fc: float, time: np.ndarray, Tcin: Optional[float] = None):
        """Compute the linearized step response from F_C to T_R."""

        import control

        transfer_function = self.linearized_transfer_function(state, F, Fc, Tcin=Tcin)
        response_time, response = control.step_response(transfer_function, T=time)
        return np.asarray(response_time, dtype=float), np.asarray(response, dtype=float)
