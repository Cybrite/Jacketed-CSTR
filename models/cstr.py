"""Nonlinear jacketed exothermic CSTR model.

The reactor is modeled with a single irreversible exothermic reaction

    A -> B

under the standard assumptions of perfect mixing, constant volume, constant
density, constant heat capacity, and first-order irreversible kinetics. The
state vector is

    x = [Ca, T]^T

with manipulated feed flow rate F and cooling-jacket temperature Tc treated as
an external disturbance.
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
    T: float
    Tc: float
    eigenvalues: np.ndarray


class CSTRModel:
    """Nonlinear jacketed CSTR model and linearization utilities."""

    def __init__(self, params: CSTRParameters):
        self.params = params

    def reaction_rate(self, Ca: float, T: float) -> float:
        """Arrhenius reaction rate rA = k(T) Ca."""

        p = self.params
        return p.k0 * np.exp(-p.E / (p.R * T)) * Ca

    def dynamics(
        self,
        t: float,
        state: np.ndarray,
        F: float,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
        Tc: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the nonlinear state derivatives at time t."""

        p = self.params
        Ca, T = float(state[0]), float(state[1])
        flow_rate = float(F)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        Tc_value = p.Tc0 if Tc is None else float(Tc)

        reaction = self.reaction_rate(Ca, T)
        flow_term = flow_rate / p.V
        heat_release = (-p.deltaH) / (p.rho * p.Cp)
        heat_transfer = p.UA / (p.rho * p.Cp * p.V)

        # Accumulation = in - out - consumption.
        dCa_dt = flow_term * (Ca_feed - Ca) - reaction
        # Energy accumulation = in - out + heat generated - heat removed.
        dT_dt = flow_term * (T_feed - T) + heat_release * reaction - heat_transfer * (T - Tc_value)

        return np.array([dCa_dt, dT_dt], dtype=float)

    def simulate_open_loop(
        self,
        time_span: Tuple[float, float],
        initial_state: np.ndarray,
        F_profile: Callable[[float], float],
        Tc_profile: Optional[Callable[[float], float]] = None,
        Ca0_profile: Optional[Callable[[float], float]] = None,
        T0_profile: Optional[Callable[[float], float]] = None,
        time_eval: Optional[np.ndarray] = None,
    ) -> solve_ivp:
        """Simulate the nonlinear reactor with solve_ivp.

        Disturbances are represented as callable profiles so step changes in feed
        temperature or feed concentration can be introduced cleanly.
        """

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.dynamics(
                t=t,
                state=x,
                F=F_profile(t),
                Ca0=None if Ca0_profile is None else Ca0_profile(t),
                T0=None if T0_profile is None else T0_profile(t),
                Tc=self.params.Tc0 if Tc_profile is None else Tc_profile(t),
            )

        return solve_ivp(rhs, time_span, np.asarray(initial_state, dtype=float), t_eval=time_eval, method="RK45")

    def steady_state(
        self,
        F: Optional[float] = None,
        Tc: Optional[float] = None,
        guess: Optional[Iterable[float]] = None,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the nonlinear steady-state equations for a given operating point."""

        p = self.params
        flow_rate = p.F if F is None else float(F)
        Tc_value = p.Tc0 if Tc is None else float(Tc)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        x0 = np.array([0.5, 350.0], dtype=float) if guess is None else np.asarray(list(guess), dtype=float)

        def residual(x: np.ndarray) -> np.ndarray:
            Ca, T = float(x[0]), float(x[1])
            reaction = self.reaction_rate(Ca, T)
            flow_term = flow_rate / p.V
            heat_release = (-p.deltaH) / (p.rho * p.Cp)
            heat_transfer = p.UA / (p.rho * p.Cp * p.V)
            return np.array(
                [
                    flow_term * (Ca_feed - Ca) - reaction,
                    flow_term * (T_feed - T) + heat_release * reaction - heat_transfer * (T - Tc_value),
                ],
                dtype=float,
            )

        solution = root(residual, x0, method="hybr")
        if not solution.success:
            raise RuntimeError(f"Steady-state solver failed: {solution.message}")
        state = np.asarray(solution.x, dtype=float)
        A, _, _, _ = self.linearize(state, flow_rate, Tc_value, Ca0=Ca_feed, T0=T_feed)
        eigenvalues = np.linalg.eigvals(A)
        return state, eigenvalues

    def find_steady_state_candidates(
        self,
        F: Optional[float] = None,
        Tc: Optional[float] = None,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
    ) -> List[SteadyStateSolution]:
        """Search for multiple steady states from a grid of initial guesses."""

        p = self.params
        flow_rate = p.F if F is None else float(F)
        Tc_value = p.Tc0 if Tc is None else float(Tc)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        candidates: List[SteadyStateSolution] = []
        guesses = [
            np.array([ca_guess, t_guess], dtype=float)
            for ca_guess in (0.05, 0.2, 0.5, 0.8, 1.0)
            for t_guess in (320.0, 340.0, 360.0, 380.0, 400.0, 420.0)
        ]

        for guess in guesses:
            try:
                state, eigenvalues = self._steady_state_from_guess(guess, flow_rate, Tc_value, Ca_feed, T_feed)
            except RuntimeError:
                continue
            if any(np.linalg.norm(state - np.array([candidate.Ca, candidate.T])) < 1e-4 for candidate in candidates):
                continue
            candidates.append(
                SteadyStateSolution(Ca=float(state[0]), T=float(state[1]), Tc=Tc_value, eigenvalues=eigenvalues)
            )
        candidates.sort(key=lambda item: item.T)
        return candidates

    def _steady_state_from_guess(
        self,
        guess: np.ndarray,
        F: float,
        Tc: float,
        Ca0: float,
        T0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params

        def residual(x: np.ndarray) -> np.ndarray:
            Ca, T = float(x[0]), float(x[1])
            reaction = self.reaction_rate(Ca, T)
            flow_term = F / p.V
            heat_release = (-p.deltaH) / (p.rho * p.Cp)
            heat_transfer = p.UA / (p.rho * p.Cp * p.V)
            return np.array(
                [
                    flow_term * (Ca0 - Ca) - reaction,
                    flow_term * (T0 - T) + heat_release * reaction - heat_transfer * (T - Tc),
                ],
                dtype=float,
            )

        solution = root(residual, guess, method="hybr")
        if not solution.success:
            raise RuntimeError(solution.message)
        state = np.asarray(solution.x, dtype=float)
        A, _, _, _ = self.linearize(state, F, Tc, Ca0=Ca0, T0=T0)
        eigenvalues = np.linalg.eigvals(A)
        return state, eigenvalues

    def linearize(
        self,
        state: np.ndarray,
        F: float,
        Tc: float,
        Ca0: Optional[float] = None,
        T0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the continuous-time state-space matrices A, B, C, D."""

        p = self.params
        Ca, T = float(state[0]), float(state[1])
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)

        reaction = self.reaction_rate(Ca, T)
        exp_term = np.exp(-p.E / (p.R * T))
        dr_dCa = p.k0 * exp_term
        dr_dT = reaction * (p.E / (p.R * T**2))

        flow_term = float(F) / p.V
        heat_release = (-p.deltaH) / (p.rho * p.Cp)
        heat_transfer = p.UA / (p.rho * p.Cp * p.V)

        A = np.array(
            [
                [-flow_term - dr_dCa, -dr_dT],
                [heat_release * dr_dCa, -flow_term + heat_release * dr_dT - heat_transfer],
            ],
            dtype=float,
        )
        B = np.array([[(Ca_feed - Ca) / p.V], [(T_feed - T) / p.V]], dtype=float)
        C = np.array([[0.0, 1.0]], dtype=float)
        D = np.array([[0.0]], dtype=float)
        return A, B, C, D

    def linearized_transfer_function(self, state: np.ndarray, F: float, Tc: float):
        """Return the transfer function from feed-flow deviation to reactor temperature."""

        import control

        A, B, C, D = self.linearize(state, F, Tc)
        state_space = control.ss(A, B, C, D)
        return control.ss2tf(state_space)

    def step_response_linear(self, state: np.ndarray, F: float, Tc: float, time: np.ndarray):
        """Compute the linearized step response from F to T."""

        import control

        transfer_function = self.linearized_transfer_function(state, F, Tc)
        response_time, response = control.step_response(transfer_function, T=time)
        return np.asarray(response_time, dtype=float), np.asarray(response, dtype=float)