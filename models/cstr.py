"""Nonlinear jacketed exothermic CSTR model."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from models.parameters import CSTRParameters

@dataclass(frozen=True)
class SteadyStateSolution:
    Ca: float; TR: float; TJ: float; Fc: float; eigenvalues: np.ndarray

class CSTRModel:
    def __init__(self, params: CSTRParameters):
        self.params = params

    def reaction_rate(self, Ca: float, TR: float) -> float:
        """Arrhenius reaction rate with absolute mathematical overflow protection."""
        p = self.params
        TR_safe = max(1.0, float(TR)) 
        
        # THE FIX: This mathematically shields the exponent, preventing the NaN crash
        exponent = np.clip(-p.E / (p.R * TR_safe), -500.0, 100.0)
        return p.k0 * np.exp(exponent) * Ca

    def dynamics(self, t: float, state: np.ndarray, Fc: float, F: Optional[float] = None, Ca0: Optional[float] = None, T0: Optional[float] = None, Tcin: Optional[float] = None) -> np.ndarray:
        p = self.params
        Ca, TR, TJ = float(state[0]), float(state[1]), float(state[2])
        flow_rate = p.F if F is None else float(F)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)

        reaction = self.reaction_rate(Ca, TR)
        feed_flow_term = flow_rate / p.V
        jacket_flow_term = float(Fc) / p.Vj
        heat_release = (-p.deltaH) / (p.rho * p.Cp)
        heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
        heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)

        dCa_dt = feed_flow_term * (Ca_feed - Ca) - reaction
        dTR_dt = feed_flow_term * (T_feed - TR) + heat_release * reaction - heat_transfer_reactor * (TR - TJ)
        dTJ_dt = jacket_flow_term * (Tcin_value - TJ) + heat_transfer_jacket * (TR - TJ)
        return np.array([dCa_dt, dTR_dt, dTJ_dt], dtype=float)

    def simulate_open_loop(self, time_span: Tuple[float, float], initial_state: np.ndarray, Fc_profile: Callable[[float], float], F_profile: Optional[Callable[[float], float]] = None, Tcin_profile: Optional[Callable[[float], float]] = None, Ca0_profile: Optional[Callable[[float], float]] = None, T0_profile: Optional[Callable[[float], float]] = None, time_eval: Optional[np.ndarray] = None) -> solve_ivp:
        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.dynamics(t=t, state=x, Fc=Fc_profile(t), F=None if F_profile is None else F_profile(t), Ca0=None if Ca0_profile is None else Ca0_profile(t), T0=None if T0_profile is None else T0_profile(t), Tcin=None if Tcin_profile is None else Tcin_profile(t))
        return solve_ivp(rhs, time_span, np.asarray(initial_state, dtype=float), t_eval=time_eval, method="RK45")

    def steady_state(self, F=None, Fc=None, Tcin=None, guess=None, Ca0=None, T0=None):
        p = self.params
        flow_rate = p.F if F is None else float(F)
        coolant_flow = p.Fc if Fc is None else float(Fc)
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)
        Ca_feed = p.Ca0 if Ca0 is None else float(Ca0)
        T_feed = p.T0 if T0 is None else float(T0)
        x0 = np.array([0.5, 350.0, 305.0], dtype=float) if guess is None else np.asarray(list(guess), dtype=float)

        def residual(x: np.ndarray) -> np.ndarray:
            Ca, TR, TJ = float(x[0]), float(x[1]), float(x[2])
            reaction = self.reaction_rate(Ca, TR)
            feed_flow_term = flow_rate / p.V
            jacket_flow_term = coolant_flow / p.Vj
            heat_release = (-p.deltaH) / (p.rho * p.Cp)
            heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
            heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)
            return np.array([feed_flow_term * (Ca_feed - Ca) - reaction, feed_flow_term * (T_feed - TR) + heat_release * reaction - heat_transfer_reactor * (TR - TJ), jacket_flow_term * (Tcin_value - TJ) + heat_transfer_jacket * (TR - TJ)], dtype=float)

        solution = root(residual, x0, method="hybr")
        if not solution.success: raise RuntimeError(f"Steady-state solver failed: {solution.message}")
        state = np.asarray(solution.x, dtype=float)
        A, _, _, _ = self.linearize(state, flow_rate, coolant_flow, Ca0=Ca_feed, T0=T_feed, Tcin=Tcin_value)
        return state, np.linalg.eigvals(A)

    def find_steady_state_candidates(self, F=None, Fc=None, Tcin=None, Ca0=None, T0=None):
        guesses = [np.array([c, tr, tj], dtype=float) for c in (0.05, 0.2, 0.5, 0.8, 1.0) for tr in (320.0, 340.0, 360.0, 380.0, 400.0, 420.0) for tj in (295.0, 300.0, 305.0, 310.0, 320.0)]
        candidates = []
        for guess in guesses:
            try: state, eigenvalues = self._steady_state_from_guess(guess, F if F is not None else self.params.F, Fc if Fc is not None else self.params.Fc, Tcin if Tcin is not None else self.params.Tcin0, Ca0 if Ca0 is not None else self.params.Ca0, T0 if T0 is not None else self.params.T0)
            except RuntimeError: continue
            if not any(np.linalg.norm(state - np.array([c.Ca, c.TR, c.TJ])) < 1e-4 for c in candidates):
                candidates.append(SteadyStateSolution(Ca=float(state[0]), TR=float(state[1]), TJ=float(state[2]), Fc=Fc if Fc is not None else self.params.Fc, eigenvalues=eigenvalues))
        candidates.sort(key=lambda item: item.TR)
        return candidates

    def _steady_state_from_guess(self, guess, F, Fc, Tcin, Ca0, T0):
        def residual(x):
            Ca, TR, TJ = float(x[0]), float(x[1]), float(x[2])
            reaction = self.reaction_rate(Ca, TR)
            return np.array([F/self.params.V * (Ca0 - Ca) - reaction, F/self.params.V * (T0 - TR) + (-self.params.deltaH)/(self.params.rho*self.params.Cp) * reaction - self.params.UA/(self.params.rho*self.params.Cp*self.params.V) * (TR - TJ), Fc/self.params.Vj * (Tcin - TJ) + self.params.UA/(self.params.rho_j*self.params.Cp_j*self.params.Vj) * (TR - TJ)], dtype=float)
        solution = root(residual, guess, method="hybr")
        if not solution.success: raise RuntimeError(solution.message)
        state = np.asarray(solution.x, dtype=float)
        A, _, _, _ = self.linearize(state, F, Fc, Ca0=Ca0, T0=T0, Tcin=Tcin)
        return state, np.linalg.eigvals(A)

    def coolant_flow_for_steady_state(self, state, Tcin=None):
        p = self.params
        _, TR, TJ = float(state[0]), float(state[1]), float(state[2])
        Tcin_value = p.Tcin0 if Tcin is None else float(Tcin)
        return float(- (p.UA * (TR - TJ)) / (p.rho_j * p.Cp_j * (Tcin_value - TJ)))

    def linearize(self, state, F, Fc, Ca0=None, T0=None, Tcin=None):
        p = self.params
        Ca, TR, TJ = float(state[0]), float(state[1]), float(state[2])
        reaction = self.reaction_rate(Ca, TR)
        TR_safe = max(1.0, float(TR))
        dr_dCa = p.k0 * np.exp(np.clip(-p.E / (p.R * TR_safe), -500.0, 100.0))
        dr_dTR = reaction * (p.E / (p.R * TR_safe**2))

        feed_flow_term = float(F) / p.V
        jacket_flow_term = float(Fc) / p.Vj
        heat_release = (-p.deltaH) / (p.rho * p.Cp)
        heat_transfer_reactor = p.UA / (p.rho * p.Cp * p.V)
        heat_transfer_jacket = p.UA / (p.rho_j * p.Cp_j * p.Vj)

        A = np.array([[-feed_flow_term - dr_dCa, -dr_dTR, 0.0], [heat_release * dr_dCa, -feed_flow_term + heat_release * dr_dTR - heat_transfer_reactor, heat_transfer_reactor], [0.0, heat_transfer_jacket, -jacket_flow_term - heat_transfer_jacket]], dtype=float)
        B = np.array([[0.0], [0.0], [((p.Tcin0 if Tcin is None else float(Tcin)) - TJ) / p.Vj]], dtype=float)
        C = np.array([[0.0, 1.0, 0.0]], dtype=float)
        D = np.array([[0.0]], dtype=float)
        return A, B, C, D

    def linearized_transfer_function(self, state, F, Fc, Tcin=None):
        import control
        A, B, C, D = self.linearize(state, F, Fc, Tcin=Tcin)
        return control.ss2tf(control.ss(A, B, C, D))

    def step_response_linear(self, state, F, Fc, time, Tcin=None):
        import control
        response_time, response = control.step_response(self.linearized_transfer_function(state, F, Fc, Tcin=Tcin), T=time)
        return np.asarray(response_time, dtype=float), np.asarray(response, dtype=float)