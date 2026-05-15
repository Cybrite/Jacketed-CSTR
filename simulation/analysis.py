"""Simulation and closed-loop analysis routines for the CSTR project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import control
import numpy as np

from controllers.pi import PIController, PIGains
from models.cstr import CSTRModel
from utils.metrics import ResponseMetrics, step_response_metrics


@dataclass
class LinearAnalysisResult:
    """Container for linearized model data and response arrays."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    transfer_function: object
    poles: np.ndarray
    zeros: np.ndarray
    step_time: np.ndarray
    step_response: np.ndarray



def simulate_open_loop_disturbance(
    model: CSTRModel,
    steady_state: np.ndarray,
    time_final: float = 60.0,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the nonlinear reactor with feed disturbances and fixed coolant temperature."""

    time_eval = np.arange(0.0, time_final + dt, dt)

    def coolant_temperature(t: float) -> float:
        return model.params.Tc0

    def feed_temperature(t: float) -> float:
        if t < 15.0:
            return model.params.T0
        if t < 35.0:
            return model.params.T0 + 8.0
        return model.params.T0 + 5.0

    def feed_concentration(t: float) -> float:
        if t < 25.0:
            return model.params.Ca0
        return model.params.Ca0 * 1.15

    result = model.simulate_open_loop(
        time_span=(0.0, time_final),
        initial_state=steady_state,
        Tc_profile=coolant_temperature,
        Ca0_profile=feed_concentration,
        T0_profile=feed_temperature,
        time_eval=time_eval,
    )
    Tc = np.full_like(result.t, model.params.Tc0, dtype=float)
    Ca0 = np.array([feed_concentration(t) for t in result.t], dtype=float)
    T0 = np.array([feed_temperature(t) for t in result.t], dtype=float)
    return result.t, result.y[0], result.y[1], Tc, np.vstack([Ca0, T0])



def linear_analysis(model: CSTRModel, operating_point: np.ndarray, Tc: float, time_final: float = 50.0) -> LinearAnalysisResult:
    """Linearize the CSTR and generate the corresponding step response."""

    time = np.linspace(0.0, time_final, 500)
    A, B, C, D = model.linearize(operating_point, Tc)
    state_space = control.ss(A, B, C, D)
    transfer_function = control.ss2tf(state_space)
    poles = control.poles(transfer_function)
    zeros = control.zeros(transfer_function)
    step_time, step_response = control.step_response(transfer_function, T=time)

    return LinearAnalysisResult(
        A=A,
        B=B,
        C=C,
        D=D,
        transfer_function=transfer_function,
        poles=np.asarray(poles, dtype=complex),
        zeros=np.asarray(zeros, dtype=complex),
        step_time=np.asarray(step_time, dtype=float),
        step_response=np.asarray(step_response, dtype=float),
    )



def build_pi_transfer_function(gains: PIGains):
    """Return the continuous-time PI controller transfer function."""

    s = control.TransferFunction.s
    return gains.Kc * (1.0 + 1.0 / (gains.tauI * s))



def closed_loop_linear_response(plant_tf, gains: PIGains, time: np.ndarray):
    """Construct GcGp/(1+GcGp) and simulate the step response."""

    controller_tf = build_pi_transfer_function(gains)
    closed_loop_tf = control.feedback(controller_tf * plant_tf, 1)
    step_time, step_response = control.step_response(closed_loop_tf, T=time)
    return closed_loop_tf, np.asarray(step_time, dtype=float), np.asarray(step_response, dtype=float)



def simulate_closed_loop_step(
    model: CSTRModel,
    gains: PIGains,
    setpoint: float,
    operating_point: np.ndarray,
    time_final: float = 80.0,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a nonlinear closed-loop PI controlled reactor."""

    controller = PIController(gains.Kc, gains.tauI, bias=model.params.Tc0)
    controller.reset()
    time = np.arange(0.0, time_final + dt, dt)
    state = np.asarray(operating_point, dtype=float).copy()
    temperatures = np.empty_like(time)
    coolants = np.empty_like(time)

    for index, _ in enumerate(time):
        temperatures[index] = state[1]
        coolants[index] = controller.compute(setpoint=setpoint, measurement=state[1], dt=dt)
        state = _rk4_step(model, state, coolants[index], dt)
        if not np.all(np.isfinite(state)):
            temperatures[index:] = np.nan
            coolants[index:] = np.nan
            break
    return time, temperatures, coolants



def closed_loop_metrics(time: np.ndarray, response: np.ndarray, setpoint: float) -> ResponseMetrics:
    """Compute standard performance indices and transient-response metrics."""

    return step_response_metrics(time, response, setpoint)



def classical_closed_loop_analysis(
    model: CSTRModel,
    operating_point: np.ndarray,
    tuning_map: Dict[str, PIGains],
    setpoint: float,
) -> Dict[str, Dict[str, object]]:
    """Simulate all classical PI tuning rules and return trajectories and metrics."""

    results: Dict[str, Dict[str, object]] = {}
    for name, gains in tuning_map.items():
        time, temperature, coolant = simulate_closed_loop_step(
            model=model,
            gains=gains,
            setpoint=setpoint,
            operating_point=operating_point,
        )
        metrics = closed_loop_metrics(time, temperature, setpoint)
        results[name] = {
            "time": time,
            "temperature": temperature,
            "coolant": coolant,
            "metrics": metrics,
            "gains": gains,
        }
    return results



def dominant_second_order_characteristics(poles: np.ndarray) -> Tuple[float, float]:
    """Estimate damping ratio and natural frequency from a pole pair."""

    complex_poles = [pole for pole in poles if np.imag(pole) != 0]
    if complex_poles:
        pole = sorted(complex_poles, key=lambda value: abs(np.real(value)))[0]
        wn = float(np.abs(pole))
        zeta = float(-np.real(pole) / max(wn, 1e-9))
        return zeta, wn
    pole = poles[np.argmax(np.real(poles))]
    wn = float(abs(np.real(pole)))
    zeta = 1.0
    return zeta, wn



def _rk4_step(model: CSTRModel, state: np.ndarray, Tc: float, dt: float) -> np.ndarray:
    """Fixed-step RK4 integrator for fast RL rollouts."""

    k1 = model.dynamics(0.0, state, Tc)
    k2 = model.dynamics(0.0, state + 0.5 * dt * k1, Tc)
    k3 = model.dynamics(0.0, state + 0.5 * dt * k2, Tc)
    k4 = model.dynamics(0.0, state + dt * k3, Tc)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)