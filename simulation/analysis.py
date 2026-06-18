"""Simulation and closed-loop analysis routines for the CSTR project."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import control
import numpy as np

from controllers.pi import PIController, PIGains
from models.cstr import CSTRModel
from utils.metrics import step_response_metrics

@dataclass
class LinearAnalysisResult:
    A: np.ndarray; B: np.ndarray; C: np.ndarray; D: np.ndarray
    transfer_function: object; poles: np.ndarray; zeros: np.ndarray
    step_time: np.ndarray; step_response: np.ndarray

def simulate_open_loop_disturbance(model: CSTRModel, steady_state: np.ndarray, time_final: float = 60.0, dt: float = 0.1):
    time_eval = np.arange(0.0, time_final + dt, dt)
    def coolant_flow(t): return model.params.Fc * (0.90 if 18.0 <= t < 38.0 else (1.08 if t >= 38.0 else 1.0))
    def feed_concentration(t): return model.params.Ca0 * (1.12 if t >= 28.0 else 1.0)
    result = model.simulate_open_loop(time_span=(0.0, time_final), initial_state=steady_state, Fc_profile=coolant_flow, F_profile=lambda t: model.params.F, Tcin_profile=lambda t: model.params.Tcin0, Ca0_profile=feed_concentration, T0_profile=lambda t: model.params.T0, time_eval=time_eval)
    Fc = np.array([coolant_flow(t) for t in result.t], dtype=float)
    Ca0 = np.array([feed_concentration(t) for t in result.t], dtype=float)
    return result.t, result.y[0], result.y[1], result.y[2], Fc, np.vstack([Ca0, np.full_like(Ca0, model.params.T0), np.full_like(Ca0, model.params.Tcin0)])

def linear_analysis(model, operating_point, F, Fc, Tcin, time_final=50.0):
    time = np.linspace(0.0, time_final, 500)
    A, B, C, D = model.linearize(operating_point, F, Fc, Tcin=Tcin)
    state_space = control.ss(A, B, C, D); tf = control.ss2tf(state_space)
    step_time, step_response = control.step_response(tf, T=time)
    return LinearAnalysisResult(A, B, C, D, tf, np.asarray(control.poles(tf), dtype=complex), np.asarray(control.zeros(tf), dtype=complex), step_time, step_response)

def build_pi_transfer_function(gains: PIGains): return gains.Kc * (1.0 + 1.0 / (gains.tauI * control.TransferFunction.s))

def closed_loop_linear_response(plant_tf, gains: PIGains, time: np.ndarray):
    closed_loop_tf = control.feedback(build_pi_transfer_function(gains) * plant_tf, 1)
    t, y = control.step_response(closed_loop_tf, T=time)
    return closed_loop_tf, t, y

def _rk4_step_pure(model, state, Fc, dt, Ca0=None, T0=None, Tcin=None):
    k1 = model.dynamics(0.0, state, Fc=Fc, F=model.params.F, Ca0=Ca0, T0=T0, Tcin=Tcin)
    k2 = model.dynamics(0.0, state + 0.5 * dt * k1, Fc=Fc, F=model.params.F, Ca0=Ca0, T0=T0, Tcin=Tcin)
    k3 = model.dynamics(0.0, state + 0.5 * dt * k2, Fc=Fc, F=model.params.F, Ca0=Ca0, T0=T0, Tcin=Tcin)
    k4 = model.dynamics(0.0, state + dt * k3, Fc=Fc, F=model.params.F, Ca0=Ca0, T0=T0, Tcin=Tcin)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def simulate_closed_loop_step(model, gains, setpoint, operating_point, time_final=80.0, dt=0.1):
    controller = PIController(gains.Kc, gains.tauI, bias=model.params.Fc, u_min=model.params.Fc_min, u_max=model.params.Fc_max)
    time = np.arange(0.0, time_final + dt, dt); state = np.asarray(operating_point, dtype=float).copy()
    temp, flow = np.empty_like(time), np.empty_like(time)
    for i in range(len(time)):
        temp[i] = state[1]; flow[i] = controller.compute(setpoint, state[1], dt)
        state = _rk4_step_pure(model, state, flow[i], dt)
        if state[1] > 600.0 or state[1] < 150.0:
            temp[i:], flow[i:] = model.params.T_safe, model.params.Fc_max; break
    return time, temp, flow

def simulate_closed_loop_disturbance_rejection(model, gains, setpoint, operating_point, disturbance, time_final=40.0, dt=0.1, disturbance_time=20.0):
    controller = PIController(gains.Kc, gains.tauI, bias=model.params.Fc, u_min=model.params.Fc_min, u_max=model.params.Fc_max)
    time = np.arange(0.0, time_final + dt, dt); state = np.asarray(operating_point, dtype=float).copy()
    temp, flow, trace = np.empty_like(time), np.empty_like(time), np.empty_like(time)
    for i, current_time in enumerate(time):
        Ca0, T0, Tcin = model.params.Ca0, model.params.T0, model.params.Tcin0
        if disturbance == "ca0": Ca0 = trace[i] = model.params.Ca0 * (1.12 if current_time >= disturbance_time else 1.0)
        elif disturbance == "t0": T0 = trace[i] = model.params.T0 + (8.0 if current_time >= disturbance_time else 0.0)
        elif disturbance == "tcin": Tcin = trace[i] = model.params.Tcin0 + (5.0 if current_time >= disturbance_time else 0.0)
        temp[i] = state[1]; flow[i] = controller.compute(setpoint, state[1], dt)
        state = _rk4_step_pure(model, state, flow[i], dt, Ca0, T0, Tcin)
        if state[1] > 600.0 or state[1] < 150.0:
            temp[i:], flow[i:], trace[i:] = model.params.T_safe, model.params.Fc_max, trace[i]; break
    return time, temp, flow, trace

def _rk4_step_shielded(model, state, Fc, dt, Ca0=None, T0=None, Tcin=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        def rhs(x):
            xs = np.copy(x)
            xs[1] = np.clip(xs[1], 150.0, 600.0)
            dx = model.dynamics(0.0, xs, Fc=Fc, F=model.params.F, Ca0=Ca0, T0=T0, Tcin=Tcin)
            return np.nan_to_num(dx)
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        res = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.nan_to_num(res, nan=model.params.T_safe)

def simulate_closed_loop_policy(model, policy_fn, setpoint, operating_point, time_final=80.0, dt=0.1, disturbance=None, disturbance_time=20.0, observation_horizon=20.0, baseline_Kc=-5.0, baseline_tauI=2.0, delta_Kc=2.5, delta_tauI=1.0):
    time = np.arange(0.0, time_final + dt, dt); state = np.asarray(operating_point, dtype=float).copy()
    temp, flow, trace, gains_trace = np.empty_like(time), np.empty_like(time), np.empty_like(time), np.empty((len(time), 2))
    prev_err, int_action = setpoint - state[1], 0.0
    for i, current_time in enumerate(time):
        Ca0, T0, Tcin = model.params.Ca0, model.params.T0, model.params.Tcin0
        if disturbance == "ca0": Ca0 = trace[i] = model.params.Ca0 * (1.12 if current_time >= disturbance_time else 1.0)
        elif disturbance == "t0": T0 = trace[i] = model.params.T0 + (8.0 if current_time >= disturbance_time else 0.0)
        else: trace[i] = np.nan
        temp[i], error = state[1], setpoint - state[1]
        obs = np.clip([error / 10.0, int_action / 50.0, (error - prev_err) / (dt * 50.0)], -10.0, 10.0).astype(np.float32)
        
        act = np.clip(np.asarray(policy_fn(obs)).reshape(-1), -1.0, 1.0)
        Kc = baseline_Kc + act[0] * delta_Kc
        tauI = max(0.1, baseline_tauI + act[1] * delta_tauI)
        gains_trace[i] = [Kc, tauI]
        
        int_action += (Kc / tauI) * error * dt
        flow_raw = model.params.Fc + Kc * error + int_action
        flow[i] = float(np.clip(flow_raw, model.params.Fc_min, model.params.Fc_max))
        if flow_raw > model.params.Fc_max or flow_raw < model.params.Fc_min: int_action -= (Kc / tauI) * error * dt 
            
        state = _rk4_step_shielded(model, state, flow[i], dt, Ca0, T0, Tcin)
        prev_err = error
        if state[1] > 550.0 or state[1] < 150.0:
            temp[i:], flow[i:], gains_trace[i:] = model.params.T_safe, model.params.Fc_max, [Kc, tauI]
            if not np.isnan(trace[i]): trace[i:] = trace[i]
            break
    return time, np.clip(temp, 100.0, 600.0), flow, gains_trace, trace

def closed_loop_metrics(time, response, setpoint): return step_response_metrics(time, response, setpoint)

def classical_closed_loop_analysis(model, operating_point, tuning_map, setpoint):
    return {name: {"time": t, "temperature": y, "flow": u, "metrics": closed_loop_metrics(t, y, setpoint), "gains": g} for name, g in tuning_map.items() for t, y, u in [simulate_closed_loop_step(model, g, setpoint, operating_point)]}

def dominant_second_order_characteristics(poles):
    c_poles = [p for p in poles if np.imag(p) != 0]
    if c_poles:
        p = sorted(c_poles, key=lambda v: abs(np.real(v)))[0]
        return float(-np.real(p) / max(float(np.abs(p)), 1e-9)), float(np.abs(p))
    p = poles[np.argmax(np.real(poles))]
    return 1.0, float(abs(np.real(p)))