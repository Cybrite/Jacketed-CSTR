"""Custom Gymnasium environment for Residual PI gain optimization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
import warnings
from gymnasium import spaces
from models.cstr import CSTRModel

@dataclass
class EnvironmentConfig:
    dt: float = 0.1
    episode_steps: int = 200
    safety_temperature: float = 500.0
    setpoint_temperature: float = 355.0
    # Residual Control Anchors (updated dynamically from main.py)
    baseline_Kc: float = -5.0
    baseline_tauI: float = 2.0
    delta_Kc: float = 2.5
    delta_tauI: float = 1.0

class CSTRPITuningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model: CSTRModel, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        self.model = model
        self.config = config or EnvironmentConfig(
            dt=model.params.dt, episode_steps=model.params.episode_steps,
            safety_temperature=model.params.T_safe, setpoint_temperature=model.params.T0 + 5.0,
        )
        
        # The AI only thinks in pure mathematical normalized nudges [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._state = np.zeros(3, dtype=float)
        self._previous_error = 0.0
        self._integral_action = 0.0
        self._step_count = 0
        self._rng = np.random.default_rng()
        
        candidates = self.model.find_steady_state_candidates()
        stable = [c for c in candidates if np.max(np.real(c.eigenvalues)) < 0.0] if candidates else []
        if stable:
            preferred = sorted(stable, key=lambda item: (abs(item.TR - 325.0), abs(np.max(np.real(item.eigenvalues)))))[0]
            self._ss_state = np.array([preferred.Ca, preferred.TR, preferred.TJ], dtype=float)
        else:
            self._ss_state = np.array([0.0317, 405.47, 335.16], dtype=float)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None: self._rng = np.random.default_rng(seed)
            
        base_state = self._ss_state.copy()
        base_state[1] = self.config.setpoint_temperature - self._rng.uniform(1.0, 6.0)
        self._state = base_state + np.array([self._rng.normal(0.0, 0.001), self._rng.normal(0.0, 1.0), self._rng.normal(0.0, 0.5)])
        self._state[0] = max(1e-6, self._state[0])
        self._step_count = 0
        self._previous_error = self.config.setpoint_temperature - self._state[1]
        self._integral_action = 0.0
        return self._get_observation(), {}

    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= (self.config.episode_steps // 10)

        # THE FIX: Residual Mapping. Action=0 equals perfect IMC. Action=1 is +delta.
        act = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        Kc = self.config.baseline_Kc + act[0] * self.config.delta_Kc
        tauI = max(0.1, self.config.baseline_tauI + act[1] * self.config.delta_tauI)
        
        total_reward = 0.0
        instability = False
        hold_steps = 10  
        
        for _ in range(hold_steps):
            error = self.config.setpoint_temperature - self._state[1]
            self._integral_action += (Kc / max(tauI, 1e-9)) * error * self.config.dt
            flow_raw = self.model.params.Fc + Kc * error + self._integral_action
            flow = float(np.clip(flow_raw, self.model.params.Fc_min, self.model.params.Fc_max))
            
            if flow_raw > self.model.params.Fc_max or flow_raw < self.model.params.Fc_min:
                self._integral_action -= (Kc / max(tauI, 1e-9)) * error * self.config.dt

            next_state = self._rk4_step(self._state, flow)
            
            if next_state[1] > 480.0 or next_state[1] < 200.0 or not np.all(np.isfinite(next_state)):
                instability = True
                break

            # Pure IAE tracking reward. Simple and powerful.
            total_reward -= abs(error) * self.config.dt
            self._previous_error = error
            self._state = next_state

        terminated = bool(instability)
        if terminated: total_reward -= 100.0 

        return self._get_observation(), float(total_reward), terminated, truncated, {}

    def _get_observation(self) -> np.ndarray:
        error = self.config.setpoint_temperature - self._state[1]
        derivative_error = (error - self._previous_error) / self.config.dt
        return np.clip(np.array([error / 10.0, self._integral_action / 50.0, derivative_error / 50.0], dtype=np.float32), -10.0, 10.0)

    def _rk4_step(self, state: np.ndarray, flow: float) -> np.ndarray:
        dt = self.config.dt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            def rhs(x):
                xs = np.copy(x)
                xs[1] = np.clip(xs[1], 150.0, 600.0)
                dx = self.model.dynamics(0.0, xs, Fc=flow, F=self.model.params.F, Ca0=self.model.params.Ca0, T0=self.model.params.T0, Tcin=self.model.params.Tcin0)
                return np.nan_to_num(dx)
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return np.nan_to_num(next_state, nan=self.config.safety_temperature)