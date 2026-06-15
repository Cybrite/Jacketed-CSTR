"""Custom Gymnasium environment for PI gain optimization on the CSTR."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from models.cstr import CSTRModel

@dataclass
class EnvironmentConfig:
    dt: float = 0.1
    episode_steps: int = 200
    safety_temperature: float = 500.0
    setpoint_temperature: float = 355.0
    Kc_bounds: Tuple[float, float] = (-20.0, -0.5)
    tauI_bounds: Tuple[float, float] = (1.0, 20.0)

class CSTRPITuningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model: CSTRModel, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        self.model = model
        self.config = config or EnvironmentConfig(
            dt=model.params.dt, episode_steps=model.params.episode_steps,
            safety_temperature=model.params.T_safe, setpoint_temperature=model.params.T0 + 5.0,
        )
        
        self.action_space = spaces.Box(
            low=np.array([self.config.Kc_bounds[0], self.config.tauI_bounds[0]], dtype=np.float32),
            high=np.array([self.config.Kc_bounds[1], self.config.tauI_bounds[1]], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._state = np.zeros(3, dtype=float)
        self._previous_error = 0.0
        self._integral_action = 0.0
        self._step_count = 0
        self._is_in_purgatory = False
        self._rng = np.random.default_rng()
        
        candidates = self.model.find_steady_state_candidates(F=self.model.params.F, Fc=self.model.params.Fc, Tcin=self.model.params.Tcin0)
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
        
        perturbation = np.array([self._rng.normal(0.0, 0.001), self._rng.normal(0.0, 1.0), self._rng.normal(0.0, 0.5)])
        self._state = base_state + perturbation
        self._state[0] = max(1e-6, self._state[0])
        self._step_count = 0
        self._previous_error = self.config.setpoint_temperature - self._state[1]
        self._integral_action = 0.0
        self._is_in_purgatory = False
        return self._get_observation(), {}

    def step(self, action):
        self._step_count += 1
        truncated = self._step_count >= (self.config.episode_steps // 10)
        
        # Purgatory Loop: If the agent breaks the reactor, it is trapped receiving constant negative rewards.
        # This prevents "Lazy reward hacking" by forcing smooth negative gradients.
        if self._is_in_purgatory:
            return self._get_observation(), -5.0, False, truncated, {}

        gains = np.clip(np.asarray(action, dtype=float), self.action_space.low, self.action_space.high)
        Kc, tauI = float(gains[0]), float(gains[1])
        
        total_reward = 0.0
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
                self._is_in_purgatory = True
                self._state[1] = self.config.safety_temperature
                break

            total_reward -= abs(error) / 5.0
            self._previous_error = error
            self._state = next_state

        if self._is_in_purgatory:
            total_reward -= 100.0  

        return self._get_observation(), float(total_reward), False, truncated, {}

    def _get_observation(self) -> np.ndarray:
        error = self.config.setpoint_temperature - self._state[1]
        derivative_error = (error - self._previous_error) / self.config.dt
        obs = np.array([error / 10.0, self._integral_action / 50.0, derivative_error / 50.0], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _rk4_step(self, state: np.ndarray, flow: float) -> np.ndarray:
        dt = self.config.dt
        def rhs(x):
            xs = np.clip(np.nan_to_num(x), -1e5, 1e5)
            dx = self.model.dynamics(0.0, xs, Fc=flow, F=self.model.params.F, Ca0=self.model.params.Ca0, T0=self.model.params.T0, Tcin=self.model.params.Tcin0)
            return np.clip(np.nan_to_num(dx), -1e5, 1e5)

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = np.nan_to_num(next_state, nan=self.config.safety_temperature)
        next_state[0] = np.clip(next_state[0], 0.0, 10.0)
        next_state[1] = np.clip(next_state[1], 150.0, 600.0)
        next_state[2] = np.clip(next_state[2], 150.0, 600.0)
        return next_state