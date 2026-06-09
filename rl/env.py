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
    """Environment-level configuration values."""

    dt: float = 0.1
    episode_steps: int = 200
    safety_temperature: float = 500.0
    setpoint_temperature: float = 355.0
    Kc_bounds: Tuple[float, float] = (-25.0, -0.1)
    tauI_bounds: Tuple[float, float] = (0.2, 50.0)


class CSTRPITuningEnv(gym.Env):
    """Continuous control environment optimized for smooth macro-horizon tuning."""

    metadata = {"render_modes": []}

    def __init__(self, model: CSTRModel, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        self.model = model
        self.config = config or EnvironmentConfig(
            dt=model.params.dt,
            episode_steps=model.params.episode_steps,
            safety_temperature=model.params.T_safe,
            setpoint_temperature=model.params.T0 + 5.0,
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
        self._step_count = 0
        self._state = np.zeros(3, dtype=float)
        self._previous_error = 0.0
        self._integral_error = 0.0
        self._disturbance_offset_T = 0.0
        self._disturbance_offset_Ca = 0.0
        self._rng = np.random.default_rng()
        
        candidates = self.model.find_steady_state_candidates(F=self.model.params.F, Fc=self.model.params.Fc, Tcin=self.model.params.Tcin0)
        if candidates:
            stable = [c for c in candidates if np.max(np.real(c.eigenvalues)) < 0.0]
            ranked = stable if stable else candidates
            preferred = sorted(ranked, key=lambda item: (abs(item.TR - 325.0), abs(np.max(np.real(item.eigenvalues)))))[0]
            self._ss_state = np.array([preferred.Ca, preferred.TR, preferred.TJ], dtype=float)
        else:
            self._ss_state = np.array([0.0317, 405.47, 335.16], dtype=float)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        base_state = self._ss_state.copy()
        base_state[1] = self.config.setpoint_temperature - 3.0  
        
        perturbation = np.array([
            self._rng.normal(0.0, 0.001),
            self._rng.normal(0.0, 1.0),
            self._rng.normal(0.0, 0.5),
        ])
        
        self._state = base_state + perturbation
        self._state[0] = max(1e-6, self._state[0])
        self._step_count = 0
        self._previous_error = self.config.setpoint_temperature - self._state[1]
        self._integral_error = 0.0
        self._disturbance_offset_T = float(self._rng.uniform(-1.0, 2.0))
        self._disturbance_offset_Ca = float(self._rng.uniform(-0.02, 0.02))
        return self._get_observation(), {}

    def step(self, action):
        gains = np.clip(np.asarray(action, dtype=float), self.action_space.low, self.action_space.high)
        Kc, tauI = float(gains[0]), float(gains[1])
        
        total_reward = 0.0
        instability = False
        hold_steps = 10  
        
        for _ in range(hold_steps):
            error = self.config.setpoint_temperature - self._state[1]
            self._integral_error += error * self.config.dt
            derivative_error = (error - self._previous_error) / self.config.dt

            flow = self.model.params.Fc + Kc * error + (Kc / max(tauI, 1e-9)) * self._integral_error
            flow = float(np.clip(flow, self.model.params.Fc_min, self.model.params.Fc_max))

            next_state = self._rk4_step(self._state, flow)
            
            # Catch intermediate integration divergence before the loop finishes
            instability = not np.all(np.isfinite(next_state)) or next_state[1] > 600.0 or next_state[1] < 150.0
            if instability:
                next_state = np.array([0.0, self.config.safety_temperature, self.model.params.Tcin0], dtype=float)

            next_error = self.config.setpoint_temperature - next_state[1]
            overshoot = max(0.0, next_state[1] - self.config.setpoint_temperature)
            
            error_scale = 10.0
            normalized_error = next_error / error_scale
            normalized_overshoot = overshoot / error_scale
            normalized_delta_error = (next_error - error) / error_scale

            step_reward = -(5.0 * normalized_error**2 + 12.0 * normalized_overshoot**2 + 0.4 * normalized_delta_error**2)
            total_reward += step_reward

            self._previous_error = error
            self._state = next_state
            
            if instability:
                break

        self._step_count += 1

        if self._state[1] > 0.95 * self.config.safety_temperature:
            total_reward -= 20.0

        terminated = bool(
            instability 
            or self._state[1] >= self.config.safety_temperature 
            or self._state[1] <= self.model.params.T_min
        )
        truncated = self._step_count >= (self.config.episode_steps // hold_steps)

        if terminated:
            total_reward -= 150.0
        elif truncated:
            total_reward += 15.0

        observation = self._get_observation()
        info = {
            "Kc": Kc,
            "tauI": tauI,
            "control": flow,
            "temperature": float(self._state[1]),
            "error": float(self.config.setpoint_temperature - self._state[1]),
        }
        return observation, float(total_reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        error = self.config.setpoint_temperature - self._state[1]
        derivative_error = (error - self._previous_error) / self.config.dt
        error_scale = 10.0
        integral_scale = error_scale * max(self.config.dt * self.config.episode_steps, 1.0)
        derivative_scale = 50.0
        observation = np.array(
            [
                error / error_scale,
                self._integral_error / integral_scale,
                derivative_error / derivative_scale,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -10.0, 10.0)

    def _rk4_step(self, state: np.ndarray, flow: float) -> np.ndarray:
        dt = self.config.dt
        feed_temperature = self.model.params.T0 + self._disturbance_offset_T
        feed_concentration = self.model.params.Ca0 * (1.0 + self._disturbance_offset_Ca)

        def rhs(x: np.ndarray) -> np.ndarray:
            return self.model.dynamics(
                0.0, x, Fc=flow, F=self.model.params.F, 
                Ca0=feed_concentration, T0=feed_temperature, Tcin=self.model.params.Tcin0
            )

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[0] = max(self.model.params.Ca_min, next_state[0])
        return next_state