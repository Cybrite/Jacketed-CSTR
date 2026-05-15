"""Custom Gymnasium environment for Q-learning based PI gain optimization on the CSTR."""

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
    Kc_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0)
    tauI_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    tc_bounds: Tuple[float, float] = (250.0, 450.0)


class CSTRPITuningEnv(gym.Env):
    """Discrete control environment where actions select PI gains.

    The state vector is [error, integral_error, derivative_error]. The action
    is an integer index into a predefined gain table. This is the practical
    Q-learning formulation because standard tabular Q-learning and DQN require
    a discrete action set.
    """

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
        self.gain_table = [(float(kc), float(tau_i)) for kc in self.config.Kc_grid for tau_i in self.config.tauI_grid]
        self.action_space = spaces.Discrete(len(self.gain_table))
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._state = np.zeros(2, dtype=float)
        self._previous_error = 0.0
        self._integral_error = 0.0
        self._disturbance_offset_T = 0.0
        self._disturbance_offset_Ca = 0.0
        self._last_control = self.model.params.Tc0
        self._rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        base_state = np.array([0.5, self.config.setpoint_temperature - 8.0], dtype=float)
        perturbation = np.array([
            self._rng.normal(0.0, 0.02),
            self._rng.normal(0.0, 2.0),
        ])
        self._state = base_state + perturbation
        self._state[0] = max(1e-6, self._state[0])
        self._step_count = 0
        self._previous_error = self.config.setpoint_temperature - self._state[1]
        self._integral_error = 0.0
        self._disturbance_offset_T = float(self._rng.uniform(0.0, 8.0))
        self._disturbance_offset_Ca = float(self._rng.uniform(0.0, 0.2))
        self._last_control = self.model.params.Tc0
        return self._get_observation(), {}

    def step(self, action):
        action_index = int(np.asarray(action).item())
        action_index = int(np.clip(action_index, 0, self.action_space.n - 1))
        Kc, tauI = self.gain_table[action_index]
        error = self.config.setpoint_temperature - self._state[1]
        self._integral_error += error * self.config.dt
        derivative_error = (error - self._previous_error) / self.config.dt

        proportional = Kc * error
        integral = Kc / max(tauI, 1e-9) * self._integral_error
        control = self.model.params.Tc0 + proportional + integral
        control = float(np.clip(control, self.config.tc_bounds[0], self.config.tc_bounds[1]))

        next_state = self._rk4_step(self._state, control)
        self._step_count += 1
        next_error = self.config.setpoint_temperature - next_state[1]
        overshoot = max(0.0, next_state[1] - self.config.setpoint_temperature)
        control_effort = (control - self.model.params.Tc0) ** 2
        oscillation = (next_error - error) ** 2

        reward = -(
            1.0 * next_error**2
            + 0.02 * control_effort
            + 0.4 * overshoot**2
            + 0.1 * oscillation
        )

        terminated = bool(
            next_state[1] >= self.config.safety_temperature
            or next_state[1] <= self.model.params.T_min
            or not np.all(np.isfinite(next_state))
        )
        truncated = self._step_count >= self.config.episode_steps

        self._previous_error = error
        self._state = next_state
        self._last_control = control

        observation = self._get_observation()
        info = {
            "action_index": action_index,
            "Kc": Kc,
            "tauI": tauI,
            "control": control,
            "temperature": float(next_state[1]),
            "error": float(next_error),
        }
        return observation, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        error = self.config.setpoint_temperature - self._state[1]
        derivative_error = (error - self._previous_error) / self.config.dt
        observation = np.array([error, self._integral_error, derivative_error], dtype=np.float32)
        return observation

    def _rk4_step(self, state: np.ndarray, coolant_temperature: float) -> np.ndarray:
        dt = self.config.dt
        t0 = 0.0
        feed_temperature = self.model.params.T0 + self._disturbance_offset_T
        feed_concentration = self.model.params.Ca0 * (1.0 + self._disturbance_offset_Ca)

        def rhs(x: np.ndarray) -> np.ndarray:
            return self.model.dynamics(t0, x, coolant_temperature, Ca0=feed_concentration, T0=feed_temperature)

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[0] = max(self.model.params.Ca_min, next_state[0])
        return next_state