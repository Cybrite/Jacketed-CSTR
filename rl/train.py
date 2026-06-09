"""DDPG training utilities for PI gain optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl.env import CSTRPITuningEnv, EnvironmentConfig


@dataclass
class TrainingHistory:
    """Container for training and evaluation traces."""

    episode_rewards: List[float]
    action_trace: List[List[float]]
    best_gains: Tuple[float, float]


class RewardActionLogger(BaseCallback):
    """Log episode rewards and actions during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: List[float] = []
        self.action_trace: List[List[float]] = []
        self._episode_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        actions = self.locals.get("actions")
        if rewards is not None:
            self._episode_reward += float(np.asarray(rewards).mean())
        if actions is not None:
            self.action_trace.append(np.asarray(actions).reshape(-1).tolist())
        dones = self.locals.get("dones")
        if dones is not None and bool(np.asarray(dones).any()):
            self.episode_rewards.append(self._episode_reward)
            self._episode_reward = 0.0
        return True


def train_ddpg_agent(
    model_env: CSTRPITuningEnv,
    total_timesteps: int = 25000,  # Increased slightly to give the policy network time to map the curves
    model_path: Path | None = None,
) -> Tuple[DDPG, TrainingHistory]:
    """Train a DDPG agent on the PI tuning environment."""

    config = model_env.config
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, config)])
    vec_env = VecMonitor(vec_env)
    
    # FIX: Completely removed VecNormalize. 
    # Your custom manual scaling in env.py is already pristine. 
    # Removing this prevents early-episode explosion variance from blinding the agent to tracking errors.

    # Scaled noise parameters to match your action space dimensions smoothly
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=np.array([0.3, 0.5]))
    
    model = DDPG(
        "MlpPolicy",
        vec_env,
        action_noise=action_noise,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
    )

    callback = RewardActionLogger()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

    best_gains = evaluate_policy_gains(model, model_env.model, config)
    history = TrainingHistory(
        episode_rewards=callback.episode_rewards,
        action_trace=callback.action_trace,
        best_gains=best_gains,
    )
    return model, history


def evaluate_policy_gains(
    model: DDPG,
    reactor_model,
    config: EnvironmentConfig,
    episodes: int = 3,
) -> Tuple[float, float]:
    """Estimate the optimized PI gains by rolling out the learned policy."""

    env = CSTRPITuningEnv(reactor_model, config)
    gains: List[np.ndarray] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            gains.append(np.asarray(action, dtype=float))
    if not gains:
        return -5.0, 10.0
    tail = np.asarray(gains[-min(len(gains), 50):], dtype=float)
    return float(np.median(tail[:, 0])), float(np.median(tail[:, 1]))