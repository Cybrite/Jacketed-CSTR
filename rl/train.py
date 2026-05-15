"""DQN training utilities for PI gain optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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
        infos = self.locals.get("infos")
        if rewards is not None:
            self._episode_reward += float(np.asarray(rewards).mean())
        if infos is not None and len(infos) > 0:
            latest_info = infos[0]
            if isinstance(latest_info, dict) and "Kc" in latest_info and "tauI" in latest_info:
                self.action_trace.append([float(latest_info["Kc"]), float(latest_info["tauI"])])
        dones = self.locals.get("dones")
        if dones is not None and bool(np.asarray(dones).any()):
            self.episode_rewards.append(self._episode_reward)
            self._episode_reward = 0.0
        return True



def train_dqn_agent(
    model_env: CSTRPITuningEnv,
    total_timesteps: int = 10000,
    model_path: Path | None = None,
) -> Tuple[DQN, TrainingHistory]:
    """Train a DQN agent on the discretized PI tuning environment."""

    config = model_env.config
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, config)])
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
    )

    callback = RewardActionLogger()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

    best_gains, rollout_trace = evaluate_policy_gains(model, model_env.model, config)
    history = TrainingHistory(
        episode_rewards=callback.episode_rewards,
        action_trace=rollout_trace if rollout_trace else callback.action_trace,
        best_gains=best_gains,
    )
    return model, history



def evaluate_policy_gains(
    model: DQN,
    reactor_model,
    config: EnvironmentConfig,
    episodes: int = 3,
) -> Tuple[Tuple[float, float], List[List[float]]]:
    """Estimate the optimized PI gains by rolling out the learned policy."""

    env = CSTRPITuningEnv(reactor_model, config)
    gains: List[np.ndarray] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            action_index = int(np.asarray(action).item())
            gains.append(np.asarray(env.gain_table[action_index], dtype=float))
    if not gains:
        return (1.0, 1.0), []
    tail = np.asarray(gains[-min(len(gains), 50):], dtype=float)
    return (float(np.median(tail[:, 0])), float(np.median(tail[:, 1]))), [g.tolist() for g in gains]