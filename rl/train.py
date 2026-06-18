"""Continuous Deep RL training utilities for DDPG and SAC."""

from __future__ import annotations
import numpy as np
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.env import CSTRPITuningEnv

def train_ddpg_agent(model_env: CSTRPITuningEnv, total_timesteps: int = 25000) -> DDPG:
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, model_env.config)])
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.2 * np.ones(2))
    model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, verbose=0, learning_rate=1e-3, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, learning_starts=1000, seed=42)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model

def train_sac_agent(model_env: CSTRPITuningEnv, total_timesteps: int = 25000) -> SAC:
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, model_env.config)])
    model = SAC("MlpPolicy", vec_env, verbose=0, learning_rate=1e-3, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, ent_coef='auto', learning_starts=1000, seed=42)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model