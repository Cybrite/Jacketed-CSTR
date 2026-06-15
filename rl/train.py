"""Continuous Deep RL training utilities for DDPG and SAC."""

from __future__ import annotations
import numpy as np
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.env import CSTRPITuningEnv

def train_ddpg_agent(model_env: CSTRPITuningEnv, total_timesteps: int = 25000) -> DDPG:
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, model_env.config)])
    n_actions = model_env.action_space.shape[-1]
    action_range = (model_env.action_space.high - model_env.action_space.low)
    
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * action_range)
    
    # THE FIX: Added seed=42 to guarantee zero variance
    model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, verbose=0, learning_rate=1e-3, buffer_size=50000, batch_size=128, gamma=0.98, tau=0.01, learning_starts=1000, seed=42)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model

def train_sac_agent(model_env: CSTRPITuningEnv, total_timesteps: int = 25000) -> SAC:
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, model_env.config)])
    
    # THE FIX: Added seed=42 to guarantee zero variance
    model = SAC("MlpPolicy", vec_env, verbose=0, learning_rate=1e-3, buffer_size=50000, batch_size=128, gamma=0.98, tau=0.01, ent_coef='auto', learning_starts=1000, seed=42)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model