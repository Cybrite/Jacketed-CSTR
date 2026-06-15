"""Continuous Deep RL training utilities with Expert Offline Pre-training."""

from __future__ import annotations
import numpy as np
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from controllers.pi import PIGains
from rl.env import CSTRPITuningEnv

def prefill_expert_buffer(model, expert_gains: PIGains, n_steps=5000):
    """Fills the RL Replay Buffer with optimal IMC tracking data before live training."""
    vec_env = model.env
    obs = vec_env.reset()
    
    # The perfect classical parameters
    expert_action = np.array([[expert_gains.Kc, expert_gains.tauI]])
    
    for _ in range(n_steps):
        # We add a tiny bit of "noise" (Gaussian cloud) around the perfect IMC parameters.
        # This teaches the neural network what the gradient (slope) looks like near the optimum!
        noise = np.random.normal(loc=0.0, scale=[0.5, 0.5], size=expert_action.shape)
        noisy_action = np.clip(expert_action + noise, vec_env.action_space.low, vec_env.action_space.high)
        
        next_obs, rewards, dones, infos = vec_env.step(noisy_action)
        
        # SB3 automatically resets environments on 'done', hiding the true final observation.
        # We must extract the true 'terminal' observation to prevent neural network corruption.
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d and "terminal_observation" in infos[idx]:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
                
        # Inject the expert memory directly into the agent's brain
        model.replay_buffer.add(obs, real_next_obs, noisy_action, rewards, dones, infos)
        obs = next_obs

def train_ddpg_agent(model_env: CSTRPITuningEnv, expert_gains: PIGains, total_timesteps: int = 25000) -> DDPG:
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, model_env.config)])
    n_actions = model_env.action_space.shape[-1]
    action_range = (model_env.action_space.high - model_env.action_space.low)
    
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * action_range)
    
    model = DDPG(
        "MlpPolicy", vec_env, action_noise=action_noise, verbose=0, 
        learning_rate=3e-4, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, 
        learning_starts=0, seed=42  # learning_starts=0 because the buffer is already full!
    )
    
    # Pre-train the neural network on the IMC database
    prefill_expert_buffer(model, expert_gains, n_steps=5000)
    
    # Fine-tune live
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model

def train_sac_agent(model_env: CSTRPITuningEnv, expert_gains: PIGains, total_timesteps: int = 25000) -> SAC:
    vec_env = DummyVecEnv([lambda: CSTRPITuningEnv(model_env.model, model_env.config)])
    
    model = SAC(
        "MlpPolicy", vec_env, verbose=0, 
        learning_rate=3e-4, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, 
        ent_coef='auto', learning_starts=0, seed=42
    )
    
    prefill_expert_buffer(model, expert_gains, n_steps=5000)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model