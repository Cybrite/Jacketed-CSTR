"""Tabular Q-learning implementation for CSTR PI tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np

from rl.env import CSTRPITuningEnv


@dataclass
class QLearningHistory:
    episode_rewards: List[float]
    best_gains: Tuple[float, float]


class TabularQLearningAgent:
    def __init__(self, kc_actions: np.ndarray, tau_actions: np.ndarray):
        self.kc_actions = kc_actions
        self.tau_actions = tau_actions
        
        # Build 10 discrete quantization bins per observation dimension
        self.num_bins = 10
        self.state_bins = [
            np.linspace(-2.0, 2.0, self.num_bins - 1),  # Normalized Error
            np.linspace(-2.0, 2.0, self.num_bins - 1),  # Normalized Integral Error
            np.linspace(-2.0, 2.0, self.num_bins - 1)   # Normalized Derivative Error
        ]
        
        # Q-Table Shape: (10, 10, 10, len(Kc), len(tauI))
        self.q_table = np.zeros((self.num_bins, self.num_bins, self.num_bins, len(kc_actions), len(tau_actions)))
        self.alpha = 0.12
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_decay = 0.992
        self.epsilon_min = 0.02

    def discretize_observation(self, obs: np.ndarray) -> Tuple[int, int, int]:
        """Map continuous environment observations into discrete grid indices."""
        return (
            int(np.digitize(obs[0], self.state_bins[0])),
            int(np.digitize(obs[1], self.state_bins[1])),
            int(np.digitize(obs[2], self.state_bins[2]))
        )

    def select_action_indices(self, state_idx: Tuple[int, int, int], explore: bool = True) -> Tuple[int, int]:
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.kc_actions)), np.random.randint(0, len(self.tau_actions))
        
        q_slice = self.q_table[state_idx]
        max_idx = np.unravel_index(np.argmax(q_slice), q_slice.shape)
        return int(max_idx[0]), int(max_idx[1])

    def update(self, state_idx: Tuple[int, int, int], action_idx: Tuple[int, int], reward: float, next_state_idx: Tuple[int, int, int], done: bool):
        current_q = self.q_table[state_idx + action_idx]
        max_next_q = np.max(self.q_table[next_state_idx]) if not done else 0.0
        target = reward + self.gamma * max_next_q
        self.q_table[state_idx + action_idx] += self.alpha * (target - current_q)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning_agent(model_env: CSTRPITuningEnv, episodes: int = 1200) -> Tuple[TabularQLearningAgent, QLearningHistory]:
    """Train a discrete tabular Q-learning matrix over the non-linear reactor bounds."""
    # Action grid structured directly within the operational bounds
    kc_actions = np.linspace(-20.0, -1.0, 12)
    tau_actions = np.linspace(0.2, 4.0, 12)
    
    agent = TabularQLearningAgent(kc_actions, tau_actions)
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = model_env.reset(seed=episode)
        state_idx = agent.discretize_observation(obs)
        total_reward = 0.0
        done = False

        while not done:
            kc_idx, tau_idx = agent.select_action_indices(state_idx, explore=True)
            action = np.array([kc_actions[kc_idx], tau_actions[tau_idx]])
            
            next_obs, reward, terminated, truncated, _ = model_env.step(action)
            next_state_idx = agent.discretize_observation(next_obs)
            done = terminated or truncated

            agent.update(state_idx, (kc_idx, tau_idx), reward, next_state_idx, done)
            
            state_idx = next_state_idx
            total_reward += reward

        agent.decay_exploration()
        episode_rewards.append(total_reward)

    # Extract final optimal greedy parameter choices
    final_obs, _ = model_env.reset()
    final_state = agent.discretize_observation(final_obs)
    k_idx, t_idx = agent.select_action_indices(final_state, explore=False)
    best_gains = (float(kc_actions[k_idx]), float(tau_actions[t_idx]))

    return agent, QLearningHistory(episode_rewards=episode_rewards, best_gains=best_gains)