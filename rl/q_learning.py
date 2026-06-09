"""Tabular Q-learning implementation with Compact High-Exploration State Grid."""

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
        
        # Compact high-density thresholds clustering near zero error 
        # This keeps the total state-space fully explorable within 2,000 episodes
        self.state_bins = [
            np.array([-0.5, -0.05, 0.05, 0.5]),  # Normalized Error
            np.array([-0.5, -0.05, 0.05, 0.5]),  # Normalized Integral Error
            np.array([-1.0, -0.1, 0.1, 1.0])    # Normalized Derivative Error
        ]
        
        self.num_bins_err = len(self.state_bins[0]) + 1
        self.num_bins_int = len(self.state_bins[1]) + 1
        self.num_bins_der = len(self.state_bins[2]) + 1
        
        # FIX: Pessimistic Initialization. Defaulting cells to -10000.0 forces the greedy policy 
        # to select known stable trajectories instead of picking unvisited zero-value boundaries.
        self.q_table = np.full(
            (self.num_bins_err, self.num_bins_int, self.num_bins_der, len(kc_actions), len(tau_actions)),
            -10000.0,
            dtype=float
        )
        self.alpha = 0.15
        self.gamma = 0.96  
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def discretize_observation(self, obs: np.ndarray) -> Tuple[int, int, int]:
        """Map continuous observations into precise logarithmic coordinate bins."""
        return (
            int(np.clip(np.digitize(obs[0], self.state_bins[0]), 0, self.num_bins_err - 1)),
            int(np.clip(np.digitize(obs[1], self.state_bins[1]), 0, self.num_bins_int - 1)),
            int(np.clip(np.digitize(obs[2], self.state_bins[2]), 0, self.num_bins_der - 1))
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


def train_q_learning_agent(model_env: CSTRPITuningEnv, episodes: int = 2000) -> Tuple[TabularQLearningAgent, QLearningHistory]:
    """Train a high-resolution tabular matrix over the non-linear reactor bounds."""
    # Action spaces bounded securely around your classical baselines to stop arithmetic overflows
    kc_actions = np.array([-16.0, -13.0, -10.0, -8.0, -5.0, -2.0])
    tau_actions = np.array([0.2, 0.35, 0.5, 1.0, 5.0, 20.0])
    
    agent = TabularQLearningAgent(kc_actions, tau_actions)
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = model_env.reset(seed=episode)
        state_idx = agent.discretize_observation(obs)
        total_reward = 0.0
        done = False
        prev_action_idx = None

        while not done:
            kc_idx, tau_idx = agent.select_action_indices(state_idx, explore=True)
            action = np.array([kc_actions[kc_idx], tau_actions[tau_idx]])
            
            next_obs, reward, terminated, truncated, _ = model_env.step(action)
            
            # Penalize radical controller parameter changes between steps to stabilize actuation
            if prev_action_idx is not None:
                gain_change_penalty = 1.5 * (abs(kc_idx - prev_action_idx[0]) + abs(tau_idx - prev_action_idx[1]))
                reward -= gain_change_penalty
            
            next_state_idx = agent.discretize_observation(next_obs)
            done = terminated or truncated

            agent.update(state_idx, (kc_idx, tau_idx), reward, next_state_idx, done)
            
            state_idx = next_state_idx
            prev_action_idx = (kc_idx, tau_idx)
            total_reward += reward

        agent.decay_exploration()
        episode_rewards.append(total_reward)

    final_obs, _ = model_env.reset()
    final_state = agent.discretize_observation(final_obs)
    k_idx, t_idx = agent.select_action_indices(final_state, explore=False)
    best_gains = (float(kc_actions[k_idx]), float(tau_actions[t_idx]))

    return agent, QLearningHistory(episode_rewards=episode_rewards, best_gains=best_gains)