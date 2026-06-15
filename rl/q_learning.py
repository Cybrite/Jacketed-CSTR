"""Tabular Q-learning implementation with Optimistic Initialization."""

from __future__ import annotations
from dataclasses import dataclass
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
        
        self.state_bins = [
            np.array([-0.6, -0.1, 0.1, 0.6]),
            np.array([-0.6, -0.1, 0.1, 0.6]),
            np.array([-1.2, -0.2, 0.2, 1.2])
        ]
        
        self.q_table = np.zeros(
            (len(self.state_bins[0])+1, len(self.state_bins[1])+1, len(self.state_bins[2])+1, len(kc_actions), len(tau_actions)),
            dtype=float
        )
        self.alpha = 0.15
        self.gamma = 0.98  
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def discretize_observation(self, obs: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(np.clip(np.digitize(obs[0], self.state_bins[0]), 0, len(self.state_bins[0]))),
            int(np.clip(np.digitize(obs[1], self.state_bins[1]), 0, len(self.state_bins[1]))),
            int(np.clip(np.digitize(obs[2], self.state_bins[2]), 0, len(self.state_bins[2])))
        )

    def select_action_indices(self, state_idx: Tuple[int, int, int], explore: bool = True) -> Tuple[int, int]:
        q_slice = self.q_table[state_idx]
        
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.kc_actions)), np.random.randint(0, len(self.tau_actions))
        
        if not explore and np.all(q_slice == 0.0):
            return int(len(self.kc_actions) // 2), int(len(self.tau_actions) // 2)
        
        max_idx = np.unravel_index(np.argmax(q_slice), q_slice.shape)
        return int(max_idx[0]), int(max_idx[1])

    def update(self, state_idx, action_idx, reward, next_state_idx, done):
        current_q = self.q_table[state_idx + action_idx]
        max_next_q = np.max(self.q_table[next_state_idx]) if not done else 0.0
        self.q_table[state_idx + action_idx] += self.alpha * (reward + self.gamma * max_next_q - current_q)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_q_learning_agent(model_env: CSTRPITuningEnv, episodes: int = 1500) -> Tuple[TabularQLearningAgent, QLearningHistory]:
    np.random.seed(42) # Global Seed
    kc_actions = np.linspace(-20.0, -0.5, 10)
    tau_actions = np.linspace(1.0, 20.0, 10)
    agent = TabularQLearningAgent(kc_actions, tau_actions)
    episode_rewards = []

    for episode in range(episodes):
        obs, _ = model_env.reset(seed=42 + episode)
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

    return agent, QLearningHistory(episode_rewards=episode_rewards, best_gains=(0.0, 0.0))