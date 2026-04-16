from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import gymnasium as gym
import numpy as np


class NormalizeObsWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        node_features = obs['node_features']
        denom = np.maximum(np.max(node_features, axis=0, keepdims=True), 1.0)
        out = dict(obs)
        out['node_features'] = (node_features / denom).astype(np.float32)
        return out


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, stack_k: int = 4):
        super().__init__(env)
        self.stack_k = int(stack_k)
        self.frames: Deque[np.ndarray] = deque(maxlen=self.stack_k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack_k):
            self.frames.append(obs['node_features'])
        obs = dict(obs)
        obs['node_features'] = np.concatenate(list(self.frames), axis=1)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs['node_features'])
        obs = dict(obs)
        obs['node_features'] = np.concatenate(list(self.frames), axis=1)
        return obs, reward, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        self.episode_length += 1
        info = dict(info)
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        return obs, reward, terminated, truncated, info
