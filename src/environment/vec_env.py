"""
Vectorized Trading Environment
================================
Wraps N independent ConfigIntegratedTradingEnv instances so that the PPO agent
can make a single batched GPU forward-pass per step instead of N sequential ones.

Why this matters
----------------
Without VecEnv the training loop looks like:

    for step in range(2048):
        action = trader.select_action(obs)   # 1 GPU call: (1, state_dim)
        obs, reward = env.step(action)       # 1 CPU call

That is 2048 tiny GPU kernels with CPU bubbles between each one — GPU utilisation
ends up at ~10 % on a Colab T4.

With VecEnv and N=8:

    for step in range(256):                              # 8× fewer iterations
        actions = trader.select_action_batch(obs_batch) # 1 GPU call: (8, state_dim)
        obs_batch, rewards, dones = vec_env.step(actions)

GPU utilisation goes to ~60 % because each kernel does 8× more work and the
CPU↔GPU transfer ratio drops by 8×.

API
---
    vec_env = VecTradingEnv(price_data, features, config, n_envs=8)
    obs = vec_env.reset()                          # (N, state_dim)
    obs, rewards, dones, infos = vec_env.step(actions)  # actions: (N,) int array

Thread safety
-------------
Each sub-environment runs its own copy of the price/feature data and internal
state — there is no shared mutable state.  We use a ThreadPoolExecutor for the
env.step() calls so that they run concurrently on the CPU while the GPU is doing
the forward pass of the previous step.

Note: Python's GIL is not a bottleneck here because gymnasium environments are
mostly numpy arithmetic (no pure-Python loops in the hot path), so threads give
real parallelism from the GIL-releasing numpy operations.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any

from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
from src.environment.config_system import EnvironmentConfig


class VecTradingEnv:
    """
    Synchronous vectorised wrapper for ConfigIntegratedTradingEnv.

    Parameters
    ----------
    price_data : pd.DataFrame
        OHLCV price data (shared — each env gets an independent copy).
    features : pd.DataFrame
        Pre-computed feature data (shared — each env gets an independent copy).
    config : EnvironmentConfig
        Environment configuration.  Each env instance receives a deep copy so
        that internal state (regime factors, etc.) does not leak between envs.
    n_envs : int
        Number of parallel environments.  Recommended: 8 on Colab T4.
    seed_offset : int
        Added to the per-env seed so that stochastic resets are independent.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        config: EnvironmentConfig,
        n_envs: int = 8,
        seed_offset: int = 0,
    ):
        self.n_envs = n_envs
        self._seed_offset = seed_offset

        # Build N independent environment instances.
        # copy.deepcopy on EnvironmentConfig is safe — it is a dataclass with
        # only primitive / nested-dataclass fields (no file handles, sockets, …).
        self._envs: List[ConfigIntegratedTradingEnv] = [
            ConfigIntegratedTradingEnv(
                price_data.copy(),
                features.copy(),
                copy.deepcopy(config),
            )
            for _ in range(n_envs)
        ]

        # Derive state_dim from the first environment
        _sample_obs, _ = self._envs[0].reset()
        self.state_dim: int = int(_sample_obs.shape[0])
        self.n_actions: int = int(self._envs[0].action_space.n)

        # Internal tracking — used for episode-boundary resets
        self._obs: np.ndarray = np.zeros((n_envs, self.state_dim), dtype=np.float32)
        self._infos: List[Dict] = [{} for _ in range(n_envs)]

        # Thread pool — reuse across calls to avoid thread-creation overhead
        self._executor = ThreadPoolExecutor(max_workers=n_envs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset all environments.

        Returns
        -------
        obs : np.ndarray  shape (N, state_dim)
        """

        def _reset_one(i: int) -> Tuple[int, np.ndarray, Dict]:
            obs, info = self._envs[i].reset()
            return i, obs.astype(np.float32), info

        futures = [self._executor.submit(_reset_one, i) for i in range(self.n_envs)]
        for f in as_completed(futures):
            i, obs, info = f.result()
            self._obs[i] = obs
            self._infos[i] = info

        return self._obs.copy()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with their respective actions.

        Parameters
        ----------
        actions : np.ndarray  shape (N,)  dtype int
            One action per environment.

        Returns
        -------
        obs     : np.ndarray  shape (N, state_dim)  — next observations
        rewards : np.ndarray  shape (N,)            — step rewards
        dones   : np.ndarray  shape (N,)  dtype bool — episode boundaries
        infos   : List[Dict]  length N
        """
        next_obs = np.zeros_like(self._obs)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos: List[Dict] = [{}] * self.n_envs

        def _step_one(i: int, action: int) -> Tuple[int, np.ndarray, float, bool, Dict]:
            obs, reward, terminated, truncated, info = self._envs[i].step(action)
            done = terminated or truncated
            if done:
                # Auto-reset: return new episode's first observation
                obs, info = self._envs[i].reset()
            return i, obs.astype(np.float32), float(reward), done, info

        futures = [
            self._executor.submit(_step_one, i, int(actions[i]))
            for i in range(self.n_envs)
        ]
        for f in as_completed(futures):
            i, obs, reward, done, info = f.result()
            next_obs[i] = obs
            rewards[i] = reward
            dones[i] = done
            infos[i] = info

        self._obs = next_obs
        self._infos = infos
        return next_obs.copy(), rewards, dones, infos

    def close(self) -> None:
        """Shut down the thread pool and close all environments."""
        self._executor.shutdown(wait=False)
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        return self._envs[0].observation_space

    @property
    def action_space(self):
        return self._envs[0].action_space

    def __len__(self) -> int:
        return self.n_envs

    def __repr__(self) -> str:
        return (
            f"VecTradingEnv(n_envs={self.n_envs}, "
            f"state_dim={self.state_dim}, "
            f"n_actions={self.n_actions})"
        )
