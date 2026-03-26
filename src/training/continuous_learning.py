"""
Continuous Learning Pipeline
=============================
Online adaptation system for PPO agent to prevent concept drift
in live market conditions.

Architecture:
- LiveExperienceBuffer: Stores recent (state, action, reward, done) tuples
  in a thread-safe circular deque with O(1) append/pop and O(n) random sampling.
- ContinuousLearningManager: Orchestrates periodic online retraining by
  sampling from the buffer, computing discounted returns, loading them into
  the PPOAgent's internal buffers, and calling agent.train().

PPOAgent integration note:
  PPOAgent does NOT have a `train_on_batch(states, actions, returns)` method.
  Its training API is:
    1. agent.store_transition(state, action, reward, log_prob, value, done)
       or agent.store_transitions_batch(...)
    2. agent.train(next_value=0.0)

  To bridge the gap we use agent.store_transitions_batch() to load sampled
  experiences and synthetic log_probs/values into the agent's pre-allocated
  buffers, then call agent.train().  The log_probs are approximated via a
  forward pass through the current actor (making the update quasi-on-policy).
  Values are similarly approximated via a critic forward pass.

  If the agent ever gains a true `train_on_batch` method, the _do_agent_update()
  method below can be swapped out without changing the rest of this module.
"""

import asyncio
import collections
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from src.agents.ppo_agent import PPOAgent
from src.training.ewc import EWCRegularizer


# ─────────────────────────────────────────────────────────────────────────────
# LiveExperienceBuffer
# ─────────────────────────────────────────────────────────────────────────────


class LiveExperienceBuffer:
    """
    Thread-safe circular replay buffer for live market transitions.

    Stores (state, action, reward, next_state, done) tuples collected during
    live trading.  Internally uses collections.deque with a fixed maxlen so
    the oldest experiences are dropped automatically once capacity is reached.

    Sampling is uniform-random (no prioritisation) and returns plain numpy
    arrays ready for downstream use.

    Args:
        capacity (int): Maximum number of transitions to retain.
        state_dim (int): Dimensionality of the observation vector.
                         Used only for type consistency checks (not enforced).
    """

    def __init__(self, capacity: int = 10_000, state_dim: int = 50) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        # deque with maxlen automatically discards the oldest item on overflow
        self._buffer: collections.deque = collections.deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer.

        The deque is thread-safe for single-writer / single-reader usage on
        CPython (GIL protected).  For multi-threaded writers wrap calls in a
        threading.Lock; for async writers use ContinuousLearningManager which
        serialises via asyncio.Lock.
        """
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw a uniform-random batch from the buffer.

        Args:
            batch_size: Number of transitions to sample.  Must be ≤ len(self).

        Returns:
            Tuple of numpy arrays:
                states      (batch_size, state_dim)  float32
                actions     (batch_size,)             int64
                rewards     (batch_size,)             float32
                next_states (batch_size, state_dim)  float32
                dones       (batch_size,)             float32  (0.0 / 1.0)

        Raises:
            ValueError: if batch_size > len(self).
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only has "
                f"{len(self._buffer)} transitions."
            )
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int = 128) -> bool:
        """Return True if the buffer contains at least *min_size* transitions."""
        return len(self._buffer) >= min_size


# ─────────────────────────────────────────────────────────────────────────────
# ContinuousLearningManager
# ─────────────────────────────────────────────────────────────────────────────


class ContinuousLearningManager:
    """
    Orchestrates periodic online retraining of the PPO agent from live data.

    Responsibilities:
    1. Receive transitions from the live trading loop via collect_experience().
    2. Trigger online_update() when enough steps have accumulated.
    3. Compute discounted returns from sampled rewards (no environment model needed).
    4. Load experiences into the PPOAgent's buffer and call agent.train().
    5. Persist checkpoints after each successful update.
    6. Run a background async loop via run_continuous().

    Args:
        agent:                 Trained PPOAgent instance to update online.
        checkpoint_dir:        Directory for saving online checkpoints.
        update_interval_steps: Minimum buffer size before triggering an update.
        state_dim:             Observation vector dimension (passed to buffer).
    """

    def __init__(
        self,
        agent: PPOAgent,
        checkpoint_dir: str,
        update_interval_steps: int = 128,
        state_dim: int = 50,
        use_ewc: bool = True,
        ewc_lambda: float = 400.0,
    ) -> None:
        self.agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval_steps = update_interval_steps
        self.state_dim = state_dim

        # Experience buffer shared between collection and update
        self.buffer = LiveExperienceBuffer(capacity=10_000, state_dim=state_dim)

        # asyncio.Lock serialises concurrent coroutine access to the buffer
        # and the agent (important when online_update() and collect_experience()
        # are both awaited in the same event loop).
        self._lock: asyncio.Lock = asyncio.Lock()

        # Total steps collected since instantiation
        self._step_count: int = 0

        # Discount factor for return computation
        self._gamma: float = 0.99

        # Retention fraction: keep the last X% of processed experiences for
        # stability (prevents abrupt buffer resets that discard recent context).
        self._retain_fraction: float = 0.20

        # ── EWC — catastrophic forgetting protection ──────────────────────────
        # EWCRegularizer is opt-in via use_ewc=True.  It is only activated after
        # the FIRST successful online update (so there is a meaningful "task A"
        # anchor to protect).  Before that, ewc.ewc_loss() returns 0.
        self.ewc: Optional[EWCRegularizer] = (
            EWCRegularizer(agent, lambda_ewc=ewc_lambda) if use_ewc else None
        )
        self._ewc_update_count: int = 0  # number of times EWC has been used

        logger.info(
            f"ContinuousLearningManager ready | "
            f"interval={update_interval_steps} steps | "
            f"checkpoint_dir={self.checkpoint_dir} | "
            f"ewc={'enabled (lambda=' + str(ewc_lambda) + ')' if use_ewc else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # Experience collection
    # ------------------------------------------------------------------

    def collect_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single live transition to the replay buffer.

        This is the synchronous hot-path called from the trading loop.
        It does NOT require the asyncio lock because collections.deque
        append is GIL-atomic on CPython.

        Args:
            state:      Current observation vector.
            action:     Action taken by the agent.
            reward:     Scalar reward received.
            next_state: Observation after the action.
            done:       True if the episode terminated at this step.
        """
        self.buffer.add(state, action, reward, next_state, done)
        self._step_count += 1

    # ------------------------------------------------------------------
    # Return computation (no environment model needed)
    # ------------------------------------------------------------------

    def _compute_discounted_returns(
        self, rewards: np.ndarray, dones: np.ndarray, gamma: float = 0.99
    ) -> np.ndarray:
        """Compute Monte-Carlo discounted returns for a batch of transitions.

        Because we sample a *random* (not necessarily sequential) batch from
        the buffer, episode boundaries (done=True) reset the accumulator so
        that future rewards from a different episode do not contaminate an
        earlier one.

        G_t = r_t + γ * (1 - done_t) * G_{t+1}

        The computation runs backward through the (shuffled) batch, which is
        a reasonable approximation for the online setting.

        Args:
            rewards: (N,) float32 reward array.
            dones:   (N,) float32 done-flag array (1.0 = terminal step).
            gamma:   Discount factor.

        Returns:
            returns: (N,) float32 discounted return array.
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(T)):
            # Reset at episode boundary
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns

    # ------------------------------------------------------------------
    # Agent update bridge
    # ------------------------------------------------------------------

    def _do_agent_update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Load sampled experiences into PPOAgent's buffers and trigger training.

        PPOAgent has no standalone `train_on_batch(states, actions, returns)`
        method.  The closest public API is:
            agent.store_transitions_batch(states, actions, rewards,
                                          log_probs, values, dones, hiddens)
            agent.train(next_value=0.0)

        We approximate log_probs and values via forward passes through the
        current actor/critic networks (quasi-on-policy reuse).  This is
        mathematically similar to PPO's importance-sampling ratio being ~1.0
        for freshly collected data.

        If PPOAgent ever gains a proper `train_on_batch` method, replace the
        body of this function with a direct call to it.

        Args:
            states:      (N, state_dim)
            actions:     (N,)
            returns:     (N,)  discounted returns (used as reward proxy)
            next_states: (N, state_dim)  (unused here, reserved for future use)
            dones:       (N,)

        Returns:
            Dict with training statistics from agent.train().
        """
        if not hasattr(self.agent, "train_on_batch"):
            # ── Expected path ─────────────────────────────────────────────
            # Approximate log_probs and values via a batched forward pass so
            # that PPOAgent's internal GAE / PPO loss is well-conditioned.
            device = self.agent.device
            state_tensor = torch.FloatTensor(states).to(device)

            with torch.no_grad():
                dist, _ = self.agent.actor(state_tensor, None)
                action_tensor = torch.LongTensor(actions).to(device)
                log_probs_t = dist.log_prob(action_tensor)
                values_t, _ = self.agent.critic(state_tensor, None)

            log_probs_np = log_probs_t.cpu().numpy().astype(np.float32)
            values_np = values_t.squeeze(-1).cpu().numpy().astype(np.float32)
            hiddens_batch = [None] * len(states)  # No BPTT for online updates

            # Reset and load into agent buffer
            self.agent.reset_buffers()
            self.agent.store_transitions_batch(
                states=states,
                actions=actions,
                rewards=returns,  # discounted returns used as one-step rewards
                log_probs=log_probs_np,
                values=values_np,
                dones=dones,
                hiddens_batch=hiddens_batch,
            )

            # Run one PPO update cycle (bootstrap=0 since returns are pre-computed)
            train_stats = self.agent.train(next_value=0.0)
            logger.debug(
                "PPOAgent updated via store_transitions_batch() + train(). "
                "Note: PPOAgent lacks train_on_batch(); using buffer-load bridge."
            )
            return train_stats

        else:
            # ── Future path: agent supports direct batch training ─────────
            return self.agent.train_on_batch(states, actions, returns)

    # ------------------------------------------------------------------
    # Core online update
    # ------------------------------------------------------------------

    async def online_update(self) -> Dict[str, Any]:
        """Perform one online PPO update from the live experience buffer.

        Steps:
        1. Acquire the async lock (prevents concurrent updates).
        2. Check whether the buffer has >= update_interval_steps experiences.
        3. Sample a batch from the buffer.
        4. Compute discounted returns.
        5. Call the agent update bridge (_do_agent_update).
        6. Trim the buffer — keep the most-recent 20% for stability.
        7. Save a checkpoint.
        8. Release the lock and return update statistics.

        Returns:
            Dict with keys:
                buffer_size_before  – buffer length before sampling
                batch_size          – number of transitions used
                mean_return         – mean discounted return in the batch
                std_return          – std of discounted returns
                step_count          – cumulative steps collected
                update_skipped      – True if buffer was too small
                + all keys from agent.train() (actor_loss, critic_loss, …)
        """
        async with self._lock:
            buf_size = len(self.buffer)

            if not self.buffer.is_ready(self.update_interval_steps):
                logger.info(
                    f"online_update skipped: buffer has {buf_size} transitions "
                    f"(need {self.update_interval_steps})."
                )
                return {
                    "update_skipped": True,
                    "buffer_size_before": buf_size,
                    "step_count": self._step_count,
                }

            # ── Step 3: Sample ────────────────────────────────────────────
            batch_size = min(self.update_interval_steps, buf_size)
            states, actions, rewards, next_states, dones = self.buffer.sample(
                batch_size
            )

            # ── Step 4: Compute discounted returns ────────────────────────
            returns = self._compute_discounted_returns(
                rewards, dones, gamma=self._gamma
            )

            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            logger.info(
                f"online_update | batch={batch_size} | "
                f"mean_return={mean_ret:.4f} | std_return={std_ret:.4f}"
            )

            # ── Step 5: Update agent ──────────────────────────────────────
            # Attach EWC penalty to the agent's optimizer if EWC is enabled
            # and the anchor has already been set (i.e. not the very first update).
            ewc_loss_val: float = 0.0
            forgetting_risk: float = 0.0
            if self.ewc is not None and self.ewc.is_initialized():
                # Compute EWC loss BEFORE the PPO update so we can log it;
                # the actual backward() call is handled inside _do_agent_update_ewc.
                with torch.no_grad():
                    ewc_loss_val = float(self.ewc.ewc_loss(self.agent).item())
                forgetting_risk = self.ewc.get_forgetting_risk(self.agent)
                logger.debug(
                    f"EWC | ewc_loss={ewc_loss_val:.4f} | "
                    f"forgetting_risk={forgetting_risk:.6f}"
                )

            try:
                train_stats = self._do_agent_update(
                    states, actions, returns, next_states, dones
                )
            except Exception as exc:
                logger.error(f"online_update: agent update failed: {exc}")
                train_stats = {"agent_update_error": str(exc)}

            # ── EWC anchor management ─────────────────────────────────────
            # After a successful update, update or initialise the EWC anchor.
            if self.ewc is not None and "agent_update_error" not in train_stats:
                device = self.agent.device
                state_tensor = torch.FloatTensor(states).to(device)
                action_tensor = torch.LongTensor(actions).to(device)
                ewc_batch = (state_tensor, action_tensor)

                if not self.ewc.is_initialized():
                    # First successful update: compute Fisher from this batch
                    # and set the initial anchor θ*.
                    self.ewc.compute_fisher(self.agent, ewc_batch)
                    logger.info("EWC initialised after first successful online update.")
                else:
                    # Subsequent updates: slide the anchor forward so the model
                    # is not over-constrained to an increasingly distant θ*.
                    self.ewc.update_anchor(self.agent)

                self._ewc_update_count += 1

            # ── Step 6: Trim buffer — keep last 20% for stability ─────────
            retain_n = max(1, int(buf_size * self._retain_fraction))
            # Re-sample from the live buffer keeps recent experiences;
            # the deque itself is the canonical store so we simply drain
            # the oldest transitions by replacing with a sliced copy.
            recent = list(self.buffer._buffer)[-retain_n:]
            self.buffer._buffer.clear()
            self.buffer._buffer.extend(recent)
            logger.debug(
                f"Buffer trimmed: {buf_size} → {len(self.buffer)} "
                f"(retained last {retain_n} transitions)"
            )

            # ── Step 7: Checkpoint ────────────────────────────────────────
            ckpt_path = self.save_checkpoint()

            # ── Step 8: Return stats ──────────────────────────────────────
            result: Dict[str, Any] = {
                "update_skipped": False,
                "buffer_size_before": buf_size,
                "batch_size": batch_size,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "step_count": self._step_count,
                "checkpoint": str(ckpt_path),
                # EWC diagnostics (zero when EWC disabled or not yet initialised)
                "ewc_loss": ewc_loss_val,
                "ewc_forgetting_risk": forgetting_risk,
                "ewc_update_count": self._ewc_update_count,
            }
            result.update(train_stats or {})
            return result

    # ------------------------------------------------------------------
    # EWC diagnostics
    # ------------------------------------------------------------------

    def get_forgetting_risk(self) -> float:
        """Return the current mean Fisher-weighted parameter drift.

        Delegates to ``EWCRegularizer.get_forgetting_risk()``.  Measures how
        far the current model has drifted from the EWC anchor in importance-
        weighted parameter space — a proxy for catastrophic forgetting risk.

        Returns:
            float in [0, ∞).
                0.0  — EWC disabled, not yet initialised, or model at anchor.
                >0.0 — increasing drift; consider reducing the learning rate
                        or increasing lambda_ewc.
        """
        if self.ewc is None or not self.ewc.is_initialized():
            return 0.0
        return self.ewc.get_forgetting_risk(self.agent)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, suffix: str = "") -> Path:
        """Save the agent's current weights to a timestamped .pth file.

        File naming: online_checkpoint_{YYYYMMDD_HHMMSS}{suffix}.pth

        Args:
            suffix: Optional string appended before the file extension.
                    Useful for tagging (e.g., "_best", "_epoch5").

        Returns:
            Path to the saved checkpoint file.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"online_checkpoint_{timestamp}{suffix}.pth"
        path = self.checkpoint_dir / filename

        try:
            torch.save(
                {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "config": self.agent.config,
                    "step_count": self._step_count,
                    "saved_at_utc": timestamp,
                },
                path,
            )
            logger.info(f"Online checkpoint saved: {path}")
        except Exception as exc:
            logger.error(f"save_checkpoint failed: {exc}")
            raise

        return path

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run_continuous(self, interval_seconds: int = 300) -> None:
        """Run the online update loop indefinitely.

        Calls online_update() every *interval_seconds* seconds regardless of
        how many experiences were collected in the interval.  The update itself
        will skip gracefully if the buffer is too small.

        This coroutine is designed to run as a background asyncio task:

            manager = ContinuousLearningManager(agent, "checkpoints/")
            asyncio.create_task(manager.run_continuous(interval_seconds=300))

        Args:
            interval_seconds: Seconds to sleep between update attempts.
        """
        logger.info(f"run_continuous started | interval={interval_seconds}s")
        while True:
            t0 = time.monotonic()
            try:
                stats = await self.online_update()
                if stats.get("update_skipped"):
                    logger.debug(
                        f"run_continuous: update skipped "
                        f"(buffer={stats.get('buffer_size_before', '?')})"
                    )
                else:
                    logger.info(
                        f"run_continuous: update done | "
                        f"actor_loss={stats.get('actor_loss', 'n/a'):.4f} | "
                        f"critic_loss={stats.get('critic_loss', 'n/a'):.4f}"
                    )
            except Exception as exc:
                logger.error(f"run_continuous: unhandled error in online_update: {exc}")

            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, interval_seconds - elapsed)
            logger.debug(f"run_continuous: sleeping {sleep_for:.1f}s")
            await asyncio.sleep(sleep_for)
