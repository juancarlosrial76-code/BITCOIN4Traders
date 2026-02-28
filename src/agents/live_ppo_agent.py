"""
Live PPO Agent – Phase 5
=========================
Wrapper um den trainierten PPOAgent für Live/Paper-Trading.

Übersetzt:
  agent.predict(features: np.ndarray) -> int  (-1=short, 0=flat, +1=long)

Die 7 diskreten Aktionen aus dem Training werden auf 3 gemappt:
  0 (Short100%) → -1
  1 (Short50%)  → -1
  2 (Neutral)   →  0
  3 (Long33%)   → +1
  4 (Long50%)   → +1
  5 (Long75%)   → +1
  6 (Long100%)  → +1
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger("phase7.live_agent")

# Mapping: 7 Trainings-Aktionen → 3 Live-Signale
ACTION_TO_SIGNAL = {
    0: -1,  # Short 100%
    1: -1,  # Short 50%
    2: 0,  # Neutral / Flat
    3: +1,  # Long 33%
    4: +1,  # Long 50%
    5: +1,  # Long 75%
    6: +1,  # Long 100%
}


class LivePPOAgent:
    """
    Geladener PPO-Agent für Live- und Paper-Trading.
    Drop-in Ersatz für StubAgent in run.py.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cpu"):
        """
        Lädt Trader-Gewichte aus dem angegebenen Checkpoint-Verzeichnis.

        Parameters
        ----------
        checkpoint_dir : str
            Pfad zu einem der trainierten Modell-Ordner:
            - data/models/largenet/    (Phase 3 – höchster Return)
            - data/models/best/        (Phase 4 – bester Sharpe)
        device : str
            'cpu' oder 'cuda'
        """
        self._device = device
        self._hidden: Optional[Any] = None  # GRU-Hidden-State zwischen Ticks
        self._agent = None
        self._load(checkpoint_dir)

    def _load(self, checkpoint_dir: str) -> None:
        import sys

        sys.path.insert(0, "src")
        from agents.ppo_agent import PPOAgent, PPOConfig

        ckpt_dir = Path(checkpoint_dir)

        # Suche nach trader-Gewichten: final > best > letzter checkpoint
        candidates = [
            ckpt_dir / "final_model_trader.pth",
            ckpt_dir / "best_model_trader.pth",
        ]
        # auch nummerierte checkpoints als Fallback
        for p in sorted(ckpt_dir.glob("checkpoint_iter_*_trader.pth"), reverse=True):
            candidates.append(p)

        trader_path = None
        for c in candidates:
            if c.exists():
                trader_path = c
                break

        if trader_path is None:
            raise FileNotFoundError(
                f"Keine Trader-Gewichte in {checkpoint_dir} gefunden.\n"
                f"Erwartet: final_model_trader.pth oder best_model_trader.pth"
            )

        logger.info(f"[LiveAgent] Lade Gewichte: {trader_path}")

        # Checkpoint laden (enthält 'actor', 'critic', 'config' keys)
        ckpt = torch.load(trader_path, map_location=self._device)

        # Falls das Checkpoint ein verschachteltes Dict ist (actor/critic/config)
        if isinstance(ckpt, dict) and "config" in ckpt and "actor" in ckpt:
            saved_cfg = ckpt["config"]
            actor_state = ckpt["actor"]
            # PPOConfig-Objekt direkt übernehmen
            cfg = PPOConfig(
                state_dim=saved_cfg.state_dim,
                hidden_dim=saved_cfg.hidden_dim,
                n_actions=saved_cfg.n_actions,
                use_recurrent=saved_cfg.use_recurrent,
                rnn_type=saved_cfg.rnn_type,
                rnn_layers=getattr(saved_cfg, "rnn_layers", 1),
                dropout=getattr(saved_cfg, "dropout", 0.1),
                use_layer_norm=getattr(saved_cfg, "use_layer_norm", True),
            )
            logger.info(
                f"[LiveAgent] Checkpoint-Config: state_dim={cfg.state_dim}, "
                f"hidden_dim={cfg.hidden_dim}"
            )
        else:
            # Fallback: rohes state_dict, Architektur aus Gewichten ableiten
            actor_state = ckpt
            hidden_dim = 128
            for key, val in actor_state.items():
                if "rnn.weight_hh_l0" in key or "gru.weight_hh_l0" in key:
                    hidden_dim = val.shape[1]
                    break
            state_dim = 34
            for key, val in actor_state.items():
                if "feature_extractor.0.weight" in key:
                    state_dim = val.shape[1]
                    break
            cfg = PPOConfig(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                n_actions=7,
                use_recurrent=True,
                rnn_type="GRU",
            )
            logger.info(
                f"[LiveAgent] Architektur (abgeleitet): state_dim={state_dim}, "
                f"hidden_dim={hidden_dim}"
            )

        agent = PPOAgent(cfg, self._device)
        agent.actor.load_state_dict(actor_state)
        agent.actor.eval()
        self._agent = agent

        logger.info(
            f"[LiveAgent] Agent geladen ✓  (hidden={cfg.hidden_dim}, state={cfg.state_dim})"
        )

    def predict(self, features: np.ndarray) -> int:
        """
        Gibt Trading-Signal zurück: -1 (short), 0 (flat), +1 (long).

        Parameters
        ----------
        features : np.ndarray
            Feature-Vektor vom FeatureEngine.transform_single()
        """
        if self._agent is None:
            return 0

        obs = torch.FloatTensor(features).unsqueeze(0).to(self._device)

        with torch.no_grad():
            dist, self._hidden = self._agent.actor(obs, self._hidden)
            action = dist.probs.argmax().item()  # deterministisch im Live-Modus

        signal = ACTION_TO_SIGNAL.get(action, 0)
        return signal

    def reset_hidden(self) -> None:
        """Hidden-State zurücksetzen (z.B. nach Verbindungsabbruch)."""
        self._hidden = None
        logger.debug("[LiveAgent] Hidden State zurückgesetzt")
