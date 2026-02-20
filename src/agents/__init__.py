"""Extended DRL Agents module."""

from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.agents.drl_agents import (
    DQNAgent,
    DDPGAgent,
    SACAgent,
    A2CAgent,
    TD3Agent,
    ActorNetwork,
    CriticNetwork,
    GaussianActor,
    ActorCriticNetwork,
    ReplayBuffer,
    create_agent,
)

__all__ = [
    "PPOAgent",
    "PPOConfig",
    "DQNAgent",
    "DDPGAgent",
    "SACAgent",
    "A2CAgent",
    "TD3Agent",
    "ActorNetwork",
    "CriticNetwork",
    "GaussianActor",
    "ActorCriticNetwork",
    "ReplayBuffer",
    "create_agent",
]
