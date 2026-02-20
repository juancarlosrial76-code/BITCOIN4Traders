"""
AI Trading System - Main Package
"""

__version__ = "1.0.0"
__author__ = "Your CTO"

from . import data
from . import features
from . import environment
from . import math_tools as math
from . import risk
from . import agents
from . import training
from . import backtesting

__all__ = [
    "data",
    "features",
    "environment",
    "math",
    "risk",
    "agents",
    "training",
    "backtesting",
]
