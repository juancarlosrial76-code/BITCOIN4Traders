"""
User Profiling Models
=====================
Core data models for the scientific risk profiling system.

Defines the domain objects that flow through the profiling pipeline:
QuestionAnswer -> RiskProfile -> RiskProfileMapped.

These models are intentionally decoupled from persistence and scoring logic
to allow independent reuse and testing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid


class RiskCategory(str, Enum):
    """
    Four-tier risk classification based on the continuum score (0-100).

    Boundaries follow FinaMetrica's normative distribution:
        KONSERVATIV : 0-30   (conservative, capital preservation focus)
        AUSGEWOGEN  : 31-55  (balanced, moderate growth)
        WACHSTUM    : 56-75  (growth-oriented, higher volatility tolerance)
        SPEKULATIV  : 76-100 (aggressive, maximum growth, high drawdown tolerance)
    """
    KONSERVATIV = "KONSERVATIV"
    AUSGEWOGEN  = "AUSGEWOGEN"
    WACHSTUM    = "WACHSTUM"
    SPEKULATIV  = "SPEKULATIV"


@dataclass
class QuestionAnswer:
    """
    A single answered question in the risk assessment questionnaire.

    Attributes
    ----------
    question_id    : Maps to Question.id (1-25).
    answer_index   : 0-based index into the question's options list (0=a, 1=b, 2=c, 3=d).
    time_taken_sec : Seconds the user spent on this question. Used for speeding detection.
                     A value of 0.0 means timing was not recorded.
    """
    question_id: int
    answer_index: int
    time_taken_sec: float = 0.0


@dataclass
class RiskProfile:
    """
    The complete psychometric risk profile for one assessment session.

    Scores follow a two-stage normalization:
        1. raw_score       : weighted sum across all 25 answers (25–100 range).
        2. t_score         : population-normalized T-score (mean=50, SD=10).
        3. continuum_score : linear map of t_score onto 0-100 for display purposes.

    Dimension scores (0-100 each) break down the overall profile by psychological
    construct, enabling targeted advice beyond the single continuum number.

    Attributes
    ----------
    user_id          : Application-level user identifier.
    raw_score        : Weighted sum of scored answers.
    t_score          : T-score relative to normative population (mean=50, SD=10).
    continuum_score  : 0-100 value used for category classification and display.
    category         : Derived RiskCategory from continuum_score boundaries.
    tolerance_score  : Emotional risk tolerance dimension (0-100).
    capacity_score   : Financial capacity to absorb losses (0-100).
    knowledge_score  : Financial and crypto market knowledge (0-100).
    horizon_score    : Investment time horizon flexibility (0-100).
    bias_score       : Behavioral bias resistance (inverted: 100 = fully rational).
    assessed_at      : UTC timestamp of assessment completion.
    consent_given    : True if user explicitly consented to data storage.
    consent_ip       : IP address at time of consent (GDPR record).
    profile_id       : Unique UUID for this assessment session.
    speeding_detected: True if any answer was given in < 5 seconds (validity flag).
    """
    user_id: str
    raw_score: float
    t_score: float
    continuum_score: float
    category: RiskCategory
    tolerance_score: float
    capacity_score: float
    knowledge_score: float
    horizon_score: float
    bias_score: float
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    consent_given: bool = False
    consent_ip: Optional[str] = None
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    speeding_detected: bool = False


@dataclass
class RiskProfileMapped:
    """
    Risk profile with concrete RiskConfig parameters attached.

    This is the integration interface between the user profiling module and
    src/risk/risk_manager.py. Downstream code receives a RiskProfileMapped
    and can apply it directly to a RiskManager instance via
    risk_mapping.apply_to_risk_manager().

    Attributes
    ----------
    profile                  : The underlying RiskProfile.
    max_position_size        : Maximum fraction of capital per trade (0.0-1.0).
    kelly_fraction           : Fractional Kelly multiplier (0.0-1.0).
    max_drawdown_per_session : Maximum acceptable session drawdown (0.0-1.0).
    max_consecutive_losses   : Trading halt threshold on consecutive losses.
    volatility_target        : Annualized volatility target for position sizing.
    """
    profile: RiskProfile
    max_position_size: float
    kelly_fraction: float
    max_drawdown_per_session: float
    max_consecutive_losses: int
    volatility_target: float
