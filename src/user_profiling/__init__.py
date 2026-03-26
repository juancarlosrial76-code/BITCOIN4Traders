"""
user_profiling — Scientific Risk Profiling Module
==================================================
Standalone, reusable module for MiFID II / WpHG-aligned user risk profiling.

This module is part of BITCOIN4Traders but designed with clean interfaces so it
can be extracted, versioned, and developed independently.

Public API
----------
The recommended entry point is UserProfilingService:

    from src.user_profiling import UserProfilingService, QuestionAnswer

    svc = UserProfilingService()
    mapped = svc.assess_and_map(user_id, answers, consent_ip=ip)
    svc.apply_to_risk_manager(mapped, risk_manager)

Domain objects:

    RiskCategory        — KONSERVATIV | AUSGEWOGEN | WACHSTUM | SPEKULATIV
    QuestionAnswer      — single answered question (id, index, time_taken_sec)
    RiskProfile         — psychometric assessment result (scores + metadata)
    RiskProfileMapped   — profile with concrete RiskConfig parameters attached

Supporting modules (importable directly for advanced use):

    questionnaire       — 25 bilingual questions (DE/EN) with dimension tags
    scoring             — T-score normalization and category classification
    risk_mapping        — RiskCategory → RiskConfig parameter mapping table
    database            — SQLite persistence (swappable to PostgreSQL)

Integration with src/risk/risk_manager.py
-----------------------------------------
    from src.user_profiling import UserProfilingService
    from src.risk.risk_manager import RiskManager, RiskConfig

    svc = UserProfilingService()
    mapped = svc.get_current_profile(user_id)
    if mapped:
        svc.apply_to_risk_manager(mapped, risk_manager_instance)

Version
-------
    1.0.0  — Initial implementation (2026-03-26)
             25-question FinaMetrica-inspired instrument, DE/EN i18n,
             SQLite persistence, RiskManager integration interface.
"""

from .models import (
    RiskCategory,
    QuestionAnswer,
    RiskProfile,
    RiskProfileMapped,
)
from .service import UserProfilingService
from .risk_mapping import RISK_PROFILE_MAPPING
from .questionnaire import QUESTIONS, SUPPORTED_LANGUAGES

__all__ = [
    # Core service
    "UserProfilingService",
    # Domain models
    "RiskCategory",
    "QuestionAnswer",
    "RiskProfile",
    "RiskProfileMapped",
    # Data
    "RISK_PROFILE_MAPPING",
    "QUESTIONS",
    "SUPPORTED_LANGUAGES",
]

__version__ = "1.0.0"
