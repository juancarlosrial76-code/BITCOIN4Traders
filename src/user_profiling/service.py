"""
UserProfilingService — Facade for the Risk Profiling Module
============================================================
Orchestrates scoring, mapping, and persistence behind a single clean interface.

This is the primary entry point for all external code (backend API, live engine,
tests) that needs to interact with the risk profiling system.

Integration example
-------------------
    from src.user_profiling import UserProfilingService, QuestionAnswer

    svc = UserProfilingService()

    # After user completes questionnaire:
    answers = [
        QuestionAnswer(question_id=i, answer_index=a, time_taken_sec=t)
        for i, a, t in raw_answers
    ]
    mapped = svc.assess_and_map("user-uuid", answers, consent_ip="1.2.3.4")

    # Apply to live risk manager:
    svc.apply_to_risk_manager(mapped, risk_manager_instance)

    # Retrieve later:
    mapped = svc.get_current_profile("user-uuid")
    if mapped is None:
        # User has not completed assessment yet
        ...
"""

from __future__ import annotations

from typing import List, Optional

from .models import QuestionAnswer, RiskProfile, RiskProfileMapped
from .scoring import score_assessment
from .risk_mapping import map_profile, apply_to_risk_manager as _apply
from .database import init_db, save_profile, load_profile, load_all_profiles


class UserProfilingService:
    """
    Standalone service for scientific user risk profiling.

    Designed to be instantiated once at application startup and reused across
    requests. Thread-safe: all state is stored in SQLite (file-level locking)
    and the service itself holds no mutable instance state.

    Parameters
    ----------
    auto_init_db : bool
        If True (default), calls init_db() on construction to ensure the
        SQLite table exists. Set to False in unit tests that mock the DB.
    """

    def __init__(self, auto_init_db: bool = True) -> None:
        if auto_init_db:
            init_db()

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def assess_and_map(
        self,
        user_id: str,
        answers: List[QuestionAnswer],
        consent_ip: Optional[str] = None,
    ) -> RiskProfileMapped:
        """
        Full assessment pipeline: score → classify → map → persist → return.

        Parameters
        ----------
        user_id    : Application-level user identifier (any non-empty string).
        answers    : List of 25 QuestionAnswer objects (one per questionnaire item).
        consent_ip : IP address of the client at time of consent (for GDPR audit log).

        Returns
        -------
        RiskProfileMapped with concrete RiskConfig parameters attached.

        Raises
        ------
        ValueError  : If fewer or more than 25 answers are provided.
        """
        if len(answers) != 25:
            raise ValueError(f"Expected exactly 25 answers, got {len(answers)}")

        profile = score_assessment(
            user_id=user_id,
            answers=answers,
            consent_given=True,
            consent_ip=consent_ip,
        )
        save_profile(profile)
        return map_profile(profile)

    def get_current_profile(self, user_id: str) -> Optional[RiskProfileMapped]:
        """
        Load the most recent risk profile for a user.

        Returns None if the user has not completed the assessment yet.
        """
        profile = load_profile(user_id)
        if profile is None:
            return None
        return map_profile(profile)

    def apply_to_risk_manager(
        self, mapped: RiskProfileMapped, risk_manager: object
    ) -> None:
        """
        Apply a mapped risk profile to a RiskManager instance.

        Mutates risk_manager.config in-place. The risk_manager must expose a
        .config attribute with the same fields as src/risk/risk_manager.RiskConfig.

        Parameters
        ----------
        mapped       : RiskProfileMapped returned by assess_and_map or get_current_profile.
        risk_manager : Instance of src.risk.risk_manager.RiskManager (or duck-typed equivalent).
        """
        _apply(mapped, risk_manager)

    # ------------------------------------------------------------------
    # Registration helper (used by the /register endpoint)
    # ------------------------------------------------------------------

    def register_and_assess(
        self,
        user_id: str,
        answers: List[QuestionAnswer],
        consent_ip: Optional[str] = None,
    ) -> RiskProfileMapped:
        """
        Alias for assess_and_map used by the registration endpoint to make
        intent explicit. Identical behaviour.
        """
        return self.assess_and_map(user_id, answers, consent_ip)

    # ------------------------------------------------------------------
    # Admin / analytics
    # ------------------------------------------------------------------

    def get_distribution(self, limit: int = 1000) -> dict:
        """
        Return aggregate profile statistics for admin monitoring.

        No individual (PII) data is returned — only counts and averages.

        Returns
        -------
        dict with keys: total, by_category, avg_continuum_score, avg_dimension_scores
        """
        profiles: List[RiskProfile] = load_all_profiles(limit=limit)

        if not profiles:
            return {
                "total": 0,
                "by_category": {},
                "avg_continuum_score": None,
                "avg_dimension_scores": {},
            }

        by_category: dict = {}
        continuum_scores: List[float] = []
        dim_sums = {
            "tolerance": 0.0,
            "capacity": 0.0,
            "knowledge": 0.0,
            "horizon": 0.0,
            "bias": 0.0,
        }

        for p in profiles:
            cat = p.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            continuum_scores.append(p.continuum_score)
            dim_sums["tolerance"] += p.tolerance_score
            dim_sums["capacity"] += p.capacity_score
            dim_sums["knowledge"] += p.knowledge_score
            dim_sums["horizon"] += p.horizon_score
            dim_sums["bias"] += p.bias_score

        n = len(profiles)
        return {
            "total": n,
            "by_category": by_category,
            "avg_continuum_score": round(sum(continuum_scores) / n, 2),
            "avg_dimension_scores": {k: round(v / n, 2) for k, v in dim_sums.items()},
        }
