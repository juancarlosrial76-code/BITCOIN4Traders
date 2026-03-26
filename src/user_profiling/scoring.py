"""
Psychometric Scoring Engine
============================
Implements the full scoring pipeline for the risk profiling questionnaire.

Normalization follows the T-score approach standard in psychometrics:
    raw_score       -> T-score  (mean=50, SD=10 relative to population norms)
    T-score         -> continuum score (0-100 display scale)
    continuum score -> RiskCategory

Population norms approximate FinaMetrica's global normative dataset (N > 600,000).
The mean raw score of 62.5 corresponds to 25 questions each averaging 2.5 on the
1-4 scale, reflecting a roughly symmetric population distribution.

Speeding detection: any answer delivered in fewer than 5 seconds (but more than 0)
is flagged. A full assessment with speeding should be treated with caution and may
warrant re-administration.

References
----------
Nunnally, J.C. & Bernstein, I.H. (1994). Psychometric Theory (3rd ed.). McGraw-Hill.
FinaMetrica Pty Ltd. (2013). FinaMetrica Risk Profiling Technical Manual.
"""

from typing import List, Optional

from .models import QuestionAnswer, RiskCategory, RiskProfile
from .questionnaire import QUESTIONS, QUESTIONS_BY_ID

# ---------------------------------------------------------------------------
# Population norms (FinaMetrica normative approximation)
# Raw score range: 25 (all 1s) to 100 (all 4s, unweighted).
# With weights the effective mean shifts slightly; we use the unweighted
# midpoint as a conservative baseline.
# ---------------------------------------------------------------------------
POPULATION_MEAN: float = 62.5   # midpoint of 25 × avg 2.5
POPULATION_STD: float = 12.5    # assumed SD (≈ 20% of range)


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def compute_dimension_score(answers: List[QuestionAnswer], dimension: str) -> float:
    """
    Compute a 0-100 score for a single psychometric dimension.

    The raw dimension score is the weighted average of answered question scores
    mapped onto the 0-100 scale, where:
        minimum possible option score = 1  (maps to 0)
        maximum possible option score = 4  (maps to 100)

    Parameters
    ----------
    answers   : Full list of QuestionAnswer objects from one assessment session.
    dimension : One of "tolerance", "capacity", "knowledge", "horizon", "bias".

    Returns
    -------
    float : Dimension score in [0, 100]. Returns 50.0 (neutral) if no matching
            questions are found (defensive fallback).
    """
    dim_questions = [q for q in QUESTIONS if q.dimension == dimension]
    if not dim_questions:
        return 50.0

    answers_by_qid = {a.question_id: a for a in answers}
    weighted_sum = 0.0
    total_weight = 0.0

    for question in dim_questions:
        answer = answers_by_qid.get(question.id)
        if answer is None:
            continue
        # Clamp answer_index to valid range
        idx = max(0, min(3, answer.answer_index))
        option_score = question.options[idx].score  # 1-4
        # Normalize to 0-100: (score - 1) / 3 * 100
        normalized = (option_score - 1) / 3.0 * 100.0
        weighted_sum += normalized * question.weight
        total_weight += question.weight

    if total_weight == 0.0:
        return 50.0

    return round(weighted_sum / total_weight, 2)


def compute_raw_score(answers: List[QuestionAnswer]) -> float:
    """
    Compute the weighted raw score across all 25 questions.

    Raw score = sum of (option_score × question.weight) for each answered question.
    Mirrors the FinaMetrica approach where each item contributes its point value
    multiplied by its psychometric weight.

    Parameters
    ----------
    answers : List of QuestionAnswer objects.

    Returns
    -------
    float : Weighted raw score. Range depends on weights but approximates [25, 100]
            for unweighted questions.
    """
    answers_by_qid = {a.question_id: a for a in answers}
    total = 0.0

    for question in QUESTIONS:
        answer = answers_by_qid.get(question.id)
        if answer is None:
            # Treat unanswered questions as neutral (score=2, midpoint)
            total += 2.0 * question.weight
        else:
            idx = max(0, min(3, answer.answer_index))
            option_score = question.options[idx].score  # 1-4
            total += option_score * question.weight

    return round(total, 4)


def compute_t_score(raw_score: float) -> float:
    """
    Convert a raw score to a T-score (population-normalized).

    T-score formula:  T = 50 + 10 × (X - μ) / σ

    where X is the raw score, μ is the population mean, and σ is the population SD.
    A T-score of 50 is exactly average; 60 is one SD above average.

    Parameters
    ----------
    raw_score : Weighted raw score from compute_raw_score().

    Returns
    -------
    float : T-score. Typical range 20-80; extreme values outside this are clipped
            at the continuum_score stage.
    """
    return round(50.0 + 10.0 * (raw_score - POPULATION_MEAN) / POPULATION_STD, 4)


def compute_continuum_score(t_score: float) -> float:
    """
    Map a T-score onto the 0-100 display continuum, clipped to [0, 100].

    Mapping: (T - 20) / 60 × 100
        T=20 -> 0   (extreme conservative)
        T=50 -> 50  (average / balanced)
        T=80 -> 100 (extreme speculative)

    Values outside [20, 80] are clipped to the boundary.

    Parameters
    ----------
    t_score : T-score from compute_t_score().

    Returns
    -------
    float : Continuum score in [0.0, 100.0].
    """
    raw = (t_score - 20.0) / 60.0 * 100.0
    return round(max(0.0, min(100.0, raw)), 2)


def classify_profile(continuum_score: float) -> RiskCategory:
    """
    Classify a continuum score into a RiskCategory.

    Boundaries:
        [0,  30]  -> KONSERVATIV
        [31, 55]  -> AUSGEWOGEN
        [56, 75]  -> WACHSTUM
        [76, 100] -> SPEKULATIV

    Parameters
    ----------
    continuum_score : Score in [0, 100] from compute_continuum_score().

    Returns
    -------
    RiskCategory
    """
    if continuum_score <= 30.0:
        return RiskCategory.KONSERVATIV
    if continuum_score <= 55.0:
        return RiskCategory.AUSGEWOGEN
    if continuum_score <= 75.0:
        return RiskCategory.WACHSTUM
    return RiskCategory.SPEKULATIV


def detect_speeding(answers: List[QuestionAnswer]) -> bool:
    """
    Return True if any answer was given in fewer than 5 seconds.

    A time_taken_sec value of 0.0 is treated as "not recorded" and ignored.
    Any value in (0.0, 5.0) exclusive triggers the flag.

    Parameters
    ----------
    answers : List of QuestionAnswer objects.

    Returns
    -------
    bool : True if speeding was detected on at least one question.
    """
    return any(0.0 < a.time_taken_sec < 5.0 for a in answers)


def score_assessment(
    user_id: str,
    answers: List[QuestionAnswer],
    consent_given: bool = False,
    consent_ip: Optional[str] = None,
) -> RiskProfile:
    """
    Full scoring pipeline: answers -> RiskProfile.

    Steps:
        1. Compute weighted raw score across all 25 questions.
        2. Normalize to T-score (population reference).
        3. Map T-score to 0-100 continuum.
        4. Classify into RiskCategory.
        5. Compute per-dimension scores (tolerance, capacity, knowledge, horizon, bias).
        6. Detect speeding.
        7. Assemble and return RiskProfile.

    Parameters
    ----------
    user_id       : Application-level user identifier.
    answers       : List of 25 QuestionAnswer objects (one per question).
                    Missing answers are scored as neutral (option score=2).
    consent_given : Whether the user has consented to data storage.
    consent_ip    : IP address string for GDPR consent record.

    Returns
    -------
    RiskProfile : Fully populated profile ready for persistence and mapping.
    """
    raw_score = compute_raw_score(answers)
    t_score = compute_t_score(raw_score)
    continuum_score = compute_continuum_score(t_score)
    category = classify_profile(continuum_score)

    tolerance_score = compute_dimension_score(answers, "tolerance")
    capacity_score = compute_dimension_score(answers, "capacity")
    knowledge_score = compute_dimension_score(answers, "knowledge")
    horizon_score = compute_dimension_score(answers, "horizon")
    bias_score = compute_dimension_score(answers, "bias")

    speeding = detect_speeding(answers)

    return RiskProfile(
        user_id=user_id,
        raw_score=raw_score,
        t_score=t_score,
        continuum_score=continuum_score,
        category=category,
        tolerance_score=tolerance_score,
        capacity_score=capacity_score,
        knowledge_score=knowledge_score,
        horizon_score=horizon_score,
        bias_score=bias_score,
        consent_given=consent_given,
        consent_ip=consent_ip,
        speeding_detected=speeding,
    )
