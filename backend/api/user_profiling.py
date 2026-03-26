"""
User Risk Profiling API — REST endpoints for registration and risk assessment.

Endpoints:
  GET  /api/user-profiling/questionnaire          — returns all 25 questions (with i18n lang param)
  POST /api/user-profiling/submit                 — submit answers, get back risk profile
  GET  /api/user-profiling/profile                — get current user's risk profile
  POST /api/user-profiling/register               — register + submit profile in one step
  GET  /api/user-profiling/admin/distribution     — admin: aggregate profile distribution (no PII)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, validator

from backend.api.login import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Optional service import — module may not exist yet during parallel build
# ---------------------------------------------------------------------------
try:
    from src.user_profiling import QuestionAnswer, UserProfilingService
    from src.user_profiling.questionnaire import QUESTIONS, get_question_text

    _service = UserProfilingService()
except Exception as _import_exc:
    logger.warning(f"UserProfilingService not available: {_import_exc}")
    _service = None
    QUESTIONS = None
    QuestionAnswer = None

# ---------------------------------------------------------------------------
# Localised risk-summary text templates
# ---------------------------------------------------------------------------
RISK_SUMMARY: Dict[str, Dict[str, str]] = {
    "KONSERVATIV": {
        "de": (
            "Ihr Risikoprofil ist KONSERVATIV. Sie bevorzugen Kapitalerhalt über Rendite. "
            "Max. {pct}% Drawdown konfiguriert."
        ),
        "en": (
            "Your risk profile is CONSERVATIVE. You prioritize capital preservation. "
            "Max {pct}% drawdown configured."
        ),
        "fr": (
            "Votre profil de risque est CONSERVATEUR. Vous privilégiez la préservation du capital. "
            "Drawdown max. {pct}% configuré."
        ),
        "es": (
            "Su perfil de riesgo es CONSERVADOR. Prioriza la preservación del capital. "
            "Drawdown máx. {pct}% configurado."
        ),
    },
    "AUSGEWOGEN": {
        "de": (
            "Ihr Risikoprofil ist AUSGEWOGEN. Sie akzeptieren moderate Schwankungen für solide Renditen. "
            "Max. {pct}% Drawdown konfiguriert."
        ),
        "en": (
            "Your risk profile is BALANCED. You accept moderate fluctuations for solid returns. "
            "Max {pct}% drawdown configured."
        ),
        "fr": (
            "Votre profil de risque est ÉQUILIBRÉ. Vous acceptez des fluctuations modérées pour des rendements solides. "
            "Drawdown max. {pct}% configuré."
        ),
        "es": (
            "Su perfil de riesgo es EQUILIBRADO. Acepta fluctuaciones moderadas para obtener rendimientos sólidos. "
            "Drawdown máx. {pct}% configurado."
        ),
    },
    "WACHSTUM": {
        "de": (
            "Ihr Risikoprofil ist WACHSTUM. Sie streben überdurchschnittliche Renditen an und tolerieren höhere Volatilität. "
            "Max. {pct}% Drawdown konfiguriert."
        ),
        "en": (
            "Your risk profile is GROWTH. You seek above-average returns and tolerate higher volatility. "
            "Max {pct}% drawdown configured."
        ),
        "fr": (
            "Votre profil de risque est CROISSANCE. Vous recherchez des rendements supérieurs à la moyenne et tolérez une volatilité plus élevée. "
            "Drawdown max. {pct}% configuré."
        ),
        "es": (
            "Su perfil de riesgo es CRECIMIENTO. Busca rendimientos superiores a la media y tolera mayor volatilidad. "
            "Drawdown máx. {pct}% configurado."
        ),
    },
    "SPEKULATIV": {
        "de": (
            "Ihr Risikoprofil ist SPEKULATIV. Sie sind auf maximales Wachstum ausgerichtet und akzeptieren erhebliche Verlustrisiken. "
            "Max. {pct}% Drawdown konfiguriert."
        ),
        "en": (
            "Your risk profile is SPECULATIVE. You are focused on maximum growth and accept significant loss risk. "
            "Max {pct}% drawdown configured."
        ),
        "fr": (
            "Votre profil de risque est SPÉCULATIF. Vous visez une croissance maximale et acceptez des risques de pertes significatifs. "
            "Drawdown max. {pct}% configuré."
        ),
        "es": (
            "Su perfil de riesgo es ESPECULATIVO. Está enfocado en el crecimiento máximo y acepta riesgos de pérdida significativos. "
            "Drawdown máx. {pct}% configurado."
        ),
    },
}

# Drawdown percentages per category (used in summary text and max-loss example)
_CATEGORY_MAX_DRAWDOWN_PCT: Dict[str, float] = {
    "KONSERVATIV": 10.0,
    "AUSGEWOGEN": 20.0,
    "WACHSTUM": 35.0,
    "SPEKULATIV": 55.0,
}

_MAX_LOSS_TEMPLATE: Dict[str, str] = {
    "de": "Bei 10.000\u20ac Kapital wären max. {loss:.0f}\u20ac Verlust möglich ({pct}% Drawdown).",
    "en": "With 10,000\u20ac capital, a maximum loss of {loss:.0f}\u20ac would be possible ({pct}% drawdown).",
    "fr": "Avec un capital de 10\u202f000\u20ac, une perte maximale de {loss:.0f}\u20ac serait possible ({pct}% drawdown).",
    "es": "Con un capital de 10.000\u20ac, sería posible una pérdida máxima de {loss:.0f}\u20ac ({pct}% drawdown).",
}


def _build_summary_text(category: str, lang: str) -> str:
    """Return localised risk-summary sentence for *category*."""
    lang = lang if lang in ("de", "en", "fr", "es") else "de"
    pct = _CATEGORY_MAX_DRAWDOWN_PCT.get(category, 20.0)
    template = RISK_SUMMARY.get(category, RISK_SUMMARY["AUSGEWOGEN"]).get(lang, "")
    return template.format(pct=pct)


def _build_max_loss_example(category: str, lang: str) -> str:
    """Return localised max-loss-at-10k-capital sentence."""
    lang = lang if lang in ("de", "en", "fr", "es") else "de"
    pct = _CATEGORY_MAX_DRAWDOWN_PCT.get(category, 20.0)
    loss = 10_000.0 * pct / 100.0
    template = _MAX_LOSS_TEMPLATE.get(lang, _MAX_LOSS_TEMPLATE["de"])
    return template.format(loss=loss, pct=pct)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class AnswerSubmission(BaseModel):
    question_id: int = Field(..., ge=1, le=25)
    answer_index: int = Field(..., ge=0, le=3)
    time_taken_sec: float = Field(default=0.0, ge=0.0)


class AssessmentRequest(BaseModel):
    answers: List[AnswerSubmission] = Field(..., min_items=25, max_items=25)
    consent_given: bool
    language: str = Field(default="de", regex="^(de|en|fr|es)$")

    @validator("consent_given")
    def consent_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Consent must be given to proceed")
        return v


class RegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)
    assessment: AssessmentRequest


class RiskProfileResponse(BaseModel):
    profile_id: str
    category: str  # KONSERVATIV | AUSGEWOGEN | WACHSTUM | SPEKULATIV
    continuum_score: float  # 0-100
    t_score: float
    dimension_scores: Dict[str, Any]  # {tolerance, capacity, knowledge, horizon, bias}
    risk_params: Dict[str, Any]  # mapped RiskConfig params
    assessed_at: str
    speeding_detected: bool
    risk_summary_text: str  # "Ihr Profil bedeutet..."
    max_loss_example: str  # "Bei 10.000€ Kapital wären max. X€ Verlust möglich"


class QuestionResponse(BaseModel):
    id: int
    dimension: str
    text: str
    options: List[str]  # option texts for the requested language


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_service() -> Any:
    """Return the module-level service instance or raise 503."""
    if _service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User profiling service is not available. Please try again later.",
        )
    return _service


def _map_profile_to_response(profile: Any, lang: str = "de") -> RiskProfileResponse:
    """Convert a domain profile object (or plain dict) to RiskProfileResponse."""
    # Support both attribute-access objects and plain dicts.
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    category: str = _get(profile, "category", "AUSGEWOGEN")
    assessed_at_raw = _get(profile, "assessed_at", None)
    if isinstance(assessed_at_raw, datetime):
        assessed_at_str = assessed_at_raw.isoformat()
    elif assessed_at_raw is None:
        assessed_at_str = datetime.utcnow().isoformat()
    else:
        assessed_at_str = str(assessed_at_raw)

    return RiskProfileResponse(
        profile_id=str(_get(profile, "profile_id", "")),
        category=category,
        continuum_score=float(_get(profile, "continuum_score", 0.0)),
        t_score=float(_get(profile, "t_score", 50.0)),
        dimension_scores=dict(_get(profile, "dimension_scores", {})),
        risk_params=dict(_get(profile, "risk_params", {})),
        assessed_at=assessed_at_str,
        speeding_detected=bool(_get(profile, "speeding_detected", False)),
        risk_summary_text=_build_summary_text(category, lang),
        max_loss_example=_build_max_loss_example(category, lang),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/questionnaire",
    response_model=List[QuestionResponse],
    summary="Get all 25 questionnaire questions",
    description="Returns the full risk-profiling questionnaire in the requested language. No authentication required.",
)
async def get_questionnaire(
    lang: str = Query(default="de", regex="^(de|en|fr|es)$"),
) -> List[QuestionResponse]:
    """Return all 25 questions localised to *lang*. Public endpoint — no auth required."""
    if QUESTIONS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Questionnaire data is not available. Please try again later.",
        )

    result: List[QuestionResponse] = []
    for q in QUESTIONS:
        # Support both attribute-access objects and dicts
        if isinstance(q, dict):
            qid = q.get("id", 0)
            dimension = q.get("dimension", "")
            # get_question_text may be available for i18n; fall back to raw text
            if callable(get_question_text):
                text, options = get_question_text(q, lang)
            else:
                text = q.get("text", "")
                options = q.get("options", [])
        else:
            qid = getattr(q, "id", 0)
            dimension = getattr(q, "dimension", "")
            if callable(get_question_text):
                text, options = get_question_text(q, lang)
            else:
                text = getattr(q, "text", "")
                options = list(getattr(q, "options", []))

        result.append(
            QuestionResponse(
                id=qid,
                dimension=dimension,
                text=text,
                options=list(options),
            )
        )

    return result


@router.post(
    "/submit",
    response_model=RiskProfileResponse,
    summary="Submit risk assessment answers",
    description="Accepts the completed 25-question assessment, persists the result, and returns the computed risk profile.",
)
async def submit_assessment(
    payload: AssessmentRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> RiskProfileResponse:
    """Submit a completed assessment for the authenticated user."""
    svc = _require_service()

    user_id: str = current_user.get("username", "")
    consent_ip: str = request.client.host if request.client else "unknown"
    lang: str = payload.language

    # Convert Pydantic answer models to the domain QuestionAnswer type
    if QuestionAnswer is not None:
        answers = [
            QuestionAnswer(
                question_id=a.question_id,
                answer_index=a.answer_index,
                time_taken_sec=a.time_taken_sec,
            )
            for a in payload.answers
        ]
    else:
        # Fall back to passing plain dicts if domain type is unavailable
        answers = [a.dict() for a in payload.answers]

    try:
        profile = svc.assess_and_map(
            user_id=user_id,
            answers=answers,
            consent_ip=consent_ip,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception("Error during risk assessment for user '%s': %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the assessment.",
        )

    return _map_profile_to_response(profile, lang=lang)


@router.get(
    "/profile",
    response_model=RiskProfileResponse,
    summary="Get current user's risk profile",
    description="Returns the most recently computed risk profile for the authenticated user.",
)
async def get_profile(
    lang: str = Query(default="de", regex="^(de|en|fr|es)$"),
    current_user: dict = Depends(get_current_user),
) -> RiskProfileResponse:
    """Retrieve the stored risk profile for the authenticated user."""
    svc = _require_service()

    user_id: str = current_user.get("username", "")

    try:
        profile = svc.get_profile(user_id=user_id)
    except Exception as exc:
        logger.exception("Error fetching profile for user '%s': %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the profile.",
        )

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No risk profile found for this user. Please complete the assessment first.",
        )

    return _map_profile_to_response(profile, lang=lang)


@router.post(
    "/register",
    response_model=RiskProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user and submit risk profile in one step",
    description=(
        "Creates a new user account and immediately persists the completed "
        "risk assessment. Returns the computed risk profile on success."
    ),
)
async def register_with_assessment(
    payload: RegistrationRequest,
    request: Request,
) -> RiskProfileResponse:
    """Register a new user and process the risk assessment atomically."""
    svc = _require_service()

    consent_ip: str = request.client.host if request.client else "unknown"
    lang: str = payload.assessment.language

    # Convert answer models
    if QuestionAnswer is not None:
        answers = [
            QuestionAnswer(
                question_id=a.question_id,
                answer_index=a.answer_index,
                time_taken_sec=a.time_taken_sec,
            )
            for a in payload.assessment.answers
        ]
    else:
        answers = [a.dict() for a in payload.assessment.answers]

    try:
        profile = svc.register_and_assess(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            answers=answers,
            consent_ip=consent_ip,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "Error during registration+assessment for user '%s': %s",
            payload.username,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while registering the user.",
        )

    return _map_profile_to_response(profile, lang=lang)


@router.get(
    "/admin/distribution",
    summary="Admin: aggregate risk-profile distribution",
    description=(
        "Returns aggregate statistics — count per category and average dimension scores — "
        "without any personally identifiable information. Restricted to admin users."
    ),
)
async def get_admin_distribution(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return aggregate profile statistics. Admin access only."""
    if current_user.get("username") != "admin" and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )

    svc = _require_service()

    try:
        distribution = svc.get_distribution()
    except Exception as exc:
        logger.exception("Error fetching profile distribution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving distribution data.",
        )

    if distribution is None:
        # Return an empty but valid structure when no profiles exist yet
        return {
            "total_profiles": 0,
            "category_counts": {
                "KONSERVATIV": 0,
                "AUSGEWOGEN": 0,
                "WACHSTUM": 0,
                "SPEKULATIV": 0,
            },
            "average_scores": {
                "continuum_score": None,
                "t_score": None,
                "dimensions": {},
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

    # Normalise: accept dict or object from service
    if isinstance(distribution, dict):
        dist_out = distribution
    else:
        dist_out = {
            "total_profiles": getattr(distribution, "total_profiles", 0),
            "category_counts": dict(getattr(distribution, "category_counts", {})),
            "average_scores": dict(getattr(distribution, "average_scores", {})),
        }

    dist_out.setdefault("generated_at", datetime.utcnow().isoformat())
    return dist_out
