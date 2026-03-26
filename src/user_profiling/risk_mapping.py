"""
Risk Category to RiskConfig Parameter Mapping
==============================================
Maps a classified RiskCategory onto concrete trading risk parameters that are
compatible with the existing RiskConfig dataclass in src/risk/risk_manager.py.

Design rationale
----------------
The mapping is intentionally conservative relative to the default RiskConfig values
for KONSERVATIV and AUSGEWOGEN profiles, and more aggressive for SPEKULATIV:

    Category     max_pos  kelly   max_dd   max_losses  vol_target
    ──────────── ──────── ─────── ──────── ─────────── ──────────
    KONSERVATIV   5%       0.25    5%        3           8%
    AUSGEWOGEN   10%       0.35   10%        4          15%
    WACHSTUM     20%       0.50   15%        5          25%
    SPEKULATIV   30%       0.75   20%        7          40%

The volatility_target is an annualized figure used by position-sizing models that
implement volatility-targeting (e.g. risk-parity, constant-volatility sizing).
It is stored in RiskProfileMapped but not in RiskConfig (which predates this module);
downstream code can use it independently.

Integration with src/risk/risk_manager.py
------------------------------------------
Call apply_to_risk_manager(mapped, risk_manager) to patch a live RiskManager
instance. The function writes directly to risk_manager.config, which is assumed
to be a mutable RiskConfig dataclass — consistent with the existing implementation.
"""

from typing import Dict

from .models import RiskCategory, RiskProfile, RiskProfileMapped


RISK_PROFILE_MAPPING: Dict[RiskCategory, dict] = {
    RiskCategory.KONSERVATIV: {
        "max_position_size": 0.05,
        "kelly_fraction": 0.25,
        "max_drawdown_per_session": 0.05,
        "max_consecutive_losses": 3,
        "volatility_target": 0.08,
    },
    RiskCategory.AUSGEWOGEN: {
        "max_position_size": 0.10,
        "kelly_fraction": 0.35,
        "max_drawdown_per_session": 0.10,
        "max_consecutive_losses": 4,
        "volatility_target": 0.15,
    },
    RiskCategory.WACHSTUM: {
        "max_position_size": 0.20,
        "kelly_fraction": 0.50,
        "max_drawdown_per_session": 0.15,
        "max_consecutive_losses": 5,
        "volatility_target": 0.25,
    },
    RiskCategory.SPEKULATIV: {
        "max_position_size": 0.30,
        "kelly_fraction": 0.75,
        "max_drawdown_per_session": 0.20,
        "max_consecutive_losses": 7,
        "volatility_target": 0.40,
    },
}


def map_profile(profile: RiskProfile) -> RiskProfileMapped:
    """
    Create a RiskProfileMapped from a scored RiskProfile.

    Looks up the parameter set for profile.category in RISK_PROFILE_MAPPING
    and returns a RiskProfileMapped that bundles the profile with those parameters.

    Parameters
    ----------
    profile : A fully scored RiskProfile (e.g. from scoring.score_assessment).

    Returns
    -------
    RiskProfileMapped
    """
    params = RISK_PROFILE_MAPPING[profile.category]
    return RiskProfileMapped(profile=profile, **params)


def apply_to_risk_manager(mapped: RiskProfileMapped, risk_manager) -> None:
    """
    Apply the mapped risk profile parameters to an existing RiskManager instance.

    This function is the clean integration point between the user profiling module
    and src/risk/risk_manager.py. It assumes risk_manager.config is a mutable
    RiskConfig dataclass with the following fields (all present in the existing
    implementation):
        - max_position_size        (float)
        - kelly_fraction           (float)
        - max_drawdown_per_session (float)
        - max_consecutive_losses   (int)

    Note: volatility_target is not part of RiskConfig and is intentionally not
    applied here. It is available on the mapped object for models that consume it.

    Parameters
    ----------
    mapped       : RiskProfileMapped produced by map_profile().
    risk_manager : A RiskManager instance whose .config attribute is RiskConfig.

    Side effects
    ------------
    Mutates risk_manager.config in-place. No return value.
    """
    risk_manager.config.max_position_size = mapped.max_position_size
    risk_manager.config.kelly_fraction = mapped.kelly_fraction
    risk_manager.config.max_drawdown_per_session = mapped.max_drawdown_per_session
    risk_manager.config.max_consecutive_losses = mapped.max_consecutive_losses
