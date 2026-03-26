"""
Persistence Layer for User Risk Profiles
=========================================
SQLite-backed storage for RiskProfile objects. Follows the same lightweight
pattern as src/data/sqlite_local.py (file-based SQLite, stdlib only).

Table schema: user_risk_profiles
    - profile_id is the PRIMARY KEY (UUID string).
    - user_id is indexed for fast lookup of the most recent profile per user.
    - All scores are stored as REAL to preserve floating-point precision.
    - Booleans are stored as INTEGER (0/1) per SQLite convention.
    - assessed_at is stored as ISO-8601 TEXT (UTC).

Swap path to PostgreSQL
-----------------------
Replace _get_conn() with a psycopg2/asyncpg connection and adjust the
CREATE TABLE statement (TEXT -> VARCHAR, INTEGER -> BOOLEAN, etc.).
The save/load interface is intentionally generic to minimise migration effort.

Privacy note
------------
consent_ip is stored as plain text. For production deployments consider
hashing or encrypting this field (GDPR / DSGVO compliance).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import RiskCategory, RiskProfile

# Default path relative to project root. Override DB_PATH before calling init_db()
# if a custom location is required.
DB_PATH: Path = Path("data/user_profiles.db")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Open (and create if necessary) the SQLite database at DB_PATH."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _row_to_profile(row: sqlite3.Row) -> RiskProfile:
    """Convert a sqlite3.Row from user_risk_profiles to a RiskProfile."""
    return RiskProfile(
        profile_id=row["profile_id"],
        user_id=row["user_id"],
        raw_score=row["raw_score"],
        t_score=row["t_score"],
        continuum_score=row["continuum_score"],
        category=RiskCategory(row["category"]),
        tolerance_score=row["tolerance_score"] if row["tolerance_score"] is not None else 50.0,
        capacity_score=row["capacity_score"] if row["capacity_score"] is not None else 50.0,
        knowledge_score=row["knowledge_score"] if row["knowledge_score"] is not None else 50.0,
        horizon_score=row["horizon_score"] if row["horizon_score"] is not None else 50.0,
        bias_score=row["bias_score"] if row["bias_score"] is not None else 50.0,
        assessed_at=datetime.fromisoformat(row["assessed_at"]),
        consent_given=bool(row["consent_given"]),
        consent_ip=row["consent_ip"],
        speeding_detected=bool(row["speeding_detected"]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create the user_risk_profiles table and index if they do not exist.

    Safe to call multiple times (idempotent). Call once at application startup
    before any save/load operations.
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_risk_profiles (
                profile_id          TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                raw_score           REAL NOT NULL,
                t_score             REAL NOT NULL,
                continuum_score     REAL NOT NULL,
                category            TEXT NOT NULL,
                tolerance_score     REAL,
                capacity_score      REAL,
                knowledge_score     REAL,
                horizon_score       REAL,
                bias_score          REAL,
                assessed_at         TEXT NOT NULL,
                consent_given       INTEGER NOT NULL DEFAULT 0,
                consent_ip          TEXT,
                speeding_detected   INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_id ON user_risk_profiles(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assessed_at ON user_risk_profiles(assessed_at)"
        )


def save_profile(profile: RiskProfile) -> None:
    """
    Persist a RiskProfile to the database.

    Uses INSERT OR REPLACE so that re-running an assessment with the same
    profile_id (UUID) is safe. In practice each assessment generates a new UUID,
    so this is equivalent to INSERT.

    Parameters
    ----------
    profile : RiskProfile to persist.
    """
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_risk_profiles (
                profile_id, user_id, raw_score, t_score, continuum_score,
                category, tolerance_score, capacity_score, knowledge_score,
                horizon_score, bias_score, assessed_at, consent_given,
                consent_ip, speeding_detected
            ) VALUES (
                :profile_id, :user_id, :raw_score, :t_score, :continuum_score,
                :category, :tolerance_score, :capacity_score, :knowledge_score,
                :horizon_score, :bias_score, :assessed_at, :consent_given,
                :consent_ip, :speeding_detected
            )
            """,
            {
                "profile_id": profile.profile_id,
                "user_id": profile.user_id,
                "raw_score": profile.raw_score,
                "t_score": profile.t_score,
                "continuum_score": profile.continuum_score,
                "category": profile.category.value,
                "tolerance_score": profile.tolerance_score,
                "capacity_score": profile.capacity_score,
                "knowledge_score": profile.knowledge_score,
                "horizon_score": profile.horizon_score,
                "bias_score": profile.bias_score,
                "assessed_at": profile.assessed_at.isoformat(),
                "consent_given": int(profile.consent_given),
                "consent_ip": profile.consent_ip,
                "speeding_detected": int(profile.speeding_detected),
            },
        )


def load_profile(user_id: str) -> Optional[RiskProfile]:
    """
    Load the most recent RiskProfile for a given user.

    "Most recent" is defined by the assessed_at timestamp (descending order).

    Parameters
    ----------
    user_id : Application-level user identifier.

    Returns
    -------
    RiskProfile if found, None otherwise.
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM user_risk_profiles
            WHERE user_id = ?
            ORDER BY assessed_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None
    return _row_to_profile(row)


def load_profile_history(user_id: str, limit: int = 10) -> List[RiskProfile]:
    """
    Load all historical profiles for a user, newest first.

    Parameters
    ----------
    user_id : Application-level user identifier.
    limit   : Maximum number of records to return.

    Returns
    -------
    List[RiskProfile] ordered by assessed_at descending.
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM user_risk_profiles
            WHERE user_id = ?
            ORDER BY assessed_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()

    return [_row_to_profile(row) for row in rows]


def load_all_profiles(limit: int = 100) -> List[RiskProfile]:
    """
    Load recent profiles across all users ordered by assessed_at descending.

    Intended for admin/analytics views. Returns full RiskProfile objects;
    callers are responsible for access control and PII handling.

    Parameters
    ----------
    limit : Maximum number of records to return. Default 100.

    Returns
    -------
    List[RiskProfile]
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM user_risk_profiles
            ORDER BY assessed_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    return [_row_to_profile(row) for row in rows]


def delete_profile(user_id: str) -> int:
    """
    Delete all stored profiles for a user (GDPR right to erasure).

    Parameters
    ----------
    user_id : Application-level user identifier.

    Returns
    -------
    int : Number of rows deleted.
    """
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM user_risk_profiles WHERE user_id = ?",
            (user_id,),
        )
        return cursor.rowcount
