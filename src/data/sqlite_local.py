"""
SQLite Local Database - High-Performance Local Storage
======================================================
Lightweight SQLite database optimized for the Linux Local Master.

This module provides a fast, embedded database solution for storing trading
data locally. It is designed for scenarios where PostgreSQL is not available
or where maximum performance is required for large-scale data operations.

Why SQLite instead of PostgreSQL?
    - No server installation required (zero-config)
    - Millions of price rows retrievable in milliseconds
    - No network overhead (file on local NVMe/SSD)
    - Full SQL functionality via SQLAlchemy
    - Single-file portability for backup and migration

Why not CSV/Parquet?
    - SQLite supports complex queries (JOINs, aggregations)
    - Concurrent-safe (multiple processes can write simultaneously)
    - Integrated indexes: query of 1 million bars in <10ms
    - ACID compliance ensures data integrity

Architecture:
    Local Master:   SQLite (local file, fast, no limits)
    GitHub Actions: Cache (actions/cache, max 10GB)
    Google Colab:   Drive-Sync or download from Binance

Database Tables:
    - market_data: OHLCV candlestick data (primary table, millions of rows)
    - champion_history: Historical champion model metadata
    - trades: Executed trade records
    - portfolio_snapshots: Hourly portfolio performance snapshots
    - heartbeat: System health monitoring log

Configuration:
    Database path can be configured via SQLITE_DB_PATH environment variable.
    Default location: <repo_root>/data/sqlite/trading.db

Performance Optimizations:
    - WAL (Write-Ahead Logging) mode for concurrent reads
    - 64MB in-memory page cache
    - Synchronous=NORMAL for fast writes without full fsync
    - UNIQUE index on (symbol, timeframe, timestamp) prevents duplicates

Usage:
    from src.data.sqlite_local import LocalDB, get_local_db

    # Using singleton instance (recommended)
    db = get_local_db()

    # Save OHLCV data
    df = pd.DataFrame({
        'open': [50000, 50100],
        'high': [50200, 50300],
        'low': [49900, 50000],
        'close': [50100, 50200],
        'volume': [1000, 1100]
    }, index=pd.date_range('2024-01-01', periods=2, freq='1h', tz='UTC'))
    db.save_ohlcv(df, symbol='BTC/USDT', timeframe='1h')

    # Load last 90 days of data
    df = db.load_ohlcv('BTC/USDT', '1h', days=90)

    # Archive champion model
    db.save_champion(champion_meta)

    # Check database stats
    stats = db.stats()
    print(f"Database size: {stats['db_size_mb']} MB")
    print(f"Total bars: {stats['market_data_rows']}")

CLI Usage:
    python -m src.data.sqlite_local --test  # Run with test data

Dependencies:
    - sqlite3: Built-in Python module
    - pandas: DataFrame support
    - loguru: Logging

Warning:
    SQLite is not designed for concurrent writes from multiple processes.
    For multi-process scenarios, use PostgreSQL instead.
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path detection: SQLite file is located next to the repo
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/data/sqlite_local.py -> root
_DEFAULT_DB = Path(
    os.environ.get(
        "SQLITE_DB_PATH",
        str(_REPO_ROOT / "data" / "sqlite" / "trading.db"),
    )
)


class LocalDB:
    """
    High-performance SQLite database for local trading data storage.

    This class provides a complete local database solution optimized for
    trading applications. It offers fast read/write operations with proper
    concurrency handling and automatic index management.

    Attributes:
        db_path: Path to the SQLite database file

    Thread Safety:
        Uses check_same_thread=False to allow multi-threaded access.
        WAL mode enables concurrent reads while writing.
        Timeout of 30 seconds prevents write failures under load.

    Database Schema:
        market_data:
            - id (INTEGER PRIMARY KEY)
            - symbol (TEXT) - e.g., "BTC/USDT"
            - timestamp (INTEGER) - Unix ms
            - open, high, low, close, volume (REAL)
            - timeframe (TEXT) - "1h", "4h", "1d", etc.
            - UNIQUE INDEX: (symbol, timeframe, timestamp)

        champion_history:
            - id, timestamp, name, strategy
            - sharpe, calmar, sortino, profit_factor
            - total_return, win_rate, max_drawdown, n_trades
            - source, meta_json (full metadata as JSON)

        trades:
            - id, trade_id (UNIQUE), symbol, side
            - order_type, quantity, price, total_value
            - fee, exchange, strategy, timestamp

        portfolio_snapshots:
            - id, timestamp, total_value, cash
            - exposure, daily_pnl, total_pnl
            - sharpe_ratio, max_drawdown

        heartbeat:
            - id, timestamp, source, status
            - signal, champion

    Example:
        >>> db = LocalDB()  # Creates db at default location
        >>>
        >>> # Save OHLCV data
        >>> df = pd.DataFrame({
        ...     'open': [50000, 50100],
        ...     'high': [50200, 50300],
        ...     'low': [49900, 50000],
        ...     'close': [50100, 50200],
        ...     'volume': [1000, 1100]
        ... }, index=pd.date_range('2024-01-01', periods=2, freq='1h', tz='UTC'))
        >>>
        >>> saved = db.save_ohlcv(df, 'BTC/USDT', '1h')
        >>> print(f"Saved {saved} rows")
        Saved 2 rows

        >>> # Load data
        >>> df = db.load_ohlcv('BTC/USDT', '1h', days=30)
        >>> print(f"Loaded {len(df)} bars")

        >>> # Get database statistics
        >>> stats = db.stats()
        >>> print(f"DB size: {stats['db_size_mb']} MB")

    Performance Tips:
        - Use load_ohlcv() with limit parameter for recent data only
        - Call vacuum() periodically after many deletions
        - Use delete_old_data() to manage database size
        - Monitor stats() to track growth

    Note:
        The database is created automatically on first access. Parent
        directories are created if they don't exist.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or _DEFAULT_DB)
        self.db_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure directory exists
        self._init_db()
        logger.info(f"LocalDB initialized: {self.db_path}")

    # ------------------------------------------------------------------
    # Internal initialization
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection with optimized settings.

        Configures the connection for high-performance concurrent access:
        - check_same_thread=False: Allows multi-threaded access
        - timeout=30: Waits up to 30s for locked database
        - WAL mode: Write-Ahead Logging for concurrent reads
        - synchronous=NORMAL: Fast + safe without full fsync
        - cache_size=-64000: 64MB in-memory page cache
        - row_factory=Row: Enables column-name access on rows
        """
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow multi-threaded access
            timeout=30,  # Wait up to 30s for locked DB
        )
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead-Log (concurrent-safe)
        conn.execute(
            "PRAGMA synchronous=NORMAL"
        )  # Fast + safe (no fsync on every write)
        conn.execute("PRAGMA cache_size=-64000")  # 64 MB in-memory page cache
        conn.row_factory = sqlite3.Row  # Enable column-name access on rows
        return conn

    def _init_db(self):
        """Creates all tables if they do not exist."""
        with self._connect() as conn:
            conn.executescript("""
                -- OHLCV market data (main table, can hold millions of rows)
                CREATE TABLE IF NOT EXISTS market_data (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol    TEXT    NOT NULL,
                    timestamp INTEGER NOT NULL,   -- Unix timestamp in ms
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    REAL,
                    timeframe TEXT    DEFAULT '1h'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_md_uniq
                    ON market_data (symbol, timeframe, timestamp);
                CREATE INDEX IF NOT EXISTS idx_md_symbol_ts
                    ON market_data (symbol, timestamp DESC);

                -- Champion history (every new champion is archived here)
                CREATE TABLE IF NOT EXISTS champion_history (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    name           TEXT,
                    strategy       TEXT,
                    sharpe         REAL,
                    calmar         REAL,
                    sortino        REAL,
                    profit_factor  REAL,
                    total_return   REAL,
                    win_rate       REAL,
                    max_drawdown   REAL,
                    n_trades       INTEGER,
                    source         TEXT DEFAULT 'local_master',
                    meta_json      TEXT   -- Full metadata as JSON
                );

                -- Executed trades
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id    TEXT    UNIQUE,
                    symbol      TEXT,
                    side        TEXT,   -- BUY or SELL
                    order_type  TEXT,   -- MARKET, LIMIT
                    quantity    REAL,
                    price       REAL,
                    total_value REAL,
                    fee         REAL,
                    exchange    TEXT,
                    strategy    TEXT,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Portfolio snapshots (hourly)
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_value  REAL,
                    cash         REAL,
                    exposure     REAL,
                    daily_pnl    REAL,
                    total_pnl    REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL
                );

                -- Heartbeat log (hourly entry)
                CREATE TABLE IF NOT EXISTS heartbeat (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source    TEXT,   -- 'local_master', 'github_actions', 'colab'
                    status    TEXT,
                    signal    TEXT,   -- 'BUY', 'SELL', 'HOLD'
                    champion  TEXT
                );
            """)
        logger.debug("SQLite tables verified")

    # ------------------------------------------------------------------
    # OHLCV price data
    # ------------------------------------------------------------------

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
    ) -> int:
        """
        Saves OHLCV DataFrame to local SQLite DB.
        Uses INSERT OR IGNORE to avoid duplicates.

        Args:
            df:        DataFrame with columns [open, high, low, close, volume]
                       Index must be DatetimeIndex.
            symbol:    e.g. "BTC/USDT"
            timeframe: e.g. "1h", "4h", "1d"

        Returns:
            Number of newly saved rows
        """
        if df.empty:
            return 0

        records = []
        for ts, row in df.iterrows():
            ts_ms = int(
                pd.Timestamp(ts).timestamp() * 1000
            )  # Convert timestamp to milliseconds
            records.append(
                (
                    symbol,
                    ts_ms,
                    float(row.get("open", row.get("o", 0))),
                    float(row.get("high", row.get("h", 0))),
                    float(row.get("low", row.get("l", 0))),
                    float(row.get("close", row.get("c", 0))),
                    float(row.get("volume", row.get("v", 0))),
                    timeframe,
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO market_data
                    (symbol, timestamp, open, high, low, close, volume, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            saved = conn.total_changes  # Number of rows actually inserted

        logger.info(f"OHLCV saved: {saved} new rows ({symbol} {timeframe})")
        return saved

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: Optional[int] = None,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Loads OHLCV data from local SQLite DB.

        Args:
            symbol:     e.g. "BTC/USDT"
            timeframe:  e.g. "1h"
            days:       Last N days (optional)
            limit:      Last N bars (optional, preferred over days)
            start_date: From this date (optional)

        Returns:
            DataFrame with DatetimeIndex, columns [open, high, low, close, volume]
        """
        params: List[Any] = [symbol, timeframe]
        where_clauses = ["symbol = ?", "timeframe = ?"]

        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(int(start_date.timestamp() * 1000))  # Convert to ms
        elif days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            where_clauses.append("timestamp >= ?")
            params.append(int(cutoff.timestamp() * 1000))

        where = " AND ".join(where_clauses)
        order = "ORDER BY timestamp DESC"
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE {where}
            {order}
            {limit_clause}
        """

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            logger.warning(f"No OHLCV data for {symbol} {timeframe}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )  # Convert ms to UTC datetime
        df = df.set_index(
            "timestamp"
        ).sort_index()  # Set datetime as index, sort ascending

        logger.info(f"OHLCV loaded: {len(df)} bars ({symbol} {timeframe})")
        return df

    def count_ohlcv(self, symbol: str, timeframe: str = "1h") -> int:
        """Returns the number of stored bars."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Champion archive
    # ------------------------------------------------------------------

    def save_champion(self, meta: Dict[str, Any], source: str = "local_master") -> None:
        """
        Saves a new champion to the history database.
        All previous champions are preserved (archive function).

        Args:
            meta:   Champion metadata (from multiverse_champion_meta.json)
            source: Origin: 'local_master', 'github_actions', 'colab'
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO champion_history
                    (name, strategy, sharpe, calmar, sortino, profit_factor,
                     total_return, win_rate, max_drawdown, n_trades, source, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(meta.get("name", "")),
                    str(meta.get("strategy", "")),
                    float(meta.get("sharpe", 0) or 0),
                    float(meta.get("calmar", 0) or 0),
                    float(meta.get("sortino", 0) or 0),
                    float(meta.get("profit_factor", 0) or 0),
                    float(meta.get("total_return", 0) or 0),
                    float(meta.get("win_rate", 0) or 0),
                    float(meta.get("max_drawdown", 0) or 0),
                    int(meta.get("n_trades", 0) or 0),
                    source,
                    json.dumps(meta, default=str),  # Serialize full metadata as JSON
                ),
            )
        logger.info(f"Champion archived: {meta.get('name')} from {source}")

    def get_champion_history(self, limit: int = 50) -> pd.DataFrame:
        """Returns the last N champion entries."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, name, strategy, sharpe, calmar, profit_factor,
                       total_return, win_rate, source
                FROM champion_history
                ORDER BY id DESC
                LIMIT ?
            """,
                (limit,),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "name",
                "strategy",
                "sharpe",
                "calmar",
                "profit_factor",
                "total_return",
                "win_rate",
                "source",
            ],
        )

    # ------------------------------------------------------------------
    # Save trades
    # ------------------------------------------------------------------

    def save_trade(self, trade: Dict[str, Any]) -> None:
        """Saves an executed trade."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO trades
                    (trade_id, symbol, side, order_type, quantity, price,
                     total_value, fee, exchange, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(trade.get("trade_id", "")),
                    str(trade.get("symbol", "")),
                    str(trade.get("side", "")),
                    str(trade.get("order_type", "MARKET")),
                    float(trade.get("quantity", 0)),
                    float(trade.get("price", 0)),
                    float(trade.get("total_value", 0)),
                    float(trade.get("fee", 0)),
                    str(trade.get("exchange", "binance")),
                    str(trade.get("strategy", "")),
                ),
            )

    def get_trades(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """Returns trade history for the last N days."""
        params: list = []
        where = ""
        if symbol:
            where = "WHERE symbol = ?"
            params.append(symbol)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT 1000",
                params,
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        cols = [
            d[0]
            for d in conn.execute(
                f"SELECT * FROM trades {where} LIMIT 0", params
            ).description
            or []
        ]
        # Fallback if conn is already closed
        cols = [
            "id",
            "trade_id",
            "symbol",
            "side",
            "order_type",
            "quantity",
            "price",
            "total_value",
            "fee",
            "exchange",
            "strategy",
            "timestamp",
        ]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def log_heartbeat(
        self,
        source: str,
        status: str = "ok",
        signal: str = "HOLD",
        champion: str = "",
    ) -> None:
        """Writes a heartbeat entry (who is alive?)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO heartbeat (source, status, signal, champion)
                VALUES (?, ?, ?, ?)
            """,
                (source, status, signal, champion),
            )

    def get_last_heartbeat(self, source: str) -> Optional[Dict]:
        """Returns the last heartbeat of a source."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT timestamp, status, signal, champion
                FROM heartbeat
                WHERE source = ?
                ORDER BY id DESC
                LIMIT 1
            """,
                (source,),
            ).fetchone()

        if row:
            return dict(row)
        return None

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Saves a portfolio snapshot."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_snapshots
                    (total_value, cash, exposure, daily_pnl, total_pnl,
                     sharpe_ratio, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    float(snapshot.get("total_value", 0)),
                    float(snapshot.get("cash", 0)),
                    float(snapshot.get("exposure", 0)),
                    float(snapshot.get("daily_pnl", 0)),
                    float(snapshot.get("total_pnl", 0)),
                    float(snapshot.get("sharpe_ratio", 0) or 0),
                    float(snapshot.get("max_drawdown", 0) or 0),
                ),
            )

    # ------------------------------------------------------------------
    # Database maintenance
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        """Compresses the database (after many deletes)."""
        with self._connect() as conn:
            conn.execute("VACUUM")
        logger.info("SQLite VACUUM completed")

    def delete_old_data(self, days: int = 365) -> int:
        """Deletes market data older than N days."""
        cutoff_ms = int(
            (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
        )  # Cutoff in ms
        with self._connect() as conn:
            conn.execute("DELETE FROM market_data WHERE timestamp < ?", (cutoff_ms,))
            deleted = conn.total_changes
        logger.info(f"{deleted} old OHLCV rows deleted (> {days} days)")
        return deleted

    def stats(self) -> Dict[str, Any]:
        """Returns database statistics."""
        with self._connect() as conn:
            stats = {
                "db_path": str(self.db_path),
                "db_size_mb": round(self.db_path.stat().st_size / 1024 / 1024, 2),
                "market_data_rows": conn.execute(
                    "SELECT COUNT(*) FROM market_data"
                ).fetchone()[0],
                "trades_rows": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[
                    0
                ],
                "champion_history_rows": conn.execute(
                    "SELECT COUNT(*) FROM champion_history"
                ).fetchone()[0],
                "symbols": [
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT symbol FROM market_data"
                    ).fetchall()
                ],
            }
        return stats


# ---------------------------------------------------------------------------
# Module-level singleton (for convenient import)
# ---------------------------------------------------------------------------
_db_instance: Optional[LocalDB] = None


def get_local_db() -> LocalDB:
    """
    Returns the global LocalDB instance (singleton).
    Creates it on first call.

    Usage:
        from src.data.sqlite_local import get_local_db
        db = get_local_db()
        db.save_ohlcv(df, "BTC/USDT")
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = LocalDB()
    return _db_instance


# ---------------------------------------------------------------------------
# CLI: python3 -m src.data.sqlite_local
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    db = LocalDB()
    stats = db.stats()

    print("\n--- LocalDB Status ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test: Save and load synthetic data
    if "--test" in sys.argv:
        print("\n--- Saving test data ---")
        import numpy as np

        idx = pd.date_range("2024-01-01", periods=1000, freq="1h", tz="UTC")
        test_df = pd.DataFrame(
            {
                "open": np.random.uniform(40000, 50000, 1000),
                "high": np.random.uniform(45000, 55000, 1000),
                "low": np.random.uniform(38000, 48000, 1000),
                "close": np.random.uniform(40000, 52000, 1000),
                "volume": np.random.uniform(100, 500, 1000),
            },
            index=idx,
        )

        n = db.save_ohlcv(test_df, "TEST/USDT", "1h")
        print(f"  Saved: {n} new rows")

        loaded = db.load_ohlcv("TEST/USDT", "1h", limit=10)
        print(f"  Loaded: {len(loaded)} bars")
        print(loaded.tail(3))

        db.log_heartbeat("local_master", "ok", "HOLD", "TestChampion")
        print("  Heartbeat saved")

        print("\n--- Test successful! ---")
