"""
Database Models and Persistence Layer
=====================================
PostgreSQL storage for trades, orders, positions, and market data.

This module provides the data persistence layer for the trading system using
SQLAlchemy ORM with PostgreSQL. It defines all database models and provides
a DatabaseManager class for common CRUD operations.

Database Configuration:
    Connection is configured via the DATABASE_URL environment variable.
    Set it in your .env file (see .env.example for the format):
        DATABASE_URL=postgresql://user:password@host:port/dbname

    Example:
        DATABASE_URL=postgresql://trader:securepass@localhost:5432/trading_db

Database Models:
    Trade: Executed trade records with fees and strategy attribution
    Order: Order history with fills and status tracking
    Position: Position snapshots for portfolio tracking
    MarketData: OHLCV candlestick data for backtesting
    PortfolioSnapshot: Historical portfolio performance metrics

Key Features:
    - SQLAlchemy ORM for database-agnostic queries
    - Automatic table creation with create_tables()
    - Efficient bulk operations for market data
    - Pandas DataFrame integration for analysis
    - Automatic timestamps with server-side defaults
    - Index optimization for common queries

Usage:
    from src.data.database import DatabaseManager, init_database

    # Initialize database with tables
    db = init_database()

    # Save a trade
    db.save_trade({
        'trade_id': 'trade_001',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 0.001,
        'price': 50000,
        'total_value': 50,
        'fee': 0.05,
        'exchange': 'binance',
    })

    # Query market data
    df = db.get_market_data(
        symbol='BTCUSDT',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31)
    )

Performance Considerations:
    - Use bulk_save_objects() for batch inserts (100x faster than individual)
    - Indexes are created automatically on foreign keys and common filters
    - Consider partitioning for large market_data tables (>10M rows)
    - Use connection pooling for high-throughput applications

Dependencies:
    - sqlalchemy: Database ORM
    - psycopg2: PostgreSQL driver
    - pandas: DataFrame support
    - loguru: Logging

Warning:
    Ensure DATABASE_URL is set before using DatabaseManager. Operations will
    fail gracefully but won't work without a valid connection string.
"""

import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import pandas as pd
from loguru import logger

# Database configuration — read from environment, never hardcoded
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    logger.warning(
        "DATABASE_URL environment variable is not set. "
        "Database operations will fail. "
        "Add DATABASE_URL to your .env file (see .env.example)."
    )
    DATABASE_URL = ""  # placeholder so module loads without crash

Base = declarative_base()  # ORM base class; all models inherit from this
engine = create_engine(DATABASE_URL)  # SQLAlchemy engine: manages connection pool
SessionLocal = sessionmaker(
    bind=engine
)  # Session factory; call SessionLocal() to get a session


class Trade(Base):
    """Store executed trades."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # BUY or SELL
    order_type = Column(String)  # MARKET, LIMIT, etc.
    quantity = Column(Float)
    price = Column(Float)
    total_value = Column(Float)
    fee = Column(Float)
    exchange = Column(String)
    strategy = Column(String, nullable=True)
    timestamp = Column(
        DateTime, default=func.now()
    )  # server-side default: DB sets current time on insert

    def to_dict(self):
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "total_value": self.total_value,
            "fee": self.fee,
            "exchange": self.exchange,
            "timestamp": self.timestamp,
        }


class Order(Base):
    """Store order history."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    order_type = Column(String)
    quantity = Column(Float)
    price = Column(Float, nullable=True)
    status = Column(String)  # PENDING, FILLED, CANCELED
    filled_qty = Column(Float, default=0.0)
    avg_price = Column(Float, nullable=True)
    exchange = Column(String)
    created_at = Column(DateTime, default=func.now())  # set once on insert
    updated_at = Column(DateTime, onupdate=func.now())  # auto-updated on every UPDATE

    def to_dict(self):
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "quantity": self.quantity,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
        }


class Position(Base):
    """Store position snapshots."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # LONG or SHORT
    size = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    timestamp = Column(DateTime, default=func.now())

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
        }


class MarketData(Base):
    """Store OHLCV market data."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    timeframe = Column(String)  # 1m, 5m, 1h, 1d, etc.

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class PortfolioSnapshot(Base):
    """Store portfolio value over time."""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    total_value = Column(Float)
    cash = Column(Float)
    exposure = Column(Float)
    margin_used = Column(Float)
    daily_pnl = Column(Float)
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)


class DatabaseManager:
    """
    Database operations manager for trading system persistence.

    Provides a high-level interface for all database operations including
    trade recording, order history, market data storage, and portfolio
    tracking. Uses SQLAlchemy ORM for database-agnostic operations.

    Attributes:
        engine: SQLAlchemy engine instance for database connections
        SessionLocal: Session factory for creating database sessions

    Database Tables:
        - trades: Executed trade records
        - orders: Order history with status tracking
        - positions: Current position snapshots
        - market_data: OHLCV candlestick data
        - portfolio_snapshots: Historical portfolio performance

    Connection Management:
        Creates a connection pool automatically. For production, consider
        adjusting pool_size and max_overflow parameters in create_engine.

    Example:
        >>> db = DatabaseManager()  # Uses DATABASE_URL env var
        >>> db.create_tables()  # Create all tables

        >>> # Save market data
        >>> df = pd.DataFrame({
        ...     'open': [50000, 50100],
        ...     'high': [50200, 50300],
        ...     'low': [49900, 50000],
        ...     'close': [50100, 50200],
        ...     'volume': [1000, 1100]
        ... }, index=pd.date_range('2024-01-01', periods=2, freq='1h'))
        >>> db.save_market_data(df, 'BTCUSDT', '1h')

        >>> # Query trades
        >>> trades = db.get_trades(symbol='BTCUSDT')
        >>> print(trades.head())

    Note:
        All methods handle their own session lifecycle - no need to manage
        sessions manually. Sessions are created, used, and closed within
        each method call.
    """

    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("DatabaseManager initialized")

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.success("Database tables created")

    def get_session(self):
        """Get database session."""
        return self.SessionLocal()

    def save_trade(self, trade_data: dict):
        """Save a trade to database."""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            logger.info(f"Trade saved: {trade_data.get('trade_id')}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trade: {e}")
        finally:
            session.close()

    def save_order(self, order_data: dict):
        """Save an order to database."""
        session = self.get_session()
        try:
            order = Order(**order_data)
            session.add(order)
            session.commit()
            logger.info(f"Order saved: {order_data.get('order_id')}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving order: {e}")
        finally:
            session.close()

    def save_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save market data from DataFrame."""
        session = self.get_session()
        try:
            records = []
            for index, row in df.iterrows():
                records.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=index
                        if isinstance(index, datetime)
                        else pd.to_datetime(index),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        timeframe=timeframe,
                    )
                )

            session.bulk_save_objects(
                records
            )  # batch INSERT: much faster than individual session.add() calls
            session.commit()  # flush all pending INSERT statements to the DB atomically
            logger.success(f"Saved {len(records)} records for {symbol}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving market data: {e}")
        finally:
            session.close()

    def get_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime = None
    ) -> pd.DataFrame:
        """Retrieve market data from database."""
        session = self.get_session()
        try:
            query = session.query(MarketData).filter(
                MarketData.symbol == symbol, MarketData.timestamp >= start_date
            )  # build SQLAlchemy query with mandatory symbol + start_date filters

            if end_date:
                query = query.filter(
                    MarketData.timestamp <= end_date
                )  # optional upper-bound filter

            results = query.order_by(
                MarketData.timestamp
            ).all()  # execute query; returns list of ORM objects

            if not results:
                return (
                    pd.DataFrame()
                )  # return empty frame rather than None to avoid caller type errors

            df = pd.DataFrame(
                [r.to_dict() for r in results]
            )  # convert ORM rows to plain dicts, then DataFrame
            df.set_index(
                "timestamp", inplace=True
            )  # use timestamp as index for time-series operations
            return df

        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def get_trades(
        self, symbol: str = None, start_date: datetime = None
    ) -> pd.DataFrame:
        """Get trade history."""
        session = self.get_session()
        try:
            query = session.query(Trade)

            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if start_date:
                query = query.filter(Trade.timestamp >= start_date)

            trades = query.order_by(Trade.timestamp.desc()).all()

            if not trades:
                return pd.DataFrame()

            df = pd.DataFrame([t.to_dict() for t in trades])
            return df

        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio performance history."""
        session = self.get_session()
        try:
            start_date = datetime.now() - pd.Timedelta(days=days)

            snapshots = (
                session.query(PortfolioSnapshot)
                .filter(PortfolioSnapshot.timestamp >= start_date)
                .order_by(PortfolioSnapshot.timestamp)
                .all()
            )

            if not snapshots:
                return pd.DataFrame()

            data = [
                {
                    "timestamp": s.timestamp,
                    "total_value": s.total_value,
                    "daily_pnl": s.daily_pnl,
                    "total_pnl": s.total_pnl,
                    "sharpe_ratio": s.sharpe_ratio,
                    "max_drawdown": s.max_drawdown,
                }
                for s in snapshots
            ]

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error retrieving portfolio history: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def delete_old_data(self, days: int = 365):
        """Delete data older than specified days."""
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days)

            # Delete old market data
            session.query(MarketData).filter(
                MarketData.timestamp < cutoff_date
            ).delete()

            # Delete old portfolio snapshots
            session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.timestamp < cutoff_date
            ).delete()

            session.commit()
            logger.info(f"Deleted data older than {days} days")
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting old data: {e}")
        finally:
            session.close()


def init_database():
    """Initialize database with tables."""
    db = DatabaseManager()
    db.create_tables()
    return db


if __name__ == "__main__":
    # Initialize database
    print("Initializing database...")
    db = init_database()
    print("Database initialized successfully!")
    url_display = DATABASE_URL.split("@")[-1] if DATABASE_URL else "(not configured)"
    print(f"Connected to: {url_display}")
