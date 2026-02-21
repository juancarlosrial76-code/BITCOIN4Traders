"""
Database Models and Persistence Layer
======================================
PostgreSQL storage for trades, orders, and market data.

Connection is configured via the DATABASE_URL environment variable.
Set it in your .env file (see .env.example for the format):
  DATABASE_URL=postgresql://user:password@host:port/dbname
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

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


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
    timestamp = Column(DateTime, default=func.now())

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
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

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
    """Manage database operations."""

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
                        timestamp=index if isinstance(index, datetime) else pd.to_datetime(index),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        timeframe=timeframe,
                    )
                )

            session.bulk_save_objects(records)
            session.commit()
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
            )

            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)

            results = query.order_by(MarketData.timestamp).all()

            if not results:
                return pd.DataFrame()

            df = pd.DataFrame([r.to_dict() for r in results])
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def get_trades(self, symbol: str = None, start_date: datetime = None) -> pd.DataFrame:
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
            session.query(MarketData).filter(MarketData.timestamp < cutoff_date).delete()

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
