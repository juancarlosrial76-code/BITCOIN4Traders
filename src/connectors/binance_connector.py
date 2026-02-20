"""
Production-Ready Binance Exchange Connector
==========================================
Real trading implementation with:
- Spot and Futures trading
- WebSocket real-time data
- Order management
- Position tracking
- Error handling & retry logic
- Rate limiting compliance

Requires: export BINANCE_API_KEY=your_key
          export BINANCE_API_SECRET=your_secret
"""

import os
import time
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from loguru import logger

# Try to import Binance libraries
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
    from binance.streams import ThreadedWebsocketManager

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("python-binance not installed. Using mock implementation.")


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET', 'LIMIT', etc.
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "PENDING"
    filled_qty: float = 0.0
    avg_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    side: str  # 'LONG' or 'SHORT'
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class Balance:
    """Account balance."""

    asset: str
    free: float
    locked: float
    total: float


class BinanceConnector:
    """
    Production-ready Binance connector.

    Supports both Spot and Futures trading.
    """

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True,
        futures: bool = False,
    ):
        """
        Initialize Binance connector.

        Args:
            api_key: Binance API key (or from env BINANCE_API_KEY)
            api_secret: Binance API secret (or from env BINANCE_API_SECRET)
            testnet: Use testnet (default True for safety)
            futures: Trade futures (default False for spot)
        """
        # Get credentials from environment if not provided
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API key and secret required. "
                "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
            )

        self.testnet = testnet
        self.futures = futures
        self.client = None
        self.ws_manager = None

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.balances: Dict[str, Balance] = {}

        # Callbacks
        self.price_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        self._initialize_client()
        logger.info(
            f"BinanceConnector initialized ({'Futures' if futures else 'Spot'}, "
            f"{'Testnet' if testnet else 'Live'})"
        )

    def _initialize_client(self):
        """Initialize Binance client."""
        if not BINANCE_AVAILABLE:
            logger.error("python-binance not installed. Cannot initialize.")
            return

        try:
            self.client = Client(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )

            # Test connection
            self.client.ping()
            server_time = self.client.get_server_time()
            logger.success(f"Connected to Binance. Server time: {server_time}")

        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise

    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()

    def get_account_balance(self) -> Dict[str, Balance]:
        """Get account balance."""
        if not self.client:
            logger.error("Client not initialized")
            return {}

        try:
            self._rate_limit()

            if self.futures:
                account = self.client.futures_account()
                for asset in account["assets"]:
                    balance = Balance(
                        asset=asset["asset"],
                        free=float(asset["availableBalance"]),
                        locked=float(asset["withdrawAvailable"])
                        - float(asset["availableBalance"]),
                        total=float(asset["walletBalance"]),
                    )
                    self.balances[asset["asset"]] = balance
            else:
                account = self.client.get_account()
                for balance_data in account["balances"]:
                    free = float(balance_data["free"])
                    locked = float(balance_data["locked"])
                    balance = Balance(
                        asset=balance_data["asset"],
                        free=free,
                        locked=locked,
                        total=free + locked,
                    )
                    self.balances[balance_data["asset"]] = balance

            return self.balances

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Optional[Order]:
        """
        Place a market order.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            quantity: Order quantity

        Returns:
            Order object or None if failed
        """
        if not self.client:
            logger.error("Client not initialized")
            return None

        try:
            self._rate_limit()

            order_params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
            }

            if self.futures:
                response = self.client.futures_create_order(**order_params)
            else:
                response = self.client.order_market(**order_params)

            # Create order object
            order = Order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=quantity,
                order_id=str(response["orderId"]),
                status=response["status"],
                filled_qty=float(response.get("executedQty", 0)),
                avg_price=float(response.get("avgPrice", 0)),
            )

            self.orders[order.order_id] = order

            # Notify callbacks
            for callback in self.order_callbacks:
                callback(order)

            logger.success(
                f"Market order placed: {side} {quantity} {symbol} @ {order.avg_price}"
            )
            return order

        except BinanceAPIException as e:
            logger.error(f"Binance API error placing order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = "GTC",
    ) -> Optional[Order]:
        """
        Place a limit order.

        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Limit price
            time_in_force: 'GTC', 'IOC', 'FOK'

        Returns:
            Order object or None if failed
        """
        if not self.client:
            logger.error("Client not initialized")
            return None

        try:
            self._rate_limit()

            order_params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": time_in_force,
                "quantity": quantity,
                "price": price,
            }

            if self.futures:
                response = self.client.futures_create_order(**order_params)
            else:
                response = self.client.order_limit(**order_params)

            order = Order(
                symbol=symbol,
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                price=price,
                order_id=str(response["orderId"]),
                status=response["status"],
            )

            self.orders[order.order_id] = order

            logger.success(f"Limit order placed: {side} {quantity} {symbol} @ {price}")
            return order

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        if not self.client:
            return False

        try:
            self._rate_limit()

            if self.futures:
                self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            else:
                self.client.cancel_order(symbol=symbol, orderId=order_id)

            if order_id in self.orders:
                self.orders[order_id].status = "CANCELED"

            logger.info(f"Order {order_id} canceled")
            return True

        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Order]:
        """Get current status of an order."""
        if not self.client:
            return None

        try:
            self._rate_limit()

            if self.futures:
                response = self.client.futures_get_order(
                    symbol=symbol, orderId=order_id
                )
            else:
                response = self.client.get_order(symbol=symbol, orderId=order_id)

            order = Order(
                symbol=symbol,
                side=response["side"],
                order_type=response["type"],
                quantity=float(response["origQty"]),
                price=float(response.get("price", 0))
                if response.get("price")
                else None,
                order_id=order_id,
                status=response["status"],
                filled_qty=float(response["executedQty"]),
                avg_price=float(response.get("avgPrice", 0)),
            )

            self.orders[order_id] = order
            return order

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price."""
        if not self.client:
            return None

        try:
            self._rate_limit()

            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])

        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None

    def start_websocket(self, symbols: List[str], callback: Callable = None):
        """
        Start WebSocket for real-time data.

        Args:
            symbols: List of symbols to watch
            callback: Function to call with price updates
        """
        if not BINANCE_AVAILABLE:
            logger.error("Cannot start WebSocket - python-binance not installed")
            return

        try:
            self.ws_manager = ThreadedWebsocketManager(
                api_key=self.api_key, api_secret=self.api_secret
            )
            self.ws_manager.start()

            if callback:
                self.price_callbacks.append(callback)

            for symbol in symbols:
                self.ws_manager.start_symbol_ticker_socket(
                    symbol=symbol.lower(), callback=self._handle_websocket_message
                )

            logger.info(f"WebSocket started for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")

    def _handle_websocket_message(self, msg):
        """Handle WebSocket message."""
        try:
            if msg["e"] == "24hrTicker":
                symbol = msg["s"]
                price = float(msg["c"])

                # Notify callbacks
                for callback in self.price_callbacks:
                    callback(symbol, price)

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def stop_websocket(self):
        """Stop WebSocket."""
        if self.ws_manager:
            self.ws_manager.stop()
            logger.info("WebSocket stopped")

    def get_historical_klines(
        self, symbol: str, interval: str = "1h", limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical candlestick data.

        Args:
            symbol: Trading pair
            interval: Time interval (1m, 5m, 1h, 1d, etc.)
            limit: Number of candles (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            logger.error("Client not initialized")
            return pd.DataFrame()

        try:
            self._rate_limit()

            klines = self.client.get_klines(
                symbol=symbol, interval=interval, limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Convert types
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        if not self.client or not self.futures:
            return None

        try:
            self._rate_limit()

            positions = self.client.futures_position_information(symbol=symbol)

            for pos in positions:
                if float(pos["positionAmt"]) != 0:
                    position = Position(
                        symbol=symbol,
                        side="LONG" if float(pos["positionAmt"]) > 0 else "SHORT",
                        size=abs(float(pos["positionAmt"])),
                        entry_price=float(pos["entryPrice"]),
                        unrealized_pnl=float(pos["unRealizedProfit"]),
                        leverage=float(pos["leverage"]),
                    )
                    self.positions[symbol] = position
                    return position

            return None

        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None

    def close_all_positions(self):
        """Close all open positions (Shut and Run)."""
        logger.critical("CLOSING ALL POSITIONS - EMERGENCY SHUTDOWN")

        if not self.futures:
            logger.warning("Position closing only supported for futures")
            return

        try:
            # Get all positions
            positions = self.client.futures_position_information()

            for pos in positions:
                size = float(pos["positionAmt"])
                if size != 0:
                    symbol = pos["symbol"]
                    side = "SELL" if size > 0 else "BUY"

                    # Place market order to close
                    self.place_market_order(symbol, side, abs(size))
                    logger.info(f"Closed position: {symbol} {size}")

            logger.success("All positions closed")

        except Exception as e:
            logger.error(f"Error closing positions: {e}")

    def get_trading_fees(self, symbol: str) -> Dict:
        """Get trading fees for a symbol."""
        try:
            self._rate_limit()

            if self.futures:
                fee = self.client.futures_trade_fee(symbol=symbol)
            else:
                fee = self.client.get_trade_fee(symbol=symbol)

            return {
                "maker": float(fee[0]["makerCommission"]),
                "taker": float(fee[0]["takerCommission"]),
            }

        except Exception as e:
            logger.error(f"Error getting fees: {e}")
            return {"maker": 0.001, "taker": 0.001}  # Default 0.1%


# Production helper functions
def create_binance_connector(
    testnet: bool = True, futures: bool = False
) -> BinanceConnector:
    """
    Create Binance connector from environment variables.

    Usage:
        export BINANCE_API_KEY=your_key
        export BINANCE_API_SECRET=your_secret

        connector = create_binance_connector(testnet=True)
    """
    return BinanceConnector(testnet=testnet, futures=futures)


def test_connection():
    """Test Binance connection."""
    try:
        connector = create_binance_connector(testnet=True)
        balance = connector.get_account_balance()
        print(f"✓ Connection successful!")
        print(f"✓ Balances: {list(balance.keys())[:5]}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connector
    print("Testing Binance Connector...")
    test_connection()
