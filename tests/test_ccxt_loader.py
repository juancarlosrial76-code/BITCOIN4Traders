"""
Tests for CCXTDataLoader (src/data/ccxt_loader.py)
===================================================
All exchange I/O is mocked. No live network calls.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from src.data.ccxt_loader import CCXTDataLoader, DataLoaderConfig


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def loader_config(tmp_path) -> DataLoaderConfig:
    return DataLoaderConfig(
        exchange_id="binance",
        exchange_type="spot",
        rate_limit_ms=100,
        cache_dir=tmp_path / "cache",
        processed_dir=tmp_path / "processed",
        compression="snappy",
    )


def _make_raw_ohlcv(n: int = 10, start_ms: int = 1_700_000_000_000):
    """Return n synthetic CCXT-format OHLCV rows (list of lists)."""
    rows = []
    for i in range(n):
        ts = start_ms + i * 3_600_000  # 1-hour intervals
        rows.append([ts, 50000.0 + i, 51000.0 + i, 49000.0 + i, 50500.0 + i, 1000.0 + i])
    return rows


@pytest.fixture
def mock_exchange():
    exch = MagicMock()
    exch.load_markets.return_value = {}
    exch.fetch_ohlcv.return_value = _make_raw_ohlcv(10)
    return exch


@pytest.fixture
def loader(loader_config, mock_exchange):
    """Return a CCXTDataLoader with the real exchange replaced by a mock."""
    with patch("ccxt.binance", return_value=mock_exchange):
        inst = CCXTDataLoader(loader_config)
    inst.exchange = mock_exchange
    return inst


# ─────────────────────────────────────────────
#  _ohlcv_to_dataframe
# ─────────────────────────────────────────────


class TestOhlcvToDataFrame:
    def test_correct_columns(self, loader):
        raw = _make_raw_ohlcv(5)
        df = loader._ohlcv_to_dataframe(raw)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_float64_dtypes(self, loader):
        raw = _make_raw_ohlcv(5)
        df = loader._ohlcv_to_dataframe(raw)
        for col in ("open", "high", "low", "close", "volume"):
            assert df[col].dtype == np.float64, f"{col} should be float64"

    def test_datetimeindex(self, loader):
        raw = _make_raw_ohlcv(5)
        df = loader._ohlcv_to_dataframe(raw)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_row_count_matches(self, loader):
        raw = _make_raw_ohlcv(7)
        df = loader._ohlcv_to_dataframe(raw)
        assert len(df) == 7

    def test_no_duplicate_index(self, loader):
        raw = _make_raw_ohlcv(5)
        raw_dup = raw + [raw[-1]]  # add a duplicate
        df = loader._ohlcv_to_dataframe(raw_dup)
        assert df.index.is_unique


# ─────────────────────────────────────────────
#  _get_cache_path
# ─────────────────────────────────────────────


class TestGetCachePath:
    def test_deterministic_same_args_same_path(self, loader):
        p1 = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", None)
        p2 = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", None)
        assert p1 == p2

    def test_different_symbol_different_path(self, loader):
        p_btc = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", None)
        p_eth = loader._get_cache_path("ETH/USDT", "1h", "2023-01-01", None)
        assert p_btc != p_eth

    def test_different_start_date_different_path(self, loader):
        p1 = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", None)
        p2 = loader._get_cache_path("BTC/USDT", "1h", "2022-01-01", None)
        assert p1 != p2

    def test_end_date_changes_path(self, loader):
        p1 = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", None)
        p2 = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", "2023-12-31")
        assert p1 != p2

    def test_returns_path_object(self, loader):
        p = loader._get_cache_path("BTC/USDT", "1h", "2023-01-01", None)
        assert isinstance(p, Path)


# ─────────────────────────────────────────────
#  _save_to_cache / load_local roundtrip
# ─────────────────────────────────────────────


class TestCacheRoundtrip:
    def test_save_and_load_equals_original(self, loader, tmp_path):
        raw = _make_raw_ohlcv(20)
        df_orig = loader._ohlcv_to_dataframe(raw)
        cache_path = tmp_path / "cache" / "test_roundtrip.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        loader._save_to_cache(df_orig, cache_path)

        df_loaded = pd.read_parquet(cache_path)
        pd.testing.assert_frame_equal(df_orig, df_loaded)

    def test_load_local_returns_same_data(self, loader):
        raw = _make_raw_ohlcv(15)
        df_orig = loader._ohlcv_to_dataframe(raw)
        cache_path = loader._get_cache_path("BTC/USDT", "1h", "2020-01-01", None)
        loader._save_to_cache(df_orig, cache_path)

        df_loaded = loader.load_local("BTC/USDT", "1h", "2020-01-01", None)
        pd.testing.assert_frame_equal(df_orig, df_loaded)

    def test_load_local_raises_if_no_cache(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_local("ETH/USDT", "4h", "2099-01-01", None)


# ─────────────────────────────────────────────
#  download_and_cache mocking
# ─────────────────────────────────────────────


class TestDownloadAndCache:
    def test_calls_fetch_ohlcv_once_on_first_download(self, loader):
        loader.exchange.fetch_ohlcv.return_value = _make_raw_ohlcv(5)
        df = loader.download_and_cache("BTC/USDT", "1h", "2023-01-01", "2023-01-10")
        assert loader.exchange.fetch_ohlcv.called

    def test_second_call_uses_cache(self, loader):
        loader.exchange.fetch_ohlcv.return_value = _make_raw_ohlcv(5)
        # First download
        loader.download_and_cache("BTC/USDT", "1h", "2023-02-01", "2023-02-05")
        call_count_first = loader.exchange.fetch_ohlcv.call_count
        # Second call — must NOT call exchange again
        loader.download_and_cache("BTC/USDT", "1h", "2023-02-01", "2023-02-05")
        assert loader.exchange.fetch_ohlcv.call_count == call_count_first

    def test_force_refresh_re_downloads(self, loader):
        loader.exchange.fetch_ohlcv.return_value = _make_raw_ohlcv(5)
        loader.download_and_cache("BTC/USDT", "1h", "2023-03-01", "2023-03-05")
        count_after_first = loader.exchange.fetch_ohlcv.call_count
        loader.download_and_cache(
            "BTC/USDT", "1h", "2023-03-01", "2023-03-05", force_refresh=True
        )
        assert loader.exchange.fetch_ohlcv.call_count > count_after_first

    def test_returns_dataframe_with_correct_columns(self, loader):
        loader.exchange.fetch_ohlcv.return_value = _make_raw_ohlcv(8)
        df = loader.download_and_cache("BTC/USDT", "1h", "2023-04-01", "2023-04-10")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns
