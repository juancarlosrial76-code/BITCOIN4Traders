"""
Pytest Configuration
====================

Shared fixtures and configuration for all tests.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root and src to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    import torch

    np.random.seed(42)
    torch.manual_seed(42)
    yield


# ---------------------------------------------------------------------------
# Shared fixtures used across multiple test modules
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_returns():
    """Generate sample multi-asset returns data."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 3) * 0.02,
        columns=["BTC", "ETH", "SOL"],
    )


@pytest.fixture
def config():
    """Provide a default TransformerConfig for edge-case tests."""
    from src.transformer.trading_transformer import TransformerConfig

    return TransformerConfig(
        input_dim=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_seq_len=100,
        output_dim=3,
    )


@pytest.fixture
def sample_input():
    """Provide a default sample tensor input for transformer tests."""
    import torch

    return torch.randn(4, 50, 10)  # batch=4, seq=50, features=10
