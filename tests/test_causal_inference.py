"""
Comprehensive Test Suite for Causal Inference Module
======================================================
Tests causal discovery, effect estimation, and counterfactual reasoning.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from src.causal.causal_inference import (
    CausalDiscovery,
    CausalEffectEstimator,
    CounterfactualReasoning,
    CausalTradingStrategy,
    CausalEffect,
    find_causal_drivers,
    estimate_causal_effect,
    analyze_trade_counterfactuals,
)


class TestCausalEffect:
    """Test CausalEffect dataclass."""

    def test_initialization(self):
        """Test CausalEffect creation."""
        effect = CausalEffect(
            treatment="X",
            outcome="Y",
            effect_size=0.5,
            p_value=0.01,
            confidence_interval=(0.3, 0.7),
            method="backdoor_adjustment",
        )

        assert effect.treatment == "X"
        assert effect.outcome == "Y"
        assert effect.effect_size == 0.5
        assert effect.p_value == 0.01
        assert effect.confidence_interval == (0.3, 0.7)
        assert effect.method == "backdoor_adjustment"


class TestCausalDiscovery:
    """Test causal discovery algorithms."""

    @pytest.fixture
    def simple_causal_data(self):
        """Generate simple causal data: X -> Y -> Z."""
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n)
        Y = 2 * X + np.random.randn(n) * 0.1  # Y caused by X
        Z = 1.5 * Y + np.random.randn(n) * 0.1  # Z caused by Y

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    @pytest.fixture
    def complex_causal_data(self):
        """Generate complex causal data with confounding."""
        np.random.seed(42)
        n = 1000

        # Confounder affects both X and Y
        confounder = np.random.randn(n)
        X = confounder + np.random.randn(n) * 0.5  # X influenced by confounder
        Y = 2 * confounder + np.random.randn(n) * 0.5  # Y influenced by same confounder
        Z = X + Y + np.random.randn(n) * 0.1  # Z depends on both X and Y

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z, "confounder": confounder})

    def test_initialization(self):
        """Test CausalDiscovery initialization."""
        discovery = CausalDiscovery(alpha=0.05)
        assert discovery.alpha == 0.05
        assert isinstance(discovery.causal_graph, nx.DiGraph)
        assert discovery.causal_graph.number_of_nodes() == 0  # Graph starts empty

    def test_pc_algorithm_simple(self, simple_causal_data):
        """Test PC algorithm on simple causal structure."""
        discovery = CausalDiscovery(alpha=0.05)
        graph = discovery.pc_algorithm(simple_causal_data, max_cond_vars=1)

        # Should have 3 nodes
        assert graph.number_of_nodes() == 3

        # Should have edges (exact count depends on algorithm)
        assert graph.number_of_edges() >= 2

        # X should cause Y (or be connected)
        assert graph.has_edge("X", "Y") or graph.has_edge("Y", "X")

        # Y should cause Z (or be connected)
        assert graph.has_edge("Y", "Z") or graph.has_edge("Z", "Y")

    def test_conditional_independence_test(self, simple_causal_data):
        """Test conditional independence testing."""
        discovery = CausalDiscovery(alpha=0.05)

        # X and Z should be dependent (marginally)
        independent = discovery._test_conditional_independence(
            simple_causal_data,
            "X",
            "Z",
            [],  # Empty conditioning set (marginal test)
        )
        assert not independent  # X and Z are marginally dependent

        # X and Z should be independent given Y (Y is the mediator)
        independent_given_y = discovery._test_conditional_independence(
            simple_causal_data,
            "X",
            "Z",
            ["Y"],  # Condition on mediator
        )
        assert independent_given_y  # Conditioning on Y should make them independent

    def test_get_causal_parents(self, simple_causal_data):
        """Test retrieving causal parents."""
        discovery = CausalDiscovery()
        discovery.pc_algorithm(simple_causal_data)

        # Get parents of Y
        parents_y = discovery.get_causal_parents("Y")
        assert isinstance(parents_y, list)

        # Get parents of Z
        parents_z = discovery.get_causal_parents("Z")
        assert isinstance(parents_z, list)

    def test_get_causal_children(self, simple_causal_data):
        """Test retrieving causal children."""
        discovery = CausalDiscovery()
        discovery.pc_algorithm(simple_causal_data)

        # Get children of X
        children_x = discovery.get_causal_children("X")
        assert isinstance(children_x, list)

        # Get children of Y
        children_y = discovery.get_causal_children("Y")
        assert isinstance(children_y, list)


class TestCausalEffectEstimator:
    """Test causal effect estimation methods."""

    @pytest.fixture
    def binary_treatment_data(self):
        """Generate data with binary treatment."""
        np.random.seed(42)
        n = 1000

        # Confounder
        Z = np.random.randn(n)

        # Treatment depends on confounder (selection bias)
        treatment_prob = 1 / (1 + np.exp(-Z))  # Logistic probability
        X = (np.random.rand(n) < treatment_prob).astype(int)  # Binary treatment

        # Outcome depends on treatment and confounder
        Y = 2 * X + 1.5 * Z + np.random.randn(n) * 0.5

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    @pytest.fixture
    def iv_data(self):
        """Generate data for instrumental variables."""
        np.random.seed(42)
        n = 1000

        # Instrument (affects treatment but not outcome directly)
        instrument = np.random.randn(n)

        # Unobserved confounder
        U = np.random.randn(n)

        # Treatment depends on instrument and confounder
        X = 1.5 * instrument + 0.5 * U + np.random.randn(n) * 0.3

        # Outcome depends on treatment and confounder
        Y = 2 * X + 1.0 * U + np.random.randn(n) * 0.5

        return pd.DataFrame(
            {
                "instrument": instrument,
                "X": X,
                "Y": Y,
            }
        )

    @pytest.fixture
    def did_data(self):
        """Generate difference-in-differences data."""
        np.random.seed(42)
        n = 1000

        # Treatment group indicator
        group = np.random.randint(0, 2, n)

        # Time period (0 = pre, 1 = post)
        time = np.random.randint(0, 2, n)

        # Outcome with treatment effect
        Y = (
            10  # baseline
            + 2 * group  # group difference (pre-existing)
            + 3 * time  # time trend (shared across groups)
            + 5 * group * time  # treatment effect (DiD interaction term)
            + np.random.randn(n) * 0.5
        )

        return pd.DataFrame(
            {
                "group": group,
                "time": time,
                "Y": Y,
            }
        )

    def test_initialization(self):
        """Test CausalEffectEstimator initialization."""
        estimator = CausalEffectEstimator()
        assert estimator.estimates == []  # Starts with empty estimate history

    def test_backdoor_adjustment(self, binary_treatment_data):
        """Test backdoor adjustment."""
        estimator = CausalEffectEstimator()

        effect = estimator.backdoor_adjustment(
            binary_treatment_data,
            treatment="X",
            outcome="Y",
            confounders=["Z"],  # Block backdoor path through Z
        )

        # Check result structure
        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "X"
        assert effect.outcome == "Y"
        assert effect.method == "backdoor"

        # Effect should be positive or zero (X positively affects Y in test data)
        assert effect.effect_size >= 0

        # Should have confidence interval
        assert len(effect.confidence_interval) == 2
        assert effect.confidence_interval[0] <= effect.confidence_interval[1]

    def test_instrumental_variables(self, iv_data):
        """Test instrumental variables estimation."""
        estimator = CausalEffectEstimator()

        effect = estimator.instrumental_variables(
            iv_data,
            treatment="X",
            outcome="Y",
            instrument="instrument",  # Valid instrument: correlated with X, not Y directly
        )

        # Check result structure
        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "X"
        assert effect.outcome == "Y"
        assert effect.method == "instrumental_variables"
        assert "instrument" in effect.valid_instruments

        # Effect should be positive
        assert effect.effect_size > 0

    def test_difference_in_differences(self, did_data):
        """Test difference-in-differences estimation."""
        estimator = CausalEffectEstimator()

        effect = estimator.difference_in_differences(
            did_data,
            treatment="group",
            outcome="Y",
            time_var="time",
            group_var="group",
        )

        # Check result structure
        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "group"
        assert effect.outcome == "Y"
        assert effect.method == "difference_in_differences"

        # Effect should be positive (we set treatment effect = 5)
        assert effect.effect_size > 0


class TestCounterfactualReasoning:
    """Test counterfactual reasoning capabilities."""

    @pytest.fixture
    def sample_trade_history(self):
        """Generate sample trade history."""
        return pd.DataFrame(
            {
                "entry_time": pd.date_range("2024-01-01", periods=10, freq="D"),
                "exit_time": pd.date_range("2024-01-02", periods=10, freq="D"),
                "entry_price": [50000] * 10,
                "exit_price": [
                    51000,
                    49000,
                    50500,
                    52000,
                    48000,
                    51500,
                    49500,
                    52500,
                    48500,
                    53000,
                ],
                "side": ["buy"] * 10,
                "size": [0.1] * 10,
                "pnl": [100, -100, 50, 200, -200, 150, -50, 250, -150, 300],
            }
        )

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = 50000 + np.cumsum(np.random.randn(30) * 100)
        return pd.DataFrame(
            {
                "close": prices,
            },
            index=dates,
        )

    def test_initialization(self):
        """Test CounterfactualReasoning initialization."""
        reasoner = CounterfactualReasoning()
        assert reasoner.history == []  # Starts with no recorded counterfactuals

    def test_estimate_counterfactual(self):
        """Test counterfactual estimation."""
        reasoner = CounterfactualReasoning()

        # Simple model that predicts outcome from action
        def model(action):
            return {"buy": 10, "sell": -5}[action]  # Predefined payoffs

        result = reasoner.estimate_counterfactual(
            actual_outcome=10,
            actual_action="buy",
            counterfactual_action="sell",
            model=model,
        )

        # Check result structure
        assert result["actual_outcome"] == 10
        assert result["actual_action"] == "buy"
        assert result["counterfactual_action"] == "sell"
        assert result["counterfactual_outcome"] == -5  # What would have happened
        assert result["regret"] == -15  # -5 - 10 = -15 (counterfactual was worse)
        assert result["optimal"] == False  # Actual was better, so no regret

    def test_analyze_trade_regret(self, sample_trade_history, sample_price_data):
        """Test trade regret analysis."""
        reasoner = CounterfactualReasoning()

        regrets = reasoner.analyze_trade_regret(
            sample_trade_history,
            sample_price_data,
            holding_period=5,  # Alternative: hold for 5 extra bars
        )

        # Check result structure
        assert isinstance(regrets, pd.DataFrame)
        assert "trade_id" in regrets.columns
        assert "actual_pnl" in regrets.columns
        assert "regret_hold_longer" in regrets.columns
        assert "regret_opposite" in regrets.columns
        assert "optimal_decision" in regrets.columns

        # Should have one row per trade
        assert len(regrets) == len(sample_trade_history)

    def test_calculate_pnl(self):
        """Test P&L calculation."""
        reasoner = CounterfactualReasoning()

        # Buy trade
        pnl_buy = reasoner._calculate_pnl(50000, 51000, "buy", 0.1)
        assert pnl_buy == 100  # (51000 - 50000) * 0.1

        # Sell trade
        pnl_sell = reasoner._calculate_pnl(50000, 51000, "sell", 0.1)
        assert pnl_sell == -100  # (50000 - 51000) * 0.1 (sell loses when price rises)


class TestCausalTradingStrategy:
    """Test complete causal trading strategy."""

    @pytest.fixture
    def market_data(self):
        """Generate market data for testing."""
        np.random.seed(42)
        n = 1000

        # Features that cause returns
        feature1 = np.random.randn(n)
        feature2 = np.random.randn(n)

        # Returns depend on features (causal relationship)
        returns = 0.5 * feature1 + 0.3 * feature2 + np.random.randn(n) * 0.1

        # Future returns for prediction
        future_returns = np.roll(returns, -1)  # Shift returns back by one period
        future_returns[-1] = 0  # Last element has no future return

        return pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,
                "returns": returns,
                "future_returns": future_returns,
            }
        )

    def test_initialization(self):
        """Test CausalTradingStrategy initialization."""
        strategy = CausalTradingStrategy()
        assert strategy.discovery is not None
        assert strategy.estimator is not None
        assert strategy.counterfactual is not None
        assert (
            strategy.causal_graph is None
        )  # No graph until discover_drivers is called

    def test_discover_drivers(self, market_data):
        """Test causal driver discovery."""
        strategy = CausalTradingStrategy()

        drivers = strategy.discover_drivers(market_data, target="returns")

        # Should return a list
        assert isinstance(drivers, list)

        # Should have discovered causal graph
        assert strategy.causal_graph is not None

    def test_generate_signal(self, market_data):
        """Test signal generation."""
        strategy = CausalTradingStrategy()

        signal = strategy.generate_signal(market_data)

        # Check result structure
        assert "signal" in signal
        assert "confidence" in signal

        # Signal should be -1, 0, or 1
        assert signal["signal"] in [-1, 0, 1]

        # Confidence should be non-negative
        assert signal["confidence"] >= 0

        # If predictors found, should have causal flag
        if "predictors" in signal:
            assert "causal" in signal
            assert signal["causal"] == True


class TestProductionFunctions:
    """Test production convenience functions."""

    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        np.random.seed(42)
        n = 500

        X = np.random.randn(n)
        Z = np.random.randn(n)
        Y = 2 * X + 1.5 * Z + np.random.randn(n) * 0.5  # Y depends on both X and Z

        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_find_causal_drivers(self, test_data):
        """Test find_causal_drivers function."""
        drivers = find_causal_drivers(test_data, target="Y")

        assert isinstance(drivers, list)

    def test_estimate_causal_effect_backdoor(self, test_data):
        """Test estimate_causal_effect with backdoor method."""
        effect = estimate_causal_effect(
            test_data,
            treatment="X",
            outcome="Y",
            method="backdoor",
            confounders=["Z"],  # Control for Z to identify X's effect
        )

        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "X"
        assert effect.outcome == "Y"
        assert effect.method == "backdoor"

    def test_estimate_causal_effect_iv(self, test_data):
        """Test estimate_causal_effect with IV method."""
        effect = estimate_causal_effect(
            test_data,
            treatment="X",
            outcome="Y",
            method="iv",
            confounders=["Z"],  # Z used as instrument here
        )

        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "X"
        assert effect.outcome == "Y"
        assert effect.method == "instrumental_variables"

    def test_estimate_causal_effect_unknown_method(self, test_data):
        """Test estimate_causal_effect with unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_causal_effect(
                test_data,
                treatment="X",
                outcome="Y",
                method="unknown",  # Should raise ValueError
            )

    def test_analyze_trade_counterfactuals(self):
        """Test analyze_trade_counterfactuals function."""
        trade_history = pd.DataFrame(
            {
                "entry_time": pd.date_range("2024-01-01", periods=5, freq="D"),
                "exit_time": pd.date_range("2024-01-02", periods=5, freq="D"),
                "entry_price": [50000] * 5,
                "exit_price": [51000, 49000, 50500, 52000, 48000],
                "side": ["buy"] * 5,
                "size": [0.1] * 5,
                "pnl": [100, -100, 50, 200, -200],
            }
        )

        price_data = pd.DataFrame(
            {
                "close": 50000 + np.cumsum(np.random.randn(20) * 100),
            },
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
        )

        result = analyze_trade_counterfactuals(trade_history, price_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(trade_history)  # One row per trade


class TestCausalInferenceIntegration:
    """Integration tests for the full causal inference pipeline."""

    @pytest.fixture
    def full_pipeline_data(self):
        """Generate data for full pipeline test."""
        np.random.seed(42)
        n = 1000

        # Causal structure: news -> sentiment -> price_change
        news_sentiment = np.random.randn(n)
        market_sentiment = (
            0.8 * news_sentiment + np.random.randn(n) * 0.3
        )  # Influenced by news
        price_change = (
            0.5 * market_sentiment + np.random.randn(n) * 0.2
        )  # Influenced by sentiment

        return pd.DataFrame(
            {
                "news_sentiment": news_sentiment,
                "market_sentiment": market_sentiment,
                "price_change": price_change,
                "volume": np.abs(np.random.randn(n)) * 1000,
            }
        )

    def test_full_causal_pipeline(self, full_pipeline_data):
        """Test complete causal inference workflow."""
        # Step 1: Discover causal structure
        discovery = CausalDiscovery()
        graph = discovery.pc_algorithm(full_pipeline_data)

        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() > 0

        # Step 2: Estimate causal effect
        estimator = CausalEffectEstimator()
        effect = estimator.backdoor_adjustment(
            full_pipeline_data,
            treatment="market_sentiment",
            outcome="price_change",
            confounders=[
                "news_sentiment"
            ],  # Control for news to isolate sentiment effect
        )

        assert isinstance(effect, CausalEffect)
        assert effect.effect_size >= 0  # Non-negative effect

        # Step 3: Counterfactual analysis
        reasoner = CounterfactualReasoning()

        def model(action):
            return {"positive": 1.0, "negative": -1.0}[action]  # Simple outcome model

        counterfactual = reasoner.estimate_counterfactual(
            actual_outcome=1.0,
            actual_action="positive",
            counterfactual_action="negative",
            model=model,
        )

        assert counterfactual["regret"] == -2.0  # -1.0 - 1.0 = -2.0

        # Step 4: Trading strategy
        strategy = CausalTradingStrategy()
        strategy.causal_graph = graph  # Inject discovered graph

        signal = strategy.generate_signal(full_pipeline_data)
        assert "signal" in signal
        assert "confidence" in signal
