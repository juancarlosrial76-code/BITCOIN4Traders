"""
ANTI-BIAS FRAMEWORK – Integration Tests
========================================
Testet alle Kernmodule im Kontext des complete_trading_system.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.validation.antibias_walkforward import (
    WalkForwardConfig,
    FoldSplit,
    PurgedWalkForwardCV,
    PurgedScaler,
    LeakDetector,
)
from src.costs.antibias_costs import (
    MarketType,
    Timeframe,
    OrderType,
    CostConfig,
    TransactionCostEngine,
    BreakEvenAnalyzer,
)
from src.reward.antibias_rewards import (
    SharpeIncrementReward,
    CalmarIncrementReward,
    CostAwareReward,
    RegimeAwareReward,
    RegimeState,
    RewardAnalyzer,
)
from src.evaluation.antibias_validator import (
    BacktestValidator,
    DeflatedSharpeRatio,
    MinTrackRecordLength,
    PermutationTest,
    compute_metrics,
)

RNG = np.random.default_rng(42)


def make_returns(n: int = 500, sharpe: float = 0.5) -> np.ndarray:
    """Erzeuge Returns mit gegebenem Sharpe."""
    mu = sharpe / np.sqrt(252)
    sig = 1 / np.sqrt(252)
    return RNG.normal(mu, sig, n).astype(np.float32)


def make_positions(n: int = 500, flip_prob: float = 0.3) -> np.ndarray:
    """Erzeuge realistische Position-Serie."""
    pos = np.ones(n)
    for i in range(1, n):
        if RNG.random() < flip_prob:
            pos[i] = RNG.choice([-1, 0, 1])
        else:
            pos[i] = pos[i - 1]
    return pos


class TestPurgedWalkForwardCV:
    """Test Purged Walk-Forward CV integration."""

    def test_no_overlap_between_train_and_test(self):
        cv = PurgedWalkForwardCV(WalkForwardConfig(n_splits=4, holdout_pct=0.1))
        folds, holdout = cv.split(1000)
        for fold in folds:
            train_set = set(fold.train_idx.tolist())
            test_set = set(fold.test_idx.tolist())
            assert train_set.isdisjoint(test_set), (
                f"Fold {fold.fold_id}: train/test overlap!"
            )

    def test_holdout_is_last_n_bars(self):
        cfg = WalkForwardConfig(holdout_pct=0.20, n_splits=3)
        cv = PurgedWalkForwardCV(cfg)
        folds, holdout = cv.split(1000)
        assert holdout[-1] == 999
        assert holdout[0] == 800

    def test_purging_removes_leaky_samples(self):
        cfg = WalkForwardConfig(n_splits=3, feature_lookback=50, purge=True)
        cv = PurgedWalkForwardCV(cfg)
        folds, _ = cv.split(1000)
        for fold in folds:
            test_start = fold.test_idx[0]
            assert not any(
                t >= test_start - cfg.feature_lookback for t in fold.train_idx
            ), f"Fold {fold.fold_id}: leaky train sample!"


class TestPurgedScaler:
    """Test Purged Scaler integration."""

    def test_fit_transform_consistent(self):
        X_train = RNG.normal(0, 1, (200, 10))
        X_test = RNG.normal(0, 1, (50, 10))
        scaler = PurgedScaler("zscore")
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        np.testing.assert_allclose(X_tr_sc.mean(axis=0), np.zeros(10), atol=0.1)

    def test_no_refitting_on_test(self):
        X_train = RNG.normal(100, 10, (200, 5))
        X_test = RNG.normal(0, 1, (50, 5))
        scaler = PurgedScaler("zscore")
        scaler.fit(X_train)
        X_te_sc = scaler.transform(X_test)
        assert X_te_sc.mean() <= -4.9


class TestTransactionCosts:
    """Test Transaction Cost Engine integration."""

    def test_futures_cheaper_than_spot_taker(self):
        spot = TransactionCostEngine(
            CostConfig(market_type=MarketType.SPOT, order_type=OrderType.MARKET)
        )
        fut = TransactionCostEngine(
            CostConfig(market_type=MarketType.FUTURES, order_type=OrderType.MARKET)
        )
        c_spot = spot.total_cost(30000, 0.1, adv=1e9).fee_one_way
        c_fut = fut.total_cost(30000, 0.1, adv=1e9).fee_one_way
        assert c_fut < c_spot

    def test_limit_cheaper_than_market(self):
        eng_mkt = TransactionCostEngine(CostConfig(order_type=OrderType.MARKET))
        eng_lmt = TransactionCostEngine(CostConfig(order_type=OrderType.LIMIT))
        c_mkt = eng_mkt.total_cost(30000, 0.1).fee_one_way
        c_lmt = eng_lmt.total_cost(30000, 0.1).fee_one_way
        assert c_lmt <= c_mkt

    def test_futures_have_funding(self):
        eng = TransactionCostEngine(
            CostConfig(
                market_type=MarketType.FUTURES, enable_funding=True, holding_bars=8
            )
        )
        cost = eng.total_cost(30000, 0.1)
        assert cost.funding_holding > 0


class TestRewardFunctions:
    """Test Reward Function integration."""

    def _base_kwargs(self, equity=10000.0):
        return dict(
            pnl=10.0, position=1.0, prev_position=0.0, equity=equity, cost_this_bar=5.0
        )

    def test_sharpe_reward_clipped(self):
        r = SharpeIncrementReward(window=20)
        for _ in range(30):
            rew = r.compute(**self._base_kwargs())
        assert -3 <= rew <= 3

    def test_cost_aware_penalizes_churning(self):
        r_churn = CostAwareReward(lambda_cost=5.0)
        r_hold = CostAwareReward(lambda_cost=5.0)
        rew_churn = r_churn.compute(
            pnl=10, position=1.0, prev_position=-1.0, equity=10000, cost_this_bar=20
        )
        rew_hold = r_hold.compute(
            pnl=10, position=1.0, prev_position=1.0, equity=10000, cost_this_bar=0
        )
        assert rew_hold > rew_churn

    def test_regime_aware_bonus_congruent(self):
        r = RegimeAwareReward(lambda_regime=1.0)
        bull = RegimeState(regime=2, vol_regime=0, trend_strength=0.8)
        r.set_regime(bull)
        rew_long = r.compute(
            pnl=10, position=1.0, prev_position=0.0, equity=10000, cost_this_bar=0
        )
        r.reset()
        r.set_regime(bull)
        rew_short = r.compute(
            pnl=10, position=-1.0, prev_position=0.0, equity=10000, cost_this_bar=0
        )
        assert rew_long > rew_short


class TestValidator:
    """Test Validator integration."""

    def test_compute_metrics_good_strategy(self):
        rets = make_returns(500, sharpe=1.5)
        m = compute_metrics(rets)
        assert m.sharpe > 0
        assert 0 <= m.win_rate <= 1
        assert m.max_drawdown >= 0

    def test_permutation_test_random_strategy(self):
        rets = RNG.normal(0, 0.01, 500)
        pos = RNG.choice([-1, 0, 1], 500)
        perm = PermutationTest(n_permutations=200)
        res = perm.test(rets, pos, "sharpe")
        assert not res.is_significant or res.p_value > 0.01

    def test_dsr_low_for_random(self):
        rets = RNG.normal(0, 0.01, 300)
        dsr_res = DeflatedSharpeRatio.compute(rets, n_trials=50)
        assert dsr_res.dsr < 0.9

    def test_mtrl_higher_sr_needs_less_data(self):
        high = MinTrackRecordLength.compute(sharpe=2.0)
        low = MinTrackRecordLength.compute(sharpe=0.5)
        assert high.n_min < low.n_min

    def test_full_validator_runs(self):
        rets = make_returns(600, sharpe=1.0)
        pos = np.ones(600)
        v = BacktestValidator(n_cpcv_splits=4, n_permutations=100)
        rep = v.validate(rets, pos)
        assert hasattr(rep, "passes_all")
        assert hasattr(rep, "cpcv")
        assert hasattr(rep, "perm")
        assert hasattr(rep, "dsr")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
