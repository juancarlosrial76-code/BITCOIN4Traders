from typing import Tuple

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator


@njit(cache=True)
def _kernel_simulate(
    signals: np.ndarray,  # int8 array  shape (n,)  values {-1, 0, 1}
    closes: np.ndarray,  # float64     shape (n,)
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[np.ndarray, int]:
    """
    Core backtesting loop - JIT compiled.

    Returns
    -------
    equity_curve : float64 array shape (n,)
    trade_count  : int
    """
    n = len(closes)
    equity = np.empty(n, dtype=np.float64)
    equity[0] = 100.0
    in_position = np.int8(0)
    trade_count = 0
    eq = 100.0

    for i in range(1, n):
        sig = signals[i]
        price_change = (closes[i] - closes[i - 1]) / closes[i - 1]

        if sig != in_position:
            cost = fee_rate + (slippage_rate if sig != 0 else 0.0)
            eq *= 1.0 - cost
            in_position = sig
            if sig != 0:
                trade_count += 1

        if in_position != 0:
            eq *= 1.0 + (in_position * price_change)

        equity[i] = eq

    return equity, trade_count


@njit(cache=True)
def _kernel_profit_factor(
    signals: np.ndarray,  # int8  shape (n,)
    closes: np.ndarray,  # float64 shape (n,)
    fee_rate: float,
    slippage_rate: float,
) -> Tuple[float, float, float, int, int]:
    """
    Compute true Profit Factor per closed trade (not per bar).

    A trade opens when signal != 0 and closes when signal changes.
    Entry/exit costs (fee + slippage) are deducted from each trade's P&L.

    Returns
    -------
    profit_factor : sum(winning_trade_pnl) / sum(losing_trade_pnl)
    avg_win       : average winning trade return
    avg_loss      : average losing trade return (negative)
    n_wins        : number of winning trades
    n_losses      : number of losing trades
    """
    n = len(closes)
    gross_wins = 0.0
    gross_losses = 0.0
    n_wins = 0
    n_losses = 0
    avg_win = 0.0
    avg_loss = 0.0

    in_position = np.int8(0)
    entry_equity = 1.0
    eq = 1.0

    for i in range(1, n):
        sig = signals[i]
        pc = (closes[i] - closes[i - 1]) / closes[i - 1]

        if sig != in_position:
            # Close existing position -> record completed trade
            if in_position != 0:
                trade_pnl = eq - entry_equity
                if trade_pnl >= 0.0:
                    gross_wins += trade_pnl
                    avg_win += trade_pnl
                    n_wins += 1
                else:
                    gross_losses += -trade_pnl
                    avg_loss += trade_pnl
                    n_losses += 1
            # Pay entry/exit cost
            cost = fee_rate + (slippage_rate if sig != 0 else 0.0)
            eq *= 1.0 - cost
            in_position = sig
            entry_equity = eq
            if sig != 0:
                pass  # trade_count tracked separately

        if in_position != 0:
            eq *= 1.0 + (in_position * pc)

    # Close final open position at last bar
    if in_position != 0:
        trade_pnl = eq - entry_equity
        if trade_pnl >= 0.0:
            gross_wins += trade_pnl
            avg_win += trade_pnl
            n_wins += 1
        else:
            gross_losses += -trade_pnl
            avg_loss += trade_pnl
            n_losses += 1

    pf = gross_wins / gross_losses if gross_losses > 0.0 else 0.0
    avg_win = avg_win / n_wins if n_wins > 0 else 0.0
    avg_loss = avg_loss / n_losses if n_losses > 0 else 0.0

    return pf, avg_win, avg_loss, n_wins, n_losses


@njit(cache=True)
def _kernel_market_regime(
    closes: np.ndarray,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> np.ndarray:
    """
    Classify market regime bar-by-bar using a simplified ADX proxy.

    Returns int8 array:
        1  = Trending  (ADX > threshold)
        0  = Sideways  (ADX <= threshold)
       -1  = Insufficient data (warmup)

    ADX proxy: ratio of directional movement to total range over `adx_period`.
    """
    n = len(closes)
    regime = np.full(n, np.int8(-1))

    for i in range(adx_period, n):
        window = closes[i - adx_period : i + 1]
        price_range = window[-1] - window[0]  # net move
        total_range = np.abs(np.diff(window)).sum()  # sum of absolute moves

        if total_range == 0.0:
            regime[i] = np.int8(0)
            continue

        # Directional efficiency ratio (0=random walk, 1=perfect trend)
        efficiency = abs(price_range) / total_range
        # Scale to ~ADX range: multiply by 100
        adx_proxy = efficiency * 100.0

        regime[i] = np.int8(1) if adx_proxy > adx_threshold else np.int8(0)

    return regime


@njit(cache=True)
def _kernel_rsi_wilder(
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Wilder's smoothed RSI via recursive EMA (alpha = 1/period).
    Returns float64 array of RSI values, NaN for warmup bars.
    """
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    alpha = 1.0 / period

    # Seed averages from first `period` differences
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            avg_gain += d
        else:
            avg_loss -= d
    avg_gain /= period
    avg_loss /= period

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Recursive Wilder smoothing
    for i in range(period + 1, n):
        d = closes[i] - closes[i - 1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


@njit(cache=True)
def _kernel_ema(closes: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA with alpha = 2/(span+1)."""
    n = len(closes)
    ema = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1.0 - alpha) * ema[i - 1]
    return ema


@njit(cache=True)
def _kernel_rolling_mean_std(closes: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling mean and sample std (Bessel corrected) in one pass."""
    n = len(closes)
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        m = 0.0
        for v in window:
            m += v
        m /= period
        var = 0.0
        for v in window:
            diff = v - m
            var += diff * diff
        var /= period - 1
        means[i] = m
        stds[i] = var**0.5
    return means, stds


@njit(cache=True)
def _kernel_signals_rsi(
    rsi: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Convert RSI array to signal array {-1, 0, 1}."""
    n = len(rsi)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(rsi[i]):
            continue
        if rsi[i] < lower:
            signals[i] = np.int8(1)
        elif rsi[i] > upper:
            signals[i] = np.int8(-1)
    return signals


@njit(cache=True)
def _kernel_signals_macd(
    fast_ema: np.ndarray,
    slow_ema: np.ndarray,
    signal_ema: np.ndarray,
) -> np.ndarray:
    """MACD histogram zero-cross signals."""
    n = len(fast_ema)
    signals = np.zeros(n, dtype=np.int8)
    hist_prev = np.nan
    for i in range(1, n):
        hist = (fast_ema[i] - slow_ema[i]) - signal_ema[i]
        if not np.isnan(hist_prev):
            if hist_prev < 0.0 < hist:
                signals[i] = np.int8(1)
            elif hist_prev > 0.0 > hist:
                signals[i] = np.int8(-1)
        hist_prev = hist
    return signals


@njit(cache=True)
def _kernel_signals_bollinger(
    closes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    num_std: float,
    reversion_mode: bool,  # True = reversion, False = breakout
) -> np.ndarray:
    """Bollinger Band signals."""
    n = len(closes)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if np.isnan(means[i]):
            continue
        upper = means[i] + num_std * stds[i]
        lower = means[i] - num_std * stds[i]
        price = closes[i]
        if reversion_mode:
            if price < lower:
                signals[i] = np.int8(1)
            elif price > upper:
                signals[i] = np.int8(-1)
        else:
            if price > upper:
                signals[i] = np.int8(1)
            elif price < lower:
                signals[i] = np.int8(-1)
    return signals


@njit(cache=True)
def _kernel_signals_ema_cross(
    fast_ema: np.ndarray,
    slow_ema: np.ndarray,
) -> np.ndarray:
    """
    Dual-EMA trend signals.
    Hold position direction (not just crossover bars) to stay in trend.
    """
    n = len(fast_ema)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if fast_ema[i] > slow_ema[i]:
            signals[i] = np.int8(1)
        else:
            signals[i] = np.int8(-1)
    return signals


@njit(cache=True)
def _kernel_gru_step(
    x: np.ndarray,  # input vector  shape (input_size,)
    h: np.ndarray,  # hidden state  shape (hidden_size,)
    Wz: np.ndarray,  # update gate weights  shape (hidden_size, input_size + hidden_size)
    Wr: np.ndarray,  # reset gate weights   shape (hidden_size, input_size + hidden_size)
    Wh: np.ndarray,  # candidate weights    shape (hidden_size, input_size + hidden_size)
    bz: np.ndarray,  # update gate bias     shape (hidden_size,)
    br: np.ndarray,  # reset gate bias      shape (hidden_size,)
    bh: np.ndarray,  # candidate bias       shape (hidden_size,)
) -> np.ndarray:
    """
    Single GRU cell forward step.
    Fully implemented in Numba @njit — zero PyTorch dependency.
    """
    input_size = len(x)
    hidden_size = len(h)

    # Concatenate input + hidden state
    xh = np.empty(input_size + hidden_size, dtype=np.float64)
    for i in range(input_size):
        xh[i] = x[i]
    for i in range(hidden_size):
        xh[input_size + i] = h[i]

    # Update gate z
    z = np.empty(hidden_size, dtype=np.float64)
    for i in range(hidden_size):
        val = bz[i]
        for j in range(input_size + hidden_size):
            val += Wz[i, j] * xh[j]
        z[i] = 1.0 / (1.0 + np.exp(-val))  # sigmoid

    # Reset gate r
    r = np.empty(hidden_size, dtype=np.float64)
    for i in range(hidden_size):
        val = br[i]
        for j in range(input_size + hidden_size):
            val += Wr[i, j] * xh[j]
        r[i] = 1.0 / (1.0 + np.exp(-val))  # sigmoid

    # Gated concat: concat(x, r * h)
    xrh = np.empty(input_size + hidden_size, dtype=np.float64)
    for i in range(input_size):
        xrh[i] = x[i]
    for i in range(hidden_size):
        xrh[input_size + i] = r[i] * h[i]

    # Candidate hidden state h_
    h_cand = np.empty(hidden_size, dtype=np.float64)
    for i in range(hidden_size):
        val = bh[i]
        for j in range(input_size + hidden_size):
            val += Wh[i, j] * xrh[j]
        h_cand[i] = np.tanh(val)

    # New hidden state
    h_new = np.empty(hidden_size, dtype=np.float64)
    for i in range(hidden_size):
        h_new[i] = (1.0 - z[i]) * h[i] + z[i] * h_cand[i]

    return h_new


@njit(cache=True)
def _kernel_rnn_forward(
    features: np.ndarray,  # shape (n_bars, input_size)  — normalised feature matrix
    Wz: np.ndarray,
    Wr: np.ndarray,
    Wh: np.ndarray,
    bz: np.ndarray,
    br: np.ndarray,
    bh: np.ndarray,
    Wo: np.ndarray,  # output weights  shape (1, hidden_size)
    bo: float,  # output bias     scalar
) -> np.ndarray:
    """
    Full GRU forward pass over T timesteps.
    """
    T = features.shape[0]
    hidden_size = len(bz)

    h = np.zeros(hidden_size, dtype=np.float64)
    outputs = np.zeros(T, dtype=np.float64)

    for t in range(T):
        x = features[t]
        h = _kernel_gru_step(x, h, Wz, Wr, Wh, bz, br, bh)
        # Linear output: scalar = Wo @ h + bo
        out = bo
        for i in range(hidden_size):
            out += Wo[0, i] * h[i]
        outputs[t] = out

    return outputs


@njit(cache=True)
def _kernel_signals_rnn(
    outputs: np.ndarray,
    long_thresh: float,
    short_thresh: float,
) -> np.ndarray:
    """
    Convert unbounded RNN outputs to trading signals.
    """
    n = len(outputs)
    signals = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if outputs[i] > long_thresh:
            signals[i] = np.int8(1)
        elif outputs[i] < short_thresh:
            signals[i] = np.int8(-1)
    return signals


@njit(cache=True)
def _kernel_build_features(
    closes: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Build normalised feature matrix from raw close prices.
    """
    n = len(closes)
    features = np.zeros((n, 4), dtype=np.float64)

    # Rolling mean/std for z-score (window=20)
    win = 20
    for i in range(win, n):
        # Feature 0: 1-bar log-return
        if closes[i - 1] > 0.0:
            lr1 = np.log(closes[i] / closes[i - 1])
        else:
            lr1 = 0.0

        # Feature 1: 5-bar log-return
        if i >= 5 and closes[i - 5] > 0.0:
            lr5 = np.log(closes[i] / closes[i - 5])
        else:
            lr5 = 0.0

        # Feature 2: 20-bar rolling z-score
        mu = 0.0
        for k in range(i - win, i):
            mu += closes[k]
        mu /= win
        var = 0.0
        for k in range(i - win, i):
            d = closes[k] - mu
            var += d * d
        sigma = (var / win) ** 0.5
        zscore = (closes[i] - mu) / (sigma + 1e-9)

        # Feature 3: 10-bar momentum
        if i >= 10 and closes[i - 10] > 0.0:
            mom = closes[i] / closes[i - 10] - 1.0
        else:
            mom = 0.0

        # Clip all features to [-3, 3]
        features[i, 0] = max(-3.0, min(3.0, lr1))
        features[i, 1] = max(-3.0, min(3.0, lr5))
        features[i, 2] = max(-3.0, min(3.0, zscore))
        features[i, 3] = max(-3.0, min(3.0, mom))

    return features
