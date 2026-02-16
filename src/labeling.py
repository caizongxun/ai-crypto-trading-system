import pandas as pd
import numpy as np


def triple_barrier_labeling(
    df: pd.DataFrame,
    horizon: int = 12,
    pt_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.5,
    atr_col: str = 'atr'
) -> pd.DataFrame:
    """
    Triple-barrier labeling for 15m trading.

    Label is based on which barrier is hit first within the horizon:
    -  1: upper barrier hit first
    - -1: lower barrier hit first
    -  0: neither hit within horizon (or ambiguous)

    All labels are created only for training/validation.

    Args:
        df: DataFrame with columns ['open_time','open','high','low','close', atr_col]
        horizon: look-ahead candles (e.g., 12 = 3 hours on 15m)
        pt_atr_mult: profit-taking barrier in ATR multiples
        sl_atr_mult: stop-loss barrier in ATR multiples
        atr_col: ATR column name

    Returns:
        DataFrame trimmed to exclude the last horizon rows, with a 'target' column.
    """
    if horizon <= 0:
        raise ValueError('horizon must be > 0')

    required_cols = {'high', 'low', 'close'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    out = df.copy()

    if atr_col not in out.columns:
        out[atr_col] = np.nan

    close = out['close'].to_numpy(dtype=float)
    high = out['high'].to_numpy(dtype=float)
    low = out['low'].to_numpy(dtype=float)
    atr = out[atr_col].to_numpy(dtype=float)

    n = len(out)
    y = np.zeros(n, dtype=int)

    last_i = n - horizon - 1
    for i in range(0, last_i + 1):
        entry = close[i]
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            a = entry * 0.01

        up = entry + pt_atr_mult * a
        dn = entry - sl_atr_mult * a

        label = 0
        for j in range(i + 1, i + horizon + 1):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn

            if hit_up and hit_dn:
                label = 0
                break
            if hit_up:
                label = 1
                break
            if hit_dn:
                label = -1
                break

        y[i] = label

    trimmed = out.iloc[: n - horizon].copy()
    trimmed['target'] = y[: n - horizon]
    return trimmed
