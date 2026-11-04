"""Example custom trend rule. Run with:
    python backtest.py --signal-file examples/my_rules.py --signal dual-momentum
"""

import pandas as pd

from signals import signal


@signal("dual-momentum", history=lambda p: p["long"], trend_filter=True)
def dual_momentum(window: pd.DataFrame, short: int = 63, long: int = 252) -> pd.DataFrame:
    """Average of 3-month and 12-month returns; positive only if both are up."""
    short_ret = window.iloc[-1] / window.iloc[-short - 1] - 1
    long_ret = window.iloc[-1] / window.iloc[0] - 1
    score = (short_ret + long_ret) / 2
    both_up = (short_ret > 0) & (long_ret > 0)
    return pd.DataFrame({"short_ret": short_ret, "long_ret": long_ret,
                         "score": score.where(both_up, -score.abs())})
