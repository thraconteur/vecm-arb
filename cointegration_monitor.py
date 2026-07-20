"""
cointegration_monitor.py
=========================
Cointegration Breakdown Protocol.

A basket that was cointegrated in-sample is not guaranteed to stay
cointegrated: regime shifts, a merger/delisting, or a structural change in
one asset's liquidity provider can silently break the relationship. If you
keep trading the old beta after that happens, your "mean-reverting spread"
becomes a random walk and you are just carrying naked directional risk while
believing you're market-neutral.

This module re-runs the Johansen rank test on a rolling window as new data
arrives. If the estimated rank drops below the rank the live position was
built on, that's a structural break -> the protocol signals an automated
liquidation of the basket rather than waiting for a stop-loss to catch the
resulting P&L damage after the fact.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import pandas as pd

from engine.alpha_engine import AlphaEngine


@dataclasses.dataclass
class MonitorEvent:
    timestamp: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    estimated_rank: int
    reference_rank: int
    breakdown: bool


class CointegrationMonitor:
    def __init__(self, engine: AlphaEngine, reference_rank: int, window: int = 250, step: int = 5,
                 confirm_windows: int = 3):
        """
        confirm_windows : number of CONSECUTIVE rolling windows that must all
            show rank < reference_rank before a breakdown is confirmed. The
            Johansen trace statistic is noisy near its own critical value, so
            a single dipped window is expected to happen periodically by
            chance even when the true relationship is intact -- liquidating
            the whole basket on one blip would be its own source of
            unnecessary trading cost. Requiring a short run of consecutive
            breaches is the standard way to trade off false-trigger rate
            against detection latency.
        """
        self.engine = engine
        self.reference_rank = reference_rank
        self.window = window
        self.step = step
        self.confirm_windows = confirm_windows
        self.history: list[MonitorEvent] = []

    def run(self, log_prices: pd.DataFrame) -> list[MonitorEvent]:
        n = len(log_prices)
        self.history = []
        for end in range(self.window, n, self.step):
            start = end - self.window
            win = log_prices.iloc[start:end]
            jres = self.engine.johansen_test(win)
            event = MonitorEvent(
                timestamp=log_prices.index[end - 1],
                window_start=log_prices.index[start],
                window_end=log_prices.index[end - 1],
                estimated_rank=jres.rank,
                reference_rank=self.reference_rank,
                breakdown=jres.rank < self.reference_rank,
            )
            self.history.append(event)
        return self.history

    def first_breakdown(self) -> Optional[MonitorEvent]:
        """
        Returns the event at which a breakdown was CONFIRMED, i.e. the last
        event of the first run of >= confirm_windows consecutive breaches.
        Isolated single-window dips are ignored by design (see confirm_windows).
        """
        streak = 0
        for ev in self.history:
            streak = streak + 1 if ev.breakdown else 0
            if streak >= self.confirm_windows:
                return ev
        return None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(e) for e in self.history])
