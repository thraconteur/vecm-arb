"""
backtester.py
==============
Microstructure-Aware Backtester.

A stat-arb backtest that ignores trading frictions is fiction: this module
explicitly charges the strategy for the three biggest sources of slippage
between a paper P&L and a real one:

1. Maker/taker fees       - flat bps on notional traded, charged every time
                             the basket rebalances (entries, exits, resizes).
2. Short borrow cost      - annualized bps * short-leg notional, accrued
                             daily for as long as a short position is held.
                             This alone can turn a "profitable" mean-reversion
                             strategy negative if it's short hard-to-borrow
                             names for long holding periods.
3. Square-root market impact - the industry-standard nonlinear model:

        impact_bps = Y * sigma_daily_bps * sqrt(trade_notional / ADV_notional)

   Impact grows with the SQUARE ROOT of trade size relative to average
   daily volume (ADV), not linearly -- doubling your order size does not
   double your slippage, but it doesn't stay flat either. This term is what
   ultimately caps how much AUM a stat-arb strategy can run before its own
   trading erodes the edge (the "capacity" question in the tear sheet).

Trading logic
-------------
Given a z-scored spread signal from the Alpha Engine:
    z > +entry_z  -> spread too high  -> SHORT the spread (short the
                     positive-beta leg, long the negative-beta legs)
    z < -entry_z  -> spread too low   -> LONG the spread
    |z| < exit_z  -> flatten
Position sizing scales the basket to a fixed gross-leverage target; actual
per-asset dollar weights come straight from the VECM beta vector.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class CostModel:
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 5.0
    impact_coefficient: float = 0.6   # "Y" in the square-root model
    use_taker: bool = True            # stat-arb basket legs are usually crossing the spread


@dataclasses.dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame          # dollar exposure per asset per day
    trade_log: pd.DataFrame
    sharpe: float
    max_drawdown: float
    cagr: float
    total_fee_cost: float
    total_borrow_cost: float
    total_impact_cost: float


class MicrostructureBacktester:
    def __init__(
        self,
        cost_model: CostModel,
        adv_notional: pd.Series,           # average daily $ volume per asset
        borrow_cost_bps: pd.Series,        # annualized bps per asset (for short legs)
        aum: float = 10_000_000,
        gross_leverage: float = 1.0,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        self.cost_model = cost_model
        self.adv_notional = adv_notional
        self.borrow_cost_bps = borrow_cost_bps
        self.aum = aum
        self.gross_leverage = gross_leverage
        self.entry_z = entry_z
        self.exit_z = exit_z

    def _signal_from_zscore(self, z: pd.Series) -> pd.Series:
        pos = pd.Series(0, index=z.index, dtype=float)
        state = 0.0
        for t, zt in z.items():
            if np.isnan(zt):
                pos[t] = state
                continue
            if state == 0:
                if zt > self.entry_z:
                    state = -1.0   # short the spread
                elif zt < -self.entry_z:
                    state = 1.0    # long the spread
            else:
                if abs(zt) < self.exit_z:
                    state = 0.0
            pos[t] = state
        return pos

    def run(self, price_levels: pd.DataFrame, hedge_weights: pd.Series, zscore: pd.Series) -> BacktestResult:
        assets = list(price_levels.columns)
        w = hedge_weights.reindex(assets)
        w_norm = w / w.abs().sum() * self.gross_leverage  # normalize gross exposure to target

        signal = self._signal_from_zscore(zscore.reindex(price_levels.index))

        dollar_positions = pd.DataFrame(0.0, index=price_levels.index, columns=assets)
        for a in assets:
            dollar_positions[a] = signal * w_norm[a] * self.aum

        rets = price_levels.pct_change().fillna(0.0)

        # --- gross P&L before costs (yesterday's position earns today's return) ---
        gross_pnl = (dollar_positions.shift(1).fillna(0.0) * rets).sum(axis=1)

        # --- trading costs whenever the position changes ---
        traded_notional = dollar_positions.diff().abs().fillna(dollar_positions.abs())
        fee_bps = self.cost_model.taker_fee_bps if self.cost_model.use_taker else self.cost_model.maker_fee_bps
        fee_cost = (traded_notional * fee_bps / 1e4).sum(axis=1)

        sigma_daily_bps = rets.rolling(20).std().fillna(rets.std()) * 1e4
        impact_cost = pd.DataFrame(0.0, index=price_levels.index, columns=assets)
        for a in assets:
            adv = max(self.adv_notional.get(a, self.aum * 0.5), 1.0)
            participation = (traded_notional[a] / adv).clip(lower=0)
            impact_bps = self.cost_model.impact_coefficient * sigma_daily_bps[a] * np.sqrt(participation)
            impact_cost[a] = traded_notional[a] * impact_bps / 1e4
        impact_cost_total = impact_cost.sum(axis=1)

        # --- borrow cost: accrues daily on short-leg notional held ---
        short_notional = dollar_positions.clip(upper=0).abs()
        daily_borrow_rate = self.borrow_cost_bps.reindex(assets).fillna(50) / 1e4 / 252
        borrow_cost = (short_notional * daily_borrow_rate).sum(axis=1)

        net_pnl = gross_pnl - fee_cost - impact_cost_total - borrow_cost
        equity = (1 + net_pnl / self.aum).cumprod() * self.aum

        daily_ret = net_pnl / self.aum
        sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min())
        n_years = len(equity) / 252
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        trade_log = pd.DataFrame(
            {
                "signal": signal,
                "gross_pnl": gross_pnl,
                "fee_cost": fee_cost,
                "impact_cost": impact_cost_total,
                "borrow_cost": borrow_cost,
                "net_pnl": net_pnl,
            }
        )

        return BacktestResult(
            equity_curve=equity,
            daily_returns=daily_ret,
            positions=dollar_positions,
            trade_log=trade_log,
            sharpe=sharpe,
            max_drawdown=max_dd,
            cagr=cagr,
            total_fee_cost=float(fee_cost.sum()),
            total_borrow_cost=float(borrow_cost.sum()),
            total_impact_cost=float(impact_cost_total.sum()),
        )

    def capacity_curve(self, price_levels: pd.DataFrame, hedge_weights: pd.Series, zscore: pd.Series,
                        aum_levels: list[float]) -> pd.DataFrame:
        """
        Re-runs the backtest across a range of AUM levels (holding everything
        else fixed) to find the point where market impact cost eats through
        the strategy's edge -- i.e. the strategy's real capacity.
        """
        rows = []
        original_aum = self.aum
        for aum in aum_levels:
            self.aum = aum
            res = self.run(price_levels, hedge_weights, zscore)
            rows.append(
                {
                    "aum": aum,
                    "sharpe": res.sharpe,
                    "cagr": res.cagr,
                    "total_impact_cost": res.total_impact_cost,
                    "impact_cost_pct_of_aum": res.total_impact_cost / aum,
                    "net_pnl": res.equity_curve.iloc[-1] - res.equity_curve.iloc[0],
                }
            )
        self.aum = original_aum
        return pd.DataFrame(rows)
