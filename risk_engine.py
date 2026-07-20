"""
risk_engine.py
================
Dynamic Risk & Hedging Engine.

The VECM gives you a hedge ratio (beta) that is only correct when ALL k
assets are tradeable. In production, one of them routinely isn't:
    - a trading halt (exchange circuit breaker, LULD halt, etc.)
    - the asset becomes hard-to-borrow (can't open/maintain the short leg)
    - the position hits an exchange-imposed or internal risk limit

When that happens you cannot just "drop the leg" -- doing so silently turns
a market-neutral basket into a directionally-exposed one. Instead we
re-solve, via convex optimization, for the best possible weighting over the
remaining N-1 (or fewer) assets that:
    1. stays as close as possible to the original VECM hedge ratios
       (minimize tracking error to the "true" spread), while
    2. is subject to explicit constraints: dollar-neutrality, per-asset
       gross exposure caps, and (optionally) a borrow-cost-aware penalty
       that further shrinks weights on assets that are merely expensive
       to borrow rather than fully halted.

This is a small QP, solved with CVXPY, and cheap enough to re-run every
time a risk event fires.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclasses.dataclass
class RebalanceResult:
    weights: pd.Series           # new weights over ALL original assets (0 for excluded)
    dropped_assets: list[str]
    tracking_error: float        # ||w_new - w_original|| achieved (objective value)
    dollar_neutral_slack: float  # |sum(w_new)| at the optimum (~0 if constraint satisfied)
    status: str


class RiskEngine:
    def __init__(self, gross_cap: float = 2.5, per_asset_cap: float = 1.0):
        """
        gross_cap      : max sum(|w_i|) across the basket (leverage control)
        per_asset_cap  : max |w_i| for any single asset (concentration control)
        """
        self.gross_cap = gross_cap
        self.per_asset_cap = per_asset_cap

    def rebalance(
        self,
        original_weights: pd.Series,
        halted_or_hard_to_borrow: list[str],
        borrow_cost_bps: Optional[pd.Series] = None,
        borrow_penalty_lambda: float = 0.0,
    ) -> RebalanceResult:
        """
        original_weights : hedge ratios from the VECM (beta), one entry per asset,
                            already scaled to a target dollar-neutral basket
                            (sum ~ 0).
        halted_or_hard_to_borrow : asset names to hard-exclude (weight forced to 0)
        borrow_cost_bps  : optional per-asset annualized borrow cost, used as a soft
                            penalty so the optimizer prefers cheaper-to-borrow assets
                            when there's freedom left over, without hard-excluding them
        borrow_penalty_lambda : weight on the borrow-cost penalty term (0 = ignore)
        """
        assets = list(original_weights.index)
        k = len(assets)
        w0 = original_weights.values
        excluded_mask = np.array([a in halted_or_hard_to_borrow for a in assets])

        w = cp.Variable(k)

        # --- objective: minimize tracking error to the VECM-implied basket ---
        tracking_term = cp.sum_squares(w - w0)

        objective_terms = [tracking_term]
        if borrow_penalty_lambda > 0 and borrow_cost_bps is not None:
            costs = borrow_cost_bps.reindex(assets).fillna(0).values / 1e4
            objective_terms.append(borrow_penalty_lambda * (costs @ cp.abs(w)))

        objective = cp.Minimize(sum(objective_terms))

        constraints = [
            cp.sum(w) == 0,                       # dollar (market) neutrality
            cp.norm1(w) <= self.gross_cap,         # leverage cap
            cp.abs(w) <= self.per_asset_cap,       # concentration cap
        ]
        # hard-exclude halted / hard-to-borrow names
        if excluded_mask.any():
            constraints.append(w[excluded_mask] == 0)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if w.value is None:
            # infeasible (e.g. too many names excluded to remain neutral within caps)
            return RebalanceResult(
                weights=pd.Series(0.0, index=assets),
                dropped_assets=list(halted_or_hard_to_borrow),
                tracking_error=float("inf"),
                dollar_neutral_slack=float("nan"),
                status=problem.status,
            )

        new_w = pd.Series(np.round(w.value, 8), index=assets)
        return RebalanceResult(
            weights=new_w,
            dropped_assets=list(halted_or_hard_to_borrow),
            tracking_error=float(tracking_term.value),
            dollar_neutral_slack=abs(float(new_w.sum())),
            status=problem.status,
        )
