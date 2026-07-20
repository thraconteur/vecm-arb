"""
alpha_engine.py
================
N-Dimensional Alpha Engine for VECM-ARB.

Responsibilities
-----------------
1. Run the Johansen trace test on a panel of k asset log-prices to estimate
   the cointegration rank r (how many stationary linear combinations exist).
2. Fit a Vector Error Correction Model (VECM) given that rank to recover:
       - beta  (k x r)  : the cointegrating vectors -> hedge ratios
       - alpha (k x r)  : speed-of-adjustment coefficients (how fast each
                           asset "snaps back" toward equilibrium)
       - gamma          : short-run dynamics (lagged differences)
3. Project the live price vector onto the cointegrating space to get the
   "spread" (equilibrium error) and turn that into a z-scored trading signal.

Why Johansen instead of Engle-Granger?
---------------------------------------
Engle-Granger only works cleanly for a *single* cointegrating relationship
between two series and is sensitive to which variable you regress on.  Once
you go to k > 2 assets there can be multiple (up to k-1) independent
cointegrating relationships simultaneously. The Johansen procedure estimates
all of them at once via an eigenvalue decomposition of the VECM, which is why
it's the standard tool for basket/multi-asset stat-arb.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


@dataclasses.dataclass
class JohansenResult:
    trace_stats: np.ndarray          # trace statistic for H(r) <= 0,1,2,...
    crit_values_95: np.ndarray       # 95% critical values per rank
    eigenvalues: np.ndarray          # ordered eigenvalues (largest first)
    rank: int                        # estimated cointegration rank r

    def summary(self) -> pd.DataFrame:
        k = len(self.trace_stats)
        return pd.DataFrame(
            {
                "H0: rank <=": range(k),
                "trace_stat": self.trace_stats,
                "crit_val_95": self.crit_values_95,
                "reject_H0": self.trace_stats > self.crit_values_95,
                "eigenvalue": self.eigenvalues,
            }
        )


@dataclasses.dataclass
class VECMFit:
    rank: int
    beta: np.ndarray          # (k, r) cointegrating vectors
    alpha: np.ndarray         # (k, r) speed-of-adjustment
    assets: list[str]
    lag_order: int
    fitted_model: VECM = dataclasses.field(repr=False)

    def hedge_ratios(self, vector_idx: int = 0) -> pd.Series:
        """
        Normalize a single cointegrating vector so the first asset has unit
        weight -> gives interpretable "for 1 unit of asset_0, hold -w_i of
        asset_i" hedge ratios.
        """
        vec = self.beta[:, vector_idx]
        vec = vec / vec[0]
        return pd.Series(vec, index=self.assets, name=f"beta_{vector_idx}")


class AlphaEngine:
    """
    Wraps statsmodels' Johansen/VECM machinery with the interfaces the rest
    of the system (risk engine, backtester, cointegration monitor) actually
    needs.
    """

    def __init__(self, det_order: int = 0, k_ar_diff: int = 1):
        """
        det_order : deterministic trend assumption for the Johansen test.
                    0  -> no trend, but a constant is allowed in the
                          cointegrating relation (the standard assumption
                          for asset price spreads that should mean-revert
                          around a constant, not drift).
        k_ar_diff : number of lagged differences in the VAR representation
                    (i.e. VECM lag order p-1). 1 is a reasonable default for
                    daily equity/crypto data; raise it if Q-Q/ACF diagnostics
                    on residuals show leftover autocorrelation.
        """
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff

    # ------------------------------------------------------------------ #
    # Step 1: rank discovery
    # ------------------------------------------------------------------ #
    def johansen_test(self, log_prices: pd.DataFrame) -> JohansenResult:
        """
        log_prices: (T, k) DataFrame of log price levels (NOT returns -
        the whole point of Johansen is to work on the non-stationary levels).
        """
        result = coint_johansen(log_prices.values, self.det_order, self.k_ar_diff)

        trace_stats = result.lr1          # trace statistic sequence
        crit_95 = result.cvt[:, 1]        # column 1 = 95% critical value
        eigenvalues = result.eig

        # Sequential testing procedure: walk up the rank hypotheses,
        # stop at the first rank we FAIL to reject (that's our estimate).
        rank = 0
        for stat, cv in zip(trace_stats, crit_95):
            if stat > cv:
                rank += 1
            else:
                break

        return JohansenResult(trace_stats, crit_95, eigenvalues, rank)

    # ------------------------------------------------------------------ #
    # Step 2: fit VECM at the discovered (or overridden) rank
    # ------------------------------------------------------------------ #
    def fit_vecm(self, log_prices: pd.DataFrame, rank: Optional[int] = None) -> VECMFit:
        if rank is None:
            rank = self.johansen_test(log_prices).rank
        rank = max(rank, 1)  # a VECM needs at least rank 1 to be meaningful

        model = VECM(
            log_prices.values,
            k_ar_diff=self.k_ar_diff,
            coint_rank=rank,
            deterministic="co",  # constant restricted to cointegration space
        )
        fitted = model.fit()

        return VECMFit(
            rank=rank,
            beta=fitted.beta,     # (k, r)
            alpha=fitted.alpha,   # (k, r)
            assets=list(log_prices.columns),
            lag_order=self.k_ar_diff,
            fitted_model=fitted,
        )

    # ------------------------------------------------------------------ #
    # Step 3: project live prices onto the cointegrating space -> spread
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_spread(log_prices: pd.DataFrame, beta_vector: np.ndarray) -> pd.Series:
        """
        spread_t = beta' * y_t   (the equilibrium error the VECM is
        designed to correct back toward zero / its mean)
        """
        spread = log_prices.values @ beta_vector
        return pd.Series(spread, index=log_prices.index, name="spread")

    @staticmethod
    def zscore_signal(spread: pd.Series, window: int = 60) -> pd.DataFrame:
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()
        z = (spread - roll_mean) / roll_std
        return pd.DataFrame({"spread": spread, "zscore": z})
