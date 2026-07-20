"""
tests/test_pipeline.py
========================
Smoke tests. Run with: python3 -m pytest tests/ -v
(or just: python3 tests/test_pipeline.py)

These are deliberately built around the synthetic ground-truth generator:
the whole point is to fail loudly if the Johansen/VECM machinery, the risk
engine's constraints, or the pub/sub buffering ever silently break.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import pandas as pd

from data.data_sources import generate_vecm_panel
from engine.alpha_engine import AlphaEngine
from engine.cointegration_monitor import CointegrationMonitor
from risk.risk_engine import RiskEngine
from pipeline.data_pipeline import InMemoryBroker, FeedMicroservice, SynchronizedBuffer, consume_and_buffer


def test_johansen_recovers_known_rank():
    prices, true_beta, true_alpha = generate_vecm_panel(n_assets=6, n_obs=1200, rank=2, seed=1)
    jres = AlphaEngine().johansen_test(prices)
    assert jres.rank == 2, f"expected rank 2, got {jres.rank}"


def test_vecm_recovers_beta_within_tolerance_rank1():
    prices, true_beta, true_alpha = generate_vecm_panel(n_assets=5, n_obs=1200, rank=1, seed=2)
    eng = AlphaEngine()
    fit = eng.fit_vecm(prices, rank=1)
    recovered = fit.hedge_ratios(0).values
    true_norm = true_beta[:, 0] / true_beta[0, 0]
    err = np.abs(recovered - true_norm).max()
    assert err < 0.35, f"beta recovery error too large: {err}"


def test_risk_engine_stays_dollar_neutral_after_halt():
    w0 = pd.Series([1.0, -0.4, -0.3, -0.2, -0.1], index=[f"A{i}" for i in range(5)])
    res = RiskEngine().rebalance(w0, halted_or_hard_to_borrow=["A2"])
    assert res.status == "optimal"
    assert abs(res.weights.sum()) < 1e-6
    assert res.weights["A2"] == 0.0


def test_risk_engine_respects_caps():
    w0 = pd.Series([1.0, -1.0], index=["A0", "A1"])
    eng = RiskEngine(gross_cap=1.0, per_asset_cap=0.6)
    res = eng.rebalance(w0, halted_or_hard_to_borrow=[])
    assert res.weights.abs().max() <= 0.6 + 1e-6
    assert res.weights.abs().sum() <= 1.0 + 1e-6


def test_cointegration_monitor_flags_injected_break():
    prices, true_beta, true_alpha = generate_vecm_panel(n_assets=6, n_obs=1200, rank=2, seed=3)
    broken = prices.copy()
    rng = np.random.default_rng(5)
    broken.iloc[1000:, 0] += np.cumsum(rng.normal(0, 0.03, size=200))
    eng = AlphaEngine()
    ref_rank = eng.johansen_test(prices.iloc[:800]).rank
    monitor = CointegrationMonitor(eng, reference_rank=ref_rank, window=200, step=20, confirm_windows=2)
    monitor.run(broken)
    assert any(ev.breakdown for ev in monitor.history), "monitor failed to flag any breakdown window"


def test_pipeline_delivers_all_ticks_nonblocking():
    async def _run():
        broker = InMemoryBroker()
        buf = SynchronizedBuffer(["X", "Y"])
        received = []
        task = asyncio.create_task(
            consume_and_buffer(broker, "t", buf, lambda s: received.append(s), min_updates_before_signal=2, max_messages=20)
        )
        await asyncio.sleep(0.05)
        feeds = [
            FeedMicroservice(broker, "X", [{"mid_price": 1.0 + i} for i in range(10)], channel="t"),
            FeedMicroservice(broker, "Y", [{"mid_price": 2.0 + i} for i in range(10)], channel="t"),
        ]
        await asyncio.gather(*(f.run() for f in feeds))
        await asyncio.sleep(0.05)
        task.cancel()
        return buf

    buf = asyncio.run(_run())
    assert buf.update_counts["X"] == 10
    assert buf.update_counts["Y"] == 10


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
