import pytest

from tradingapp.bars import BarBuilder

from .helpers import trade

M = 60_000
T0 = 1_700_000_040_000 - (1_700_000_040_000 % M)   # a minute boundary


def test_ohlcv_of_one_bar():
    b = BarBuilder(M)
    assert b.update(trade(1, T0 + 1_000, 100.0, 2.0)) == []
    assert b.update(trade(2, T0 + 2_000, 105.0, 1.0)) == []
    assert b.update(trade(3, T0 + 3_000, 98.0, 1.0)) == []
    assert b.update(trade(4, T0 + 59_999, 101.0, 4.0)) == []
    (bar,) = b.update(trade(5, T0 + M, 102.0, 1.0))
    assert (bar.start_ms, bar.end_ms) == (T0, T0 + M)
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 105.0, 98.0, 101.0)
    assert bar.volume == 8.0 and bar.n_trades == 4
    assert bar.vwap == pytest.approx((200 + 105 + 98 + 404) / 8)


def test_trade_exactly_on_boundary_opens_the_next_bar():
    b = BarBuilder(M)
    b.update(trade(1, T0 + 59_999, 100.0))
    (closed,) = b.update(trade(2, T0 + M, 200.0))
    assert closed.close == 100.0
    assert b.flush().open == 200.0 and b.flush().start_ms == T0 + M


def test_quiet_minutes_become_empty_bars_carrying_the_last_close():
    b = BarBuilder(M)
    b.update(trade(1, T0 + 10, 100.0))
    bars = b.update(trade(2, T0 + 3 * M + 10, 110.0))
    assert [x.start_ms for x in bars] == [T0, T0 + M, T0 + 2 * M]
    assert [x.n_trades for x in bars] == [1, 0, 0]
    assert all(x.open == x.high == x.low == x.close == 100.0 for x in bars[1:])
    assert b.stats.empty_bars == 2


def test_late_trade_is_dropped_and_counted():
    b = BarBuilder(M)
    b.update(trade(1, T0 + 10, 100.0))
    b.update(trade(2, T0 + M + 10, 101.0))
    assert b.update(trade(3, T0 + 20, 999.0)) == []
    assert b.stats.late_dropped == 1
    assert b.flush().high == 101.0


def test_huge_gap_is_not_filled():
    b = BarBuilder(M, max_fill_bars=10)
    b.update(trade(1, T0, 100.0))
    bars = b.update(trade(2, T0 + 1_000_000 * M, 100.0))
    assert len(bars) == 1
    assert b.stats.unfilled_gaps == 1 and b.stats.empty_bars == 0


def test_rejects_bad_interval():
    with pytest.raises(ValueError):
        BarBuilder(0)


def test_live_mode_drops_the_partial_first_bar_only():
    b = BarBuilder(M, skip_first_bar=True)
    b.update(trade(1, T0 + 30_000, 100.0))            # joined mid-minute
    assert b.update(trade(2, T0 + M + 1, 101.0)) == []
    (bar,) = b.update(trade(3, T0 + 2 * M + 1, 102.0))
    assert bar.start_ms == T0 + M and bar.open == 101.0
    assert b.stats.partial_dropped == 1
