"""Plain data types passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    """One executed trade as reported by the exchange."""

    trade_id: int          # exchange sequence number, increases by 1 per trade
    ts_ms: int             # exchange trade time, milliseconds since epoch (event time)
    price: float
    qty: float
    buyer_is_maker: bool   # True means the aggressor was a seller
    recv_ns: int           # local monotonic clock when the message arrived


@dataclass(frozen=True)
class Bar:
    """OHLCV summary of all trades with start_ms <= ts_ms < end_ms."""

    start_ms: int
    end_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float          # sum of qty
    notional: float        # sum of price * qty, used for VWAP
    n_trades: int          # 0 means no trade happened; prices are carried forward

    @property
    def vwap(self) -> float:
        return self.notional / self.volume if self.volume > 0 else self.close
