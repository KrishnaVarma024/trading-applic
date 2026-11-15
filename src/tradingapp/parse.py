"""Turn a raw exchange message into a Trade, or reject it.

Everything arriving from the network is untrusted. This module is the only
place that touches raw JSON, so every later stage can assume clean values.
"""
from __future__ import annotations

import json
import math

from .models import Trade


class ParseError(ValueError):
    """The message claimed to be a trade but its fields were unusable."""


def parse_trade(raw: str | bytes, recv_ns: int) -> Trade | None:
    """Parse one Binance-format trade message.

    Returns None for messages that are valid but not trades (subscription
    acknowledgements and the like). Raises ParseError for broken trades.
    """
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ParseError(f"not JSON: {exc}") from exc

    if not isinstance(msg, dict) or msg.get("e") != "trade":
        return None

    try:
        trade = Trade(
            trade_id=int(msg["t"]),
            ts_ms=int(msg["T"]),
            price=float(msg["p"]),
            qty=float(msg["q"]),
            buyer_is_maker=bool(msg["m"]),
            recv_ns=recv_ns,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ParseError(f"bad trade fields: {exc}") from exc

    if not (math.isfinite(trade.price) and trade.price > 0):
        raise ParseError(f"bad price: {trade.price}")
    if not (math.isfinite(trade.qty) and trade.qty >= 0):
        raise ParseError(f"bad qty: {trade.qty}")
    if trade.ts_ms <= 0 or trade.trade_id < 0:
        raise ParseError("bad timestamp or trade id")
    return trade
