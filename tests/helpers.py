import json

from tradingapp.models import Trade


def trade(trade_id: int, ts_ms: int, price: float, qty: float = 1.0) -> Trade:
    return Trade(trade_id=trade_id, ts_ms=ts_ms, price=price, qty=qty, buyer_is_maker=False, recv_ns=0)


def msg(trade_id: int, ts_ms: int, price: str = "100.0", qty: str = "1.0") -> str:
    return json.dumps({"e": "trade", "E": ts_ms, "s": "BTCUSDT", "t": trade_id,
                       "p": price, "q": qty, "T": ts_ms, "m": False, "M": True})
