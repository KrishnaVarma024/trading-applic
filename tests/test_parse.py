import pytest

from tradingapp.parse import ParseError, parse_trade

from .helpers import msg


def test_valid_trade():
    t = parse_trade(msg(7, 1_700_000_000_000, "95000.10", "0.25"), recv_ns=123)
    assert (t.trade_id, t.ts_ms, t.price, t.qty, t.recv_ns) == (7, 1_700_000_000_000, 95000.10, 0.25, 123)


def test_non_trade_message_is_ignored():
    assert parse_trade('{"result": null, "id": 1}', 0) is None
    assert parse_trade("[1, 2, 3]", 0) is None


@pytest.mark.parametrize("raw", [
    "not json",
    '{"e": "trade", "t": 1}',                                   # missing fields
    msg(1, 1_700_000_000_000, price="-5"),
    msg(1, 1_700_000_000_000, price="nan"),
    msg(1, 1_700_000_000_000, price="abc"),
    msg(1, 1_700_000_000_000, qty="-1"),
    msg(1, 0),
])
def test_broken_trades_raise(raw):
    with pytest.raises(ParseError):
        parse_trade(raw, 0)
