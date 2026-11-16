"""Fetch recent 1-minute klines over REST to warm the feature engine at start-up.

The engine needs a week of bars before it can forecast. Waiting a week is not
an option, so the history comes from the exchange's public REST endpoint.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable

DEFAULT_REST_BASE = "https://data-api.binance.vision"
MINUTE_MS = 60_000
PAGE = 1000

Kline = tuple[int, float, float, float]      # start_ms, close, volume, trades


def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode()


def fetch_klines(
    symbol: str, last_start_ms: int, count: int,
    base: str = DEFAULT_REST_BASE, get: Callable[[str], str] = _http_get,
) -> list[Kline]:
    """Return `count` consecutive 1-minute klines ending with the one that starts at last_start_ms.

    Raises if the exchange returns anything other than a gap-free run, because
    a silent hole in the warm-up history would corrupt a week of features.
    """
    out: list[Kline] = []
    end = last_start_ms
    while len(out) < count:
        want = min(PAGE, count - len(out))
        url = (f"{base}/api/v3/klines?symbol={symbol.upper()}&interval=1m"
               f"&endTime={end + MINUTE_MS - 1}&limit={want}")
        page = [(int(k[0]), float(k[4]), float(k[5]), float(k[8])) for k in json.loads(get(url))]
        if not page:
            break
        out = page + out
        end = page[0][0] - MINUTE_MS
    out = out[-count:]
    if len(out) != count or out[-1][0] != last_start_ms:
        raise RuntimeError(f"backfill incomplete: wanted {count} klines ending {last_start_ms}, got {len(out)}")
    for a, b in zip(out, out[1:]):
        if b[0] - a[0] != MINUTE_MS:
            raise RuntimeError(f"backfill has a hole between {a[0]} and {b[0]}")
    return out
