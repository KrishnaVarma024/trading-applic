"""Load archived 1-minute klines and reduce them to an hourly series.

Output per hour: realized variance (sum of squared 1-minute log returns) and
the hour's log return. An hour with too few minutes is dropped and counted,
because a variance summed over 40 minutes is not comparable to one over 60.
"""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MINUTE_MS = 60_000
HOUR_MS = 3_600_000


def _to_ms(ts: int) -> int:
    # The archive switched from milliseconds to microseconds in 2025.
    return ts // 1000 if ts > 10**14 else ts


def load_minutes(directory: str | Path, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (open_time_ms, close) for every 1-minute kline, sorted and de-duplicated."""
    times: list[int] = []
    closes: list[float] = []
    files = sorted(Path(directory).glob(f"{symbol}-1m-*.zip"))
    if not files:
        raise FileNotFoundError(f"no {symbol} kline archives in {directory}; run the downloader first")
    for path in files:
        with zipfile.ZipFile(path) as zf:
            with zf.open(zf.namelist()[0]) as raw:
                for row in csv.reader(io.TextIOWrapper(raw)):
                    if not row or not row[0].isdigit():
                        continue  # header line in some archive vintages
                    times.append(_to_ms(int(row[0])))
                    closes.append(float(row[4]))
    t = np.asarray(times, dtype=np.int64)
    c = np.asarray(closes, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, c = t[order], c[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], c[keep]


@dataclass
class Hourly:
    start_ms: np.ndarray   # hour start, ms
    rv: np.ndarray         # realized variance of the hour
    ret: np.ndarray        # log return over the hour
    dropped_hours: int     # hours discarded for missing minutes
    missing_minutes: int


def to_hourly(t_ms: np.ndarray, close: np.ndarray, min_minutes: int = 58) -> Hourly:
    gaps = np.diff(t_ms) // MINUTE_MS - 1
    missing = int(gaps[gaps > 0].sum())

    r = np.diff(np.log(close))
    r_time = t_ms[1:]
    # A return that spans a hole is not a 1-minute return. Leave it out.
    contiguous = np.diff(t_ms) == MINUTE_MS
    r, r_time = r[contiguous], r_time[contiguous]

    hour = r_time // HOUR_MS
    uniq, first, counts = np.unique(hour, return_index=True, return_counts=True)
    rv = np.add.reduceat(r * r, first)
    ret = np.add.reduceat(r, first)

    full = counts >= min_minutes
    # Dropped hours leave a small discontinuity: the row after a dropped hour
    # treats the hour before it as "previous". The count is reported so the
    # reader can judge whether that matters.
    return Hourly(
        start_ms=(uniq[full] * HOUR_MS).astype(np.int64),
        rv=rv[full], ret=ret[full],
        dropped_hours=int((~full).sum()), missing_minutes=missing,
    )
