"""Append-only CSV files, one per UTC day.

CSV is deliberate for phase 1: you can open it and check the pipeline by eye.
Phase 2 moves research data to Parquet. Writes here are synchronous and sit
on the hot path; phase 4 measures that cost and moves them off it.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Sequence


class DailyCsvSink:
    def __init__(self, directory: str | Path, prefix: str, header: Sequence[str], flush_every: int = 1) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix, self.header, self.flush_every = prefix, list(header), flush_every
        self._day: str | None = None
        self._file: IO[str] | None = None
        self._writer = None
        self._pending = 0

    def _rotate(self, day: str) -> None:
        self.close()
        path = self.directory / f"{self.prefix}-{day}.csv"
        is_new = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(self.header)
        self._day = day

    def write(self, ts_ms: int, row: Sequence[object]) -> None:
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if day != self._day:
            self._rotate(day)
        self._writer.writerow(row)
        self._pending += 1
        if self._pending >= self.flush_every:
            self._file.flush()
            self._pending = 0

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


BAR_HEADER = ["start_ms", "end_ms", "open", "high", "low", "close", "volume", "notional", "n_trades"]
TICK_HEADER = ["trade_id", "ts_ms", "price", "qty", "buyer_is_maker"]
