"""Collect per-stage latency samples and summarise them as percentiles.

Percentiles, not the mean: a mean hides the slow tail, and the tail is what a
latency-sensitive system is judged on.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

QUANTILES = {"p50": 50, "p90": 90, "p99": 99, "p99.9": 99.9}


class LatencyRecorder:
    def __init__(self) -> None:
        self._ns: dict[str, list[int]] = defaultdict(list)

    def add(self, **stages_ns: int) -> None:
        for name, ns in stages_ns.items():
            self._ns[name].append(ns)

    def extend(self, other: "LatencyRecorder") -> None:
        for name, samples in other._ns.items():
            self._ns[name].extend(samples)

    def count(self, stage: str = "total") -> int:
        return len(self._ns[stage])

    def summary(self) -> dict:
        """Microseconds per stage: count, percentiles and the worst sample."""
        out = {}
        for name, samples in self._ns.items():
            if not samples:
                continue
            us = np.asarray(samples, dtype=np.float64) / 1000
            row = {"n": len(us)}
            row.update({k: round(float(np.percentile(us, q)), 1) for k, q in QUANTILES.items()})
            row["max"] = round(float(us.max()), 1)
            out[name] = row
        return out

    def to_markdown(self) -> str:
        s = self.summary()
        lines = ["| Stage | n | p50 | p90 | p99 | p99.9 | max |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in s.items():
            lines.append(f"| {name} | {r['n']:,} | {r['p50']} | {r['p90']} | {r['p99']} | {r['p99.9']} | {r['max']} |")
        return "\n".join(lines) + "\n\nAll values in microseconds."

    def save(self, path: str | Path, **context) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"context": context, "latency_us": self.summary()}, indent=2) + "\n")
