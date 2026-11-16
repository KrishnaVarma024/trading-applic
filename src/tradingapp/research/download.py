"""Download monthly 1-minute klines from the exchange's public archive.

    uv run python -m tradingapp.research.download --symbol BTCUSDT --start 2025-11 --end 2026-10

Each file is verified against the SHA-256 checksum the archive publishes
beside it. A file that fails the check is deleted, never used.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly/klines"


def month_range(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def download_month(symbol: str, month: str, out_dir: Path) -> Path | None:
    name = f"{symbol}-1m-{month}.zip"
    path = out_dir / name
    if path.exists():
        return path
    url = f"{BASE}/{symbol}/1m/{name}"
    try:
        blob = _get(url)
        expected = _get(url + ".CHECKSUM").decode().split()[0]
    except urllib.error.HTTPError as exc:
        print(f"{month}: not available ({exc.code})")
        return None
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{name}: checksum mismatch, expected {expected}, got {actual}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(path)          # atomic: a crash mid-write never leaves a half file under the real name
    print(f"{month}: {len(blob) / 1e6:.1f} MB, checksum ok")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    p.add_argument("--out", default="data/hist")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for month in month_range(args.start, args.end):
        download_month(args.symbol, month, out)


if __name__ == "__main__":
    main()
