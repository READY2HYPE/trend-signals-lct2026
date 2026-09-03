"""Манифест среза: что реально выгружено и совпадает ли это у всех.

Расхождение в данных между участниками — самая трудноуловимая причина «у меня
работает, у тебя нет». Манифест коммитится, полная выгрузка — нет.

    uv run python -m src.collect.manifest          # собрать
    uv run python -m src.collect.manifest --check  # сверить свою выгрузку с манифестом
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from src.collect import config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "неизвестна"


def parquet_files(cfg: config.Slice) -> list[Path]:
    return sorted(cfg.raw.rglob("*.parquet")) + sorted(cfg.sample.glob("*.parquet"))


def build(cfg: config.Slice) -> dict:
    aggregates = json.loads((cfg.aggregates / "works-by-year.json").read_text(encoding="utf-8"))
    expected = aggregates["years"]["slice_all"]

    years: dict[str, dict] = {}
    for part in sorted((cfg.raw / "openalex").rglob("*.parquet")):
        year = part.parent.name.removeprefix("year=")
        rec = years.setdefault(year, {"rows": 0, "files": 0, "bytes": 0})
        rec["rows"] += pq.ParquetFile(part).metadata.num_rows
        rec["files"] += 1
        rec["bytes"] += part.stat().st_size
    for year, rec in years.items():
        exp = expected.get(year, expected.get(int(year), 0))
        rec["expected"] = exp
        rec["gap"] = round(abs(rec["rows"] - exp) / exp, 4) if exp else None

    files = {
        str(p.relative_to(cfg.raw.parent)).replace("\\", "/"): {
            "bytes": p.stat().st_size, "sha256": sha256(p),
        }
        for p in parquet_files(cfg)
    }

    return {
        "slice_id": cfg.slice_id,
        "fetched_on": datetime.now(UTC).date().isoformat(),
        "code_version": code_version(),
        "source": "openalex",
        "query": {
            "subfield_ids": cfg.subfield_ids,
            "years": [cfg.year_start, cfg.year_end],
            "types": cfg.types,
            "per_page": cfg.per_page,
        },
        "rows_total": sum(r["rows"] for r in years.values()),
        "years": dict(sorted(years.items())),
        "files": files,
    }


def check(cfg: config.Slice) -> int:
    """Сверяет выгрузку на диске с записанным манифестом. Возвращает число расхождений."""
    saved = json.loads(cfg.manifest.read_text(encoding="utf-8"))
    problems = 0
    for name, meta in saved["files"].items():
        path = cfg.raw.parent / name
        if not path.exists():
            print(f"  нет файла: {name}")
            problems += 1
        elif sha256(path) != meta["sha256"]:
            print(f"  не совпала контрольная сумма: {name}")
            problems += 1
    extra = {str(p.relative_to(cfg.raw.parent)).replace("\\", "/")
             for p in parquet_files(cfg)} - set(saved["files"])
    for name in sorted(extra):
        print(f"  лишний файл: {name}")
        problems += 1
    print("срез совпадает с манифестом" if not problems
          else f"расхождений: {problems} — срез отличается от общего")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Манифест среза")
    ap.add_argument("--check", action="store_true", help="сверить, а не пересобрать")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()
    if args.check:
        raise SystemExit(1 if check(cfg) else 0)

    data = build(cfg)
    cfg.manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{data['rows_total']:,} записей, {len(data['files'])} файлов -> {cfg.manifest}")
    bad = [y for y, r in data["years"].items() if (r["gap"] or 0) > 0.02]
    if bad:
        print(f"годы с расхождением больше 2%: {bad}")


if __name__ == "__main__":
    main()
