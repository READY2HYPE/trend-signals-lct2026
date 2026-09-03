"""Подвыборка для разработки: конвейер должен отрабатывать на ней за секунды.

Пропорционально по годам, чтобы динамика на подвыборке была похожа на настоящую.
Состав записей не подчищается: доля без аннотации остаётся такой же, как в срезе,
иначе тесты пройдут на данных, которых не бывает.

    uv run python -m src.collect.sample
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from src.collect import config, schema

SEED = 20260904


def main() -> None:
    ap = argparse.ArgumentParser(description="Подвыборка из среза")
    ap.add_argument("--rows", type=int, default=40_000, help="сколько записей всего")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()

    years = sorted(d for d in (cfg.raw / "openalex").glob("year=*") if d.is_dir())
    if not years:
        raise SystemExit("Срез не выгружен: сначала uv run python -m src.collect.openalex")

    counts = {d.name: sum(pq.ParquetFile(p).metadata.num_rows for p in d.glob("*.parquet"))
              for d in years}
    total = sum(counts.values())

    parts = []
    for d in years:
        share = round(args.rows * counts[d.name] / total)
        if not share:
            continue
        table = pq.read_table(d, schema=schema.WORKS)
        take = min(table.num_rows, share)
        idx = pa.compute.array_sort_indices(
            pa.compute.random(table.num_rows, initializer=SEED + int(d.name[-4:]))
        )[:take]
        parts.append(table.take(idx))
        print(f"  {d.name}: {take:,} из {table.num_rows:,}")

    sample = pa.concat_tables(parts)
    cfg.sample.mkdir(parents=True, exist_ok=True)
    out = cfg.sample / "openalex-sample.parquet"
    pq.write_table(sample, out, compression="zstd")
    size = out.stat().st_size / 1024 / 1024
    with_abstract = sample.column("abstract").is_valid().to_pylist().count(True)
    print(f"\n{sample.num_rows:,} записей, {size:.1f} МБ, "
          f"с аннотацией {with_abstract / sample.num_rows:.0%} -> {out}")


if __name__ == "__main__":
    main()
