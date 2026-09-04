"""Критерии приёмки этапа сбора, исполняемые.

Смысл — чтобы «этап сдан» проверялось командой, а не на словах.

    uv run python -m src.collect.check
"""
from __future__ import annotations

import json

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.collect import arxiv, config, manifest, schema


def main() -> None:
    config.setup_console()
    cfg = config.load()
    root = cfg.raw / "openalex"
    if not root.exists():
        raise SystemExit("Срез не выгружен.")

    table = pq.read_table(root, schema=schema.WORKS)
    n = table.num_rows
    results: list[tuple[bool, str]] = []

    results.append((n >= 100_000, f"записей в срезе: {n:,} (нужно от 100 000)"))

    share = pc.sum(pc.is_valid(table["abstract"])).as_py() / n
    results.append((share >= 0.70, f"с непустой аннотацией: {share:.1%} (нужно от 70%)"))

    no_date = pc.sum(pc.is_null(table["publication_date"])).as_py()
    no_year = pc.sum(pc.is_null(table["publication_year"])).as_py()
    results.append((no_date == 0 and no_year == 0,
                    f"без даты {no_date:,}, без года {no_year:,} (нужно 0)"))

    ids = table["id"].to_pylist()
    dup_id = len(ids) - len(set(ids))
    dois = [d for d in table["doi"].to_pylist() if d]
    dup_doi = len(dois) - len(set(dois))
    results.append((dup_id == 0, f"дублей по идентификатору: {dup_id:,} (нужно 0)"))
    results.append((True, f"записей с общим DOI: {dup_doi:,} — сводятся на сшивке"))

    aggregates = json.loads((cfg.aggregates / "works-by-year.json").read_text(encoding="utf-8"))
    expected = aggregates["years"]["slice_all"]
    worst_year, worst_gap = None, 0.0
    for year, exp in expected.items():
        got = pc.sum(pc.equal(table["publication_year"], int(year))).as_py() or 0
        gap = abs(got - exp) / exp if exp else 0
        if gap > worst_gap:
            worst_year, worst_gap = year, gap
    results.append((worst_gap <= 0.02,
                    f"худшее расхождение с агрегатом: {worst_gap:.2%} в {worst_year} (нужно до 2%)"))

    if (cfg.raw / "arxiv").exists():
        pre = arxiv.read_preprints(cfg)
        created = pre["created"].to_pylist()
        no_created = sum(1 for c in created if not c)
        ours = sum(1 for c in created if c and str(cfg.year_start) <= c[:4] <= str(cfg.year_end))
        pids = pre["id"].to_pylist()
        results.append((no_created == 0,
                        f"препринтов {pre.num_rows:,}, без даты первой версии {no_created:,}"))
        results.append((len(pids) == len(set(pids)), "повторов среди препринтов нет"))
        results.append((True, f"препринтов за {cfg.year_start}-{cfg.year_end}: {ours:,}"))

        linked = {a for a in table["arxiv_id"].to_pylist() if a} & set(pids)
        total_arx = len({a for a in table["arxiv_id"].to_pylist() if a})
        results.append((True, f"сшивка по arXiv: {len(linked):,} из {total_arx:,}"))
    else:
        results.append((False, "препринтов нет: uv run python -m src.collect.arxiv"))

    if cfg.manifest.exists():
        problems = manifest.check(cfg)
        results.append((problems == 0, f"расхождений с манифестом: {problems}"))
    else:
        results.append((False, "манифеста нет: uv run python -m src.collect.manifest"))

    sample = cfg.sample / "openalex-sample.parquet"
    results.append((sample.exists(),
                    f"подвыборка: {sample.stat().st_size / 1e6:.1f} МБ" if sample.exists()
                    else "подвыборки нет"))

    print()
    for ok, line in results:
        print(f"  {'+' if ok else '-'} {line}")
    failed = [line for ok, line in results if not ok]
    print(f"\n{'этап сдан' if not failed else f'не выполнено пунктов: {len(failed)}'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
