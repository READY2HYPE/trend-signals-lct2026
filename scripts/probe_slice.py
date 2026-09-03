"""Замер среза до выгрузки: объём под фильтрами и полнота полей на выборке.

Ничего не качает в data/ — только печатает. Нужен, чтобы обсуждать фильтр цифрами.

    uv run python scripts/probe_slice.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect import config, schema
from src.collect.openalex import Client


def main() -> None:
    config.setup_console()
    cfg = config.load()
    client = Client(cfg)

    base = "primary_topic.subfield.id:" + "|".join(str(i) for i in cfg.subfield_ids)
    steps = [
        (base, "подобласть целиком, все годы"),
        (cfg.works_filter(), f"+ {cfg.year_start}-{cfg.year_end}, {'/'.join(cfg.types)}"),
        (cfg.works_filter() + ",has_abstract:true", "+ непустая аннотация"),
    ]
    print("объём под последовательными фильтрами")
    for filt, label in steps:
        print(f"  {client.count(filt):>10,}  {label}")

    print("\nполнота полей на случайной выборке 200 записей")
    data = client._page({
        "filter": cfg.works_filter(), "select": schema.SELECT,
        "per-page": 200, "sample": 200, "seed": 42,
    })
    rows = [schema.to_row(w) for w in data["results"]]
    n = len(rows)
    checks = {
        "есть аннотация": lambda r: bool(r["abstract"]),
        "есть DOI": lambda r: bool(r["doi"]),
        "есть arxiv_id": lambda r: bool(r["arxiv_id"]),
        "дата — 1 января (день неизвестен)": lambda r: r["date_is_placeholder"],
        "препринт": lambda r: r["type"] == "preprint",
        "есть организации": lambda r: bool(r["institution_ids"]),
        f"слабая тема (score < {cfg.min_topic_score})":
            lambda r: (r["primary_topic"] or {}).get("score", 0) < cfg.min_topic_score,
    }
    for label, fn in checks.items():
        print(f"  {sum(1 for r in rows if fn(r)) / n:>5.0%}  {label}")
    print("  языки:", Counter(r["language"] for r in rows).most_common(4))


if __name__ == "__main__":
    main()
