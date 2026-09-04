"""Данные для витрины корпуса: сводка по годам, темам, странам плюс выборка работ.

Витрина нужна, чтобы корпус можно было посмотреть глазами, а не верить числам
на слово. Считается по всему срезу, работы для просмотра берутся из подвыборки.

    uv run python scripts/make_catalog.py
    uv run python scripts/make_catalog.py --page витрина.html

Готовая страница в репозиторий не кладётся: она целиком выводится из шаблона
`scripts/catalog.html` и данных `data/index/catalog.json`, которые там уже есть.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect import arxiv, config, schema

WORKS_SHOWN = 4000        # столько работ уходит в витрину для просмотра
TOPICS_SHOWN = 40


def build_page(data_path: Path, out: Path) -> None:
    """Подставляет данные в шаблон страницы."""
    tpl = (Path(__file__).parent / "catalog.html").read_text(encoding="utf-8")
    if "__CATALOG__" not in tpl:
        raise SystemExit("В шаблоне нет метки __CATALOG__.")
    out.write_text(tpl.replace("__CATALOG__", data_path.read_text(encoding="utf-8")),
                   encoding="utf-8")
    print(f"страница -> {out} ({out.stat().st_size / 1024 / 1024:.2f} МБ)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Данные для витрины корпуса")
    ap.add_argument("--page", type=Path, help="заодно собрать готовую страницу по этому пути")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()

    by_year: Counter[int] = Counter()
    by_type: Counter[str] = Counter()
    by_country: Counter[str] = Counter()
    with_abstract: Counter[int] = Counter()
    topics: dict[str, dict] = defaultdict(lambda: {"name": None, "years": Counter()})

    for year_dir in sorted((cfg.raw / "openalex").glob("year=*")):
        table = pq.read_table(year_dir, schema=schema.WORKS, columns=[
            "publication_year", "type", "abstract", "primary_topic", "institution_countries",
        ])
        years = table["publication_year"].to_pylist()
        by_year.update(years)
        by_type.update(table["type"].to_pylist())
        for year, abstract, topic, countries in zip(
            years, table["abstract"].to_pylist(),
            table["primary_topic"].to_pylist(), table["institution_countries"].to_pylist(),
        ):
            if abstract:
                with_abstract[year] += 1
            if topic and topic.get("id"):
                rec = topics[topic["id"]]
                rec["name"] = rec["name"] or topic.get("name")
                rec["years"][year] += 1
            by_country.update(set(countries or []))
        del table
    print(f"работ в срезе: {sum(by_year.values()):,}")

    pre = arxiv.read_preprints(cfg)
    pre_years = Counter(c[:4] for c in pre["created"].to_pylist() if c)
    print(f"препринтов: {pre.num_rows:,}")

    years = sorted(by_year)
    half = len(years) // 2
    top_topics = []
    for tid, rec in topics.items():
        total = sum(rec["years"].values())
        if total < 2000:
            continue
        early = sum(rec["years"].get(y, 0) for y in years[:half]) or 1
        late = sum(rec["years"].get(y, 0) for y in years[half:])
        top_topics.append({
            "id": tid, "name": rec["name"], "works": total,
            "growth": round(late / early, 2),
            "series": [rec["years"].get(y, 0) for y in years],
        })
    top_topics.sort(key=lambda t: -t["growth"])
    top_topics = top_topics[:TOPICS_SHOWN]

    sample = pq.read_table(cfg.sample / "openalex-sample.parquet", schema=schema.WORKS)
    step = max(1, sample.num_rows // WORKS_SHOWN)
    rows = sample.to_pylist()[::step][:WORKS_SHOWN]
    works = []
    for r in rows:
        topic = r["primary_topic"] or {}
        works.append({
            "t": (r["title"] or "без заголовка")[:160],
            "y": r["publication_year"],
            "k": "препринт" if r["type"] == "preprint" else "статья",
            "tp": topic.get("name") or "—",
            "c": r["cited_by_count"],
            "a": bool(r["abstract"]),
            "o": (r["institution_names"] or [None])[0],
            "cn": (r["institution_countries"] or [None])[0],
            "u": r["doi"] or f"https://openalex.org/{r['id']}",
        })

    out = {
        "slice_id": cfg.slice_id,
        "direction": cfg.direction_name,
        "years": years,
        "totals": {
            "works": sum(by_year.values()),
            "preprints": pre.num_rows,
            "with_abstract": sum(with_abstract.values()),
            "countries": len(by_country),
            "topics": len(topics),
        },
        "by_year": [by_year[y] for y in years],
        "abstract_by_year": [with_abstract[y] for y in years],
        "preprints_by_year": [pre_years.get(str(y), 0) for y in years],
        "by_type": dict(by_type.most_common()),
        "by_country": dict(by_country.most_common(20)),
        "topics": top_topics,
        "works": works,
    }

    path = Path("data/index/catalog.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(works):,} работ, {len(top_topics)} тем -> {path} "
          f"({path.stat().st_size / 1024 / 1024:.1f} МБ)")
    if args.page:
        build_page(path, args.page)


if __name__ == "__main__":
    main()
