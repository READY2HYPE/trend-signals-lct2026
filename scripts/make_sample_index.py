"""Образец индекса для фронтенда и бэкенда — настоящая форма, ненастоящие тренды.

Пока кластеризации нет, группы берутся по темам самого OpenAlex. Числа при этом
честные: ряды по годам, первый год, организации, страны и ссылки посчитаны по
выгруженному корпусу. Верстать и писать эндпоинты по этому файлу можно сразу,
а когда заработает ядро, подменится только источник.

Поля, которые ядро заполнит позже, помечены внутри файла: мотивация, кейс-пример
и результаты проверки на прошлом.

    uv run python scripts/make_sample_index.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect import config, schema

TOP = 15
MIN_WORKS = 400          # ниже этого процент роста — шум
PLACEHOLDER = (
    "Заполняется ядром на этапе описания кластеров: какую проблему решает тренд "
    "и какое преимущество даёт. Здесь показана только правдоподобная длина текста, "
    "чтобы вёрстка не поехала, когда придёт настоящее описание."
)


def collect_topics(cfg: config.Slice) -> tuple[dict, dict[int, int]]:
    """Собирает по темам: ряды по годам, организации, страны, лучшие работы."""
    topics: dict[str, dict] = defaultdict(lambda: {
        "name": None, "years": Counter(), "orgs": Counter(), "countries": Counter(),
        "best": [],
    })
    corpus_by_year: Counter[int] = Counter()

    for year_dir in sorted((cfg.raw / "openalex").glob("year=*")):
        table = pq.read_table(year_dir, schema=schema.WORKS, columns=[
            "id", "doi", "title", "publication_year", "primary_topic",
            "institution_names", "institution_countries", "cited_by_count", "venue",
        ])
        for row in table.to_pylist():
            topic = row["primary_topic"] or {}
            tid, year = topic.get("id"), row["publication_year"]
            if not tid or not year:
                continue
            corpus_by_year[year] += 1
            rec = topics[tid]
            rec["name"] = rec["name"] or topic.get("name")
            rec["years"][year] += 1
            rec["orgs"].update(n for n in (row["institution_names"] or []) if n)
            rec["countries"].update(c for c in (row["institution_countries"] or []) if c)
            rec["best"].append((row["cited_by_count"] or 0, row["title"], row["doi"],
                                year, row["venue"], row["id"]))
        del table
    return topics, corpus_by_year


def link(doi: str | None, work_id: str) -> str:
    return doi if doi else f"https://openalex.org/{work_id}"


def build_card(tid: str, rec: dict, corpus: dict[int, int], years: list[int]) -> dict:
    series, total = [], 0
    for year in years:
        works = rec["years"].get(year, 0)
        total += works
        base = corpus.get(year, 0)
        series.append({
            "year": year,
            "works": works,
            # доля темы в корпусе того же года: нормировка, без неё растёт всё
            "per_10k": round(works / base * 10_000, 1) if base else 0.0,
        })

    half = len(series) // 2
    early = sum(s["per_10k"] for s in series[:half]) or 0.01
    late = sum(s["per_10k"] for s in series[half:])
    growth = late / early

    # только по цитируемости: заголовок или издание бывают пустыми и ломают сравнение
    best = sorted(rec["best"], key=lambda b: -b[0])[:6]
    case = best[0]
    return {
        "id": tid.rsplit("/", 1)[-1],
        "title": rec["name"],
        "summary": f"Группа работ по теме «{rec['name']}».",
        "motivation": PLACEHOLDER,
        "motivation_is_placeholder": True,
        "case_example": {
            "title": case[1], "url": link(case[2], case[5]),
            "year": case[3], "venue": case[4], "cited_by": case[0],
            "is_placeholder": True,
        },
        "sources": [
            {"title": t, "url": link(d, wid), "year": y, "venue": v, "cited_by": c}
            for c, t, d, y, v, wid in best[1:]
        ],
        "first_year": min(rec["years"]) if rec["years"] else None,
        "works_total": total,
        "series": series,
        "features": {
            "normalized_growth": round(growth, 2),
            "organizations": len(rec["orgs"]),
            "countries": len(rec["countries"]),
        },
        "score": round(min(growth / 3, 1.0), 3),
        "top_organizations": [n for n, _ in rec["orgs"].most_common(5)],
        "top_countries": [c for c, _ in rec["countries"].most_common(5)],
    }


def main() -> None:
    config.setup_console()
    cfg = config.load()
    topics, corpus = collect_topics(cfg)
    years = sorted(corpus)
    print(f"тем в корпусе: {len(topics):,}, годы {years[0]}-{years[-1]}")

    cards = [build_card(tid, rec, corpus, years)
             for tid, rec in topics.items()
             if sum(rec["years"].values()) >= MIN_WORKS]
    cards.sort(key=lambda c: c["features"]["normalized_growth"], reverse=True)
    cards = cards[:TOP]
    for n, card in enumerate(cards, 1):
        card["rank"] = n

    index = {
        "is_sample": True,
        "note": "Образец формы. Группы взяты по темам OpenAlex, а не нашей "
                "кластеризацией. Числа посчитаны по настоящему корпусу; мотивация, "
                "кейс-пример и проверка на прошлом заполняются ядром позже.",
        "index_version": "sample-" + datetime.now(UTC).date().isoformat(),
        "slice_id": cfg.slice_id,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "direction": {"id": "ai", "name": cfg.direction_name,
                      "years": [years[0], years[-1]]},
        "trends": cards,
        "validation": {
            "is_placeholder": True,
            "cutoff_year": 2021,
            "metric": None,
            "baseline": {"name": "самые частые термины", "metric": None},
            "negative_controls": [
                {"name": "метавселенная", "rank": None},
                {"name": "NFT", "rank": None},
            ],
        },
    }

    out = Path("data/index/sample-index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cards)} карточек, {out.stat().st_size / 1024:.0f} КБ -> {out}")
    for c in cards[:5]:
        print(f"  {c['rank']:>2}. рост x{c['features']['normalized_growth']:<5} "
              f"{c['works_total']:>6,} работ  {c['title'][:48]}")


if __name__ == "__main__":
    main()
