"""Агрегаты по годам — знаменатель для нормировки.

Публикаций год от года больше, поэтому формально растёт всё. Рост темы имеет смысл
только относительно роста корпуса за тот же период. Считается отдельным запросом
group_by: скачивать ради этого сам корпус не нужно.

    uv run python -m src.collect.aggregates
"""
from __future__ import annotations

import json

from src.collect import config
from src.collect.openalex import Client


def by_year(client: Client, filt: str) -> dict[int, int]:
    data = client._page({"filter": filt, "group_by": "publication_year", "per_page": 200})
    return {int(g["key"]): g["count"] for g in data["group_by"]}


def main() -> None:
    config.setup_console()
    cfg = config.load()
    client = Client(cfg)
    years = f"publication_year:{cfg.year_start}-{cfg.year_end}"

    result = {
        "slice_id": cfg.slice_id,
        "source": "openalex",
        "years": {
            # весь корпус источника — знаменатель нормировки
            "corpus_all": by_year(client, years),
            # то же, но только статьи и препринты: сопоставимо с нашим срезом по типам
            "corpus_articles": by_year(client, f"{years},type:" + "|".join(cfg.types)),
            # наш срез целиком и его часть с аннотацией
            "slice_all": by_year(client, cfg.works_filter()),
            "slice_with_abstract": by_year(client, cfg.works_filter() + ",has_abstract:true"),
        },
    }

    cfg.aggregates.mkdir(parents=True, exist_ok=True)
    out = cfg.aggregates / "works-by-year.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    y = result["years"]
    print(f"{'год':>5} {'корпус':>12} {'статьи':>12} {'срез':>10} {'с аннот.':>10} {'доля':>7}")
    for year in cfg.years:
        s, a = y["slice_all"].get(year, 0), y["slice_with_abstract"].get(year, 0)
        print(f"{year:>5} {y['corpus_all'].get(year, 0):>12,} "
              f"{y['corpus_articles'].get(year, 0):>12,} {s:>10,} {a:>10,} "
              f"{a / s if s else 0:>7.0%}")
    print(f"\nзаписано в {out}")


if __name__ == "__main__":
    main()
