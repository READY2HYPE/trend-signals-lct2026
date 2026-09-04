"""Агрегаты по годам — знаменатель для нормировки.

Публикаций год от года больше, поэтому формально растёт всё. Рост темы имеет смысл
только относительно роста корпуса за тот же период. Считается отдельным запросом
group_by: скачивать ради этого сам корпус не нужно.

    uv run python -m src.collect.aggregates
"""
from __future__ import annotations

import argparse
import json

import pyarrow.parquet as pq

from src.collect import config, schema
from src.collect.openalex import Client


def by_year(client: Client, filt: str) -> dict[int, int]:
    data = client._page({"filter": filt, "group_by": "publication_year", "per_page": 200})
    return {int(g["key"]): g["count"] for g in data["group_by"]}


def embeddable_by_year(cfg: config.Slice) -> dict[int, int]:
    """Сколько записей доживает до кластеризации: есть аннотация и подходит язык.

    Это и есть знаменатель для рядов внутри групп. Доля доживших меняется от 47%
    в 2016 до 75% в 2025, поэтому нормировка на нефильтрованный корпус даёт рост
    у всех групп разом. Считается по выгрузке, а не запросом: «язык не проставлен»
    в фильтре API не выражается.
    """
    root = cfg.raw / "openalex"
    if not root.exists():
        return {}
    out: dict[int, int] = {}
    for year_dir in sorted(root.glob("year=*")):
        table = pq.read_table(year_dir, schema=schema.WORKS,
                              columns=["language", "abstract", "publication_year"])
        n = sum(1 for lang, abstract in zip(table["language"].to_pylist(),
                                            table["abstract"].to_pylist())
                if abstract and lang in (None, "en"))
        out[int(year_dir.name.removeprefix("year="))] = n
        del table
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ряды по годам для нормировки")
    ap.add_argument("--force", action="store_true",
                    help="перезапросить ряды из API, затерев замороженные")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()
    out = cfg.aggregates / "works-by-year.json"

    # Ряды из API — часть замороженного среза. Источник дописывает записи задним
    # числом: пересчёт через две недели поднял расхождение с выгрузкой с нуля до 1,91%
    # при пороге приёмки в 2%. Поэтому по умолчанию они не трогаются, а обновляется
    # только ряд, который считается по самой выгрузке.
    if out.exists() and not args.force:
        saved = json.loads(out.read_text(encoding="utf-8"))
        saved["years"]["slice_embeddable"] = embeddable_by_year(cfg)
        out.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"ряды из API оставлены как есть, пересчитан slice_embeddable -> {out}")
        print("перезапросить всё: --force")
        return

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
            # знаменатель для рядов внутри групп: то, что реально идёт в кластеризацию
            "slice_embeddable": embeddable_by_year(cfg),
        },
    }

    cfg.aggregates.mkdir(parents=True, exist_ok=True)
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
