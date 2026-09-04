"""Просмотр выгрузки из терминала: parquet двойным кликом не открывается.

    uv run python -m src.collect.peek                       десять работ подряд
    uv run python -m src.collect.peek --year 2015 -n 3      конкретный год
    uv run python -m src.collect.peek --search transformer  поиск по заголовку
    uv run python -m src.collect.peek --topic graph         поиск по теме
    uv run python -m src.collect.peek --preprints           то же по препринтам
    uv run python -m src.collect.peek --stats               сводка по срезу
    uv run python -m src.collect.peek --columns             какие есть столбцы

Читает подвыборку, если полной выгрузки нет: она лежит в репозитории и приходит
вместе с клоном.
"""
from __future__ import annotations

import argparse
import textwrap

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.collect import arxiv, config, schema


def source(cfg: config.Slice, year: int | None, preprints: bool):
    """Возвращает таблицу и подпись, откуда она взялась."""
    if preprints:
        if not (cfg.raw / "arxiv").exists():
            raise SystemExit("Препринтов нет: uv run python -m src.collect.arxiv")
        return arxiv.read_preprints(cfg), "препринты arXiv"

    root = cfg.raw / "openalex"
    if year is not None:
        path = root / f"year={year}"
        if not path.exists():
            raise SystemExit(f"Года {year} нет в выгрузке.")
        return pq.read_table(path, schema=schema.WORKS), f"работы за {year}"
    if root.exists():
        return pq.read_table(root, schema=schema.WORKS), "весь срез"
    sample = cfg.sample / "openalex-sample.parquet"
    if sample.exists():
        return pq.read_table(sample, schema=schema.WORKS), "подвыборка (полной выгрузки нет)"
    raise SystemExit("Данных нет. Выгрузка: uv run python -m src.collect.openalex")


def show_work(r: dict) -> None:
    topic = r.get("primary_topic") or {}
    print(f"\n  {textwrap.shorten(r['title'] or 'без заголовка', 88)}")
    print(f"  {r['publication_date']}  ·  {r['type']}  ·  {r['language'] or '?'}"
          f"  ·  ссылок {r['cited_by_count']}")
    if topic.get("name"):
        print(f"  тема: {topic['name']} ({topic.get('score', 0):.2f})")
    orgs = r.get("institution_names") or []
    if orgs:
        cc = ", ".join(dict.fromkeys(c for c in (r.get("institution_countries") or []) if c))
        print(f"  {textwrap.shorten(', '.join(orgs), 74)}  [{cc}]")
    if r.get("abstract"):
        print(f"  {textwrap.shorten(r['abstract'], 200)}")
    else:
        print("  аннотации нет")
    print(f"  {r.get('doi') or 'https://openalex.org/' + r['id']}"
          + (f"  ·  arXiv {r['arxiv_id']}" if r.get("arxiv_id") else ""))


def show_preprint(r: dict) -> None:
    print(f"\n  {textwrap.shorten(r['title'] or 'без заголовка', 88)}")
    print(f"  первая версия {r['created']}  ·  правка {r['updated'] or '—'}"
          f"  ·  авторов {r['author_count']}")
    print(f"  категории: {' '.join(r['categories'] or [])}")
    print(f"  {textwrap.shorten(r['abstract'] or 'аннотации нет', 200)}")
    print(f"  https://arxiv.org/abs/{r['id']}")


def stats(table, label: str) -> None:
    n = table.num_rows
    print(f"\n{label}: {n:,} записей\n")
    if "publication_year" in table.column_names:
        years = table["publication_year"].to_pylist()
        print(f"  годы            {min(years)}–{max(years)}")
        print(f"  с аннотацией    {pc.sum(pc.is_valid(table['abstract'])).as_py() / n:.0%}")
        print(f"  с DOI           {pc.sum(pc.is_valid(table['doi'])).as_py() / n:.0%}")
        print(f"  связаны с arXiv {pc.sum(pc.is_valid(table['arxiv_id'])).as_py() / n:.0%}")
        kinds = pc.value_counts(table["type"])
        print("  типы            " + ", ".join(
            f"{v['values']} {v['counts']:,}" for v in kinds.to_pylist()))
    else:
        created = [c[:4] for c in table["created"].to_pylist() if c]
        print(f"  первая версия   {min(created)}–{max(created)}")
        print(f"  с аннотацией    {pc.sum(pc.is_valid(table['abstract'])).as_py() / n:.0%}")
        print(f"  с DOI           {pc.sum(pc.is_valid(table['doi'])).as_py() / n:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Посмотреть выгрузку глазами")
    ap.add_argument("-n", type=int, default=10, help="сколько записей показать")
    ap.add_argument("--year", type=int, help="только этот год")
    ap.add_argument("--search", help="подстрока в заголовке")
    ap.add_argument("--topic", help="подстрока в названии темы или категории")
    ap.add_argument("--preprints", action="store_true", help="смотреть препринты, а не работы")
    ap.add_argument("--stats", action="store_true", help="сводка вместо записей")
    ap.add_argument("--columns", action="store_true", help="какие есть столбцы")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()
    table, label = source(cfg, args.year, args.preprints)

    if args.columns:
        print(f"\n{label} — столбцы:\n")
        for f in table.schema:
            print(f"  {f.name:<24} {f.type}")
        return
    if args.stats:
        stats(table, label)
        return

    rows = table.to_pylist()
    if args.search:
        s = args.search.lower()
        rows = [r for r in rows if s in (r["title"] or "").lower()]
    if args.topic:
        s = args.topic.lower()
        rows = [r for r in rows if s in (
            " ".join(r["categories"] or []) if args.preprints
            else (r["primary_topic"] or {}).get("name") or "").lower()]

    print(f"\n{label}: подходит {len(rows):,} из {table.num_rows:,}, показаны первые {args.n}")
    show = show_preprint if args.preprints else show_work
    for r in rows[:args.n]:
        show(r)
    if not rows:
        print("\n  ничего не нашлось")
    print()


if __name__ == "__main__":
    main()
