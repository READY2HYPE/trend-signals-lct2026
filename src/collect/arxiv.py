"""Выгрузка метаданных препринтов arXiv через открытый протокол OAI-PMH.

Учётная запись не нужна, лимита по числу запросов нет — только просьба не частить.
Страница отдаёт 1300 записей, продолжение идёт по токену из ответа, как курсор
у OpenAlex. Забираем весь раздел информатики целиком, отбор по подкатегориям —
дальше по конвейеру: наборы протокола крупные, тоньше он не режет.

Берём `created` — дату первой версии. Нас интересует момент первого появления
работы, а не дата последней правки.

    uv run python -m src.collect.arxiv
"""
from __future__ import annotations

import argparse
import json
import time
from xml.etree import ElementTree

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from src.collect import config

OAI = "https://oaipmh.arxiv.org/oai"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
ARX_NS = "{http://arxiv.org/OAI/arXiv/}"
PAUSE = 3.0               # просьба arXiv не частить
PART_ROWS = 100_000

PREPRINTS = pa.schema([
    ("id", pa.string()),                  # 2104.14399 или cs/0501001
    ("created", pa.string()),             # дата первой версии — ради неё всё и затевалось
    ("updated", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("primary_category", pa.string()),
    ("doi", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("author_count", pa.int16()),
])


def _text(node, tag: str) -> str | None:
    found = node.find(ARX_NS + tag)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split())


def parse_record(record) -> dict | None:
    header = record.find(OAI_NS + "header")
    if header is not None and header.get("status") == "deleted":
        return None
    meta = record.find(f"{OAI_NS}metadata/{ARX_NS}arXiv")
    if meta is None:
        return None

    cats = (_text(meta, "categories") or "").split()
    authors = []
    block = meta.find(ARX_NS + "authors")
    for a in (block if block is not None else []):
        name = " ".join(filter(None, [
            (a.findtext(ARX_NS + "forenames") or "").strip(),
            (a.findtext(ARX_NS + "keyname") or "").strip(),
        ]))
        if name:
            authors.append(name)

    return {
        "id": _text(meta, "id"),
        "created": _text(meta, "created"),
        "updated": _text(meta, "updated"),
        "title": _text(meta, "title"),
        "abstract": _text(meta, "abstract"),
        "categories": cats,
        "primary_category": cats[0] if cats else None,
        "doi": _text(meta, "doi"),
        "authors": authors,
        "author_count": len(authors),
    }


def fetch_page(http: httpx.Client, params: dict) -> str:
    """Протокол просит подождать и повторить, когда сервер занят: это 503 с Retry-After."""
    for _ in range(10):
        r = http.get(OAI, params=params)
        if r.status_code == 503:
            pause = int(r.headers.get("retry-after") or 20)
            print(f"  сервер занят, жду {pause} с", flush=True)
            time.sleep(pause + 1)
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError("arXiv десять раз подряд ответил «занято»")


def main() -> None:
    ap = argparse.ArgumentParser(description="Метаданные препринтов arXiv")
    ap.add_argument("--set", default="cs", help="раздел arXiv (по умолчанию информатика)")
    ap.add_argument("--limit", type=int, help="остановиться после N записей (для пробы)")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()
    out = cfg.raw / "arxiv"
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "_state.json"

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get("done"):
        print(f"уже выгружено: {state['rows']:,} записей")
        return

    token = state.get("token")
    rows_done, part = state.get("rows", 0), state.get("part", 0)
    buffer: list[dict] = []
    http = httpx.Client(timeout=180.0,
                        headers={"User-Agent": f"trend-signals/0.1 (mailto:{cfg.mailto})"})

    def flush(next_token: str | None, done: bool) -> None:
        nonlocal buffer, rows_done, part
        if buffer:
            pq.write_table(pa.Table.from_pylist(buffer, schema=PREPRINTS),
                           out / f"part-{part:03d}.parquet", compression="zstd")
            part += 1
            rows_done += len(buffer)
            buffer = []
        state_file.write_text(json.dumps(
            {"token": next_token, "part": part, "rows": rows_done, "done": done},
            ensure_ascii=False))

    print(f"раздел {args.set}, продолжаем с {rows_done:,}")
    pending = token
    try:
        while True:
            params = ({"verb": "ListRecords", "resumptionToken": token} if token
                      else {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": args.set})
            tree = ElementTree.fromstring(fetch_page(http, params))
            page = [r for r in (parse_record(rec)
                                for rec in tree.iter(OAI_NS + "record")) if r]
            buffer.extend(page)

            node = tree.find(f"{OAI_NS}ListRecords/{OAI_NS}resumptionToken")
            token = node.text if node is not None and node.text else None
            pending = token
            if len(buffer) >= PART_ROWS:
                flush(token, done=False)
                print(f"  {rows_done:,}", flush=True)
            if not token or (args.limit and rows_done + len(buffer) >= args.limit):
                break
            time.sleep(PAUSE)
    except BaseException:
        flush(pending, done=False)
        print(f"  сохранено {rows_done:,} до обрыва")
        raise

    flush(None, done=not args.limit)
    print(f"\nвсего {rows_done:,} препринтов в {out}")


if __name__ == "__main__":
    main()
