"""Выгрузка метаданных препринтов arXiv через открытый протокол OAI-PMH.

Учётная запись не нужна, лимита по числу запросов нет — только просьба не частить.
Страница отдаёт 1300 записей, продолжение идёт по токену из ответа, как курсор
у OpenAlex. Забираем разделы целиком, отбор по подкатегориям — дальше по конвейеру:
наборы протокола крупные, тоньше он не режет.

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

# Раздела информатики мало: заметная часть работ по машинному обучению лежит
# в stat.ML, обработка сигналов и речи — в eess. Проверено сшивкой с OpenAlex.
SETS = ("cs", "stat", "eess")

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


def read_preprints(cfg: config.Slice) -> pa.Table:
    """Читает все разделы как один корпус, без повторов.

    Работа, помеченная сразу в двух разделах, приходит из каждого — и это не
    ошибка выгрузки, а устройство протокола. Хранение остаётся пораздельным
    (так возобновляется выгрузка), а повторы снимаются на чтении.
    """
    table = pq.read_table(cfg.raw / "arxiv", schema=PREPRINTS)
    seen: set[str] = set()
    keep = [n for n, i in enumerate(table["id"].to_pylist())
            if not (i in seen or seen.add(i))]
    return table.take(keep) if len(keep) < table.num_rows else table


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


def harvest(cfg: config.Slice, oai_set: str, limit: int | None = None) -> int:
    """Выгружает один раздел arXiv в свою папку. Возвращает число записей."""
    out = cfg.raw / "arxiv" / f"set={oai_set}"
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "_state.json"

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get("done"):
        print(f"  {oai_set}: уже выгружен, {state['rows']:,} записей")
        return state["rows"]

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

    print(f"  {oai_set}: продолжаем с {rows_done:,}", end="", flush=True)
    pending = token
    try:
        while True:
            params = ({"verb": "ListRecords", "resumptionToken": token} if token
                      else {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": oai_set})
            tree = ElementTree.fromstring(fetch_page(http, params))
            page = [r for r in (parse_record(rec)
                                for rec in tree.iter(OAI_NS + "record")) if r]
            buffer.extend(page)

            node = tree.find(f"{OAI_NS}ListRecords/{OAI_NS}resumptionToken")
            token = node.text if node is not None and node.text else None
            pending = token
            if len(buffer) >= PART_ROWS:
                flush(token, done=False)
                print(f" .{rows_done:,}", end="", flush=True)
            if not token or (limit and rows_done + len(buffer) >= limit):
                break
            time.sleep(PAUSE)
    except BaseException:
        flush(pending, done=False)
        print(f" -> сохранено {rows_done:,} до обрыва")
        raise

    # Пробный запуск сохраняет токен: иначе следующий начнёт с начала и повторно
    # выгрузит уже лежащее на диске.
    flush(token if limit else None, done=not limit)
    print(f" -> {rows_done:,}")
    return rows_done


def main() -> None:
    ap = argparse.ArgumentParser(description="Метаданные препринтов arXiv")
    ap.add_argument("--sets", default=",".join(SETS),
                    help="разделы arXiv через запятую")
    ap.add_argument("--limit", type=int, help="остановиться после N записей (для пробы)")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    print(f"разделы: {', '.join(sets)}")
    total = sum(harvest(cfg, s, args.limit) for s in sets)
    print(f"\nвсего {total:,} препринтов в {cfg.raw / 'arxiv'}")


if __name__ == "__main__":
    main()
