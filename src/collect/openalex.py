"""Выгрузка работ OpenAlex курсором, с возобновлением после обрыва.

Проход делается отдельно на каждый год: курсор отдаёт записи вперемешку, а раскладывать
их надо по папкам year=YYYY. Побочная польза — прогресс и сверка с агрегатом видны
по годам, а не только в конце.

    uv run python -m src.collect.openalex           # весь срез
    uv run python -m src.collect.openalex --year 2015 --limit 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.collect import config, schema

API = "https://api.openalex.org/works"
PART_ROWS = 25_000        # столько строк в одном файле part-XXX.parquet
RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class Client:
    def __init__(self, cfg: config.Slice) -> None:
        self.cfg = cfg
        self.http = httpx.Client(
            timeout=90.0,
            headers={"User-Agent": f"trend-signals/0.1 (mailto:{cfg.mailto})"},
        )

    def _page(self, params: dict) -> dict:
        @retry(
            retry=retry_if_exception_type(RETRYABLE),
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=self.cfg.backoff_seconds, max=120),
            reraise=True,
        )
        def call() -> dict:
            r = self.http.get(API, params={**params, "mailto": self.cfg.mailto})
            # 429 и 5xx считаем временными, 4xx остальные — ошибкой запроса
            if r.status_code == 429 or r.status_code >= 500:
                r.raise_for_status()
            if r.status_code >= 400:
                raise SystemExit(f"OpenAlex вернул {r.status_code}: {r.text[:300]}")
            return r.json()

        return call()

    def count(self, filt: str) -> int:
        return self._page({"filter": filt, "per_page": 1})["meta"]["count"]

    def pages(self, filt: str, cursor: str = "*"):
        """Идём курсором. Первый запрос с '*', дальше — с меткой из ответа.
        Курсор возвращается наружу, чтобы его можно было сохранить и продолжить с него."""
        while cursor:
            data = self._page({
                "filter": filt, "select": schema.SELECT,
                "per-page": self.cfg.per_page, "cursor": cursor,
            })
            cursor = (data.get("meta") or {}).get("next_cursor")
            yield data["results"], cursor
            if not data["results"]:
                break


def _state_path(cfg: config.Slice, year: int) -> Path:
    return cfg.raw / "openalex" / "_state" / f"year={year}.json"


def fetch_year(client: Client, year: int, limit: int | None = None) -> dict:
    """Выгружает один год. Повторный запуск продолжает с сохранённого курсора."""
    cfg = client.cfg
    out = cfg.raw / "openalex" / f"year={year}"
    out.mkdir(parents=True, exist_ok=True)
    state_file = _state_path(cfg, year)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get("done"):
        print(f"  {year}: уже выгружен, {state['rows']:,} записей")
        return state

    filt = cfg.works_filter(year)
    expected = client.count(filt)
    cursor = state.get("cursor", "*")
    rows_done = state.get("rows", 0)
    part = state.get("part", 0)
    buffer: list[dict] = []

    def flush(next_cursor: str | None, done: bool) -> None:
        nonlocal part, buffer, rows_done
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema.WORKS)
            pq.write_table(table, out / f"part-{part:03d}.parquet", compression="zstd")
            part += 1
            rows_done += len(buffer)
            buffer = []
        state_file.write_text(json.dumps(
            {"cursor": next_cursor, "part": part, "rows": rows_done,
             "expected": expected, "done": done}, ensure_ascii=False))

    print(f"  {year}: ожидается {expected:,}", end="", flush=True)
    for results, next_cursor in client.pages(filt, cursor):
        buffer.extend(schema.to_row(w) for w in results)
        if len(buffer) >= PART_ROWS:
            flush(next_cursor, done=False)
            print(f" .{rows_done:,}", end="", flush=True)
        if limit and rows_done + len(buffer) >= limit:
            break
        if not next_cursor:
            break
    flush(None, done=not limit)

    if limit:                       # пробный запуск, полноту не проверяем
        print(f" -> выгружено {rows_done:,} (проба)")
        return {"rows": rows_done, "expected": expected, "gap": 0.0}

    gap = abs(rows_done - expected) / expected if expected else 0
    mark = "ok" if gap <= 0.02 else f"РАСХОЖДЕНИЕ {gap:.1%}"
    print(f" -> выгружено {rows_done:,} [{mark}]")
    return {"rows": rows_done, "expected": expected, "gap": gap}


def main() -> None:
    ap = argparse.ArgumentParser(description="Выгрузка среза OpenAlex")
    ap.add_argument("--year", type=int, help="только один год")
    ap.add_argument("--limit", type=int, help="остановиться после N записей (для пробы)")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()
    client = Client(cfg)
    years = [args.year] if args.year else list(cfg.years)
    print(f"срез {cfg.slice_id}: подобласти {cfg.subfield_ids}, годы {years[0]}-{years[-1]}")

    total, bad = 0, []
    for year in years:
        res = fetch_year(client, year, args.limit)
        total += res.get("rows", 0)
        if res.get("gap", 0) > 0.02:
            bad.append(year)

    print(f"\nвсего {total:,} записей в {cfg.raw / 'openalex'}")
    if bad:
        print(f"годы с расхождением больше 2%: {bad} — выгрузка неполная", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
