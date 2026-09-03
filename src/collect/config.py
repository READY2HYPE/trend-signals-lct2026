"""Чтение config/slice.yaml и .env. Единственное место, где срез описан словами."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Slice:
    slice_id: str
    subfield_ids: list[int]
    year_start: int
    year_end: int
    types: list[str]
    min_topic_score: float
    per_page: int
    max_retries: int
    backoff_seconds: float
    raw: Path
    aggregates: Path
    sample: Path
    manifest: Path
    mailto: str

    @property
    def years(self) -> range:
        return range(self.year_start, self.year_end + 1)

    def works_filter(self, year: int | None = None) -> str:
        """Строка filter= для OpenAlex. Записи без аннотации не отсекаем:
        они не пойдут в кластеризацию, но нужны в счётчиках по годам."""
        parts = [
            "primary_topic.subfield.id:" + "|".join(str(i) for i in self.subfield_ids),
            "type:" + "|".join(self.types),
        ]
        parts.append(f"publication_year:{year}" if year else
                     f"publication_year:{self.year_start}-{self.year_end}")
        return ",".join(parts)


def setup_console() -> None:
    """Консоль Windows по умолчанию не в UTF-8, и русский вывод её роняет."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path | str = ROOT / "config" / "slice.yaml") -> Slice:
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if not mailto or "example.com" in mailto:
        raise SystemExit(
            "Не задан OPENALEX_MAILTO. Скопировать .env.example в .env и вписать почту:\n"
            "без неё OpenAlex режет скорость и выгрузка растянется на сутки."
        )
    p = cfg["paths"]
    return Slice(
        slice_id=cfg["slice_id"],
        subfield_ids=cfg["direction"]["openalex_subfield_ids"],
        year_start=cfg["years"]["start"],
        year_end=cfg["years"]["end"],
        types=cfg["filters"]["types"],
        min_topic_score=float(cfg["filters"].get("min_primary_topic_score", 0.0)),
        per_page=int(cfg["fetch"]["per_page"]),
        max_retries=int(cfg["fetch"]["max_retries"]),
        backoff_seconds=float(cfg["fetch"]["backoff_seconds"]),
        raw=ROOT / p["raw"],
        aggregates=ROOT / p["aggregates"],
        sample=ROOT / p["sample"],
        manifest=ROOT / p["manifest"],
        mailto=mailto,
    )
