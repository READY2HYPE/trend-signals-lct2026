"""Схема таблицы работ и приведение ответа API к ней.

Схема объявлена здесь один раз и не меняется по ходу: расхождение схем между
участниками даёт ошибки, которые всплывают в последний день.
"""
from __future__ import annotations

import re
from typing import Any

import pyarrow as pa

TOPIC = pa.struct([("id", pa.string()), ("name", pa.string()), ("score", pa.float32())])

WORKS = pa.schema([
    ("id", pa.string()),                       # W2741809807
    ("doi", pa.string()),
    ("arxiv_id", pa.string()),                 # ключ сшивки с препринтами
    ("title", pa.string()),
    ("abstract", pa.string()),                 # восстановлен из инвертированного индекса
    ("publication_year", pa.int16()),
    ("publication_date", pa.string()),         # YYYY-MM-DD
    ("date_is_placeholder", pa.bool_()),       # дата ровно 1 января: день неизвестен
    ("type", pa.string()),
    ("language", pa.string()),
    ("is_oa", pa.bool_()),
    ("cited_by_count", pa.int32()),            # вспомогательный вес, не признак раннего роста
    ("primary_topic", TOPIC),
    ("primary_subfield_id", pa.int32()),
    ("topics", pa.list_(TOPIC)),               # переход между областями
    ("institution_ids", pa.list_(pa.string())),   # распространение
    ("institution_names", pa.list_(pa.string())),
    ("institution_countries", pa.list_(pa.string())),
    ("author_count", pa.int16()),
    ("venue", pa.string()),
])

# Поля, которые просим у API. Список процитированных работ не берём: он огромный
# и раздувает объём в разы, а для наших признаков не нужен.
SELECT = (
    "id,doi,ids,title,abstract_inverted_index,publication_year,publication_date,"
    "type,language,open_access,cited_by_count,primary_topic,topics,authorships,"
    "primary_location"
)

_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s/?#]+)|arxiv[:/]\s*([0-9]{4}\.[0-9]{4,5})",
                    re.IGNORECASE)


def restore_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Аннотации отдаются инвертированным индексом (слово -> позиции в тексте).
    Разворачиваем обратно в строку."""
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, places in inverted.items():
        for p in places:
            positions[p] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _short_id(url: str | None) -> str | None:
    return url.rsplit("/", 1)[-1] if url else None


def _arxiv_id(work: dict[str, Any]) -> str | None:
    haystack = " ".join(str(v) for v in (work.get("ids") or {}).values())
    loc = work.get("primary_location") or {}
    haystack += " " + str(loc.get("landing_page_url") or "") + " " + str(loc.get("pdf_url") or "")
    m = _ARXIV.search(haystack)
    if not m:
        return None
    return (m.group(1) or m.group(2)).removesuffix(".pdf")


def _topic(t: dict[str, Any] | None) -> dict[str, Any] | None:
    if not t:
        return None
    return {"id": _short_id(t.get("id")), "name": t.get("display_name"),
            "score": t.get("score")}


def to_row(work: dict[str, Any]) -> dict[str, Any]:
    """Одна запись API -> одна строка таблицы."""
    date = work.get("publication_date") or ""
    primary = work.get("primary_topic") or {}
    subfield = (primary.get("subfield") or {}).get("id") or ""

    inst_id, inst_name, inst_country = [], [], []
    seen = set()
    for a in work.get("authorships") or []:
        for inst in a.get("institutions") or []:
            key = inst.get("id")
            if key and key not in seen:
                seen.add(key)
                inst_id.append(_short_id(key))
                inst_name.append(inst.get("display_name"))
                inst_country.append(inst.get("country_code"))

    source = ((work.get("primary_location") or {}).get("source") or {})
    return {
        "id": _short_id(work.get("id")),
        "doi": work.get("doi"),
        "arxiv_id": _arxiv_id(work),
        "title": work.get("title"),
        "abstract": restore_abstract(work.get("abstract_inverted_index")),
        "publication_year": work.get("publication_year"),
        "publication_date": date or None,
        "date_is_placeholder": date.endswith("-01-01"),
        "type": work.get("type"),
        "language": work.get("language"),
        "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
        "cited_by_count": work.get("cited_by_count") or 0,
        "primary_topic": _topic(primary),
        "primary_subfield_id": int(subfield.rsplit("/", 1)[-1]) if subfield else None,
        "topics": [_topic(t) for t in (work.get("topics") or [])],
        "institution_ids": inst_id,
        "institution_names": inst_name,
        "institution_countries": inst_country,
        "author_count": len(work.get("authorships") or []),
        "venue": source.get("display_name"),
    }
