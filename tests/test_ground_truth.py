"""Проверка формата эталонных файлов для ретроспективной проверки (docs/02-методология.md).

Файлы статичные и коммитятся в репозиторий — тест ловит не смысловые ошибки,
а поломку формата при будущей правке (пропущенное поле, не тот тип).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground-truth"


def test_breakthrough_technologies_shape():
    data = json.loads((GROUND_TRUTH / "mit_breakthrough_technologies.json").read_text(encoding="utf-8"))
    assert data["entries"], "список пуст"
    for e in data["entries"]:
        assert isinstance(e["year"], int)
        assert 2000 < e["year"] < 2030
        assert e["name"]

    years = {e["year"] for e in data["entries"]}
    # 2002 — задокументированный пробел (см. gap_years_note), не ошибка сборки.
    assert 2001 in years
    assert max(years) >= 2025


def test_negative_controls_shape():
    data = json.loads((GROUND_TRUTH / "negative_controls.json").read_text(encoding="utf-8"))
    assert data["entries"]
    names = {e["name"] for e in data["entries"]}
    # Три примера, названные прямо в методологии — их отсутствие означает,
    # что кто-то случайно затёр файл при правке.
    assert "метавселенная" in names
    assert "NFT" in names
    for e in data["entries"]:
        assert isinstance(e["hype_year"], int)
        assert e["note"]
