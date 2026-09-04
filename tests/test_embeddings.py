"""Тесты модуля эмбеддингов без sentence-transformers/torch — только чистая логика.

Тяжёлые ML-зависимости (uv sync --extra ml) в этом окружении не установлены
намеренно: модуль обязан импортироваться и тестироваться без них, падать должен
только реальный вызов load_model()/build_embeddings().
"""
from __future__ import annotations

import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.core.embeddings as embeddings_module
from src.core.embeddings import (
    EMBEDDING_DIM,
    build_embeddings,
    build_text,
    iter_embeddable_rows,
)
from src.core.embeddings import _resolve_part_files as resolve_part_files


def test_build_text_joins_title_and_abstract_with_sep_token():
    text = build_text("Deep Learning", "A survey of methods.", "[SEP]")
    assert text == "Deep Learning[SEP]A survey of methods."


def test_build_text_handles_missing_title():
    text = build_text(None, "Just an abstract.", "[SEP]")
    assert text == "[SEP]Just an abstract."


def test_build_text_strips_surrounding_whitespace():
    text = build_text("  Title  ", "  Abstract text.  ", "[SEP]")
    assert text == "Title[SEP]Abstract text."


def test_iter_embeddable_rows_skips_empty_abstract():
    table = pa.table({
        "id": ["W1", "W2", "W3"],
        "title": ["A", "B", "C"],
        "abstract": ["real abstract", None, "   "],
        "language": ["en", "en", "en"],
    })
    rows = list(iter_embeddable_rows(table))
    assert [r["id"] for r in rows] == ["W1"]
    assert rows[0]["abstract"] == "real abstract"


def test_iter_embeddable_rows_strips_whitespace():
    table = pa.table({
        "id": ["W1"], "title": ["T"], "abstract": ["  padded text  "], "language": ["en"],
    })
    rows = list(iter_embeddable_rows(table))
    assert rows[0]["abstract"] == "padded text"


def test_iter_embeddable_rows_keeps_missing_title():
    table = pa.table({
        "id": ["W1"], "title": [None], "abstract": ["has abstract"], "language": ["en"],
    })
    rows = list(iter_embeddable_rows(table))
    assert rows[0]["title"] is None
    assert rows[0]["id"] == "W1"


def test_iter_embeddable_rows_keeps_english_and_unset_language():
    # language in ("en", None), не == "en": непроставленный язык (~5% корпуса) на 99%
    # это обычные английские статьи, где источник просто не заполнил поле — см.
    # docs/09-тз-обработка.md, раздел «1. Тексты».
    table = pa.table({
        "id": ["W1", "W2"],
        "title": ["A", "B"],
        "abstract": ["abstract one", "abstract two"],
        "language": ["en", None],
    })
    rows = list(iter_embeddable_rows(table))
    assert [r["id"] for r in rows] == ["W1", "W2"]


def test_iter_embeddable_rows_drops_other_languages():
    # Измерено: индонезийский и японский на 92-98% лежат в рубриках, не относящихся
    # к ИИ (педагогика, образовательная робототехника) — исключение обоснованно.
    # Русский устроен как нормальная периодика, но выборки (0,5%, ~70 работ на рубрику)
    # всё равно не хватит на собственный кластер, а с англоязычной моделью такой текст
    # даёт бессмысленный вектор и слипается «по языку», а не по смыслу.
    table = pa.table({
        "id": ["W1", "W2", "W3"],
        "title": ["A", "B", "C"],
        "abstract": ["текст на русском", "日本語のテキスト", "teks bahasa Indonesia"],
        "language": ["ru", "ja", "id"],
    })
    rows = list(iter_embeddable_rows(table))
    assert rows == []


def test_resolve_part_files_accepts_single_file(tmp_path):
    source = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"id": ["W1"]}), source)
    assert resolve_part_files(source) == [str(source)]


def test_resolve_part_files_globs_year_partitions(tmp_path):
    for year in (2010, 2011):
        d = tmp_path / f"year={year}"
        d.mkdir()
        pq.write_table(pa.table({"id": ["W1"]}), d / "part-000.parquet")
    files = resolve_part_files(tmp_path)
    assert len(files) == 2
    assert files == sorted(files)


def test_resolve_part_files_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_part_files(tmp_path)


def _write_fake_corpus(path, ids: list[str]) -> None:
    table = pa.table({
        "id": ids,
        "title": [f"title {i}" for i in ids],
        "abstract": [f"abstract {i}" for i in ids],
        "language": ["en"] * len(ids),
    })
    pq.write_table(table, path)


def test_build_embeddings_writes_chunked_parts_and_skips_on_resume(tmp_path, monkeypatch):
    source = tmp_path / "sample.parquet"
    _write_fake_corpus(source, ["W1", "W2", "W3", "W4", "W5"])
    out_dir = tmp_path / "emb"

    calls = {"load_model": 0, "embed_batch": 0}

    class FakeTokenizer:
        sep_token = "[SEP]"

    class FakeModel:
        tokenizer = FakeTokenizer()

    def fake_load_model(model_name=None, **kw):
        calls["load_model"] += 1
        return FakeModel()

    def fake_embed_batch(model, texts, batch_size=64):
        calls["embed_batch"] += 1
        return np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")

    monkeypatch.setattr(embeddings_module, "load_model", fake_load_model)
    monkeypatch.setattr(embeddings_module, "embed_batch", fake_embed_batch)

    n = build_embeddings(source, out_dir, chunk_size=2)

    assert n == 5
    part_files = sorted(out_dir.glob("part-*.parquet"))
    assert len(part_files) == 3  # пачки по 2, 2, 1
    assert calls == {"load_model": 1, "embed_batch": 3}

    # Повторный запуск: все пачки уже на диске — ни модель не грузится,
    # ни эмбеддинг не считается заново.
    n_again = build_embeddings(source, out_dir, chunk_size=2)
    assert n_again == 5
    assert calls == {"load_model": 1, "embed_batch": 3}


def test_build_embeddings_resumes_only_missing_chunks(tmp_path, monkeypatch):
    source = tmp_path / "sample.parquet"
    _write_fake_corpus(source, ["W1", "W2", "W3", "W4"])
    out_dir = tmp_path / "emb"
    out_dir.mkdir()

    # Пачка 0 (W1, W2) как будто уже посчитана в прошлом запуске.
    existing = pa.table(
        {"id": ["W1", "W2"], "embedding": [[0.0] * EMBEDDING_DIM, [0.0] * EMBEDDING_DIM]},
        schema=embeddings_module.EMBEDDINGS,
    )
    pq.write_table(existing, out_dir / "part-0000.parquet")

    calls = {"embed_batch": 0}

    class FakeModel:
        tokenizer = type("T", (), {"sep_token": "[SEP]"})()

    def fake_embed_batch(model, texts, batch_size=64):
        calls["embed_batch"] += 1
        return np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")

    monkeypatch.setattr(embeddings_module, "load_model", lambda model_name=None, **kw: FakeModel())
    monkeypatch.setattr(embeddings_module, "embed_batch", fake_embed_batch)

    n = build_embeddings(source, out_dir, chunk_size=2)

    assert n == 4
    assert calls["embed_batch"] == 1  # только вторая пачка (W3, W4) реально посчиталась


def test_module_imports_without_sentence_transformers():
    # Ключевая проверка этого захода: sentence_transformers не установлен
    # (uv sync --extra ml — отдельно), но модуль должен собираться.
    import src.core.embeddings  # noqa: F401


def test_load_model_gives_actionable_error_without_ml_extra(monkeypatch):
    # Импорт блокируется явно, а не проверкой «библиотеки нет в процессе»: после
    # uv sync --extra ml она есть, и такая проверка молча пропускала бы тест —
    # ровно там, где он и нужен. None в sys.modules заставляет импорт упасть.
    import src.core.embeddings as embeddings_module

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="uv sync --extra ml"):
        embeddings_module.load_model()
