"""Векторные представления работ для кластеризации.

Без этого шага нечего кластеризовать по смыслу: у зарождающейся темы устоявшегося
названия ещё нет (docs/02-методология.md), поэтому отбор по ключевым словам не
работает — по словам её пока не существует. Эмбеддинги делают «близость по смыслу»
измеримым числом.

Модель — allenai-specter: обучена контрастно на графе цитирования (работы, которые
цитируют друг друга, оказываются близко в пространстве представлений). Это ровно
задача «сгруппировать научные работы по теме», а не общая языковая близость текста,
которую даёт модель общего назначения.

Тяжёлые зависимости (sentence-transformers, torch) ставятся отдельно:

    uv sync --extra ml
    uv run python scripts/build_embeddings.py

Пишет пачками в data/embeddings/openalex/part-NNNN.parquet и пропускает уже
готовые пачки при повторном запуске — можно спокойно остановить и продолжить.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

MODEL_NAME = "sentence-transformers/allenai-specter"
EMBEDDING_DIM = 768

# Замерено на RTX 2060 Super, 1024 текста подвыборки: 32 -> 348 текстов/с,
# 64 -> 290, 96 -> 273, 256 -> 13. Крупная пачка здесь не ускоряет, а тормозит:
# на 256 занято 5 ГБ из 8, и драйвер начинает вытеснять память карты в
# оперативную. Значение подобрано измерением, менять его стоит тоже измерением.
BATCH_SIZE = 32

# Компромисс: крупные пачки эффективнее для GPU, но при обрыве (машина без присмотра
# час-два) переделывать придётся не больше одной пачки, а не всё заново.
CHUNK_SIZE = 5000

EMBEDDINGS = pa.schema([
    ("id", pa.string()),
    ("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
])


def build_text(title: str | None, abstract: str, sep_token: str) -> str:
    """SPECTER обучена на строке `title + [SEP] + abstract` — порядок и разделитель
    это конвенция самой модели, взятая из её обучающих данных, а не произвольный
    выбор форматирования."""
    return f"{(title or '').strip()}{sep_token}{abstract.strip()}"


# language in ("en", None), не == "en" — см. docs/09-тз-обработка.md, раздел «1. Тексты».
# Строгое сравнение с "en" молча теряет ~53 тыс. работ (4,9% корпуса) с непроставленным
# language, из которых 99% на латинице — обычные английские статьи, просто источник не
# заполнил поле. Языки, реально не относящиеся к теме (индонезийский, японский), измерены
# отдельно и на 92-98% сидят в не-ИИ рубриках — их отсечение обоснованно, а не произвольно.
EMBEDDABLE_LANGUAGES = ("en", None)


def iter_embeddable_rows(table: pa.Table) -> Iterator[dict]:
    """Отдаёт только записи с непустой аннотацией на английском (или без языка,
    что практически всегда тоже английский — см. EMBEDDABLE_LANGUAGES).

    Записи без аннотации не отсекаются при сборе корпуса (src/collect/config.py:
    "они не пойдут в кластеризацию, но нужны в счётчиках по годам") — здесь именно
    та точка, где это решение реализуется: эмбеддинг по одному заголовку не несёт
    содержательного сигнала для кластеризации, поэтому такие записи пропускаются,
    а не эмбеддятся частично.

    Не-английский текст (кроме непроставленного языка) исключён сознательно, не
    только из-за состава подобласти: с англоязычной моделью представлений такой
    текст даёт бессмысленный вектор, и такие записи слипаются в кластер «по языку»,
    а не по смыслу — это хуже, чем их отсутствие. Язык самих карточек и интерфейса
    этот фильтр не затрагивает — он всегда русский, независимо от языка источников.
    """
    for row in table.select(["id", "title", "abstract", "language"]).to_pylist():
        if row.get("language") not in EMBEDDABLE_LANGUAGES:
            continue
        abstract = (row.get("abstract") or "").strip()
        if not abstract:
            continue
        yield {"id": row["id"], "title": row.get("title"), "abstract": abstract}


def load_model(model_name: str = MODEL_NAME, *, half: bool | None = None):
    """Импорт sentence_transformers внутри функции, не на уровне модуля: модуль
    должен собираться и тестироваться даже без поставленного `ml`-экстра — падать
    это должно только в момент реального запуска, с понятным сообщением.

    На видеокарте модель переводится в половинную точность (half=None означает
    «решить по наличию видеокарты»). Это втрое быстрее: 348 текстов/с против 103
    на той же карте, полный корпус 47 минут вместо 158. Проверено, что счёт от
    этого не портится: на 4096 векторах ни одного «не числа», косинус с полной
    точностью не ниже 0,9995, а из десяти ближайших соседей совпадает в среднем
    9,95 — то есть соседство, на котором держится кластеризация, сохраняется.
    На процессоре половинная точность бесполезна (часть операций там считается
    медленнее полной), поэтому включается только вместе с картой.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers не установлен. Тяжёлые ML-зависимости ставятся "
            "отдельно: uv sync --extra ml"
        ) from exc

    model = SentenceTransformer(model_name)
    on_gpu = torch.cuda.is_available()
    if half is None:
        half = on_gpu
    if half:
        model.half()

    # Точность и устройство печатаются, а не подразумеваются: молчаливый откат
    # на процессор выглядит как «просто медленно идёт» и стоит суток счёта.
    print(f"устройство: {model.device}, точность: {'half' if half else 'full'}"
          + ("" if on_gpu else " (видеокарта не найдена — считает процессор)"), flush=True)
    return model


def embed_batch(model, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    # normalize_embeddings=True: кластеризация ниже по конвейеру (UMAP) по умолчанию
    # работает в евклидовой метрике — на единичных векторах она эквивалентна
    # косинусной близости, которая для эмбеддингов текста содержательнее евклидовой.
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def _chunk_path(out_dir: Path, chunk_index: int) -> Path:
    return out_dir / f"part-{chunk_index:04d}.parquet"


def _resolve_part_files(source: Path) -> list[str]:
    """source — либо один файл (подвыборка), либо каталог с партициями
    year=*/part-*.parquet (полный сырой корпус). Оба варианта — один и тот же
    формат таблицы (src/collect/schema.py:WORKS), различается только раскладка
    по файлам, так что дальше по коду они не различаются вовсе."""
    source = Path(source)
    if source.is_file():
        return [str(source)]
    part_files = sorted(str(p) for p in source.glob("year=*/part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"нет parquet-файлов под {source} — сначала соберите корпус")
    return part_files


def build_embeddings(
    source: Path,
    out_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE,
    half: bool | None = None,
) -> int:
    """Читает работы из source (файл подвыборки или каталог data/raw/openalex/),
    эмбеддит записи на английском (или без языка) с непустой аннотацией, пишет
    id + вектор пачками в out_dir/part-NNNN.parquet. Возвращает число
    заэмбеженных записей.

    Пишет по частям и пропускает уже готовые part-файлы при повторном запуске:
    процесс идёт без присмотра, час-два на GPU (или ночь на CPU), и терять всю
    работу из-за обрыва посреди — не вариант. Пачки строятся от отсортированного
    по id списка, поэтому разбиение стабильно между запусками, пока не меняются
    chunk_size или сам корпус.
    """
    part_files = _resolve_part_files(source)
    dataset = ds.dataset(part_files, format="parquet")
    table = dataset.to_table(columns=["id", "title", "abstract", "language"])

    rows = list(iter_embeddable_rows(table))
    if not rows:
        raise ValueError("ни одной записи с непустой аннотацией на английском — нечего эмбеддить")
    rows.sort(key=lambda r: r["id"])

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    model = None
    sep_token = None
    total_done = 0

    for idx, chunk in enumerate(chunks):
        part_path = _chunk_path(out_dir, idx)
        if part_path.exists():
            total_done += pq.ParquetFile(part_path).metadata.num_rows
            continue

        if model is None:  # модель грузится один раз, лениво — не нужна вообще, если всё уже посчитано
            model = load_model(model_name, half=half)
            sep_token = getattr(model.tokenizer, "sep_token", None) or "[SEP]"

        texts = [build_text(r["title"], r["abstract"], sep_token) for r in chunk]
        vectors = embed_batch(model, texts, batch_size)

        chunk_table = pa.Table.from_pylist(
            [{"id": r["id"], "embedding": v.tolist()} for r, v in zip(chunk, vectors, strict=True)],
            schema=EMBEDDINGS,
        )
        tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
        pq.write_table(chunk_table, tmp_path, compression="zstd")
        tmp_path.replace(part_path)  # атомарно — как и остальная запись в конвейере

        total_done += len(chunk)
        print(f"  пачка {idx + 1}/{len(chunks)}: {total_done:,}/{len(rows):,}", flush=True)

    return total_done
