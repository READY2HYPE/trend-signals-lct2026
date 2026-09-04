"""Строит эмбеддинги корпуса для кластеризации.

Тяжёлые ML-зависимости ставятся отдельно:

    uv sync --extra ml

Дальше — на подвыборке (быстрая проверка, что вообще работает) или на полном
корпусе (час-два на видеокарте, ночь на процессоре):

    uv run python scripts/build_embeddings.py --sample
    uv run python scripts/build_embeddings.py

Пишет пачками в data/embeddings/<sample|openalex>/part-NNNN.parquet. Если процесс
прервать (или он упадёт), повторный запуск той же командой продолжит с последней
незаконченной пачки — уже готовые part-файлы не пересчитываются.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect import config
from src.core.embeddings import BATCH_SIZE, CHUNK_SIZE, MODEL_NAME, build_embeddings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true",
                     help="подвыборка data/sample/ вместо полного data/raw/openalex/")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--fp16", action="store_true",
                     help="половинная точность: втрое быстрее на видеокарте, "
                          "для быстрых прогонов на подвыборке")
    args = ap.parse_args()

    config.setup_console()
    cfg = config.load()

    if args.sample:
        source = cfg.sample / "openalex-sample.parquet"
        out_dir = Path("data/embeddings/sample")
    else:
        source = cfg.raw / "openalex"
        out_dir = Path("data/embeddings/openalex")

    print(f"источник: {source}")
    print(f"модель: {args.model}, пачка: {args.batch_size}, чанк: {args.chunk_size}")
    started = time.monotonic()
    n = build_embeddings(source, out_dir, args.model, args.batch_size, args.chunk_size,
                         half=args.fp16)
    elapsed = time.monotonic() - started
    print(f"\nготово: {n:,} эмбеддингов -> {out_dir}, {elapsed / 60:.1f} мин")


if __name__ == "__main__":
    main()
