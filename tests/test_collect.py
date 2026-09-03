"""Быстрые проверки на подвыборке — прогоняются перед слиянием веток."""
from __future__ import annotations

import pytest

from src.collect import config, schema


def test_abstract_restored_from_inverted_index():
    inverted = {"Deep": [0], "models": [2], "learning": [1], "scale": [3]}
    assert schema.restore_abstract(inverted) == "Deep learning models scale"


def test_abstract_missing_gives_none():
    assert schema.restore_abstract(None) is None
    assert schema.restore_abstract({}) is None


@pytest.mark.parametrize("ids, expected", [
    ({"openalex": "https://openalex.org/W1", "arxiv": "https://arxiv.org/abs/2104.14399"},
     "2104.14399"),
    ({"openalex": "https://openalex.org/W2", "doi": "https://doi.org/10.1/x"}, None),
])
def test_arxiv_id_extracted_from_links(ids, expected):
    assert schema.to_row({"ids": ids})["arxiv_id"] == expected


def test_january_first_marked_as_placeholder():
    assert schema.to_row({"publication_date": "2015-01-01"})["date_is_placeholder"]
    assert not schema.to_row({"publication_date": "2015-03-12"})["date_is_placeholder"]


def test_institutions_deduplicated_within_work():
    work = {"authorships": [
        {"institutions": [{"id": "https://openalex.org/I1", "display_name": "МГУ",
                           "country_code": "RU"}]},
        {"institutions": [{"id": "https://openalex.org/I1", "display_name": "МГУ",
                           "country_code": "RU"}]},
    ]}
    row = schema.to_row(work)
    assert row["institution_ids"] == ["I1"]
    assert row["author_count"] == 2


def test_year_filter_is_single_year_when_asked():
    cfg = config.load()
    assert "publication_year:2015" in cfg.works_filter(2015)
    assert f"publication_year:{cfg.year_start}-{cfg.year_end}" in cfg.works_filter()


def test_sample_matches_schema_if_present():
    cfg = config.load()
    path = cfg.sample / "openalex-sample.parquet"
    if not path.exists():
        pytest.skip("подвыборка ещё не собрана")
    import pyarrow.parquet as pq
    assert pq.read_schema(path).equals(schema.WORKS)
