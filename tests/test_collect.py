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


ARXIV_RECORD = """<record xmlns="http://www.openarchives.org/OAI/2.0/">
  <header><identifier>oai:arXiv.org:2104.14399</identifier></header>
  <metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/">
    <id>2104.14399</id><created>2021-04-27</created><updated>2022-01-10</updated>
    <authors><author><keyname>Иванов</keyname><forenames>Пётр</forenames></author>
             <author><keyname>Smith</keyname><forenames>Jane</forenames></author></authors>
    <title>A study on
       weak signals</title>
    <categories>cs.AI cs.LG</categories>
    <abstract>Короткая аннотация.</abstract>
  </arXiv></metadata></record>"""

DELETED_RECORD = """<record xmlns="http://www.openarchives.org/OAI/2.0/">
  <header status="deleted"><identifier>oai:arXiv.org:0704.0001</identifier></header>
</record>"""


def test_preprint_parsed_with_first_version_date():
    from xml.etree import ElementTree

    from src.collect import arxiv
    row = arxiv.parse_record(ElementTree.fromstring(ARXIV_RECORD))
    assert row["id"] == "2104.14399"
    assert row["created"] == "2021-04-27"      # первая версия, не последняя правка
    assert row["updated"] == "2022-01-10"
    assert row["categories"] == ["cs.AI", "cs.LG"]
    assert row["primary_category"] == "cs.AI"
    assert row["authors"] == ["Пётр Иванов", "Jane Smith"]
    assert row["title"] == "A study on weak signals"    # перенос строки схлопнут
    assert row["doi"] is None


def test_deleted_preprint_skipped():
    from xml.etree import ElementTree

    from src.collect import arxiv
    assert arxiv.parse_record(ElementTree.fromstring(DELETED_RECORD)) is None


def test_preprints_deduplicated_across_sets(tmp_path):
    """Работа, помеченная в двух разделах, приходит из каждого — на чтении повтор снимается."""
    import dataclasses

    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.collect import arxiv

    def row(pid):
        return {"id": pid, "created": "2021-04-27", "updated": None, "title": "t",
                "abstract": "a", "categories": ["cs.LG", "stat.ML"],
                "primary_category": "cs.LG", "doi": None, "authors": [], "author_count": 0}

    for name, ids in [("cs", ["1", "2"]), ("stat", ["2", "3"])]:
        d = tmp_path / "arxiv" / f"set={name}"
        d.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row(i) for i in ids], schema=arxiv.PREPRINTS),
                       d / "part-000.parquet")

    cfg = dataclasses.replace(config.load(), raw=tmp_path)
    table = arxiv.read_preprints(cfg)
    assert table.num_rows == 3
    assert sorted(table["id"].to_pylist()) == ["1", "2", "3"]
