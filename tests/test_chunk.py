from langchain_core.documents import Document

from threadneedle.indexing.chunk import _is_noise, sanitize_metadata


def test_noise_drops_short_and_numeric_rows():
    assert _is_noise("ok", min_chars=200) is True
    assert _is_noise("12.3  4.5  6.7  8.9  10.1", min_chars=10) is True
    prose = "Services inflation remains elevated relative to the 2% target. " * 8
    assert _is_noise(prose, min_chars=200) is False


def test_sanitize_metadata_drops_none_and_flattens_lists():
    chunks = [
        Document(
            page_content="x",
            metadata={
                "title": "MPR",
                "empty": None,
                "tags": ["a", "b"],
                "page": 3,
            },
        )
    ]
    clean = sanitize_metadata(chunks)
    assert "empty" not in clean[0].metadata
    assert clean[0].metadata["tags"] == "a, b"
    assert clean[0].metadata["page"] == 3
