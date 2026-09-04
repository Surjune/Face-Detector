"""Yandex reverse image search.

Yandex is here for one reason: Google restricts public face matching for
private individuals, so Lens alone works for celebrities and fails for
everyone else. These tests cover the parsing; whether Yandex finds a given
person is a property of its index, not of this code.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.search.yandex as yandex_module
from src.config import MAX_CANDIDATES
from src.errors import SearchNotConfigured, SearchProviderError
from src.search.provider import Source
from src.search.yandex import (
    YandexReverseImageProvider,
    extract_identity,
    parse_yandex_response,
)


def result(index: int, **overrides: Any) -> dict[str, Any]:
    entry = {
        "title": f"Result {index}",
        "link": f"https://www.instagram.com/p/POST{index}/",
        "source": "instagram.com",
        "original_image": {"link": f"https://cdn.example.com/full/{index}.jpg"},
        "thumbnail": {"link": f"https://cdn.example.com/thumb/{index}.jpg"},
    }
    entry.update(overrides)
    return entry


def payload(images: int = 2, similar: int = 2) -> dict[str, Any]:
    return {
        "image_results": [result(i) for i in range(images)],
        "similar_images": [
            result(100 + i, link=f"https://www.tiktok.com/@u/video/{i}")
            for i in range(similar)
        ],
    }


class FakeJson:
    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self._data = data
        self.text = str(data)

    def json(self) -> dict[str, Any]:
        return self._data


class TestParsing:
    def test_reads_image_results(self) -> None:
        origins = {item.origin for item in parse_yandex_response(payload())}
        assert origins == {Source.YANDEX_IMAGE}

    def test_ignores_similar_images(self) -> None:
        """They carry no source page, so they can never be citable evidence.

        Their `link` points back into yandex.com/images/search rather than at
        the page the photograph came from, so a match there could not be
        anchored as a social media post however well its face scored.
        """
        data = {
            "similar_images": [
                {
                    "image": {"link": "https://avatars.mds.yandex.net/i?id=abc"},
                    "link": "https://yandex.com/images/search?url=x&img_url=y",
                }
            ]
        }
        assert parse_yandex_response(data) == []

    def test_unwraps_the_nested_image_links(self) -> None:
        """Yandex nests its URLs one level deeper than Lens does."""
        candidate = parse_yandex_response(payload(1, 0))[0]
        assert candidate.image_urls == (
            "https://cdn.example.com/full/0.jpg",
            "https://cdn.example.com/thumb/0.jpg",
        )

    def test_prefers_the_full_image_then_the_thumbnail(self) -> None:
        candidate = parse_yandex_response(payload(1, 0))[0]
        assert candidate.image_url.startswith("https://cdn.example.com/full/")

    def test_accepts_a_plain_string_image_url(self) -> None:
        """The field is sometimes a bare string rather than an object."""
        data = {"image_results": [result(0, original_image="https://cdn.example.com/x.jpg")]}
        assert parse_yandex_response(data)[0].image_urls[0] == "https://cdn.example.com/x.jpg"

    def test_skips_entries_with_no_image(self) -> None:
        data = {"image_results": [result(0, original_image=None, thumbnail=None)]}
        assert parse_yandex_response(data) == []

    def test_skips_entries_with_no_link(self) -> None:
        data = {"image_results": [result(0, link=None)]}
        assert parse_yandex_response(data) == []

    def test_deduplicates_by_page_url(self) -> None:
        data = {"image_results": [result(0), result(0)]}
        assert len(parse_yandex_response(data)) == 1

    def test_respects_the_candidate_cap(self) -> None:
        assert len(parse_yandex_response(payload(MAX_CANDIDATES + 10, 10))) == MAX_CANDIDATES

    def test_handles_an_empty_response(self) -> None:
        assert parse_yandex_response({}) == []

    def test_ignores_malformed_entries(self) -> None:
        assert len(parse_yandex_response({"image_results": ["nope", result(0)]})) == 1


class TestProvider:
    def test_requires_a_key(self) -> None:
        with pytest.raises(SearchNotConfigured):
            YandexReverseImageProvider("")

    def test_returns_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            yandex_module.requests, "get", lambda *a, **k: FakeJson(payload())
        )
        response = YandexReverseImageProvider("k").search("https://example.com/f.jpg")
        assert len(response.candidates) == 2

    def test_no_results_is_not_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Yandex reports "nothing found" as an error.

        For an obscure subject that is the expected outcome, and it must not
        abort a run whose other engine did find something.
        """
        monkeypatch.setattr(
            yandex_module.requests,
            "get",
            lambda *a, **k: FakeJson({"error": "Yandex hasn't returned any results"}),
        )
        response = YandexReverseImageProvider("k").search("https://example.com/f.jpg")
        assert response.candidates == []

    def test_surfaces_a_transport_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            yandex_module.requests, "get", lambda *a, **k: FakeJson({}, status_code=500)
        )
        with pytest.raises(SearchProviderError, match="500"):
            YandexReverseImageProvider("k").search("https://example.com/f.jpg")

    def test_sends_the_yandex_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_get(url: str, **kwargs: Any) -> FakeJson:
            captured.update(kwargs.get("params") or {})
            return FakeJson(payload(1, 0))

        monkeypatch.setattr(yandex_module.requests, "get", fake_get)
        YandexReverseImageProvider("k").search("https://example.com/f.jpg")
        assert captured["engine"] == "yandex_images"
        assert captured["url"] == "https://example.com/f.jpg"


class TestIdentity:
    def test_reads_a_knowledge_graph_title(self) -> None:
        identity = extract_identity({"knowledge_graph": {"title": "Ada Lovelace"}})
        assert identity is not None
        assert identity.name == "Ada Lovelace"

    def test_absent_for_an_ordinary_person(self) -> None:
        """Expected, not a failure: no entity exists for a private individual."""
        assert extract_identity({}) is None
        assert extract_identity({"knowledge_graph": {}}) is None
