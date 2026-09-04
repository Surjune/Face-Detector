"""Targeted per-platform search, and the YouTube Data API.

Expansion is what turns "Lens happened to return some social results" into
"every requested platform was actually looked at". It is adaptive because a
query for a platform already covered spends one of 250 monthly searches to
rediscover a post already in hand.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.search.expand as expand_module
import src.search.youtube as youtube_module
from src.errors import SearchNotConfigured, SearchProviderError
from src.search.expand import missing_platforms, parse_web_results, search_platform
from src.search.platforms import Platform
from src.search.provider import Candidate, Source
from src.search.youtube import parse_search_results, search_videos


def candidate(url: str, index: int = 0) -> Candidate:
    return Candidate(
        position=index + 1,
        title=f"Post {index}",
        page_url=url,
        image_urls=("https://cdn.example.com/x.jpg",),
        source="test",
    )


class FakeJson:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class TestMissingPlatforms:
    def test_lists_platforms_with_no_candidate(self) -> None:
        found = [
            candidate("https://www.instagram.com/p/a/"),
            candidate("https://www.facebook.com/x/posts/1", 1),
        ]
        missing = missing_platforms(found, budget=10)

        assert Platform.INSTAGRAM not in missing
        assert Platform.FACEBOOK not in missing
        assert Platform.TIKTOK in missing
        assert Platform.THREADS in missing

    def test_respects_the_search_budget(self) -> None:
        """Each query costs one of a small monthly allowance."""
        assert len(missing_platforms([], budget=2)) == 2

    def test_is_deterministic(self) -> None:
        """Same input, same queries — so a re-run costs the same and does the same."""
        assert missing_platforms([], budget=3) == missing_platforms([], budget=3)

    def test_returns_nothing_when_all_platforms_are_covered(self) -> None:
        found = [
            candidate(f"https://{domain}/x", index)
            for index, domain in enumerate(
                (
                    "www.facebook.com",
                    "x.com",
                    "www.threads.net",
                    "www.linkedin.com",
                    "www.youtube.com",
                    "www.instagram.com",
                    "www.tiktok.com",
                )
            )
        ]
        assert missing_platforms(found, budget=10) == []


class TestSitePlatformSearch:
    def test_builds_a_site_scoped_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_get(url: str, **kwargs: Any) -> FakeJson:
            captured.update(kwargs.get("params") or {})
            return FakeJson({"organic_results": []})

        monkeypatch.setattr(expand_module.requests, "get", fake_get)
        search_platform("key", Platform.INSTAGRAM, "Ada Lovelace", start_position=1)

        assert captured["q"] == 'site:instagram.com "Ada Lovelace"'

    def test_keeps_only_results_on_the_requested_platform(self) -> None:
        """A site: query still returns the occasional aggregator page."""
        payload = {
            "organic_results": [
                {
                    "title": "Real post",
                    "link": "https://www.instagram.com/p/abc/",
                    "thumbnail": "https://cdn.example.com/a.jpg",
                },
                {
                    "title": "Aggregator quoting instagram.com",
                    "link": "https://example.com/roundup",
                    "thumbnail": "https://cdn.example.com/b.jpg",
                },
            ]
        }
        results = parse_web_results(payload, Platform.INSTAGRAM, 1)

        assert len(results) == 1
        assert results[0].platform is Platform.INSTAGRAM
        assert results[0].origin is Source.SITE_SEARCH

    def test_skips_results_with_no_image(self) -> None:
        payload = {"organic_results": [{"link": "https://www.instagram.com/p/a/"}]}
        assert parse_web_results(payload, Platform.INSTAGRAM, 1) == []

    def test_treats_a_no_results_error_as_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SerpApi reports "nothing found" as an error; being absent is normal."""
        monkeypatch.setattr(
            expand_module.requests,
            "get",
            lambda *a, **k: FakeJson({"error": "hasn't returned any results"}),
        )
        assert search_platform("key", Platform.TIKTOK, "Ada", start_position=1) == []

    def test_raises_on_a_transport_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            expand_module.requests, "get", lambda *a, **k: FakeJson({}, status_code=500)
        )
        with pytest.raises(SearchProviderError, match="500"):
            search_platform("key", Platform.X, "Ada", start_position=1)


class TestYouTubeApi:
    def test_requires_a_key(self) -> None:
        with pytest.raises(SearchNotConfigured):
            search_videos("", "Ada Lovelace", start_position=1)

    def test_parses_videos_into_candidates(self) -> None:
        payload = {
            "items": [
                {
                    "id": {"videoId": "pDdrA46xBRk"},
                    "snippet": {
                        "title": "An interview",
                        "channelTitle": "Some Channel",
                        "thumbnails": {
                            "high": {"url": "https://i.ytimg.com/vi/x/hq.jpg"},
                            "default": {"url": "https://i.ytimg.com/vi/x/def.jpg"},
                        },
                    },
                }
            ]
        }
        results = parse_search_results(payload, 1)

        assert len(results) == 1
        assert results[0].page_url == "https://www.youtube.com/watch?v=pDdrA46xBRk"
        assert results[0].platform is Platform.YOUTUBE
        assert results[0].origin is Source.YOUTUBE_API

    def test_prefers_the_largest_thumbnail(self) -> None:
        """A bigger crop carries more facial detail to verify against."""
        payload = {
            "items": [
                {
                    "id": {"videoId": "abc"},
                    "snippet": {
                        "title": "t",
                        "thumbnails": {
                            "default": {"url": "https://i.ytimg.com/small.jpg"},
                            "maxres": {"url": "https://i.ytimg.com/big.jpg"},
                        },
                    },
                }
            ]
        }
        assert parse_search_results(payload, 1)[0].image_urls[0].endswith("big.jpg")

    def test_skips_items_with_no_thumbnail(self) -> None:
        payload = {"items": [{"id": {"videoId": "a"}, "snippet": {"title": "t"}}]}
        assert parse_search_results(payload, 1) == []

    def test_handles_an_empty_response(self) -> None:
        assert parse_search_results({}, 1) == []

    def test_reports_an_exhausted_quota_clearly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """403 covers both a spent quota and a key not enabled for this API."""
        monkeypatch.setattr(
            youtube_module.requests,
            "get",
            lambda *a, **k: FakeJson({}, status_code=403),
        )
        with pytest.raises(SearchProviderError, match="quota"):
            search_videos("key", "Ada", start_position=1)
