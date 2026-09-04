"""Harvesting every result set from a Google Lens response.

These run against the **real recorded payload** committed with the published
demo run, not a hand-written fixture. That payload is exactly what Lens
returned for the probe image, and it is the direct evidence for the defect
these tests pin down: the social media posts were present in the response all
along, sitting in two arrays the parser did not read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import REPO_ROOT
from src.search.platforms import Platform
from src.search.provider import Source
from src.search.serpapi_lens import extract_identity, parse_lens_response

RECORDED_RUN = REPO_ROOT / "evidence" / "run_2026-09-03T16-10-03Z" / "search_response.json"


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    if not RECORDED_RUN.exists():
        pytest.skip("the published demo run is not present")
    return json.loads(RECORDED_RUN.read_text(encoding="utf-8"))["raw"]


class TestRecordedPayload:
    def test_the_payload_has_all_three_result_sets(self, recorded: dict[str, Any]) -> None:
        """Guards the premise: if Lens stops returning these, the tests below lie."""
        assert recorded.get("visual_matches")
        assert recorded.get("organic_results")
        assert recorded.get("short_videos")


class TestHarvest:
    def test_recovers_candidates_from_every_result_set(
        self, recorded: dict[str, Any]
    ) -> None:
        origins = {item.origin for item in parse_lens_response(recorded)}
        assert Source.LENS_VISUAL in origins
        assert Source.LENS_ORGANIC in origins
        assert Source.LENS_VIDEO in origins

    def test_recovers_the_x_post_from_organic_results(
        self, recorded: dict[str, Any]
    ) -> None:
        """An X post was in the response and was previously discarded."""
        found = [
            item
            for item in parse_lens_response(recorded)
            if item.platform is Platform.X
        ]
        assert found, "no X post recovered from organic_results"

    def test_recovers_instagram_and_facebook_from_short_videos(
        self, recorded: dict[str, Any]
    ) -> None:
        platforms = {item.platform for item in parse_lens_response(recorded)}
        assert Platform.INSTAGRAM in platforms
        assert Platform.FACEBOOK in platforms

    def test_finds_several_social_platforms(self, recorded: dict[str, Any]) -> None:
        social = {
            item.platform for item in parse_lens_response(recorded) if item.is_social
        }
        assert len(social) >= 3, f"only found {social}"

    def test_deduplicates_by_page_url(self, recorded: dict[str, Any]) -> None:
        """The same post appears in more than one result set."""
        candidates = parse_lens_response(recorded)
        urls = [item.page_url for item in candidates]
        assert len(urls) == len(set(urls))

    def test_positions_are_unique_and_sequential(self, recorded: dict[str, Any]) -> None:
        """Positions name the candidate image files, so they must not collide."""
        positions = [item.position for item in parse_lens_response(recorded)]
        assert positions == list(range(1, len(positions) + 1))


class TestImageFallbackChain:
    """The fix for the Facebook and Instagram matches that were being lost.

    Meta publishes a post's canonical image through `lookaside.fbsbx.com` and
    `lookaside.instagram.com`, which answer with HTML for anything that is not
    a recognised crawler. Lens supplies its own working thumbnail of the same
    post alongside it. Keeping only the first URL threw those posts away.
    """

    def test_a_lookaside_entry_keeps_the_thumbnail_as_a_fallback(
        self, recorded: dict[str, Any]
    ) -> None:
        lookaside_entries = [
            entry
            for entry in recorded["visual_matches"]
            if "lookaside" in str(entry.get("image", ""))
        ]
        assert lookaside_entries, "the recorded payload has no lookaside entries"

        candidates = parse_lens_response({"visual_matches": lookaside_entries})
        assert candidates

        for candidate in candidates:
            assert len(candidate.image_urls) >= 2, "no fallback URL kept"
            assert "lookaside" in candidate.image_urls[0], "full image should be tried first"
            assert any("gstatic.com" in url for url in candidate.image_urls[1:])

    def test_a_single_url_entry_still_parses(self, recorded: dict[str, Any]) -> None:
        """short_videos entries carry only a thumbnail, and it is a working one."""
        candidates = parse_lens_response({"short_videos": recorded["short_videos"]})
        assert candidates
        assert all(len(item.image_urls) == 1 for item in candidates)

    def test_short_videos_are_where_the_reels_are(self, recorded: dict[str, Any]) -> None:
        platforms = {
            item.platform
            for item in parse_lens_response({"short_videos": recorded["short_videos"]})
        }
        assert Platform.INSTAGRAM in platforms
        assert Platform.FACEBOOK in platforms


class TestIdentityFromLens:
    def test_reads_the_subject_name(self, recorded: dict[str, Any]) -> None:
        """Lens resolves the face to a Knowledge Graph entity, for free."""
        identity = extract_identity(recorded)
        assert identity is not None
        assert identity.name == "Sundar Pichai"
        assert identity.origin == "lens_related_content"

    def test_returns_none_without_related_content(self) -> None:
        assert extract_identity({}) is None
        assert extract_identity({"related_content": []}) is None

    def test_ignores_malformed_related_content(self) -> None:
        assert extract_identity({"related_content": ["nope", {}, {"query": ""}]}) is None


class TestReplayCarriesIdentity:
    """A replay must reproduce the recorded search, identity included.

    Dropping it silently downgraded identity from Google's own resolved entity
    to a guess made from title frequency.
    """

    def test_replaying_a_recording_keeps_the_identity(
        self, recorded: dict[str, Any], tmp_path: Path
    ) -> None:
        from src.search.provider import SearchResponse
        from src.search.replay import ReplayProvider, record_response

        record_response(
            tmp_path,
            SearchResponse(
                provider="serpapi_google_lens",
                query_image_url="https://example.com/probe.jpg",
                retrieved_at="2026-09-03T16:10:20Z",
                candidates=[],
                raw=recorded,
            ),
        )
        replayed = ReplayProvider(tmp_path).search("ignored")

        assert replayed.identity is not None
        assert replayed.identity.name == "Sundar Pichai"
        assert replayed.identity.origin == "lens_related_content"
