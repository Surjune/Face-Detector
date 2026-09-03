"""Search stage: parsing, download guards, replay, and candidate scoring.

Nothing here touches the network. The provider, the downloader and the face
encoder are all substituted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import src.search.fetch as fetch_module
import src.search.filter as filter_module
from src.config import EMBEDDING_DIM, MAX_CANDIDATES, MAX_DOWNLOAD_BYTES
from src.errors import DownloadError, SearchNotConfigured, SearchProviderError
from src.search.fetch import download_image
from src.search.filter import score_candidates
from src.search.provider import Candidate, SearchResponse
from src.search.replay import ReplayProvider, record_response
from src.search.serpapi_lens import SerpApiLensProvider, parse_visual_matches


def lens_payload(count: int = 3, **extra: Any) -> dict[str, Any]:
    return {
        "visual_matches": [
            {
                "position": index + 1,
                "title": f"Result {index}",
                "link": f"https://example.com/post/{index}",
                "source": "example.com",
                "thumbnail": f"https://cdn.example.com/thumb/{index}.jpg",
                "image": f"https://cdn.example.com/full/{index}.jpg",
                **extra,
            }
            for index in range(count)
        ]
    }


class TestParseVisualMatches:
    def test_reads_every_match(self) -> None:
        assert len(parse_visual_matches(lens_payload(4))) == 4

    def test_prefers_the_full_image_over_the_thumbnail(self) -> None:
        """A thumbnail is downscaled enough to cost real recognition accuracy."""
        candidate = parse_visual_matches(lens_payload(1))[0]
        assert candidate.image_url == "https://cdn.example.com/full/0.jpg"

    def test_falls_back_to_the_thumbnail(self) -> None:
        payload = lens_payload(1)
        del payload["visual_matches"][0]["image"]
        assert parse_visual_matches(payload)[0].image_url.startswith(
            "https://cdn.example.com/thumb/"
        )

    def test_skips_matches_with_no_usable_image(self) -> None:
        payload = lens_payload(2)
        del payload["visual_matches"][0]["image"]
        del payload["visual_matches"][0]["thumbnail"]
        assert len(parse_visual_matches(payload)) == 1

    def test_skips_matches_with_no_page_link(self) -> None:
        payload = lens_payload(2)
        del payload["visual_matches"][0]["link"]
        assert len(parse_visual_matches(payload)) == 1

    def test_caps_the_candidate_list(self) -> None:
        """Each candidate costs a download and a forward pass."""
        assert len(parse_visual_matches(lens_payload(MAX_CANDIDATES + 10))) == MAX_CANDIDATES

    def test_handles_a_response_with_no_matches(self) -> None:
        assert parse_visual_matches({}) == []
        assert parse_visual_matches({"visual_matches": None}) == []

    def test_ignores_malformed_entries(self) -> None:
        payload = lens_payload(1)
        payload["visual_matches"].append("not a dict")
        assert len(parse_visual_matches(payload)) == 1


class TestSerpApiProvider:
    def test_refuses_to_construct_without_a_key(self) -> None:
        with pytest.raises(SearchNotConfigured):
            SerpApiLensProvider("")

    def test_surfaces_an_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.search.serpapi_lens.requests.get",
            lambda *a, **k: FakeJsonResponse(status_code=401, payload={}),
        )
        with pytest.raises(SearchProviderError, match="401"):
            SerpApiLensProvider("key").search("https://example.com/face.jpg")

    def test_surfaces_a_provider_error_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SerpApi reports a used-up quota as HTTP 200 with an error field."""
        monkeypatch.setattr(
            "src.search.serpapi_lens.requests.get",
            lambda *a, **k: FakeJsonResponse(payload={"error": "Your account has run out"}),
        )
        with pytest.raises(SearchProviderError, match="run out"):
            SerpApiLensProvider("key").search("https://example.com/face.jpg")

    def test_returns_candidates_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.search.serpapi_lens.requests.get",
            lambda *a, **k: FakeJsonResponse(payload=lens_payload(2)),
        )
        response = SerpApiLensProvider("key").search("https://example.com/face.jpg")
        assert len(response.candidates) == 2
        assert response.raw["visual_matches"]


class TestDownloadGuards:
    def test_rejects_a_non_image_content_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A candidate URL can quietly serve an HTML error page."""
        _serve(monkeypatch, content_type="text/html", body=b"<html></html>")
        with pytest.raises(DownloadError, match="Not an image"):
            download_image("https://example.com/x")

    def test_rejects_an_unsupported_image_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, content_type="image/svg+xml", body=b"<svg/>")
        with pytest.raises(DownloadError, match="Unsupported"):
            download_image("https://example.com/x")

    def test_rejects_an_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, status_code=404)
        with pytest.raises(DownloadError, match="404"):
            download_image("https://example.com/x")

    def test_stops_at_the_size_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Guards against a decompression bomb or a mislabelled video."""
        _serve(monkeypatch, body=b"x" * (MAX_DOWNLOAD_BYTES + 1))
        with pytest.raises(DownloadError, match="ceiling"):
            download_image("https://example.com/x")

    def test_rejects_an_empty_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, body=b"")
        with pytest.raises(DownloadError, match="Empty"):
            download_image("https://example.com/x")

    def test_returns_the_bytes_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, body=b"jpegbytes")
        assert download_image("https://example.com/x") == b"jpegbytes"


class TestReplay:
    def test_round_trips_a_recorded_response(self, tmp_path: Path) -> None:
        """This is how someone with no API key reproduces a published run."""
        response = SearchResponse(
            provider="serpapi_google_lens",
            query_image_url="https://example.com/face.jpg",
            retrieved_at="2026-09-03T15:40:00Z",
            candidates=parse_visual_matches(lens_payload(3)),
            raw=lens_payload(3),
        )
        record_response(tmp_path, response)

        replayed = ReplayProvider(tmp_path).search("ignored")
        assert len(replayed.candidates) == 3
        assert replayed.query_image_url == response.query_image_url
        assert replayed.retrieved_at == response.retrieved_at

    def test_labels_itself_as_a_replay(self, tmp_path: Path) -> None:
        """A recording must never be reported as a live query."""
        record_response(
            tmp_path,
            SearchResponse(
                provider="serpapi_google_lens",
                query_image_url="https://example.com/face.jpg",
                retrieved_at="2026-09-03T15:40:00Z",
                candidates=[],
                raw=lens_payload(1),
            ),
        )
        assert "replay" in ReplayProvider(tmp_path).name

    def test_refuses_a_directory_with_no_recording(self, tmp_path: Path) -> None:
        with pytest.raises(SearchNotConfigured):
            ReplayProvider(tmp_path)

    def test_rejects_a_truncated_recording(self, tmp_path: Path) -> None:
        (tmp_path / "search_response.json").write_text(json.dumps({"provider": "x"}))
        with pytest.raises(SearchProviderError, match="missing fields"):
            ReplayProvider(tmp_path)


class TestScoreCandidates:
    """The re-recognition step, with the downloader and encoder substituted."""

    def test_splits_matches_from_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scores = {0: 0.81, 1: 0.12, 2: 0.63}
        _stub_scoring(monkeypatch, scores)

        scored = score_candidates(
            _reference(), _candidates(3), tmp_path / "candidates", threshold=0.55
        )

        assert [item.status for item in scored] == ["match", "match", "below_threshold"]

    def test_orders_matches_by_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_scoring(monkeypatch, {0: 0.61, 1: 0.92, 2: 0.75})
        scored = score_candidates(
            _reference(), _candidates(3), tmp_path / "candidates", threshold=0.55
        )
        assert [item.similarity for item in scored] == [0.92, 0.75, 0.61]

    def test_keeps_every_reject_with_its_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rejected candidates are the evidence the search was real."""
        _stub_scoring(monkeypatch, {0: 0.10, 1: 0.20, 2: 0.30})
        scored = score_candidates(
            _reference(), _candidates(3), tmp_path / "candidates", threshold=0.55
        )
        assert len(scored) == 3
        assert all(item.similarity is not None for item in scored)

    def test_an_unreachable_candidate_does_not_abort_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Search results routinely contain dead links."""
        _stub_scoring(monkeypatch, {0: 0.81, 2: 0.40}, unreachable={1})
        scored = score_candidates(
            _reference(), _candidates(3), tmp_path / "candidates", threshold=0.55
        )
        statuses = {item.candidate.position: item.status for item in scored}
        assert statuses[2] == "unreachable"
        assert statuses[1] == "match"

    def test_a_candidate_with_no_face_is_recorded_not_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lens returns scenery and product shots alongside people."""
        _stub_scoring(monkeypatch, {0: 0.81}, faceless={1, 2})
        scored = score_candidates(
            _reference(), _candidates(3), tmp_path / "candidates", threshold=0.55
        )
        assert sum(item.status == "no_face" for item in scored) == 2

    def test_writes_each_candidate_image_to_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_scoring(monkeypatch, {0: 0.81, 1: 0.10, 2: 0.10})
        images = tmp_path / "candidates"
        score_candidates(_reference(), _candidates(3), images, threshold=0.55)
        assert len(list(images.iterdir())) == 3


# --- helpers -------------------------------------------------------------


class FakeJsonResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeStreamResponse:
    def __init__(self, status_code: int, content_type: str, body: bytes) -> None:
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self._body = body

    def __enter__(self) -> FakeStreamResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int = 200,
    content_type: str = "image/jpeg",
    body: bytes = b"jpegbytes",
) -> None:
    monkeypatch.setattr(
        fetch_module.requests,
        "get",
        lambda *a, **k: FakeStreamResponse(status_code, content_type, body),
    )


def _reference() -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.0
    return vector


def _candidates(count: int) -> list[Candidate]:
    return [
        Candidate(
            position=index + 1,
            title=f"Result {index}",
            page_url=f"https://example.com/post/{index}",
            image_url=f"https://cdn.example.com/{index}.jpg",
            source="example.com",
        )
        for index in range(count)
    ]


def _stub_scoring(
    monkeypatch: pytest.MonkeyPatch,
    scores: dict[int, float],
    *,
    unreachable: set[int] | None = None,
    faceless: set[int] | None = None,
) -> None:
    """Replace the downloader and encoder, keyed by candidate index."""
    unreachable = unreachable or set()
    faceless = faceless or set()

    def fake_download(url: str) -> bytes:
        index = int(url.rsplit("/", 1)[-1].split(".")[0])
        if index in unreachable:
            raise DownloadError("HTTP 403", url=url)
        return f"image-{index}".encode()

    def fake_best_match(reference: np.ndarray, image_bytes: bytes) -> Any:
        index = int(image_bytes.decode().rsplit("-", 1)[-1])
        if index in faceless:
            return None
        return scores[index], object()

    monkeypatch.setattr(filter_module, "download_image", fake_download)
    monkeypatch.setattr(filter_module, "encode_best_match", fake_best_match)
