"""Publishing the probe image so a visual search can fetch it.

The dangerous failure here is silent. If the repository still serves a
previously committed photograph at the same path, searching that URL returns
results for the wrong person and every downstream stage — face verification
included — behaves perfectly while answering a question nobody asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.search.imghost as imghost
from src.errors import ImageHostError
from src.search.imghost import public_url_for, upload_to_catbox

RAW_BASE = "https://raw.githubusercontent.com/Surjune/Face-Detector/main"


class FakeGet:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def __enter__(self) -> FakeGet:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


@pytest.fixture
def repo_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A probe image that appears to live at the repository root."""
    monkeypatch.setattr(imghost, "REPO_ROOT", tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "probe.jpg"
    path.write_bytes(b"the-new-photograph")
    return path


class TestPublishedImageVerification:
    def test_reuses_the_raw_url_when_the_bytes_match(
        self, repo_probe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            imghost.requests, "get", lambda *a, **k: FakeGet(200, b"the-new-photograph")
        )
        url, how = public_url_for(repo_probe, RAW_BASE)

        assert url == f"{RAW_BASE}/inputs/probe.jpg"
        assert "repository" in how

    def test_uploads_when_the_repo_still_serves_the_old_image(
        self, repo_probe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replacing probe.jpg without committing must not search the old one.

        This is the whole reason the published bytes are verified rather than
        assumed: the URL resolves, returns 200, and serves a different person.
        """
        monkeypatch.setattr(
            imghost.requests, "get", lambda *a, **k: FakeGet(200, b"the-old-photograph")
        )
        monkeypatch.setattr(
            imghost, "upload_to_catbox", lambda path: "https://files.catbox.moe/new.jpg"
        )
        url, how = public_url_for(repo_probe, RAW_BASE)

        assert url == "https://files.catbox.moe/new.jpg"
        assert "catbox" in how

    def test_uploads_when_the_raw_url_is_missing(
        self, repo_probe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file added locally but never pushed 404s at its raw URL."""
        monkeypatch.setattr(imghost.requests, "get", lambda *a, **k: FakeGet(404, b""))
        monkeypatch.setattr(
            imghost, "upload_to_catbox", lambda path: "https://files.catbox.moe/new.jpg"
        )
        assert public_url_for(repo_probe, RAW_BASE)[0].startswith("https://files.catbox.moe")

    def test_uploads_when_the_check_cannot_be_made(
        self, repo_probe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline, unverifiable. The safe answer is to upload, not to assume."""

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise imghost.requests.RequestException("no network")

        monkeypatch.setattr(imghost.requests, "get", boom)
        monkeypatch.setattr(
            imghost, "upload_to_catbox", lambda path: "https://files.catbox.moe/new.jpg"
        )
        assert public_url_for(repo_probe, RAW_BASE)[0].startswith("https://files.catbox.moe")

    def test_uploads_when_the_image_is_outside_the_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "elsewhere.jpg"
        outside.write_bytes(b"somebody")
        monkeypatch.setattr(imghost, "REPO_ROOT", tmp_path / "repo")
        monkeypatch.setattr(
            imghost, "upload_to_catbox", lambda path: "https://files.catbox.moe/x.jpg"
        )
        assert public_url_for(outside, RAW_BASE)[0].startswith("https://files.catbox.moe")

    def test_uploads_when_no_raw_base_is_configured(
        self, repo_probe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            imghost, "upload_to_catbox", lambda path: "https://files.catbox.moe/x.jpg"
        )
        assert public_url_for(repo_probe, None)[0].startswith("https://files.catbox.moe")

    def test_reports_an_unreadable_image(self, tmp_path: Path) -> None:
        with pytest.raises(ImageHostError):
            public_url_for(tmp_path / "absent.jpg", RAW_BASE)


class TestCatboxUpload:
    def test_returns_the_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "p.jpg"
        path.write_bytes(b"x")

        class Response:
            status_code = 200
            text = "https://files.catbox.moe/abc.jpg\n"

        monkeypatch.setattr(imghost.requests, "post", lambda *a, **k: Response())
        assert upload_to_catbox(path) == "https://files.catbox.moe/abc.jpg"

    def test_rejects_a_non_url_body(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "p.jpg"
        path.write_bytes(b"x")

        class Response:
            status_code = 200
            text = "something went wrong"

        monkeypatch.setattr(imghost.requests, "post", lambda *a, **k: Response())
        with pytest.raises(ImageHostError, match="unexpected body"):
            upload_to_catbox(path)
