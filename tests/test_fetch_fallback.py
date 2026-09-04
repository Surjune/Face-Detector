"""Walking the image URL fallback chain.

A candidate carries several image URLs because the first one frequently does
not work: Meta's crawler endpoints return HTML to non-crawlers. Trying only the
first URL is what caused verified social posts to be recorded `unreachable`.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.search.fetch as fetch_module
from src.errors import DownloadError
from src.search.fetch import download_first_available

LOOKASIDE = "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=1"
THUMBNAIL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9Gc"


class FakeResponse:
    def __init__(self, status_code: int, content_type: str, body: bytes) -> None:
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self._body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


def serve(monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[int, str, bytes]]) -> list[str]:
    """Route each URL to a canned response, recording the order tried."""
    attempted: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        attempted.append(url)
        status, content_type, body = routes.get(url, (404, "text/html", b""))
        return FakeResponse(status, content_type, body)

    monkeypatch.setattr(fetch_module.requests, "get", fake_get)
    return attempted


class TestFallbackChain:
    def test_uses_the_first_url_when_it_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempted = serve(monkeypatch, {LOOKASIDE: (200, "image/jpeg", b"real")})
        payload, used = download_first_available((LOOKASIDE, THUMBNAIL))

        assert payload == b"real"
        assert used == LOOKASIDE
        assert attempted == [LOOKASIDE], "must not fetch the fallback needlessly"

    def test_falls_back_when_the_first_serves_html(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly the Facebook and Instagram case."""
        attempted = serve(
            monkeypatch,
            {
                LOOKASIDE: (200, "text/html", b"<html></html>"),
                THUMBNAIL: (200, "image/jpeg", b"thumb"),
            },
        )
        payload, used = download_first_available((LOOKASIDE, THUMBNAIL))

        assert payload == b"thumb"
        assert used == THUMBNAIL
        assert attempted == [LOOKASIDE, THUMBNAIL]

    def test_falls_back_on_an_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        serve(
            monkeypatch,
            {
                LOOKASIDE: (403, "image/jpeg", b""),
                THUMBNAIL: (200, "image/jpeg", b"thumb"),
            },
        )
        assert download_first_available((LOOKASIDE, THUMBNAIL))[0] == b"thumb"

    def test_reports_every_failure_when_all_urls_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The evidence should say why, not just 'unreachable'."""
        serve(
            monkeypatch,
            {
                LOOKASIDE: (200, "text/html", b"<html></html>"),
                THUMBNAIL: (404, "image/jpeg", b""),
            },
        )
        with pytest.raises(DownloadError) as excinfo:
            download_first_available((LOOKASIDE, THUMBNAIL))

        message = excinfo.value.message
        assert "lookaside.fbsbx.com" in message
        assert "gstatic.com" in message

    def test_rejects_an_empty_chain(self) -> None:
        with pytest.raises(DownloadError):
            download_first_available(())

    def test_guards_still_apply_to_every_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fallback URL is no more trusted than the first one."""
        serve(
            monkeypatch,
            {
                LOOKASIDE: (200, "text/html", b"<html></html>"),
                THUMBNAIL: (200, "image/svg+xml", b"<svg/>"),
            },
        )
        with pytest.raises(DownloadError, match="Unsupported"):
            download_first_available((LOOKASIDE, THUMBNAIL))

    def test_sends_the_page_as_referer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Several media CDNs serve a placeholder without one."""
        captured: dict[str, Any] = {}

        def fake_get(url: str, **kwargs: Any) -> FakeResponse:
            captured.update(kwargs.get("headers") or {})
            return FakeResponse(200, "image/jpeg", b"ok")

        monkeypatch.setattr(fetch_module.requests, "get", fake_get)
        download_first_available((THUMBNAIL,), referer="https://www.instagram.com/p/x/")

        assert captured.get("Referer") == "https://www.instagram.com/p/x/"
