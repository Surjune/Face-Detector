"""Google Lens through SerpApi.

Google publishes no reverse-image-search API. SerpApi's `google_lens` engine is
the wrapper around it with a genuinely recurring free tier (250 searches a
month, no card), which is why it is the primary provider here.

Note the shape of the call: Lens takes an image *URL*, never an upload, so the
probe image has to be reachable on the public web first. See imghost.py.
"""

from __future__ import annotations

from typing import Any

import requests

from src.config import (
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_CANDIDATES,
    SERPAPI_ENDPOINT,
    SERPAPI_LENS_ENGINE,
)
from src.errors import SearchNotConfigured, SearchProviderError
from src.search.provider import Candidate, SearchResponse, utc_now


class SerpApiLensProvider:
    """Live Google Lens search."""

    name = "serpapi_google_lens"

    def __init__(self, api_key: str, endpoint: str = SERPAPI_ENDPOINT) -> None:
        if not api_key:
            raise SearchNotConfigured("SERPAPI_KEY is not set; see .env.example")
        self._api_key = api_key
        self._endpoint = endpoint

    def search(self, image_url: str) -> SearchResponse:
        """Query Lens and normalise its visual matches into candidates."""
        params = {
            "engine": SERPAPI_LENS_ENGINE,
            "url": image_url,
            "api_key": self._api_key,
        }
        try:
            response = requests.get(
                self._endpoint, params=params, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            raise SearchProviderError(f"Could not reach SerpApi: {exc}") from exc

        if response.status_code != 200:
            raise SearchProviderError(
                f"SerpApi returned HTTP {response.status_code}",
                body=response.text[:200],
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise SearchProviderError(f"SerpApi returned invalid JSON: {exc}") from exc

        if "error" in payload:
            raise SearchProviderError(f"SerpApi error: {payload['error']}")

        return SearchResponse(
            provider=self.name,
            query_image_url=image_url,
            retrieved_at=utc_now(),
            candidates=parse_visual_matches(payload),
            raw=payload,
        )


def parse_visual_matches(payload: dict[str, Any]) -> list[Candidate]:
    """Pull candidates out of a Google Lens response.

    Lens gives each match a page link plus one or two image URLs. The full-size
    `image` is preferred over `thumbnail`: thumbnails are heavily downscaled and
    a face crop taken from one carries much less signal.
    """
    matches = payload.get("visual_matches")
    if not isinstance(matches, list):
        return []

    candidates: list[Candidate] = []
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            continue
        image_url = match.get("image") or match.get("thumbnail")
        page_url = match.get("link")
        if not image_url or not page_url:
            continue
        candidates.append(
            Candidate(
                position=int(match.get("position", index + 1)),
                title=str(match.get("title", "")),
                page_url=str(page_url),
                image_url=str(image_url),
                source=str(match.get("source", "")),
            )
        )
        if len(candidates) >= MAX_CANDIDATES:
            break
    return candidates
