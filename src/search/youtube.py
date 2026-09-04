"""YouTube Data API v3.

The only first-party social platform API in this pipeline with a free tier that
needs no payment card and no approval process: 10,000 quota units a day, of
which one `search.list` costs 100.

It searches by *text*, not by face — no platform API searches by face. What it
returns is a set of leads whose thumbnails then go through the same face
verification as every other candidate, so a YouTube result earns its place the
same way a Lens result does.
"""

from __future__ import annotations

from typing import Any

import requests

from src.config import (
    DOWNLOAD_TIMEOUT_SECONDS,
    YOUTUBE_MAX_RESULTS,
    YOUTUBE_SEARCH_ENDPOINT,
)
from src.errors import SearchNotConfigured, SearchProviderError
from src.search.platforms import Platform
from src.search.provider import Candidate, Source

WATCH_URL = "https://www.youtube.com/watch?v="

# Thumbnail sizes YouTube publishes, largest first. A bigger crop carries more
# facial detail, and the smaller ones exist for every video as a fallback.
THUMBNAIL_SIZES = ("maxres", "standard", "high", "medium", "default")


def search_videos(
    api_key: str,
    query: str,
    *,
    start_position: int,
    endpoint: str = YOUTUBE_SEARCH_ENDPOINT,
) -> list[Candidate]:
    """Search YouTube for videos matching a query.

    Raises:
        SearchNotConfigured: no API key.
        SearchProviderError: the request failed, the quota is exhausted, or the
            response was unusable.
    """
    if not api_key:
        raise SearchNotConfigured("YOUTUBE_API_KEY is not set; see .env.example")

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": str(YOUTUBE_MAX_RESULTS),
        "key": api_key,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise SearchProviderError(f"Could not reach the YouTube API: {exc}") from exc

    if response.status_code == 403:
        # 403 covers both an exhausted daily quota and a key that is not
        # authorised for this API; the body distinguishes them for the user.
        raise SearchProviderError(
            "YouTube API refused the request (quota exhausted, or the key is "
            "not enabled for YouTube Data API v3)",
            body=response.text[:200],
        )
    if response.status_code != 200:
        raise SearchProviderError(
            f"YouTube API returned HTTP {response.status_code}", body=response.text[:200]
        )

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        raise SearchProviderError(f"YouTube API returned invalid JSON: {exc}") from exc

    return parse_search_results(payload, start_position)


def parse_search_results(payload: dict[str, Any], start_position: int) -> list[Candidate]:
    """Convert a search.list response into candidates."""
    items = payload.get("items")
    if not isinstance(items, list):
        return []

    candidates: list[Candidate] = []
    for offset, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet")
        if not video_id or not isinstance(snippet, dict):
            continue

        thumbnails = _thumbnail_chain(snippet.get("thumbnails"))
        if not thumbnails:
            continue

        candidates.append(
            Candidate(
                position=start_position + offset,
                title=str(snippet.get("title", "")),
                page_url=f"{WATCH_URL}{video_id}",
                image_urls=thumbnails,
                source=str(snippet.get("channelTitle") or Platform.YOUTUBE.label),
                origin=Source.YOUTUBE_API,
            )
        )
    return candidates


def _thumbnail_chain(thumbnails: Any) -> tuple[str, ...]:
    """Thumbnail URLs for one video, largest first."""
    if not isinstance(thumbnails, dict):
        return ()

    chain: list[str] = []
    for size in THUMBNAIL_SIZES:
        entry = thumbnails.get(size)
        if isinstance(entry, dict):
            url = entry.get("url")
            if isinstance(url, str) and url and url not in chain:
                chain.append(url)
    return tuple(chain)
