"""Google Lens through SerpApi.

Google publishes no reverse-image-search API. SerpApi's `google_lens` engine is
the wrapper around it with a genuinely recurring free tier (250 searches a
month, no card), which is why it is the primary provider here.

Note the shape of the call: Lens takes an image *URL*, never an upload, so the
probe image has to be reachable on the public web first. See imghost.py.

A Lens response carries candidates in three separate arrays, and the social
media results are concentrated in the two that are easy to overlook:

    visual_matches   pages showing a similar image  (news, encyclopaedias, some social)
    organic_results  ordinary web results           (where X posts turn up)
    short_videos     video results                  (Instagram reels, Facebook videos, Shorts)

All three are harvested from the single call we already pay for.
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
from src.search.provider import Candidate, Identity, SearchResponse, Source, utc_now

# Result arrays in a Lens payload, and the provenance recorded for each.
RESULT_SETS: tuple[tuple[str, Source], ...] = (
    ("visual_matches", Source.LENS_VISUAL),
    ("organic_results", Source.LENS_ORGANIC),
    ("short_videos", Source.LENS_VIDEO),
)


class SerpApiLensProvider:
    """Live Google Lens search."""

    name = "serpapi_google_lens"

    def __init__(self, api_key: str, endpoint: str = SERPAPI_ENDPOINT) -> None:
        if not api_key:
            raise SearchNotConfigured("SERPAPI_KEY is not set; see .env.example")
        self._api_key = api_key
        self._endpoint = endpoint

    def search(self, image_url: str) -> SearchResponse:
        """Query Lens and normalise every result set into candidates."""
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
            candidates=parse_lens_response(payload),
            raw=payload,
            identity=extract_identity(payload),
        )


def parse_lens_response(payload: dict[str, Any]) -> list[Candidate]:
    """Pull candidates out of every result set in a Lens response.

    The sets are interleaved rather than read one after another. A real
    response returns around thirty visual matches, eight organic results and
    ten short videos; draining them in order fills the candidate budget with
    visual matches alone and never reaches the other two — which is precisely
    where the social media posts live. Taking one from each set in turn
    guarantees every set is represented within the same budget.

    Results are de-duplicated by page URL, keeping the first occurrence, since
    the same post routinely appears in more than one set.
    """
    queues = [
        (origin, entries)
        for key, origin in RESULT_SETS
        if isinstance(entries := payload.get(key), list) and entries
    ]
    if not queues:
        return []

    candidates: list[Candidate] = []
    seen: set[str] = set()
    longest = max(len(entries) for _, entries in queues)

    for index in range(longest):
        for origin, entries in queues:
            if index >= len(entries):
                continue
            if len(candidates) >= MAX_CANDIDATES:
                return candidates

            candidate = _to_candidate(entries[index], index, origin, len(candidates) + 1)
            if candidate is None or candidate.page_url in seen:
                continue
            seen.add(candidate.page_url)
            candidates.append(candidate)

    return candidates


def _to_candidate(
    entry: Any, index: int, origin: Source, position: int
) -> Candidate | None:
    """Convert one raw result into a candidate, or None if unusable."""
    if not isinstance(entry, dict):
        return None

    page_url = entry.get("link")
    if not page_url:
        return None

    image_urls = _image_chain(entry)
    if not image_urls:
        return None

    return Candidate(
        position=position,
        title=str(entry.get("title", "")),
        page_url=str(page_url),
        image_urls=image_urls,
        source=str(entry.get("source") or entry.get("displayed_link") or ""),
        origin=origin,
    )


def _image_chain(entry: dict[str, Any]) -> tuple[str, ...]:
    """Ordered image URLs to try for one result.

    The publisher's own image comes first because it is full resolution, but it
    is frequently unfetchable: Facebook and Instagram serve their `lookaside.*`
    crawler endpoints as HTML to anything without a crawler user agent. Google's
    `encrypted-tbn*.gstatic.com` thumbnail is smaller but reliably an image, so
    it is kept as the fallback rather than discarded.
    """
    chain: list[str] = []
    for key in ("image", "original", "thumbnail"):
        value = entry.get(key)
        if isinstance(value, str) and value and value not in chain:
            chain.append(value)
    return tuple(chain)


def extract_identity(payload: dict[str, Any]) -> Identity | None:
    """Read the subject's likely name out of a Lens response.

    Lens resolves a recognisable face to a Knowledge Graph entity and returns it
    under `related_content` as a ready-made search query. That is the subject's
    name, already disambiguated by Google, for no extra call and no guessing.
    """
    related = payload.get("related_content")
    if not isinstance(related, list):
        return None

    for entry in related:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query")
        if isinstance(query, str) and query.strip():
            return Identity(
                name=query.strip(),
                origin="lens_related_content",
                confidence="high",
            )
    return None


def parse_visual_matches(payload: dict[str, Any]) -> list[Candidate]:
    """Backwards-compatible alias.

    Retained because recorded responses are replayed through this name; it now
    harvests every result set rather than only `visual_matches`.
    """
    return parse_lens_response(payload)
