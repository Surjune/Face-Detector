"""Targeted per-platform search.

Google Lens surfaces whichever social posts its visual index happens to hold.
That is not the same as having *looked* on each platform, and it leaves gaps —
Threads, LinkedIn and TikTok rarely appear in a Lens response at all.

So once the subject's name is known, each platform still missing from the
harvest gets its own `site:`-scoped query. This is the "scripted search
approach" the brief describes, and it makes coverage deliberate rather than
incidental.

Adaptive by design: platforms already covered are skipped, because a query that
re-finds a post we hold costs a search from a 250-a-month budget and returns
nothing new.
"""

from __future__ import annotations

from typing import Any

import requests

from src.config import (
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_EXPANSION_SEARCHES,
    SERPAPI_ENDPOINT,
    SERPAPI_WEB_ENGINE,
    SITE_SEARCH_RESULTS,
)
from src.errors import SearchProviderError
from src.search.platforms import (
    TARGET_PLATFORMS,
    Platform,
    classify,
    site_query_domain,
)
from src.search.provider import Candidate, Source


def missing_platforms(
    candidates: list[Candidate],
    *,
    targets: tuple[Platform, ...] = TARGET_PLATFORMS,
    budget: int = MAX_EXPANSION_SEARCHES,
) -> list[Platform]:
    """Which target platforms the harvest has not produced a candidate for.

    Capped by the search budget, preserving the declared platform order so the
    choice is deterministic rather than dependent on result ordering.
    """
    found = {item.platform for item in candidates}
    return [platform for platform in targets if platform not in found][:budget]


def search_platform(
    api_key: str,
    platform: Platform,
    name: str,
    *,
    start_position: int,
    endpoint: str = SERPAPI_ENDPOINT,
) -> list[Candidate]:
    """Run one `site:`-scoped query for a platform.

    Raises:
        SearchProviderError: the request failed or the response was unusable.
    """
    query = f'site:{site_query_domain(platform)} "{name}"'
    params = {
        "engine": SERPAPI_WEB_ENGINE,
        "q": query,
        "num": str(SITE_SEARCH_RESULTS),
        "api_key": api_key,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise SearchProviderError(f"Could not reach SerpApi: {exc}", query=query) from exc

    if response.status_code != 200:
        raise SearchProviderError(
            f"SerpApi returned HTTP {response.status_code}", query=query
        )

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        raise SearchProviderError(f"SerpApi returned invalid JSON: {exc}") from exc

    if "error" in payload:
        # A site: query with no results is reported as an error, not an empty
        # list. That is a normal outcome for a platform the subject is not on.
        return []

    return parse_web_results(payload, platform, start_position)


def parse_web_results(
    payload: dict[str, Any], platform: Platform, start_position: int
) -> list[Candidate]:
    """Convert Google web results into candidates.

    Only results actually on the requested platform are kept: a `site:` query
    can still return the odd aggregator page quoting the domain.
    """
    entries = payload.get("organic_results")
    if not isinstance(entries, list):
        return []

    candidates: list[Candidate] = []
    for offset, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        link = entry.get("link")
        if not link or classify(str(link)) is not platform:
            continue

        images = tuple(
            value
            for key in ("thumbnail", "original_image", "image")
            if isinstance(value := entry.get(key), str) and value
        )
        if not images:
            continue

        candidates.append(
            Candidate(
                position=start_position + offset,
                title=str(entry.get("title", "")),
                page_url=str(link),
                image_urls=images,
                source=platform.label,
                origin=Source.SITE_SEARCH,
            )
        )
    return candidates
