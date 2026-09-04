"""Downloading candidate images.

Candidate URLs come from a search engine's index, so they are untrusted: the
response can be enormous, can be HTML rather than an image, or can simply be
gone. Every download is bounded in time and size and checked before it reaches
the image decoder.
"""

from __future__ import annotations

import requests

from src.config import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_DOWNLOAD_BYTES,
)
from src.errors import DownloadError

# Some CDNs serve a placeholder or block the request outright without a
# browser-shaped User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def download_image(url: str, *, referer: str | None = None) -> bytes:
    """Fetch one candidate image.

    Args:
        url: the image to fetch.
        referer: sent as the `Referer` header. Several media CDNs return an
            error or a placeholder for requests that arrive without one.

    Raises:
        DownloadError: the request failed, timed out, returned a non-image, or
            exceeded the size ceiling.
    """
    headers = dict(_HEADERS)
    if referer:
        headers["Referer"] = referer

    try:
        response = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            headers=headers,
            stream=True,
        )
    except requests.RequestException as exc:
        raise DownloadError(f"Request failed: {exc}", url=url) from exc

    with response:
        if response.status_code != 200:
            raise DownloadError(f"HTTP {response.status_code}", url=url)

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type and not content_type.startswith("image/"):
            raise DownloadError(f"Not an image: {content_type or 'unknown'}", url=url)
        if content_type and content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise DownloadError(f"Unsupported image type: {content_type}", url=url)

        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise DownloadError(
                        f"Larger than the {MAX_DOWNLOAD_BYTES} byte ceiling", url=url
                    )
                chunks.append(chunk)
        except requests.RequestException as exc:
            raise DownloadError(f"Transfer failed: {exc}", url=url) from exc

    if not chunks:
        raise DownloadError("Empty response", url=url)
    return b"".join(chunks)


def download_first_available(
    urls: tuple[str, ...] | list[str], *, referer: str | None = None
) -> tuple[bytes, str]:
    """Try each URL in order and return the first that yields a real image.

    This is what makes social media candidates usable. Facebook and Instagram
    publish their canonical image through `lookaside.fbsbx.com` and
    `lookaside.instagram.com`, which answer with HTML for anything that is not
    a recognised crawler; the search engine's own thumbnail of the very same
    post downloads without complaint. Trying only the first URL discards those
    posts even though a working image was supplied alongside it.

    Returns:
        The image bytes and the URL that produced them.

    Raises:
        DownloadError: every URL failed. The message names each failure so the
            evidence records why, rather than a bare "unreachable".
    """
    if not urls:
        raise DownloadError("No image URL for this candidate")

    failures: list[str] = []
    for url in urls:
        try:
            return download_image(url, referer=referer), url
        except DownloadError as exc:
            failures.append(f"{_host(url)}: {exc.message}")

    raise DownloadError("; ".join(failures), url=urls[0], attempts=len(urls))


def _host(url: str) -> str:
    """Short host label, so a multi-URL failure message stays readable."""
    from src.search.platforms import hostname_of

    return hostname_of(url) or url[:40]
