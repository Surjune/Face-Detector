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


def download_image(url: str) -> bytes:
    """Fetch one candidate image.

    Raises:
        DownloadError: the request failed, timed out, returned a non-image, or
            exceeded the size ceiling.
    """
    try:
        response = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            headers=_HEADERS,
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
