"""Making the probe image reachable by URL.

Google Lens accepts an image URL, never an upload, so a local file has to be
published somewhere the search engine can fetch it before it can be searched.

Two routes, in order of preference:

1. The image is already published at its raw URL in this repository, byte for
   byte. Nothing is uploaded anywhere, and the URL is stable and free.
2. Otherwise it is posted to catbox.moe, an anonymous keyless host.

Route 1 is only taken when the published bytes are **verified** to match the
local file. Assuming they match is not safe: replacing `inputs/probe.jpg`
locally without committing leaves the old image published at the same URL, and
the search would silently run against the previous photograph and confidently
report the wrong person. A hash comparison costs one request and removes that
whole class of failure.

Route 2 publishes the probe image to a third party. That is inherent to using a
hosted reverse image search and is called out in the README; use your own
photograph, or one whose subject has agreed to it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from src.config import (
    CATBOX_ENDPOINT,
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_DOWNLOAD_BYTES,
    REPO_ROOT,
)
from src.errors import ImageHostError


def public_url_for(image_path: Path, github_raw_base: str | None) -> tuple[str, str]:
    """Return a publicly fetchable URL for a local image, and how it was obtained.

    Raises:
        ImageHostError: the image could not be read or published.
    """
    try:
        local_bytes = image_path.read_bytes()
    except OSError as exc:
        raise ImageHostError(f"Could not read {image_path.name}: {exc}") from exc

    committed = _committed_raw_url(image_path, github_raw_base)
    if committed is not None and _serves_exactly(committed, local_bytes):
        return committed, "already published in this repository"

    return upload_to_catbox(image_path), "uploaded to catbox.moe, a public host"


def _committed_raw_url(image_path: Path, github_raw_base: str | None) -> str | None:
    """Build a raw URL if the image sits inside this repository."""
    if not github_raw_base:
        return None
    try:
        relative = image_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return f"{github_raw_base.rstrip('/')}/{relative.as_posix()}"


def _serves_exactly(url: str, expected: bytes) -> bool:
    """Whether a URL currently serves exactly these bytes.

    Any failure answers False, which routes the caller to upload instead. The
    only wrong answer here would be a false True, since that would search the
    wrong photograph.
    """
    digest = hashlib.sha256(expected).hexdigest()
    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
    except requests.RequestException:
        return False

    with response:
        if response.status_code != 200:
            return False

        received = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                received += chunk
                if len(received) > MAX_DOWNLOAD_BYTES:
                    return False
        except requests.RequestException:
            return False

    return hashlib.sha256(bytes(received)).hexdigest() == digest


def upload_to_catbox(image_path: Path) -> str:
    """Post an image to catbox.moe and return its public URL.

    Raises:
        ImageHostError: the upload failed or returned something other than a URL.
    """
    try:
        payload = image_path.read_bytes()
    except OSError as exc:
        raise ImageHostError(f"Could not read {image_path.name}: {exc}") from exc

    try:
        response = requests.post(
            CATBOX_ENDPOINT,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (image_path.name, payload)},
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ImageHostError(f"Could not reach catbox.moe: {exc}") from exc

    if response.status_code != 200:
        raise ImageHostError(f"catbox.moe returned HTTP {response.status_code}")

    url = response.text.strip()
    if not url.startswith("https://"):
        raise ImageHostError(f"catbox.moe returned an unexpected body: {url[:100]!r}")
    return url
