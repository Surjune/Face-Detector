"""Making the probe image reachable by URL.

Google Lens accepts an image URL, never an upload, so a local file has to be
published somewhere the search engine can fetch it before it can be searched.

Two ways, in order of preference:

1. The image is already committed to this repository, so its raw GitHub URL is
   used. Nothing is uploaded anywhere, and the URL is stable and free.
2. Otherwise it is posted to catbox.moe, an anonymous keyless host.

Option 2 publishes the probe image to a third party. That is inherent to using
a hosted reverse image search and is called out in the README; use your own
photograph, or one whose subject has agreed to it.
"""

from __future__ import annotations

from pathlib import Path

import requests

from src.config import CATBOX_ENDPOINT, DOWNLOAD_TIMEOUT_SECONDS, REPO_ROOT
from src.errors import ImageHostError


def public_url_for(image_path: Path, github_raw_base: str | None) -> tuple[str, str]:
    """Return a publicly fetchable URL for a local image, and how it was obtained.

    Raises:
        ImageHostError: the image could not be published.
    """
    committed = _committed_raw_url(image_path, github_raw_base)
    if committed is not None:
        return committed, "committed to the repository"
    return upload_to_catbox(image_path), "uploaded to catbox.moe"


def _committed_raw_url(image_path: Path, github_raw_base: str | None) -> str | None:
    """Build a raw GitHub URL if the image lives inside this repository.

    Only usable for a file that has actually been pushed; an uncommitted file at
    the same path would give a URL that 404s, so callers should prefer this only
    for the shipped demo input.
    """
    if not github_raw_base:
        return None
    try:
        relative = image_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return f"{github_raw_base.rstrip('/')}/{relative.as_posix()}"


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
