"""Build a labelled face dataset from Wikimedia Commons.

`FACE_MATCH_THRESHOLD` has to be a measured value, and measuring it needs
several photographs of each of several people. This script assembles that set
from Commons so the derivation is reproducible by anyone reading the repository,
rather than resting on a number nobody can check.

Only freely licensed files are kept, and each one's licence and author are
recorded in a manifest alongside the images.

    python scripts/fetch_calibration_set.py --out dataset/
    python scripts/calibrate_threshold.py --dataset dataset/

The images are not committed: the dataset directory is ignored, and the manifest
records exactly which files were used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DOWNLOAD_TIMEOUT_SECONDS, MAX_DOWNLOAD_BYTES  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia asks for a descriptive User-Agent identifying the client.
HEADERS = {"User-Agent": "face-detector-calibration/1.0 (github.com/Surjune/Face-Detector)"}

# Licence short names that permit reuse. Anything else is skipped.
ACCEPTABLE_LICENCES = (
    "cc0",
    "cc-by",
    "cc by",
    "public domain",
    "pd-",
    "attribution",
)

# People with many freely licensed photographs on Commons, taken at different
# ages, angles and lighting — which is what makes the impostor/genuine
# separation meaningful rather than trivially easy.
DEFAULT_SUBJECTS = (
    "Sundar Pichai",
    "A. R. Rahman",
    "Virat Kohli",
    "Satya Nadella",
)

# Enough images per person to form several genuine pairs, few enough to keep the
# download short.
IMAGES_PER_SUBJECT = 4

# Commons returns thumbnails at any requested width; this is large enough for
# reliable detection and small enough to download quickly.
THUMBNAIL_WIDTH = 800


def search_files(subject: str, limit: int) -> list[dict[str, Any]]:
    """Find image files on Commons matching a subject's name."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f'"{subject}"',
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": str(THUMBNAIL_WIDTH),
    }
    response = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    return list(pages.values())


def usable(page: dict[str, Any]) -> dict[str, Any] | None:
    """Return download details for a freely licensed photograph, or None."""
    info = (page.get("imageinfo") or [{}])[0]
    if not info.get("thumburl"):
        return None
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None

    metadata = info.get("extmetadata", {})
    licence = str(metadata.get("LicenseShortName", {}).get("value", "")).lower()
    if not any(token in licence for token in ACCEPTABLE_LICENCES):
        return None

    return {
        "title": page.get("title", ""),
        "url": info["thumburl"],
        "descriptionurl": info.get("descriptionurl", ""),
        "licence": metadata.get("LicenseShortName", {}).get("value", ""),
        "author": _strip_html(str(metadata.get("Artist", {}).get("value", ""))),
    }


def download(url: str, destination: Path) -> int:
    """Fetch one image. Returns the byte count written."""
    response = requests.get(url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT_SECONDS, stream=True)
    response.raise_for_status()

    payload = b""
    for chunk in response.iter_content(chunk_size=64 * 1024):
        payload += chunk
        if len(payload) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"{url} exceeds the size ceiling")
    destination.write_bytes(payload)
    return len(payload)


def _strip_html(value: str) -> str:
    """Commons returns the author as an HTML fragment."""
    import re

    return re.sub(r"<[^>]+>", "", value).strip()


def _slug(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--subjects", nargs="*", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--per-subject", type=int, default=IMAGES_PER_SUBJECT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for subject in args.subjects:
        folder = args.out / _slug(subject)
        folder.mkdir(exist_ok=True)
        print(f"{subject}:")

        kept = 0
        for page in search_files(subject, args.per_subject * 4):
            if kept >= args.per_subject:
                break
            details = usable(page)
            if details is None:
                continue

            name = f"{kept:02d}.jpg"
            try:
                size = download(details["url"], folder / name)
            except (requests.RequestException, ValueError) as exc:
                print(f"  skipped {details['title']}: {exc}")
                continue

            manifest.append({"subject": subject, "file": f"{folder.name}/{name}", **details})
            print(f"  {name}  {size // 1024} KB  {details['licence']}  {details['title']}")
            kept += 1
            time.sleep(0.2)  # be gentle with the Commons API

        if kept == 0:
            print("  nothing usable found")

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(manifest)} image(s); manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
