"""Reading and writing a run's evidence folder.

Every run leaves a self-contained directory behind: the probe image, the
provider's untouched response, every candidate image, every score, the receipt
that was hashed, and the anchoring transaction. Verification works entirely from
that folder, so it can be committed, copied or handed to someone else.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.chain.canonical import Receipt, receipt_from_dict, receipt_to_dict
from src.chain.registry import AnchorResult
from src.config import EVIDENCE_ROOT, SEPOLIA_CHAIN_ID, SEPOLIA_TX_EXPLORER
from src.errors import EvidenceError
from src.search.filter import ScoredCandidate

RECEIPT_FILENAME = "receipt.json"
CANDIDATES_FILENAME = "candidates.json"
ANCHOR_FILENAME = "anchor.json"
REPORT_FILENAME = "report.html"
INPUT_STEM = "input"


@dataclass(frozen=True)
class AnchorRecordFile:
    """What anchor.json holds."""

    digest: str
    tx_hash: str
    block_number: int
    chain_id: int
    contract_address: str
    gas_used: int
    network: str
    explorer_url: str | None


def create_run_dir(base: Path | None = None) -> Path:
    """Make a fresh, timestamped run directory."""
    root = base or EVIDENCE_ROOT
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = root / f"run_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def copy_input_image(run_dir: Path, image_path: Path) -> Path:
    """Keep the exact probe image alongside its results."""
    destination = run_dir / f"{INPUT_STEM}{image_path.suffix.lower() or '.img'}"
    try:
        shutil.copyfile(image_path, destination)
    except OSError as exc:
        raise EvidenceError(f"Could not copy the input image: {exc}") from exc
    return destination


def find_input_image(run_dir: Path) -> Path | None:
    """Locate the probe image inside a run directory."""
    for path in sorted(run_dir.glob(f"{INPUT_STEM}.*")):
        return path
    return None


def write_receipt(run_dir: Path, receipt: Receipt) -> Path:
    """Store the receipt in the same shape that gets hashed.

    Written with sorted keys and an indent: readable for a human, while the
    digest is always recomputed from `canonical_bytes`, never from this file's
    layout.
    """
    path = run_dir / RECEIPT_FILENAME
    payload = json.dumps(receipt_to_dict(receipt), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def read_receipt(run_dir: Path) -> Receipt:
    """Load a stored receipt.

    Raises:
        EvidenceError: the file is absent or malformed.
    """
    return receipt_from_dict(_read_json(run_dir / RECEIPT_FILENAME))


def write_candidates(run_dir: Path, scored: list[ScoredCandidate]) -> Path:
    """Record every candidate and its score, rejects included."""
    path = run_dir / CANDIDATES_FILENAME
    payload = [
        {
            "position": item.candidate.position,
            "status": item.status,
            "similarity": item.similarity,
            "title": item.candidate.title,
            "source": item.candidate.source,
            "page_url": item.candidate.page_url,
            "image_url": item.candidate.image_url,
            "image_sha256": item.image_sha256,
            "image_file": item.image_path.name if item.image_path else None,
            "detail": item.detail,
        }
        for item in scored
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_candidates(run_dir: Path) -> list[dict[str, Any]]:
    """Load the recorded candidate table, or an empty list if absent."""
    path = run_dir / CANDIDATES_FILENAME
    if not path.exists():
        return []
    data = _read_json(path)
    if not isinstance(data, list):
        raise EvidenceError(f"{CANDIDATES_FILENAME} should hold a list")
    return data


def write_anchor(run_dir: Path, digest: str, result: AnchorResult) -> Path:
    """Record the anchoring transaction."""
    on_public_chain = result.chain_id == SEPOLIA_CHAIN_ID
    record = {
        "digest": digest,
        "tx_hash": prefixed(result.tx_hash),
        "block_number": result.block_number,
        "chain_id": result.chain_id,
        "contract_address": result.contract_address,
        "gas_used": result.gas_used,
        "network": "sepolia" if on_public_chain else "local",
        "explorer_url": (
            f"{SEPOLIA_TX_EXPLORER}{prefixed(result.tx_hash)}" if on_public_chain else None
        ),
    }
    path = run_dir / ANCHOR_FILENAME
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_anchor(run_dir: Path) -> AnchorRecordFile:
    """Load the anchoring record.

    Raises:
        EvidenceError: the run was never anchored, or the file is malformed.
    """
    record = _read_json(run_dir / ANCHOR_FILENAME)
    try:
        return AnchorRecordFile(
            digest=record["digest"],
            tx_hash=record["tx_hash"],
            block_number=int(record["block_number"]),
            chain_id=int(record["chain_id"]),
            contract_address=record["contract_address"],
            gas_used=int(record["gas_used"]),
            network=record.get("network", "unknown"),
            explorer_url=record.get("explorer_url"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"Malformed {ANCHOR_FILENAME}: {exc}") from exc


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise EvidenceError(f"Missing {path.name}", expected=str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Could not read {path.name}: {exc}") from exc


def prefixed(value: str) -> str:
    """web3 returns hashes with and without the 0x prefix depending on version."""
    return value if value.startswith("0x") else "0x" + value
