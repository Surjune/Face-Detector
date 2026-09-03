"""Chain stage: hash a run receipt and anchor it on a public blockchain."""

from __future__ import annotations

from src.chain.canonical import (
    MatchRecord,
    Receipt,
    SearchRecord,
    canonical_bytes,
    receipt_digest,
    receipt_from_dict,
    receipt_to_dict,
    sha256_hex,
)
from src.chain.registry import (
    AnchorRecord,
    AnchorResult,
    RegistryClient,
    describe_chain,
    load_deployment,
    save_deployment,
)

__all__ = [
    "AnchorRecord",
    "AnchorResult",
    "MatchRecord",
    "Receipt",
    "RegistryClient",
    "SearchRecord",
    "canonical_bytes",
    "describe_chain",
    "load_deployment",
    "receipt_digest",
    "receipt_from_dict",
    "receipt_to_dict",
    "save_deployment",
    "sha256_hex",
]
