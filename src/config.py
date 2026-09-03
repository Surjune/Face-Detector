"""Named constants and environment-backed settings.

Every tunable in the pipeline lives here with a comment recording where the value
came from. A bare numeric literal in a stage module is a defect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_ROOT = REPO_ROOT / "evidence"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
CONTRACTS_ROOT = REPO_ROOT / "contracts"

# --------------------------------------------------------------------------
# Receipt identity
# --------------------------------------------------------------------------

# Bumping either value changes every digest, so old receipts stop verifying.
# Treat both as part of the on-chain contract with past runs.
PIPELINE_VERSION = "1.0.0"
RECEIPT_SCHEMA = "face-chain-verify/v1"

# --------------------------------------------------------------------------
# Face stage
# --------------------------------------------------------------------------

# InceptionResnetV1 (vggface2 weights) was trained on 160x160 aligned crops.
FACE_IMAGE_SIZE = 160

# Pixels of context kept around the MTCNN box before cropping. facenet-pytorch's
# own examples use 0 for vggface2 weights; the model was trained on tight crops.
FACE_CROP_MARGIN = 0

# MTCNN per-face detection probability below which a detection is discarded.
# Chosen to drop the low-confidence boxes MTCNN emits on background texture
# without rejecting genuine off-angle faces.
MIN_FACE_CONFIDENCE = 0.90

# InceptionResnetV1 emits L2-normalised 512-d embeddings, so cosine similarity is
# a plain dot product.
EMBEDDING_DIM = 512

# Decimals an embedding is rounded to before it is hashed into the receipt. Six
# is far below the noise floor of the model yet coarse enough that float32 repr
# differences between platforms cannot change the digest.
EMBEDDING_DECIMALS = 6

# Cosine similarity at or above which two faces are treated as the same person.
# For L2-normalised embeddings, cosine = 1 - (L2_distance ** 2) / 2, so the widely
# used vggface2 L2 threshold of ~1.0 corresponds to cosine ~0.50. We sit slightly
# above that: in a web search a false positive anchored on-chain is far worse than
# a missed match. Re-derive with scripts/calibrate_threshold.py.
FACE_MATCH_THRESHOLD = 0.55

# Similarity values are rounded to this many decimals before hashing. Raw float
# repr differs across platforms and would break cross-machine verification.
SIMILARITY_DECIMALS = 4

# --------------------------------------------------------------------------
# Search stage
# --------------------------------------------------------------------------

# Upper bound on candidates pulled from a provider response. Google Lens returns
# far more visual matches than are useful, and each one costs a download plus a
# forward pass.
MAX_CANDIDATES = 30

# Per-request network budget for fetching a candidate image.
DOWNLOAD_TIMEOUT_SECONDS = 15.0

# Refuse candidate images larger than this. Guards against a decompression bomb
# or a mislabelled video standing in for a JPEG.
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_IMAGE_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp", "image/bmp")

SERPAPI_ENDPOINT = "https://serpapi.com/search"
SERPAPI_LENS_ENGINE = "google_lens"

# Anonymous, keyless image host used when the probe image is not committed to the
# repository. Google Lens accepts an image URL, not an upload.
CATBOX_ENDPOINT = "https://catbox.moe/user/api.php"

# --------------------------------------------------------------------------
# Chain stage
# --------------------------------------------------------------------------

SEPOLIA_CHAIN_ID = 11155111
DEFAULT_SEPOLIA_RPC = "https://ethereum-sepolia-rpc.publicnode.com"
SEPOLIA_TX_EXPLORER = "https://sepolia.etherscan.io/tx/"
SEPOLIA_ADDRESS_EXPLORER = "https://sepolia.etherscan.io/address/"

SOLC_VERSION = "0.8.24"

# Gas ceiling for an anchor(). The call writes one struct and emits one event;
# measured cost is well under 120k, and the headroom absorbs a cold storage slot.
ANCHOR_GAS_LIMIT = 200_000

# Seconds to wait for an anchor transaction to be mined before giving up.
TX_RECEIPT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Settings:
    """Environment-provided values. All optional; commands validate what they need."""

    serpapi_key: str | None
    sepolia_rpc_url: str
    private_key: str | None
    registry_address: str | None
    github_raw_base: str | None


def load_settings() -> Settings:
    """Read settings from the environment (populated from .env if present)."""
    return Settings(
        serpapi_key=_optional("SERPAPI_KEY"),
        sepolia_rpc_url=os.environ.get("SEPOLIA_RPC_URL") or DEFAULT_SEPOLIA_RPC,
        private_key=_optional("PRIVATE_KEY"),
        registry_address=_optional("REGISTRY_ADDRESS"),
        github_raw_base=_optional("GITHUB_RAW_BASE"),
    )


def _optional(name: str) -> str | None:
    """Return an environment value, treating blank strings as absent.

    The shipped .env.example lists every key with an empty value, so a blank has
    to mean "not configured" rather than "configured as empty".
    """
    value = os.environ.get(name, "").strip()
    return value or None
