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

# Drop any photograph here and `run` needs no arguments at all. Resolved from
# the repository root rather than the working directory, so the command behaves
# the same wherever it is invoked from.
DEFAULT_PROBE_IMAGE = REPO_ROOT / "inputs" / "probe.jpg"
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
#
# Measured, not assumed. Over a labelled set of freely licensed Commons
# photographs of three people (scripts/fetch_calibration_set.py, then
# scripts/calibrate_threshold.py):
#
#     genuine pairs    n=5   min +0.6405  mean +0.7843  max +0.9370
#     impostor pairs   n=16  min -0.1949  mean +0.0286  max +0.3040
#
# The two distributions do not overlap; the empty band runs from 0.3040 to
# 0.6405, whose midpoint is 0.4722. This value sits just above that midpoint
# because the asymmetry matters: a false match gets anchored on a public
# blockchain and cannot be withdrawn, while a missed match merely ends the run.
# That leaves 0.20 of margin above the worst impostor and 0.14 below the worst
# genuine pair.
#
# The sample is small, so treat this as calibrated rather than settled, and
# re-derive it on a larger set before relying on the exact value.
FACE_MATCH_THRESHOLD = 0.50

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
# Targeted social media search
# --------------------------------------------------------------------------

SERPAPI_WEB_ENGINE = "google"

# Ceiling on site:-scoped expansion queries per run. Only platforms the Lens
# harvest missed are queried, so this is rarely reached; with the Lens call it
# caps a run at 5 of the 250 free monthly SerpApi searches (~50 runs a month).
MAX_EXPANSION_SEARCHES = 4

# Results requested per platform query. Enough to find the subject's own posts
# without paying to download a long tail of incidental mentions.
SITE_SEARCH_RESULTS = 10

# Rank a verified match on a social platform above an equal-scoring one that is
# not. The brief asks for a social media post specifically, and an exact-file
# copy on an encyclopaedia otherwise wins on raw similarity every time.
PREFER_SOCIAL_MATCH = True

YOUTUBE_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"

# One search.list call costs 100 of the 10,000 free daily quota units, so a run
# spends 1% of the day's allowance regardless of how many results are asked for.
YOUTUBE_MAX_RESULTS = 10

# --------------------------------------------------------------------------
# Optional LLM (free tiers only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMProvider:
    """A free-tier, OpenAI-compatible chat provider."""

    name: str
    env_var: str
    base_url: str
    default_model: str


# Tried in this order. Both have a permanent free tier that needs no payment
# card. The Anthropic API is deliberately absent: it is pay-as-you-go with no
# free tier, which would break this project's zero-cost constraint.
#
# Gemini is tried first because its default is a rolling alias rather than a
# pinned version. Pinned model names expire: an earlier revision of this file
# named two models that had both been withdrawn by the time the keys were
# tested, and each returned a 404 that looked like an authentication failure.
# An alias keeps a clone working months later. `LLM_MODEL` pins a version when
# reproducibility matters more than longevity.
LLM_PROVIDERS: tuple[LLMProvider, ...] = (
    LLMProvider(
        name="gemini",
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-flash-lite-latest",
    ),
    LLMProvider(
        name="groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-20b",
    ),
)

# Generous for a one-line answer, because the ceiling is not really sizing the
# answer. Groq's gpt-oss models reason before replying and that reasoning is
# billed against the same budget: at 64 tokens they hit the limit mid-thought
# and return an empty message with finish_reason "length". 512 leaves room to
# think and still answer.
LLM_MAX_TOKENS = 512
LLM_TIMEOUT_SECONDS = 30.0

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
    youtube_api_key: str | None
    llm_keys: dict[str, str]
    llm_model: str | None


def load_settings() -> Settings:
    """Read settings from the environment (populated from .env if present)."""
    return Settings(
        serpapi_key=_optional("SERPAPI_KEY"),
        sepolia_rpc_url=os.environ.get("SEPOLIA_RPC_URL") or DEFAULT_SEPOLIA_RPC,
        private_key=_optional("PRIVATE_KEY"),
        registry_address=_optional("REGISTRY_ADDRESS"),
        github_raw_base=_optional("GITHUB_RAW_BASE"),
        youtube_api_key=_optional("YOUTUBE_API_KEY"),
        llm_keys={
            provider.env_var: key
            for provider in LLM_PROVIDERS
            if (key := _optional(provider.env_var))
        },
        llm_model=_optional("LLM_MODEL"),
    )


def _optional(name: str) -> str | None:
    """Return an environment value, treating blank strings as absent.

    The shipped .env.example lists every key with an empty value, so a blank has
    to mean "not configured" rather than "configured as empty".
    """
    value = os.environ.get(name, "").strip()
    return value or None
