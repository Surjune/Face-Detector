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
# This was 0.50, derived from sixteen impostor pairs across three people which
# put the highest different-person score at 0.30. That sample was far too small
# to describe the tail, and a real run proved it: two clearly different men,
# both photographed outdoors beside a motorcycle, scored 0.6548 and the wrong
# person was anchored. The visual search returns scene-similar photographs by
# design, so the face check is the only thing standing between "looks like a
# similar picture" and "is the same person".
#
# Re-measured over 990 impostor pairs from 45 distinct individuals
# (scripts/measure_impostors.py):
#
#     mean +0.1161   median +0.1099   p95 +0.4168   p99 +0.5275   max +0.6602
#
# 1.11% of those different-person pairs scored at or above the old 0.50.
#
# Genuine pairs, measured separately, ran min +0.6405 mean +0.7843 max +0.9370,
# so the two distributions genuinely overlap and no threshold separates them
# perfectly. The choice is therefore about which error to prefer, and the two
# are not symmetric: a false match is anchored on a public blockchain and cannot
# be withdrawn, while a missed match merely ends the run with an explicit
# failure. So the cut-off sits above the measured impostor maximum, accepting
# that the hardest genuine pairs are lost with it.
FACE_MATCH_THRESHOLD = 0.70

# Matches between the threshold and this value are real matches by the measured
# distribution, but close enough to the impostor tail (max +0.6602) to deserve a
# human glance before being trusted. Reported, never silently downgraded.
FACE_MARGINAL_CEILING = 0.80

# Similarity values are rounded to this many decimals before hashing. Raw float
# repr differs across platforms and would break cross-machine verification.
SIMILARITY_DECIMALS = 4

# --------------------------------------------------------------------------
# Search stage
# --------------------------------------------------------------------------

# Upper bound on candidates pulled from a single provider response. A visual
# search returns far more matches than are useful, and each one costs a download
# plus a forward pass.
MAX_CANDIDATES = 30

# Ceiling across every engine combined. Two visual engines plus platform
# expansion can otherwise produce well over a hundred leads, and at roughly
# three seconds each that turns a two-minute run into ten.
MAX_TOTAL_CANDIDATES = 60

# Per-request network budget for fetching a candidate image.
DOWNLOAD_TIMEOUT_SECONDS = 15.0

# Candidate images are fetched concurrently. Measured on a real run, a download
# takes about 1.3s against 0.6s to embed the face, so roughly 70% of the work
# per candidate is spent waiting on the network. Fetching serially left the run
# idle for most of its duration.
#
# Eight is chosen to shorten that wait without hammering any one host: candidates
# come from many different domains, so in practice only one or two requests hit
# the same server at a time.
DOWNLOAD_WORKERS = 8

# Refuse candidate images larger than this. Guards against a decompression bomb
# or a mislabelled video standing in for a JPEG.
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_IMAGE_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp", "image/bmp")

SERPAPI_ENDPOINT = "https://serpapi.com/search"
SERPAPI_LENS_ENGINE = "google_lens"

# Yandex matches ordinary faces far better than Google, which restricts public
# face matching for private individuals. Published comparisons put Yandex around
# 65-75% at finding another photograph of the same person against Google's
# 30-40%, so both engines are queried and merged.
SERPAPI_YANDEX_ENGINE = "yandex_images"

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

# How many *unverified* result titles must carry a name before it is used as a
# platform search term. One appearance is as likely to be a passing mention as
# the subject, and a wrong name spends the search budget on the wrong person.
IDENTITY_MIN_TITLE_APPEARANCES = 2

# The same bar for titles of results the face check has already accepted. It is
# lower because the evidence is stronger: the face on that page has been matched
# to the probe, so the name beside it is about the right person. A name on one
# verified page outweighs a name on a dozen unverified ones.
IDENTITY_MIN_VERIFIED_APPEARANCES = 1

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
