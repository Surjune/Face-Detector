# Face Detector

Takes a photograph of a face, searches social media for the same person, and
anchors the post it finds on a public blockchain so the discovery can be
re-checked later and shown to be untampered.

```
input image
  → detect the face and reduce it to a 512-d embedding
  → harvest every result set from a reverse image search
  → identify the subject, then query the platforms the harvest missed
  → re-run face recognition on every candidate, keep the ones that match
  → hash the matched social post and anchor it on Ethereum Sepolia
  → verify: recompute the hash, look it up on chain
```

Seven platforms are covered deliberately: **Facebook, X, Threads, LinkedIn,
YouTube, Instagram and TikTok** (Reddit counts too when it turns up).

## See it working

`evidence/run_2026-09-04T09-23-05Z/` is a real run committed to this repository.
The search returned 30 leads, the subject was identified from Google's own
resolved entity, and the platforms the harvest missed were queried directly.
Nine candidates were face-verified on social platforms:

| Platform | Leads | Face-verified |
| --- | --- | --- |
| Instagram | 6 | 6 |
| YouTube | 3 | 2 |
| Facebook | 1 | 1 |

The anchored match is a **Facebook video post** at 0.9772 cosine similarity,
recorded on Sepolia in
[tx `0x1b1f3780…`](https://sepolia.etherscan.io/tx/0x1b1f3780482f765c08d685a7c4865991ab1d76c70fab83d94ab587eca9a34d0f).
Open `report.html` inside the folder for the full candidate table, or reproduce
the check yourself with nothing but a clean clone:

```bash
python -m src.pipeline verify --run "evidence/run_2026-09-04T09-23-05Z"
```

```bash
python -m src.pipeline tamper-demo --run "evidence/run_2026-09-04T09-23-05Z"
```

An earlier run is kept at `evidence/run_2026-09-03T16-10-03Z/`. It verifies too,
and its recorded search response is real Lens output that the test-suite parses
directly.

Neither command needs a key, a wallet, or any gas — `verify` reads a public
Sepolia RPC endpoint directly.

## Why the middle step matters

A reverse image search on its own only finds copies of the *same file*. It is an
image lookup, not a face search, and it would fail the moment the person appears
in a different photograph.

So search results are treated as leads, not answers. Every candidate image is
downloaded, embedded, and scored against the probe face; only the ones that clear
a measured similarity threshold are kept. That is what lets the pipeline match a
*different photograph of the same person*.

Every rejected candidate is kept too, with its score. That record is the evidence
the search actually ran — a hardcoded result would have nothing to show.

### No API searches social media by face

Worth stating plainly, because it shapes the whole design. Meta's Graph API reads
only Pages you own, behind app review and business verification. X removed its
free tier in February 2026. LinkedIn and TikTok are approval-gated. Nothing
legitimate accepts a face and returns Instagram posts.

What this pipeline does instead is find social posts through indexes that *are*
reachable — a visual search, and `site:`-scoped queries per platform — and then
**verify each one independently against the probe face**. That is the brief's
"API, or a scripted search approach". A claim to search inside Instagram by face
would be a lie, and the rejected candidates are here precisely so the search can
be audited rather than taken on trust.

### Finding a social post rather than settling for one

Two things make the difference between finding a social post and finding a
Wikipedia page:

- **Every result set is harvested.** A Lens response carries candidates in three
  separate arrays; the social posts concentrate in the two that are easy to
  overlook. They are interleaved rather than read in order, because the first
  array alone fills the candidate budget.
- **Each candidate carries several image URLs.** Facebook and Instagram serve
  their canonical image through crawler endpoints that answer with HTML for
  anything else, while a working thumbnail of the same post sits beside it.
  Trying only the first URL silently discards those posts.

Selection then prefers a verified social match over a higher-scoring page. If
nothing social clears the threshold the run **fails** with `no_social_match_found`
rather than quietly anchoring something that does not meet the requirement;
`--allow-non-social` overrides that deliberately.

## What goes on chain

Only a hash. The receipt — post URL, image hash, page title, similarity score,
which provider was queried and when — stays in the evidence folder. Nothing that
identifies a person is published: no image, no face embedding, no name.

A verifier recomputes the hash from their own copy of the receipt and looks it up
in the registry. A match proves the receipt is byte-identical to the one anchored
at that block's timestamp. Any edit, down to one character, changes the hash and
the lookup fails.

## Which blockchain, and why

**Ethereum Sepolia**, with an in-process chain for offline runs.

Polygon Amoy was the first choice and was dropped: the official Polygon faucet
has shut down, and the surviving Amoy faucets require the requesting wallet to
already hold real ETH on Ethereum mainnet. That fails the constraint that this
project cost nothing to run. Sepolia has the [pk910 proof-of-work
faucet](https://sepolia-faucet.pk910.de), which asks only that your browser mine
for a minute — no account, no mainnet balance, no payment.

Sepolia also has free keyless public RPC endpoints, which is what lets `verify`
run on a clean clone with no signup of any kind, and Etherscan for a link anyone
can click.

- Contract: `contracts/PostRegistry.sol`, Solidity 0.8.24
- Deployed address: [`0x56394614d21b38C0557810e1Bb1D934b4620B9C4`](https://sepolia.etherscan.io/address/0x56394614d21b38C0557810e1Bb1D934b4620B9C4)
- Deployment transaction: [`0x6ac98a8d…c29b`](https://sepolia.etherscan.io/tx/0x6ac98a8dbfd7a64148e60de0b2f8adfaeb05b09ae3b29d4ec68197f90c47c29b)

## Install

Python 3.12. (facenet-pytorch pins `torch<2.3`, which has no wheels for 3.13.)

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/pip install -r requirements.txt
```

```bash
.venv/Scripts/pip install --no-deps -r requirements-evm.txt
```

Three steps rather than one, each for a reason:

1. The CPU-only PyTorch index. Plain `pip install torch` can fetch a ~2.5 GB CUDA
   build that nothing here uses; the CPU wheel is ~200 MB.
2. Everything else, pinned to the exact versions facenet-pytorch requires so the
   resolver settles in one pass.
3. The in-process EVM, with `--no-deps`. py-evm declares a keccak backend whose C
   extension ships no Windows wheel, so pip tries to compile it and fails on any
   machine without MSVC build tools. Nothing imports it — eth-hash resolves keccak
   through pycryptodome — and every dependency that *is* needed is in
   `requirements.txt`.

Pretrained weights (~110 MB, MTCNN and InceptionResnetV1) download into the
PyTorch cache the first time a face is processed.

## Configuration

Copy `.env.example` to `.env`. Every value is optional depending on the command.

| To run | You need |
| --- | --- |
| `verify`, `tamper-demo`, `run --replay` | nothing |
| `run --chain local` | a free [SerpApi](https://serpapi.com) key |
| `run` on Sepolia | that key, plus a funded throwaway Sepolia key |

Two further keys deepen the search. Both are optional, and both are free with no
payment card — the pipeline runs identically without them.

| Key | Free tier | What it adds |
| --- | --- | --- |
| [`YOUTUBE_API_KEY`](https://console.cloud.google.com) | 10,000 units/day (~100 searches), no billing account | Searches YouTube natively instead of relying on what the visual search happened to index |
| [`GEMINI_API_KEY`](https://aistudio.google.com/apikey) or [`GROQ_API_KEY`](https://console.groq.com) | Gemini ~1,500 req/day; Groq 30 req/min | Reads the subject's name from result titles when the visual search has not already supplied it |

The default models are rolling aliases rather than pinned versions, so a clone
still works months from now; set `LLM_MODEL` to pin one.

Gemini uses the same Google account as the YouTube key, so that is one signup for
both. Only one short LLM call is made per run, and only when the free
deterministic path fails.

**No paid API is used anywhere in this project.** The Anthropic API was considered
for the identity step and rejected for exactly this reason: it is pay-as-you-go
with no free tier.

Use a key that has never held real funds. It signs one transaction on a test
network and nothing else.

## Commands

```bash
python -m src.pipeline run --image inputs/probe.jpg
```

Full pipeline: find the face, search, score every candidate, anchor the match on
Sepolia, write an evidence folder and an HTML report.

```bash
python -m src.pipeline run --image inputs/probe.jpg --chain local
```

The same, on an EVM running inside the Python process. No faucet, no gas, no
network beyond the search itself.

```bash
python -m src.pipeline verify --run evidence/<run>
```

Recompute the receipt's hash and look it up on chain. Needs no key, no wallet and
no gas — it is a read against a public endpoint. Add `--live` to also re-download
the matched image and check whether the post itself has changed since.

```bash
python -m src.pipeline tamper-demo --run evidence/<run>
```

Edit one character of the receipt in memory and show that the hash changes and
the on-chain lookup fails. The stored file is not modified.

Exit codes: `0` verified, `1` tampered or absent, `2` unverifiable (the run was
anchored on an in-process chain that no longer exists).

## How the search stays honest

Live search uses SerpApi's Google Lens engine — 250 searches a month on the free
tier, no card. Google publishes no reverse-image-search API of its own.

Every run records the provider's untouched response into its evidence folder.
`--replay <run>` re-runs the pipeline over that recording, which is how someone
without an API key reproduces a published run. A replay is labelled as a replay
everywhere it appears, including inside the receipt that gets hashed. It is a
recording of a real search, and is never presented as a live one.

If no candidate clears the threshold, the run fails with `no_match_found`. It
never falls back to a weaker match, and never invents one.

## The similarity threshold

`FACE_MATCH_THRESHOLD` is measured, not guessed. Rebuild the measurement:

```bash
python scripts/fetch_calibration_set.py --out dataset/
```

```bash
python scripts/calibrate_threshold.py --dataset dataset/
```

The first assembles a labelled set of freely licensed Wikimedia Commons
photographs; the second reports both distributions. Measured over three people:

| | n | min | mean | max |
| --- | --- | --- | --- | --- |
| genuine pairs | 5 | +0.6405 | +0.7843 | +0.9370 |
| impostor pairs | 16 | −0.1949 | +0.0286 | +0.3040 |

The distributions do not overlap. The threshold sits at 0.50, inside the empty
band and slightly above its midpoint, because the errors are not symmetric: a
false match is anchored on a public blockchain and cannot be withdrawn, while a
missed match merely ends the run.

Two rules keep that measurement meaningful. Images with more or fewer than one
detected face are skipped, because a group photograph filed under a name often
shows someone else largest. And an image that agrees with no other image in its
own folder is dropped — Commons is searched by text, so a file captioned "Meet
Google CEO Sundar Pichai" frequently shows a different person at the same event.

## Tests

```bash
.venv/Scripts/python -m pytest
```

182 tests. No test makes a network call. Two kinds are worth singling out:

- The **contract tests are not mocked**. The Solidity is genuinely compiled and
  executed on an in-process EVM, covering anchoring, lookup, rejection of a
  re-anchor, and tamper detection.
- The **harvest tests parse the real recorded Lens response** committed with the
  earlier demo run, not a hand-written fixture. They assert that the X post, the
  Instagram reels and the Facebook videos are recovered from it — the direct
  evidence that those results were in the response all along.

Cases needing the pretrained weights are marked `model` and deselected by
default; run them with `-m model`.

## Known limitations

- **It finds only what is already indexed.** If Google Lens has not seen a
  photograph of the subject, the pipeline correctly returns nothing. This is why
  the demo uses a public figure.
- **250 searches a month** on the SerpApi free tier. A run spends one search on
  the visual harvest plus up to four on the platforms it missed, so roughly 50
  full runs a month. There is no reverse image search API that is simultaneously
  free, unlimited and within its provider's terms of service.
- **Platform reachability differs, and it is not uniform.** Facebook, Instagram,
  X, YouTube, LinkedIn and Reddit posts are reachable through the visual index
  and `site:` queries. Threads and TikTok are indexed far more thinly and often
  return nothing — that is a real absence, and the coverage table reports it
  rather than hiding it.
- **A search engine's index is not the platform.** A post that exists but has
  never been crawled is invisible to this pipeline, so "not found" never means
  "does not exist".
- **False positives are possible.** The threshold is derived from five genuine
  pairs — enough to show clean separation, not enough to characterise the tail.
  Treat a match as a strong lead, not proof of identity.
- **The probe image is published.** Lens takes a URL, not an upload. An image
  inside this repository is served from GitHub; anything else is uploaded to
  catbox.moe, a public host.
- **Pillow is pinned to 10.2.x** by facenet-pytorch, and the pipeline decodes
  images downloaded from arbitrary web servers. Downloads are capped in size and
  checked for type first, but a newer Pillow would be preferable.
- **A local-chain run cannot be re-verified.** The in-process chain exists only
  for the life of the process. `verify` reports this as exit code 2, distinct
  from tampering.
- **Anyone can anchor anything.** The registry records that a digest existed at a
  point in time and who submitted it. It does not, and cannot, attest that the
  underlying search was honest — only that the receipt has not changed since.
- **No keyless search backend.** A scraper fallback needing no signup was
  considered and left out: one that breaks mid-demo is worse than none.

## Ethics

This is a face search tool, and the technique it demonstrates can be used to
identify strangers. Point it at yourself, at a consenting subject, or at a public
figure whose material is already public. The probe image shipped here is a CC BY
licensed photograph of a public figure; see `inputs/README.md`.

## Licence

MIT. See `LICENSE`.
