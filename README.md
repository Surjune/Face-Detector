# Face Detector

Takes a photograph of a face, searches the web for the same person, and anchors
what it finds on a public blockchain so the discovery can be re-checked later
and shown to be untampered.

```
input image
  → detect the face and reduce it to a 512-d embedding
  → reverse-image search the web
  → re-run face recognition on every result, keep the ones that match
  → hash the matched post and anchor the hash on Ethereum Sepolia
  → verify: recompute the hash, look it up on chain
```

## Why the middle step matters

A reverse image search on its own only finds copies of the *same file*. It is an
image lookup, not a face search, and it would fail the moment the person appears
in a different photograph.

So the search result is treated as a list of leads, not as an answer. Every
candidate image is downloaded, embedded, and scored against the probe face; only
the ones that clear a measured similarity threshold are kept. That is what lets
the pipeline match a *different photograph of the same person*.

Every rejected candidate is kept too, with its score. That record is the evidence
the search actually ran — a hardcoded result would have nothing to show.

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
- Deployed address: *pending — filled in when the demo run is published*

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

91 tests. No test makes a network call. The contract tests are not mocked: the
Solidity is genuinely compiled and executed on an in-process EVM, covering
anchoring, lookup, rejection of a re-anchor, and tamper detection.

Cases needing the pretrained weights are marked `model` and deselected by
default; run them with `-m model`.

## Known limitations

- **It finds only what is already indexed.** If Google Lens has not seen a
  photograph of the subject, the pipeline correctly returns nothing. This is why
  the demo uses a public figure.
- **250 searches a month** on the SerpApi free tier. There is no reverse image
  search API that is simultaneously free, unlimited and within its provider's
  terms of service.
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
