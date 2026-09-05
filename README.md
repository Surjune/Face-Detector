# Face Detector

**Give it a photograph of a face. It finds that person's social media post, then
records the discovery on a public blockchain so nobody can alter it afterwards.**

```
   photograph                                              Ethereum Sepolia
       │                                                          ▲
       ▼                                                          │
 ┌───────────┐      ┌────────────────┐      ┌──────────────┐      │
 │  1. FACE  │─────▶│   2. SEARCH    │─────▶│  3. ANCHOR   │──────┘
 │           │      │                │      │              │
 │  detect   │      │ 2 image engines│      │ hash the post│
 │  encode   │      │ + per-platform │      │ write it on  │
 │  512-d    │      │   queries      │      │ chain        │
 └───────────┘      └────────┬───────┘      └──────────────┘
                             │
                    every result is re-checked
                    against the probe face —
                    only real matches survive
```

Seven platforms are searched deliberately: **Facebook · X · Threads · LinkedIn ·
YouTube · Instagram · TikTok**

---

## Proof it works

A real run is committed to this repository at
`evidence/run_2026-09-04T09-23-05Z/`. Nothing here is mocked or hand-picked.

| | |
| --- | --- |
| **Anchored post** | [A Facebook video post](https://www.facebook.com/Startuphubai/videos/what-sundar-pichai-reads-shaping-googles-ai-futurecurious-about-the-mind-behind-/1122776074034916/) |
| **Face similarity** | 0.9772 cosine (97.7%) |
| **Candidates checked** | 30 — of which 27 face-verified, **7 on social platforms** |
| **Platforms hit** | Instagram ×4, YouTube ×2, Facebook ×1 |
| **Blockchain record** | [Sepolia tx `0x1b1f3780…`](https://sepolia.etherscan.io/tx/0x1b1f3780482f765c08d685a7c4865991ab1d76c70fab83d94ab587eca9a34d0f) · block 11,632,578 |
| **Contract** | [`0x56394614…B9C4`](https://sepolia.etherscan.io/address/0x56394614d21b38C0557810e1Bb1D934b4620B9C4) |

What the run actually printed:

```
[2/3] WEB / SOCIAL MEDIA SEARCH
--------------------------------------------------
✓ 30 lead(s) from the visual search
✓ Platform search term: "Sundar Pichai" (guessed from result titles)
  - Note: a query only - identity is decided after face verification
✓ site:threads search: 0 lead(s)

  score   result            platform      source
  ------------------------------------------------------------
  0.9772  match             Facebook      Facebook
  0.8391  match             Instagram     Instagram
  0.8176  match             YouTube       YouTube
  0.4979  below_threshold   Facebook      Facebook      ← rejected
  -       no_face                         Wikimedia     ← rejected

  platform          leads  verified
  ----------------------------------
  Facebook          1      1
  Instagram         6      4
  YouTube           3      2

  identity
  ----------------------------------
  name           Sundar Pichai
  basis          named on 19 of 27 face-verified result(s)
  derived from   lens_related_content

✓ Social media post found!
  - Platform: Facebook
  - Similarity: 0.9772 cosine (97.7%)

[3/3] BLOCKCHAIN ATTESTATION & RECORDING
--------------------------------------------------
✓ Payload SHA-256 digest: 0x1aa8d69d97a5ba3e851c549cc6b3a5a71d8a92791f56f72288ee4e877733c7a0
✓ Transaction mined in block #11,632,578
```

**The rejected candidates are shown on purpose.** A hardcoded answer would have
nothing to reject. Every candidate, kept or discarded, is recorded with its score
in `candidates.json`.

## Check it yourself in two commands

No API key. No wallet. No gas. Just install and run — `verify` reads a public
Sepolia endpoint directly.

```bash
python -m src.pipeline verify --run "evidence/run_2026-09-04T09-23-05Z"
```

```
anchored 0x1aa8d69d97a5ba3e851c549cc6b3a5a71d8a92791f56f72288ee4e877733c7a0
computed 0x1aa8d69d97a5ba3e851c549cc6b3a5a71d8a92791f56f72288ee4e877733c7a0

VERIFIED  the receipt matches the record anchored on chain.
```

Now prove it is tamper-evident. This edits **one character** of the receipt:

```bash
python -m src.pipeline tamper-demo --run "evidence/run_2026-09-04T09-23-05Z"
```

```
  before  "...Play on Facebook. 1:04"   digest=0x1aa8d69d…733c7a0
  after   "...Play on Facebook. 1:0a"   digest=0xa6034a9f…72b5fd95

  original  found
  tampered  not found

TAMPERED  the edited receipt has no record on chain.
```

## Run it on any face

Replace one file, run one command. Nothing else changes.

```bash
python -m src.pipeline run
```

`inputs/probe.jpg` is the only input. The pipeline has no idea who it is looking
for until the search tells it — the subject's name is derived at run time, never
configured. Use `--image path/to/photo.jpg` for a file elsewhere.

*Verified on a second person with no other change: it identified them, searched
the platforms the visual index had missed, and anchored a LinkedIn post
face-verified at 0.9600 with 10 social matches across four platforms.*

## Install

Python 3.12, three commands:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/pip install -r requirements.txt && .venv/Scripts/pip install --no-deps -r requirements-evm.txt
```

<details>
<summary>Why three commands and not one</summary>

1. **CPU-only PyTorch.** A plain `pip install torch` can pull a ~2.5 GB CUDA
   build nothing here uses; the CPU wheel is ~200 MB.
2. **Everything else**, pinned to the exact versions facenet-pytorch requires so
   the resolver settles in one pass instead of backtracking through a large
   download.
3. **The in-process EVM, with `--no-deps`.** py-evm declares a keccak backend
   whose C extension ships no Windows wheel, so pip tries to compile it and
   fails on any machine without MSVC build tools. Nothing imports it — keccak
   resolves through pycryptodome — and every dependency that *is* needed is in
   `requirements.txt`.

Pretrained weights (~110 MB) download automatically on first use.
</details>

Copy `.env.example` to `.env`. **Verification needs nothing at all.** A full
search run needs a free [SerpApi](https://serpapi.com) key (250 searches/month,
no card) and a throwaway Sepolia key funded at the
[pk910 faucet](https://sepolia-faucet.pk910.de). Two further keys deepen the
search and are optional: a free
[YouTube Data API](https://console.cloud.google.com) key, and a free
[Gemini](https://aistudio.google.com/apikey) or [Groq](https://console.groq.com)
key. **No paid API is used anywhere in this project.**

## Which blockchain, and why

**Ethereum Sepolia** — a public testnet — with an in-process chain for offline runs.

Polygon Amoy was the first choice and was dropped: the official faucet has shut
down and the surviving Amoy faucets require the wallet to already hold **real ETH
on mainnet**, which breaks the zero-cost constraint. Sepolia's proof-of-work
faucet asks only that your browser mine for a minute — no account, no mainnet
balance, no payment. It also has free keyless RPC endpoints, which is exactly why
`verify` works on a clean clone with no signup, and Etherscan for a link anyone
can click.

**Only a hash goes on chain.** The receipt — post URL, image hash, page title,
similarity score — stays in the evidence folder. No image, no face embedding, no
name is ever published. A verifier recomputes the hash from their own copy and
looks it up; a match proves the receipt is byte-identical to the one anchored at
that block's timestamp.

Contract: [`contracts/PostRegistry.sol`](contracts/PostRegistry.sol), Solidity 0.8.24.

## What makes the search genuine

A reverse image search alone only finds copies of the *same file*. That is an
image lookup, not a face search, and it fails the moment the person appears in a
different photograph.

So search results are treated as **leads, not answers**. Every candidate image is
downloaded, embedded, and scored against the probe face. Only those clearing a
measured threshold survive — which is what lets the pipeline match a *different
photograph of the same person*.

Three things make it find a social post rather than settle for a Wikipedia page:

- **Two engines, not one.** Google restricts public face matching for private
  individuals; Yandex does not. On an ordinary person Yandex verified 11 matches
  where Google managed 2.
- **Every result set is harvested.** A Lens response holds candidates in three
  arrays and the social posts concentrate in the two easiest to overlook. They
  are interleaved, because the first array alone would fill the budget.
- **Each candidate carries several image URLs.** Facebook and Instagram serve
  their images through crawler endpoints that return HTML to everyone else, while
  a working thumbnail sits beside it. Using only the first URL silently discards
  those posts.

If nothing social clears the threshold, the run **fails loudly** with
`no_social_match_found` rather than quietly anchoring something that does not meet
the requirement.

## How far it reaches — measured, not claimed

Reach depends almost entirely on who the subject is.
`scripts/compare_engines.py` measures it for any photograph; these are its real
numbers:

| Subject | Engine | Leads | Face-verified | Social | Best score | Identity |
| --- | --- | --- | --- | --- | --- | --- |
| Public figure | Google Lens | 30 | 30 | 3 | 1.0000 | resolved |
| | Yandex | 30 | 30 | 2 | 1.0000 | — |
| Ordinary person | Google Lens | 30 | **2** | 1 | 0.5217 | — |
| | Yandex | 30 | **11** | 1 | 0.6354 | — |

The ordinary subject published a CC-licensed portrait of themselves and has no
encyclopaedia entry. Run it on your own photograph:

```bash
python scripts/compare_engines.py --image path/to/your/photo.jpg
```

## Known limitations

Stated plainly, because a tool like this is easy to overclaim.

- **No API searches social media by face.** Meta's Graph API reads only Pages you
  own, behind app review. X removed its free tier in February 2026. LinkedIn and
  TikTok are approval-gated. This pipeline finds posts through indexes that *are*
  reachable, then verifies each one against the face. A claim to search inside
  Instagram by face would be a lie.
- **It finds only what is already indexed.** A private account is invisible, and a
  public post that was never crawled is too. "Not found" never means "does not
  exist".
- **Ordinary people are much harder than public figures** — see the table above.
  Identity resolution comes from Google's Knowledge Graph, which has no entry for
  a private individual, so the targeted per-platform search cannot run for them.
- **A name is never announced before the face check.** The search term used for
  the per-platform queries is scraped from the titles of pages that merely looked
  similar, so it frequently names a stranger. It is printed as a query, and the
  run reports an identity only afterwards, counted from results whose face
  actually matched. When nothing matches, the verdict is `NOT IDENTIFIED`.
- **A match is a strong lead, not proof of identity.** Different people have
  been measured as high as 0.6602, and genuine pairs start at 0.6405 — the two
  overlap. Anything between 0.70 and 0.80 is flagged as marginal and should be
  eyeballed before it is believed.
- **The visual search returns scene-similar photographs by design.** Two people
  beside a motorcycle in a field will be offered as candidates; the face check
  is the only thing separating "similar picture" from "same person".
- **250 searches a month** on the free SerpApi tier — about 50 full runs.
- **The probe image is published.** Reverse image search takes a URL, not an
  upload, so an image not already in this repository is uploaded to a public host.
- **Anyone can anchor anything.** The registry proves a receipt has not changed
  since it was recorded. It cannot attest that the search behind it was honest —
  which is exactly why every rejected candidate is kept and published.

## How it is tested

```bash
.venv/Scripts/python -m pytest
```

**209 tests, no network calls.** Two kinds are worth singling out:

- The **contract tests are not mocked** — the Solidity is genuinely compiled and
  executed on an in-process EVM, covering anchoring, lookup, rejection of a
  re-anchor, and tamper detection.
- The **harvest tests parse a real recorded search response**, not a fixture, and
  assert that the X post, Instagram reels and Facebook videos are all recovered
  from it.

The threshold is measured, and it has been measured twice.

The first attempt used 16 impostor pairs from 3 people and concluded different
faces top out at 0.30, so the cut-off was set to 0.50. A real run then matched
**two clearly different men** — both photographed outdoors beside a motorcycle —
at **0.6548**, and anchored the wrong person. Sixteen pairs say nothing about
the tail of a distribution.

Re-measured over **990 impostor pairs from 45 distinct people**
(`scripts/measure_impostors.py`):

| | Score |
| --- | --- |
| mean | +0.1161 |
| 95th percentile | +0.4168 |
| 99th percentile | +0.5275 |
| **maximum** | **+0.6602** |

1.11% of different-person pairs scored above the old 0.50 cut-off. Genuine pairs
run 0.6405–0.9370, so the distributions genuinely **overlap** — no threshold
separates them perfectly.

The cut-off is now **0.70**, above the measured impostor maximum. That loses the
hardest genuine pairs, which is the right trade: a false match is anchored on a
public blockchain and cannot be withdrawn, while a missed match merely ends the
run with an explicit failure. Matches between 0.70 and 0.80 are reported as
**marginal** and flagged for human review rather than presented as certain.

## Ethics

This is a face search tool, and the technique can be used to identify strangers.
Point it at yourself, a consenting subject, or a public figure whose material is
already public. The probe image shipped here is a CC BY licensed photograph of a
public figure; see [`inputs/README.md`](inputs/README.md).

## Licence

MIT. See [`LICENSE`](LICENSE).
