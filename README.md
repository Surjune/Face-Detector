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

Two real runs are committed to this repository. Nothing here is mocked or
hand-picked.

The first is what the shipped `inputs/probe.jpg` produces. Run the default
command on a clean clone and you get exactly this -
`evidence/run_2026-09-05T17-34-09Z/`.

| | |
| --- | --- |
| **Anchored post** | [A YouTube Shorts post](https://www.youtube.com/shorts/CSpe96dvqYk) |
| **Face similarity** | 0.9616 cosine (96.2%) |
| **Candidates checked** | 42 - of which 21 face-verified, **11 on social platforms** |
| **Platforms hit** | Instagram x8, YouTube x2, X x1 |
| **Blockchain record** | [Sepolia tx `0x874af845…`](https://sepolia.etherscan.io/tx/0x874af8455159eb1b33642857074e5c48748af0d9d894235a2bbc3e2d3022edd4) - block 11,641,819 |
| **Contract** | [`0x56394614…B9C4`](https://sepolia.etherscan.io/address/0x56394614d21b38C0557810e1Bb1D934b4620B9C4) |

What it found, reproduced from that folder's committed `candidates.json`:

```
[2/3] WEB / SOCIAL MEDIA SEARCH
--------------------------------------------------
+ 30 lead(s) from the visual search
+ Platform search term: "Suresh Raina" (guessed from result titles)
  - Note: a query only - identity is decided after face verification
+ 42 lead(s) total; re-running face recognition on each

  score   result            platform      source
  ------------------------------------------------------------
  0.9798  match                           Mashable India
  0.9616  match             YouTube       YouTube
  0.9550  match             X / Twitter   x.com
  0.6807  below_threshold   X / Twitter   X - rainaedits    <- rejected
  0.5792  below_threshold   TikTok        TikTok            <- rejected
  -       no_face           Threads       Threads           <- rejected

  platform          leads  verified
  ----------------------------------
  Facebook          3      0
  X / Twitter       2      1
  Threads           6      0
  LinkedIn          0      0
  YouTube           2      2
  Instagram         11     8
  TikTok            6      0

  identity
  ----------------------------------
  name           Suresh Raina
  basis          named on 11 of 21 face-verified result(s)
  derived from   title_frequency

+ Social media post found!
  - Platform: YouTube
  - Similarity: 0.9616 cosine (96.2%)
```

**The rejected candidates are shown on purpose.** A hardcoded answer would have
nothing to reject. Every candidate, kept or discarded, is recorded with its
score in `candidates.json`.

Look at the row rejected at **0.6807** - a real X post, a different person,
scoring just under the cut-off. Rows like that are the entire reason the
threshold sits where it does; see [how it is tested](#how-it-is-tested).

**The second run is a different subject**, `evidence/run_2026-09-04T09-23-05Z/`.
It anchors [a Facebook video post](https://www.facebook.com/Startuphubai/videos/what-sundar-pichai-reads-shaping-googles-ai-futurecurious-about-the-mind-behind-/1122776074034916/)
at 0.9772, with 27 of 30 candidates face-verified and 7 social matches across
Facebook, Instagram and YouTube -
[Sepolia tx `0x1b1f3780…`](https://sepolia.etherscan.io/tx/0x1b1f3780482f765c08d685a7c4865991ab1d76c70fab83d94ab587eca9a34d0f).
Same code, same command, a different face and a different platform.

## Check it yourself in two commands

No API key. No wallet. No gas. Just install and run — `verify` reads a public
Sepolia endpoint directly.

```bash
python -m src.pipeline verify --run "evidence/run_2026-09-05T17-34-09Z"
```

```
anchored 0x382e906aa7abbe48f964a8d3c00cdd8cd53122bba5fc063f2ba0e42b0a508048
computed 0x382e906aa7abbe48f964a8d3c00cdd8cd53122bba5fc063f2ba0e42b0a508048

on Ethereum Sepolia at 0x56394614d21b38C0557810e1Bb1D934b4620B9C4
  submitter    0x62d54EfeB2D75a37C1Ca36E09cD41E9a3519B11E
  block        11641819

VERIFIED  the receipt matches the record anchored on chain.
```

Now prove it is tamper-evident. This edits **one character** of the receipt:

```bash
python -m src.pipeline tamper-demo --run "evidence/run_2026-09-05T17-34-09Z"
```

```
  before  page_title='Happy retirement Ms dhoni x suresh raina \u2665\ufe0f - YouTube'
          digest=0x382e906aa7abbe48f964a8d3c00cdd8cd53122bba5fc063f2ba0e42b0a508048
  after   page_title='Happy retirement Ms dhoni x suresh raina \u2665\ufe0f - YouTuba'
          digest=0x813c858ca2fb7db11ca700392a589827ae26a879450eb1c8a00bf57acbd94ef5

  original  found
  tampered  not found

TAMPERED  the edited receipt has no record on chain.
```

One character changed at the end of the title, and the digest that results
shares nothing with the original.

That title really does end in a heart emoji, and `\u2665\ufe0f` is how it is
printed on a Windows console whose code page cannot encode one. Escaping it
rather than letting the write fail is deliberate: the same title crashed this
command outright until it was fixed.

## Run it on any face

Replace one file, run one command. Nothing else changes.

```bash
python -m src.pipeline run
```

`inputs/probe.jpg` is the only input. The pipeline has no idea who it is looking
for until it looks: the subject's name is derived at run time, never configured,
and it is not reported as an identity until the face check has ruled on every
candidate. Use `--image path/to/photo.jpg` for a file elsewhere.

*Pointed at a private individual it finds nothing, and says so.* A run on the
second committed probe returned 60 leads, rejected every one of them at the face
check, exited `no_match_found`, and reported the identity as `NOT IDENTIFIED`.
That is the correct answer, and being able to watch it refuse is what makes the
successes above mean anything.

## Install

Python 3.12, four commands:

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

<details>
<summary>Why four commands and not one</summary>

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
encyclopaedia entry. **These counts were measured at the earlier 0.50 cut-off**,
so at today's 0.70 they would be lower on both rows; the Google-versus-Yandex
gap is the point, and it is unaffected. Re-measure on your own photograph:

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

**220 tests, no network calls.** Two kinds are worth singling out:

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
already public.

The default probe is a photograph of a public figure, so everything a run
discovers about him was already public. Only `inputs/probe1.jpg` carries a
documented free licence, and the folder records the provenance of all three
images: see [`inputs/README.md`](inputs/README.md).

## Licence

MIT. See [`LICENSE`](LICENSE).
