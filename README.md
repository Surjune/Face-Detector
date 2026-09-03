# Face Detector

A pipeline that takes a face photograph, searches the web for the same person,
and anchors whatever it finds on a public blockchain so the discovery can be
re-verified later and shown to be untampered.

```
input image
  → detect + embed the face
  → reverse-image search the web, then re-run face recognition on every candidate
  → hash the matched post and anchor it on Ethereum Sepolia
  → re-verify: recompute the hash, look it up on-chain
```

The middle step is the point. A reverse-image search alone only finds copies of
the *same file*. Re-running face recognition on every candidate the search
returns means the pipeline matches a *different photograph of the same person* —
and the rejected candidates, kept with their similarity scores, are the evidence
that the search actually ran.

## Status

Under active development. See the commit history for what has landed.

| Stage | State |
| --- | --- |
| Face detection and embedding | done |
| Blockchain anchoring and verification | in progress |
| Web / social search | in progress |

## Requirements

Python 3.12. Install the CPU-only build of PyTorch first, or pip will pull a
~2.5 GB CUDA build you do not need:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/pip install -r requirements.txt
```

Pretrained weights (~110 MB, MTCNN and InceptionResnetV1) download automatically
into the PyTorch cache the first time a face is processed.

## Configuration

Copy `.env.example` to `.env`. Every value is optional depending on the command;
verification needs nothing at all.

## Tests

```bash
.venv/Scripts/python -m pytest
```

Tests never make network calls. Cases that need the pretrained weights are
marked `model` and deselected by default; run them with `-m model`.

## Licence

MIT.
