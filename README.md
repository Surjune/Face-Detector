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
| Web search and candidate re-recognition | done |
| Blockchain anchoring and verification | done |
| Deployed contract and a published demo run | pending |

## Requirements

Python 3.12. Install the CPU-only build of PyTorch first, or pip will pull a
~2.5 GB CUDA build you do not need:

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

The third step installs the in-process EVM. It uses `--no-deps` because py-evm
declares a keccak backend whose C extension has no Windows wheel; nothing
imports it, and every dependency that is actually needed is in
`requirements.txt`. See the comments in `requirements-evm.txt`.

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
