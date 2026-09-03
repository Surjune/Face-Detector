# Conventions

Scoped to this repository. A parent directory contains an unrelated project's
conventions file; it does not apply here.

## Layering

```
pipeline (CLI)  ->  face | search | chain | report  ->  config, errors
```

- `pipeline.py` parses arguments, sequences stages and prints. No business logic.
- Stage packages never import each other. Anything shared moves down into `config`
  or a small helper both can import.
- `config.py` and `errors.py` are leaf modules and import nothing from the stages.

## No magic numbers

Every threshold, limit, timeout and physical constant lives in `src/config.py` as a
named constant with a comment recording its source or how it was derived. A bare
numeric literal in a stage module is a defect. Exceptions: `0`, `1`, array indices.

## Errors

Every external call (network, chain, model load, file read) has explicit handling and
raises a typed exception from `src/errors.py` carrying a machine-readable `code`.
Never a bare `except:`, never `except: pass`. A missing credential or unreachable
upstream produces an explicit error, never fabricated or placeholder data — an
invented match would defeat the entire point of the project.

## Determinism

The receipt digest must be byte-reproducible on any machine, or verification proves
nothing. All canonicalization goes through `src/chain/canonical.py`; floats are
rounded and emitted as strings so platform float repr cannot drift the hash.

## Testing

`tests/` mirrors `src/`. No test makes a real network call — providers and chain
clients are mocked or run against the in-process eth-tester chain.

## Attribution

Commit messages are plain conventional commits. No AI/assistant attribution,
co-author trailers, generated-with footers, or acknowledgement sections anywhere in
this repository — including README, code comments and generated reports.
