"""Command line entry point.

    run           face -> web search -> anchor on chain
    verify        recompute a stored receipt's digest and look it up on chain
    tamper-demo   edit a stored receipt and watch verification fail

Only `run` needs credentials. `verify` and `tamper-demo` work from a committed
evidence folder with no key, no wallet and no gas.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from src import evidence
from src.chain import (
    MatchRecord,
    Receipt,
    RegistryClient,
    SearchRecord,
    describe_chain,
    load_deployment,
    receipt_digest,
    sha256_hex,
)
from src.config import (
    FACE_MATCH_THRESHOLD,
    SEPOLIA_CHAIN_ID,
    SEPOLIA_TX_EXPLORER,
    Settings,
    load_settings,
)
from src.errors import (
    ChainNotConfigured,
    DownloadError,
    NoMatchFound,
    PipelineError,
    SearchNotConfigured,
)
from src.face import encode_primary_face
from src.report.html import write_report
from src.search import (
    CANDIDATE_IMAGE_DIRNAME,
    ReplayProvider,
    ScoredCandidate,
    SearchProvider,
    SerpApiLensProvider,
    download_image,
    public_url_for,
    record_response,
    score_candidates,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Face identification, web search and blockchain verification.",
)


@app.command()
def run(
    image: Annotated[Path, typer.Option("--image", help="Photograph of the face to search for.")],
    chain: Annotated[
        str, typer.Option("--chain", help="Where to anchor: 'sepolia' or 'local'.")
    ] = "sepolia",
    replay: Annotated[
        Path | None,
        typer.Option("--replay", help="Reuse the search recorded in this run directory."),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help=f"Cosine cut-off (default {FACE_MATCH_THRESHOLD})."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to write the evidence folder.")
    ] = None,
) -> None:
    """Search the web for a face and anchor what is found."""
    settings = load_settings()
    cutoff = threshold if threshold is not None else FACE_MATCH_THRESHOLD
    run_dir = evidence.create_run_dir(out)
    print(f"Evidence: {run_dir}\n")

    # -- 1. face ---------------------------------------------------------
    print("[1/3] FACE")
    if not image.exists():
        raise PipelineError(f"No such image: {image}")
    encoding = encode_primary_face(image)
    input_digest = sha256_hex(image.read_bytes())
    evidence.copy_input_image(run_dir, image)
    left, top, right, bottom = (int(value) for value in encoding.box)
    print(
        f"      1 face  confidence={encoding.confidence:.3f}  "
        f"box=({left},{top},{right},{bottom})"
    )
    print(f"      embedding {encoding.digest[:16]}...  image {input_digest[:16]}...\n")

    # -- 2. search -------------------------------------------------------
    print("[2/3] SEARCH")
    provider, query_url = _build_provider(settings, image, replay)
    print(f"      provider: {provider.name}")
    print(f"      query image: {query_url}")

    response = provider.search(query_url)
    record_response(run_dir, response)
    print(f"      {len(response.candidates)} candidate(s); scoring each against the probe face")

    scored = score_candidates(
        encoding.embedding,
        response.candidates,
        run_dir / CANDIDATE_IMAGE_DIRNAME,
        threshold=cutoff,
    )
    evidence.write_candidates(run_dir, scored)
    _print_candidates(scored)

    matches = [item for item in scored if item.is_match]
    if not matches:
        raise NoMatchFound(
            "No candidate cleared the similarity threshold",
            candidates=len(scored),
            threshold=cutoff,
        )
    best = matches[0]
    print(f"\n      best match: {best.candidate.page_url}\n")

    # -- 3. chain --------------------------------------------------------
    print("[3/3] CHAIN")
    receipt = Receipt(
        input_image_sha256=input_digest,
        embedding_sha256=encoding.digest,
        match=MatchRecord(
            post_url=best.candidate.page_url,
            image_url=best.candidate.image_url,
            image_sha256=best.image_sha256 or "",
            page_title=best.candidate.title,
            similarity=best.similarity if best.similarity is not None else 0.0,
        ),
        search=SearchRecord(
            provider=response.provider,
            query_image_sha256=input_digest,
            candidate_count=len(scored),
            retrieved_at=response.retrieved_at,
        ),
    )
    evidence.write_receipt(run_dir, receipt)
    digest = receipt_digest(receipt)
    print(f"      receipt digest {digest}")

    client = _connect_for_anchoring(chain, settings)
    print(f"      anchoring on {describe_chain(client.chain_id)} at {client.address}")
    result = client.anchor(digest, run_dir.name)
    evidence.write_anchor(run_dir, digest, result)
    tx_hash = evidence.prefixed(result.tx_hash)
    print(f"      tx {tx_hash}  block {result.block_number}  gas {result.gas_used}")
    if result.chain_id == SEPOLIA_CHAIN_ID:
        print(f"      {SEPOLIA_TX_EXPLORER}{tx_hash}")
    else:
        print("      in-process chain: this record is gone when the process exits")

    report = write_report(run_dir)
    print(f"\nDone. Report: {report}")
    print(f"Verify with: python -m src.pipeline verify --run {run_dir}")


@app.command()
def verify(
    run: Annotated[Path, typer.Option("--run", help="Evidence folder to verify.")],
    live: Annotated[
        bool, typer.Option("--live", help="Also re-download the post image and re-hash it.")
    ] = False,
) -> None:
    """Recompute a stored receipt's digest and look it up on chain."""
    receipt = evidence.read_receipt(run)
    anchor = evidence.read_anchor(run)
    recomputed = receipt_digest(receipt)

    print(f"receipt  {run / evidence.RECEIPT_FILENAME}")
    print(f"anchored {anchor.digest}")
    print(f"computed {recomputed}")

    if recomputed != anchor.digest:
        print("\nTAMPERED  the receipt no longer hashes to the value that was anchored.")
        raise typer.Exit(code=1)

    if anchor.network != "sepolia":
        # Exit 2, not 1: the receipt is intact, there is simply no ledger left
        # to check it against. That is a different outcome from tampering.
        print(
            "\nUNVERIFIABLE  this run was anchored on an in-process chain, which no\n"
            "longer exists. The receipt still hashes to the digest that was anchored,\n"
            "so it is internally consistent, but only a public-chain run can be\n"
            "re-checked against a live ledger. Use --chain sepolia for that."
        )
        raise typer.Exit(code=2)

    settings = load_settings()
    client = RegistryClient.connect_readonly(settings.sepolia_rpc_url, anchor.contract_address)
    record = client.verify(recomputed)
    if record is None:
        print("\nNOT ANCHORED  this digest is absent from the registry.")
        raise typer.Exit(code=1)

    print(f"\non {describe_chain(client.chain_id)} at {anchor.contract_address}")
    print(f"  submitter    {record.submitter}")
    print(f"  block        {record.block_number}")
    print(f"  timestamp    {record.timestamp}")
    if anchor.explorer_url:
        print(f"  transaction  {anchor.explorer_url}")

    if live:
        _verify_live_image(receipt)

    print("\nVERIFIED  the receipt matches the record anchored on chain.")


@app.command("tamper-demo")
def tamper_demo(
    run: Annotated[Path, typer.Option("--run", help="Evidence folder to tamper with.")],
) -> None:
    """Edit one character of a stored receipt and watch verification fail.

    Nothing on disk is modified; the edit is made to an in-memory copy.
    """
    receipt = evidence.read_receipt(run)
    anchor = evidence.read_anchor(run)
    original_digest = receipt_digest(receipt)

    title = receipt.match.page_title or receipt.match.post_url
    edited_title = _flip_last_character(title)
    tampered = replace(receipt, match=replace(receipt.match, page_title=edited_title))
    tampered_digest = receipt_digest(tampered)

    print("Editing a single character of the anchored receipt.\n")
    print(f"  before  page_title={title!r}")
    print(f"          digest={original_digest}")
    print(f"  after   page_title={edited_title!r}")
    print(f"          digest={tampered_digest}\n")

    if anchor.network != "sepolia":
        print("This run was anchored on an in-process chain, so nothing is left to query.")
        print("The digests above already show the receipt cannot be edited undetected.")
        raise typer.Exit(code=0)

    settings = load_settings()
    client = RegistryClient.connect_readonly(settings.sepolia_rpc_url, anchor.contract_address)
    print(f"Looking both up on {describe_chain(client.chain_id)}:")
    print(f"  original  {'found' if client.verify(original_digest) else 'not found'}")
    print(f"  tampered  {'found' if client.verify(tampered_digest) else 'not found'}")
    print("\nTAMPERED  the edited receipt has no record on chain.")


def _verify_live_image(receipt: Receipt) -> None:
    """Re-download the matched image and compare it to the anchored hash."""
    print(f"\nre-fetching {receipt.match.image_url}")
    try:
        current = sha256_hex(download_image(receipt.match.image_url))
    except DownloadError as exc:
        print(f"  unreachable: {exc.message}")
        print("  the post may have been deleted; the on-chain record is unaffected")
        return

    if current == receipt.match.image_sha256:
        print("  unchanged since it was anchored")
    else:
        print(f"  CHANGED  anchored {receipt.match.image_sha256[:16]}..., now {current[:16]}...")
        print("  the post itself was edited after anchoring")


def _flip_last_character(text: str) -> str:
    """Change exactly one character, so the edit is as small as it can be."""
    if not text:
        return "."
    return text[:-1] + ("a" if text[-1] != "a" else "b")


def _build_provider(
    settings: Settings, image: Path, replay: Path | None
) -> tuple[SearchProvider, str]:
    """Pick a search backend and the URL it will be given."""
    if replay is not None:
        provider = ReplayProvider(replay)
        return provider, provider.search("").query_image_url

    if not settings.serpapi_key:
        raise SearchNotConfigured(
            "SERPAPI_KEY is not set. Get a free key at https://serpapi.com, "
            "or reuse a recorded search with --replay <run directory>."
        )
    url, how = public_url_for(image, settings.github_raw_base)
    print(f"      probe image {how}")
    return SerpApiLensProvider(settings.serpapi_key), url


def _connect_for_anchoring(chain: str, settings: Settings) -> RegistryClient:
    """Open a signing connection to the requested chain."""
    if chain == "local":
        return RegistryClient.local_chain()
    if chain != "sepolia":
        raise PipelineError(f"Unknown chain {chain!r}; expected 'sepolia' or 'local'")

    if not settings.private_key:
        raise ChainNotConfigured(
            "PRIVATE_KEY is not set. Fund a throwaway key at "
            "https://sepolia-faucet.pk910.de, or anchor locally with --chain local."
        )
    address = settings.registry_address or load_deployment()["address"]
    return RegistryClient.connect_signing(settings.sepolia_rpc_url, settings.private_key, address)


def _print_candidates(scored: list[ScoredCandidate]) -> None:
    """Print every candidate with its score, rejects included."""
    print()
    for item in scored:
        score = f"{item.similarity:.4f}" if item.similarity is not None else "     -"
        detail = f"  {item.detail}" if item.detail else ""
        print(f"      {score}  {item.status:<16}{item.candidate.describe()}{detail}")


def main() -> None:
    try:
        app()
    except PipelineError as error:
        print(f"\nerror [{error.code}]: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
