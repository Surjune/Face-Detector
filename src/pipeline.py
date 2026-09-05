"""Command line entry point.

    run           face -> web search -> anchor on chain
    verify        recompute a stored receipt's digest and look it up on chain
    tamper-demo   edit a stored receipt and watch verification fail

Only `run` needs credentials. `verify` and `tamper-demo` work from a committed
evidence folder with no key, no wallet and no gas.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from src import evidence, ui
from src.chain import (
    MatchRecord,
    Receipt,
    RegistryClient,
    SearchRecord,
    canonical_bytes,
    describe_chain,
    load_deployment,
    receipt_digest,
    receipt_to_dict,
    sha256_hex,
)
from src.config import (
    DEFAULT_PROBE_IMAGE,
    EMBEDDING_DIM,
    FACE_MARGINAL_CEILING,
    FACE_MATCH_THRESHOLD,
    MAX_TOTAL_CANDIDATES,
    SEPOLIA_CHAIN_ID,
    SEPOLIA_TX_EXPLORER,
    Settings,
    load_settings,
)
from src.errors import (
    ChainNotConfigured,
    DownloadError,
    EvidenceError,
    NoMatchFound,
    NoSocialMatchFound,
    NotAnchored,
    PipelineError,
    SearchError,
    SearchNotConfigured,
)
from src.face import encode_primary_face
from src.report.html import write_report
from src.search import (
    CANDIDATE_IMAGE_DIRNAME,
    Candidate,
    Identity,
    IdentityVerdict,
    Platform,
    ReplayProvider,
    ScoredCandidate,
    SearchProvider,
    SerpApiLensProvider,
    TARGET_PLATFORMS,
    YandexReverseImageProvider,
    confirm_identity,
    derive_identity,
    download_image,
    missing_platforms,
    public_url_for,
    record_response,
    score_candidates,
    search_platform,
    search_videos,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Face identification, web search and blockchain verification.",
)


@app.command()
def run(
    image: Annotated[
        Path | None,
        typer.Option("--image", help="Photograph to search for. Defaults to inputs/probe.jpg."),
    ] = None,
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
    allow_non_social: Annotated[
        bool,
        typer.Option(
            "--allow-non-social",
            help="Anchor the best match even if it is not a social media post.",
        ),
    ] = False,
) -> None:
    """Search social media for a face and anchor the post that is found."""
    settings = load_settings()
    cutoff = threshold if threshold is not None else FACE_MATCH_THRESHOLD
    run_dir = evidence.create_run_dir(out)
    print(f"Evidence: {run_dir}")

    # -- 1. face ---------------------------------------------------------
    ui.section("[1/3] FACE DETECTION & ENCODING")
    image = image or DEFAULT_PROBE_IMAGE
    if not image.exists():
        raise PipelineError(
            f"No image at {image}. Put a photograph there, or pass --image.",
        )
    ui.ok(f"Input: {image}")
    encoding = encode_primary_face(image)
    input_digest = sha256_hex(image.read_bytes())
    evidence.copy_input_image(run_dir, image)
    left, top, right, bottom = (int(value) for value in encoding.box)
    ui.ok(
        f"Face detected: bounding box "
        f"[x: {left}, y: {top}, w: {right - left}, h: {bottom - top}]"
    )
    ui.ok(f"Detection confidence: {encoding.confidence:.4f}")
    ui.ok(f"Generated {EMBEDDING_DIM}-d embedding vector")
    ui.ok(f"Embedding SHA-256: {ui.shorten(encoding.digest)}")
    ui.ok(f"Image SHA-256: {ui.shorten(input_digest)}")

    # -- 2. search -------------------------------------------------------
    ui.section("[2/3] WEB / SOCIAL MEDIA SEARCH")
    provider, query_url = _build_provider(settings, image, replay)
    ui.ok(f"Provider: {provider.name}")
    ui.ok(f"Query image: {query_url}")

    ui.ok("Executing reverse image / visual search...")
    response = provider.search(query_url)
    record_response(run_dir, response)
    ui.ok(f"{len(response.candidates)} lead(s) from the visual search")

    candidates = list(response.candidates)
    candidates += _yandex_pass(settings, query_url, candidates, replay is not None)

    identity = derive_identity(response.identity, candidates)
    if identity is not None:
        # A query string, not a conclusion. Nothing has been face-checked at
        # this point, so this name is only what the titles of visually similar
        # pages happen to say. What the run actually concludes about identity is
        # printed after the face check, from the results that survive it.
        ui.ok(f"Platform search term: \"{identity.name}\" (guessed from result titles)")
        ui.detail("Note", "a query only - identity is decided after face verification")
    else:
        ui.ok("No search term could be derived; skipping targeted platform search")

    candidates += _expand_platforms(settings, identity, candidates, replay is not None)
    if len(candidates) > MAX_TOTAL_CANDIDATES:
        ui.ok(f"capping {len(candidates)} leads at {MAX_TOTAL_CANDIDATES}")
        candidates = candidates[:MAX_TOTAL_CANDIDATES]
    ui.ok(f"{len(candidates)} lead(s) total; re-running face recognition on each")

    scored = score_candidates(
        encoding.embedding,
        candidates,
        run_dir / CANDIDATE_IMAGE_DIRNAME,
        threshold=cutoff,
    )
    evidence.write_candidates(run_dir, scored)
    _print_candidates(scored)
    _print_platform_coverage(scored)

    matches = [item for item in scored if item.is_match]
    _print_identity(confirm_identity(identity, [item.candidate for item in matches]))

    if not matches:
        raise NoMatchFound(
            "No candidate cleared the similarity threshold",
            candidates=len(scored),
            threshold=cutoff,
        )

    social = [item for item in matches if item.is_social_match]
    if not social and not allow_non_social:
        raise NoSocialMatchFound(
            "Faces were matched, but none on a social media platform. "
            "Re-run with --allow-non-social to anchor a non-social page anyway.",
            matches=len(matches),
            candidates=len(scored),
        )

    best = (social or matches)[0]
    similarity = best.similarity if best.similarity is not None else 0.0

    print()
    ui.ok("Social media post found!" if best.is_social_match else "Match found!")
    ui.detail("Platform", best.platform.label)
    ui.detail("Source", best.candidate.source or "unknown")
    ui.detail("Post URL", best.candidate.page_url)
    ui.detail("Title", best.candidate.title or "(none)")
    ui.detail("Media URL", best.image_url_used or best.candidate.image_url)
    ui.detail("Found via", best.candidate.origin.value)
    # Cosine similarity, shown as a percentage for readability. It measures the
    # angle between two embeddings; it is not a calibrated probability.
    ui.detail("Similarity", f"{similarity:.4f} cosine ({similarity * 100:.1f}%)")
    ui.detail(
        "Cleared threshold",
        f"{len(matches)} of {len(scored)} candidates at cut-off {cutoff} "
        f"({len(social)} on social platforms)",
    )
    if similarity < FACE_MARGINAL_CEILING:
        # Above the cut-off but inside the band where the impostor distribution
        # still has weight. Saying so is the honest report; silently anchoring
        # it as a confident match is how the wrong person gets recorded.
        print()
        ui.ok("MARGINAL MATCH - check this one yourself before trusting it")
        ui.detail(
            "Why",
            f"{similarity:.4f} is above the {cutoff} cut-off but below "
            f"{FACE_MARGINAL_CEILING}, and different people have been measured "
            f"as high as 0.6602",
        )

    # -- 3. chain --------------------------------------------------------
    ui.section("[3/3] BLOCKCHAIN ATTESTATION & RECORDING")
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

    canonical = canonical_bytes(receipt)
    ui.ok("Record payload:")
    ui.block(json.dumps(receipt_to_dict(receipt), indent=2, sort_keys=True))
    # The payload above is formatted for reading. The digest is taken over the
    # canonical encoding -- sorted keys, no whitespace, ASCII-escaped -- because
    # only that form is reproducible byte for byte on another machine.
    ui.ok(f"Canonical encoding: {ui.count(len(canonical))} bytes, sorted keys, compact")
    ui.ok(f"Payload SHA-256 digest: {digest}")

    client = _connect_for_anchoring(chain, settings)
    ui.ok(f"Submitting transaction to {_chain_label(client)}...")
    result = client.anchor(digest, run_dir.name)
    evidence.write_anchor(run_dir, digest, result)
    tx_hash = evidence.prefixed(result.tx_hash)

    ui.ok(f"Transaction mined in block #{ui.count(result.block_number)}")
    ui.detail("Tx hash", tx_hash)
    ui.detail("Contract address", client.address)
    ui.detail("Gas used", ui.count(result.gas_used))
    if result.chain_id == SEPOLIA_CHAIN_ID:
        ui.detail("Explorer", f"{SEPOLIA_TX_EXPLORER}{tx_hash}")
    else:
        ui.detail("Note", "in-process chain, gone when this process exits")

    _verification_routine(run_dir, client, digest)

    report = write_report(run_dir)
    print()
    ui.ok(f"Report: {report}")
    ui.ok(f"Re-verify any time: python -m src.pipeline verify --run {run_dir}")


def _verification_routine(run_dir: Path, client: RegistryClient, expected: str) -> None:
    """Prove the round trip immediately, from what was written to disk.

    The digest is recomputed from the stored receipt rather than reused from
    memory, so this exercises the same path an independent verifier takes. A
    receipt that failed to serialise back to the anchored value would be caught
    here, at the point it was created, rather than by whoever checks it later.
    """
    ui.section("[VERIFICATION ROUTINE]")

    ui.ok("Re-reading receipt from disk")
    stored = evidence.read_receipt(run_dir)

    ui.ok("Recomputing digest from its canonical form")
    recomputed = receipt_digest(stored)
    if recomputed != expected:
        raise EvidenceError(
            "The stored receipt does not reproduce the anchored digest",
            anchored=expected,
            recomputed=recomputed,
        )

    ui.ok(f"Querying registry on {_chain_label(client)}")
    record = client.verify(recomputed)
    if record is None:
        raise NotAnchored("The digest was not found in the registry", digest=recomputed)

    ui.ok("On-chain digest matches computed payload digest")
    ui.detail("Submitter", record.submitter)
    ui.detail("Block", ui.count(record.block_number))
    ui.ok("Tamper-evidence verified: MATCH")


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


def _chain_label(client: RegistryClient) -> str:
    """Name the chain for output.

    The in-process chain has a randomly generated id, so printing the number
    tells the reader nothing; say what it actually is instead.
    """
    if client.is_local:
        return "in-process test chain"
    return describe_chain(client.chain_id)


def _yandex_pass(
    settings: Settings,
    query_url: str,
    harvested: list[Candidate],
    is_replay: bool,
) -> list[Candidate]:
    """Search Yandex as well, and keep whatever Lens did not already find.

    This is the step that decides whether an ordinary person can be found at
    all. Google restricts public face matching for private individuals; Yandex
    does not, and matches faces across pose and lighting that Lens misses. For a
    well-known subject it adds little, for everyone else it is the difference
    between a result and an empty run.

    A failure here is reported and swallowed: it is a second opinion on top of a
    harvest that already stands on its own.
    """
    if is_replay or not settings.serpapi_key:
        return []

    try:
        response = YandexReverseImageProvider(settings.serpapi_key).search(query_url)
    except SearchError as exc:
        ui.ok(f"Yandex search unavailable: {exc.message}")
        return []

    seen = {item.page_url for item in harvested}
    fresh = [item for item in response.candidates if item.page_url not in seen]
    ui.ok(
        f"Yandex reverse image: {len(response.candidates)} result(s), "
        f"{len(fresh)} not already seen"
    )
    return fresh


def _expand_platforms(
    settings: Settings,
    identity: Identity | None,
    harvested: list[Candidate],
    is_replay: bool,
) -> list[Candidate]:
    """Search the target platforms the visual harvest did not reach.

    Only the missing platforms are queried: re-searching one already covered
    spends a search from a small monthly budget to rediscover a post we hold.

    Every failure here is reported and swallowed. Expansion adds coverage on top
    of a harvest that already stands on its own, so an exhausted quota or a
    provider outage must not fail a run that has candidates in hand.
    """
    if identity is None:
        return []
    if is_replay:
        # A replay reproduces one recorded search. Issuing live queries beside
        # it would make the run neither a faithful replay nor a live search.
        ui.ok("Replay: skipping live platform expansion")
        return []

    found: list[Candidate] = []
    next_position = len(harvested) + 1

    youtube_covered = any(item.platform is Platform.YOUTUBE for item in harvested)
    if settings.youtube_api_key and not youtube_covered:
        try:
            videos = search_videos(
                settings.youtube_api_key, identity.name, start_position=next_position
            )
            found.extend(videos)
            next_position += len(videos)
            ui.ok(f"YouTube Data API: {len(videos)} lead(s)")
        except SearchError as exc:
            ui.ok(f"YouTube Data API unavailable: {exc.message}")

    if not settings.serpapi_key:
        return found

    for platform in missing_platforms(harvested + found):
        try:
            results = search_platform(
                settings.serpapi_key,
                platform,
                identity.name,
                start_position=next_position,
            )
        except SearchError as exc:
            ui.ok(f"site:{platform.value} search failed: {exc.message}")
            continue

        found.extend(results)
        next_position += len(results)
        ui.ok(f"site:{platform.value} search: {len(results)} lead(s)")

    return found


def _print_platform_coverage(scored: list[ScoredCandidate]) -> None:
    """Summarise what was found per target platform.

    Printed whether or not a platform yielded anything: "searched and found
    nothing" is a different, and more honest, statement than silence.
    """
    print()
    ui.plain("platform          leads  verified")
    ui.plain("-" * 34)
    for platform in TARGET_PLATFORMS:
        on_platform = [item for item in scored if item.platform is platform]
        verified = sum(1 for item in on_platform if item.is_match)
        ui.plain(f"{platform.label:<18}{len(on_platform):<7}{verified}")

    other = [item for item in scored if item.platform is Platform.REDDIT]
    if other:
        verified = sum(1 for item in other if item.is_match)
        ui.plain(f"{Platform.REDDIT.label:<18}{len(other):<7}{verified}")


def _print_identity(verdict: IdentityVerdict) -> None:
    """State who the subject is, only once the face check has had its say.

    Before verification the pipeline holds nothing but a name scraped from pages
    a search engine thought looked similar. Printing that as an identification
    is a claim the evidence does not support: a run whose every candidate is
    later rejected would still have announced a name, and the name would belong
    to whichever stranger the search happened to surface.
    """
    print()
    ui.plain("identity")
    ui.plain("-" * 34)

    if verdict.is_confirmed:
        ui.plain(f"{'name':<15}{verdict.name}")
        ui.plain(
            f"{'basis':<15}named on {verdict.supporting} of {verdict.verified} "
            f"face-verified result(s)"
        )
        ui.plain(f"{'derived from':<15}{verdict.origin}")
        return

    ui.plain(f"{'name':<15}NOT IDENTIFIED - not sure who this is")

    if verdict.verified:
        ui.plain(
            f"{'basis':<15}{verdict.verified} result(s) matched the face, but none "
            f"of them names a person"
        )
    else:
        ui.plain(f"{'basis':<15}no result matched the face, so nothing names the subject")

    if verdict.search_term:
        ui.plain(f"{'search term':<15}\"{verdict.search_term}\" - unconfirmed")
        if verdict.verified:
            ui.plain(f"{'':<15}A guess from page titles. The results that did match")
            ui.plain(f"{'':<15}the face do not carry this name, so nothing supports it.")
        else:
            ui.plain(f"{'':<15}A guess taken from the titles of pages that merely")
            ui.plain(f"{'':<15}looked similar. Every one was rejected by the face")
            ui.plain(f"{'':<15}check, so it is not the subject's name.")


def _print_candidates(scored: list[ScoredCandidate]) -> None:
    """Print every candidate with its score.

    The rejects are printed alongside the matches on purpose: a result set with
    nothing rejected in it would be indistinguishable from a hardcoded answer.
    """
    print()
    ui.plain(f"{'score':<8}{'result':<18}{'platform':<14}source")
    ui.plain(f"{'-' * 8}{'-' * 18}{'-' * 14}{'-' * 20}")
    for item in scored:
        score = f"{item.similarity:.4f}" if item.similarity is not None else "-"
        detail = f"  ({item.detail})" if item.detail else ""
        platform = item.platform.label if item.candidate.is_social else ""
        ui.plain(
            f"{score:<8}{item.status:<18}{platform:<14}"
            f"{item.candidate.describe()}{detail}"
        )


def main() -> None:
    try:
        app()
    except PipelineError as error:
        print(f"\nerror [{error.code}]: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
