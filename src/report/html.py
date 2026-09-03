"""A self-contained HTML summary of one run.

Written into the evidence folder so it can be opened straight from a clone with
no server. Images are referenced by relative path, so the folder stays portable
as long as it is copied whole.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from src import evidence
from src.config import PIPELINE_VERSION
from src.errors import EvidenceError
from src.search.filter import CANDIDATE_IMAGE_DIRNAME

STATUS_LABELS = {
    "match": "match",
    "below_threshold": "below threshold",
    "no_face": "no face",
    "unreachable": "unreachable",
}


def write_report(run_dir: Path) -> Path:
    """Render report.html for a completed run.

    Raises:
        EvidenceError: the run directory is incomplete.
    """
    receipt = evidence.read_receipt(run_dir)
    candidates = evidence.read_candidates(run_dir)
    try:
        anchor: Any = evidence.read_anchor(run_dir)
    except EvidenceError:
        anchor = None

    input_image = evidence.find_input_image(run_dir)
    path = run_dir / evidence.REPORT_FILENAME
    path.write_text(
        _render(run_dir.name, receipt, candidates, anchor, input_image),
        encoding="utf-8",
    )
    return path


def _render(
    run_name: str,
    receipt: Any,
    candidates: list[dict[str, Any]],
    anchor: Any,
    input_image: Path | None,
) -> str:
    matched = [item for item in candidates if item.get("status") == "match"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face search run {html.escape(run_name)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 system-ui, -apple-system, Segoe UI, sans-serif;
         margin: 0 auto; padding: 2rem 1.25rem; max-width: 60rem; }}
  h1 {{ font-size: 1.35rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: .06em;
        opacity: .6; margin: 2.5rem 0 .75rem; }}
  .sub {{ opacity: .6; margin: 0 0 2rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #8883; }}
  th {{ font-weight: 600; opacity: .7; }}
  td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em;
          word-break: break-all; }}
  .tag {{ display: inline-block; padding: .1rem .45rem; border-radius: .25rem;
          font-size: .78rem; border: 1px solid #8886; }}
  .tag.match {{ border-color: #2e7d32; color: #2e7d32; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(9rem, 1fr));
           gap: .75rem; }}
  .grid figure {{ margin: 0; }}
  .grid img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: .3rem;
               background: #8882; }}
  .grid figcaption {{ font-size: .78rem; opacity: .7; margin-top: .25rem;
                      font-variant-numeric: tabular-nums; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: .35rem 1rem; margin: 0; }}
  dt {{ opacity: .6; }}
  dd {{ margin: 0; }}
  .wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>Face search run</h1>
<p class="sub">{html.escape(run_name)} &middot; pipeline {PIPELINE_VERSION}</p>

{_probe_section(input_image, receipt)}
{_match_section(receipt, matched)}
{_candidates_section(candidates)}
{_anchor_section(anchor)}

<h2>Receipt</h2>
<p class="sub">The exact structure that was hashed and anchored.</p>
<div class="wrap"><pre><code>{html.escape(_receipt_json(receipt))}</code></pre></div>
</body>
</html>
"""


def _probe_section(input_image: Path | None, receipt: Any) -> str:
    image_html = (
        f'<img src="{html.escape(input_image.name)}" alt="probe" '
        'style="max-width:12rem;border-radius:.3rem">'
        if input_image
        else "<p>not stored</p>"
    )
    return f"""<h2>Probe</h2>
{image_html}
<dl style="margin-top:1rem">
  <dt>image sha256</dt><dd><code>{html.escape(receipt.input_image_sha256)}</code></dd>
  <dt>embedding</dt><dd><code>{html.escape(receipt.embedding_sha256)}</code></dd>
</dl>"""


def _match_section(receipt: Any, matched: list[dict[str, Any]]) -> str:
    return f"""<h2>Match</h2>
<dl>
  <dt>post</dt><dd><a href="{html.escape(receipt.match.post_url)}">
      {html.escape(receipt.match.post_url)}</a></dd>
  <dt>title</dt><dd>{html.escape(receipt.match.page_title) or "&mdash;"}</dd>
  <dt>similarity</dt><dd>{receipt.match.similarity:.4f}</dd>
  <dt>image sha256</dt><dd><code>{html.escape(receipt.match.image_sha256)}</code></dd>
  <dt>provider</dt><dd>{html.escape(receipt.search.provider)}</dd>
  <dt>retrieved</dt><dd>{html.escape(receipt.search.retrieved_at)}</dd>
  <dt>candidates</dt><dd>{receipt.search.candidate_count}
      ({len(matched)} above threshold)</dd>
</dl>"""


def _candidates_section(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""

    rows = "\n".join(
        f"""<tr>
  <td class="num">{_score(item)}</td>
  <td><span class="tag {'match' if item.get('status') == 'match' else ''}">
      {html.escape(STATUS_LABELS.get(str(item.get("status")), "?"))}</span></td>
  <td>{html.escape(str(item.get("source") or ""))}</td>
  <td><a href="{html.escape(str(item.get("page_url") or ""))}">
      {html.escape(_truncate(str(item.get("title") or item.get("page_url") or "")))}</a></td>
</tr>"""
        for item in candidates
    )

    thumbs = "\n".join(
        f"""<figure>
  <img src="{html.escape(CANDIDATE_IMAGE_DIRNAME)}/{html.escape(str(item['image_file']))}"
       alt="candidate {item.get('position')}">
  <figcaption>{_score(item)} &middot;
      {html.escape(STATUS_LABELS.get(str(item.get("status")), "?"))}</figcaption>
</figure>"""
        for item in candidates
        if item.get("image_file")
    )

    return f"""<h2>Every candidate the search returned</h2>
<p class="sub">Each one was embedded and scored against the probe face. The rejects are
kept deliberately: they are the evidence that a real search ran.</p>
<div class="wrap"><table>
<thead><tr><th>score</th><th>result</th><th>source</th><th>page</th></tr></thead>
<tbody>
{rows}
</tbody>
</table></div>
<div class="grid" style="margin-top:1.25rem">
{thumbs}
</div>"""


def _anchor_section(anchor: Any) -> str:
    if anchor is None:
        return """<h2>Chain</h2>
<p>This run was not anchored.</p>"""

    explorer = (
        f'<dt>explorer</dt><dd><a href="{html.escape(anchor.explorer_url)}">'
        f"{html.escape(anchor.explorer_url)}</a></dd>"
        if anchor.explorer_url
        else "<dt>explorer</dt><dd>&mdash; in-process chain</dd>"
    )
    return f"""<h2>Chain</h2>
<dl>
  <dt>digest</dt><dd><code>{html.escape(anchor.digest)}</code></dd>
  <dt>network</dt><dd>{html.escape(anchor.network)} (chain {anchor.chain_id})</dd>
  <dt>contract</dt><dd><code>{html.escape(anchor.contract_address)}</code></dd>
  <dt>transaction</dt><dd><code>{html.escape(anchor.tx_hash)}</code></dd>
  <dt>block</dt><dd>{anchor.block_number}</dd>
  <dt>gas used</dt><dd>{anchor.gas_used}</dd>
  {explorer}
</dl>"""


def _score(item: dict[str, Any]) -> str:
    similarity = item.get("similarity")
    return f"{float(similarity):.4f}" if similarity is not None else "&mdash;"


def _truncate(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _receipt_json(receipt: Any) -> str:
    from src.chain.canonical import receipt_to_dict

    return json.dumps(receipt_to_dict(receipt), indent=2, sort_keys=True, ensure_ascii=False)
