"""Working out who the probe face belongs to.

A name is what makes targeted platform search possible: you cannot run
`site:instagram.com` against a face, only against a query. This module derives
that query from the reverse image search results.

Three routes, cheapest and most reliable first:

1. Google Lens has already resolved the face to a Knowledge Graph entity and
   returns it as a ready-made query. Free, deterministic, and disambiguated by
   Google. This succeeds for anyone with a public profile.
2. Failing that, the most frequent capitalised phrase across the result titles.
   Costs nothing and needs no key.
3. Failing that, an LLM reads the titles and names the subject — only when a
   free-tier key is configured.

The name is a *search term*, never an assertion of identity. Nothing is
anchored on it, and every candidate it turns up is still face-verified before
it can become a match.
"""

from __future__ import annotations

import re
from collections import Counter

from src import llm
from src.errors import PipelineError
from src.search.provider import Candidate, Identity

# A capitalised run of two to four words: the shape of a personal name in a
# page title. One word is too noisy (every sentence starts capitalised) and
# beyond four is a headline, not a name.
_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

# Words that match the pattern but are never the subject. Publication and
# platform names dominate result titles and would otherwise win on frequency.
_STOPWORDS = frozenset(
    {
        "business insider", "new york", "new york times", "the new york times",
        "times of india", "india today", "getty images", "yahoo finance",
        "seattle times", "the atlantic", "wikimedia commons", "google ceo",
        "chief executive", "united states", "artificial intelligence",
    }
)

_LLM_SYSTEM = (
    "You identify the person a set of web search result titles is about. "
    "Reply with only their full name, nothing else. "
    "If the titles do not clearly name one person, reply exactly: UNKNOWN"
)

# A name is a couple of words; anything longer is the model explaining itself.
_MAX_NAME_WORDS = 5


def derive_identity(
    lens_identity: Identity | None,
    candidates: list[Candidate],
    *,
    use_llm: bool = True,
) -> Identity | None:
    """Determine the subject's likely name, or None if it cannot be established.

    Args:
        lens_identity: what the Lens response already reported, if anything.
        candidates: harvested results, whose titles are the fallback evidence.
        use_llm: allow the optional LLM refinement step.
    """
    if lens_identity is not None:
        return lens_identity

    titles = [item.title for item in candidates if item.title]
    if not titles:
        return None

    frequent = _most_frequent_name(titles)
    if frequent is not None:
        return Identity(name=frequent, origin="title_frequency", confidence="medium")

    if use_llm and llm.available():
        refined = _ask_llm(titles)
        if refined is not None:
            return Identity(name=refined, origin="llm", confidence="low")

    return None


def _most_frequent_name(titles: list[str]) -> str | None:
    """The capitalised phrase appearing in the most distinct titles.

    Counted once per title rather than per occurrence, so a single title
    repeating a publication's name cannot outvote the subject.
    """
    counts: Counter[str] = Counter()
    for title in titles:
        seen = {
            match.group(1)
            for match in _NAME_PATTERN.finditer(title)
            if match.group(1).lower() not in _STOPWORDS
        }
        counts.update(seen)

    if not counts:
        return None

    name, appearances = counts.most_common(1)[0]
    # A name appearing in only one title is as likely to be a passing mention.
    return name if appearances >= 2 else None


def _ask_llm(titles: list[str]) -> str | None:
    """Ask the configured free-tier model to name the subject.

    Returns None on any failure: identity refinement is a convenience, and a
    provider being down must never take the pipeline with it.
    """
    joined = "\n".join(f"- {title}" for title in titles[:20])
    try:
        answer = llm.complete(f"Search result titles:\n{joined}", system=_LLM_SYSTEM)
    except PipelineError:
        return None

    answer = answer.strip().strip('"').strip()
    if not answer or answer.upper() == "UNKNOWN":
        return None
    if len(answer.split()) > _MAX_NAME_WORDS:
        return None
    return answer
