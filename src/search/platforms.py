"""Recognising which platform a URL belongs to.

The brief asks specifically for a social media post, so the pipeline has to be
able to tell one apart from a news article or an encyclopaedia entry, both when
ranking matches and when deciding which platforms still need searching.

Matching is on the parsed hostname, never on a substring of the URL. A naive
`"facebook.com" in url` test would accept `notfacebook.com`, an attacker-chosen
path like `example.com/?next=facebook.com`, and `x.company` — all of which would
put a non-social page forward as the social match this project exists to find.
"""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse


class Platform(str, Enum):
    """A site a candidate can come from."""

    FACEBOOK = "facebook"
    X = "x"
    THREADS = "threads"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    REDDIT = "reddit"
    OTHER = "other"

    @property
    def label(self) -> str:
        """Human-facing name for output and reports."""
        return _LABELS.get(self, self.value.title())


_LABELS = {
    Platform.X: "X / Twitter",
    Platform.YOUTUBE: "YouTube",
    Platform.TIKTOK: "TikTok",
    Platform.LINKEDIN: "LinkedIn",
    Platform.OTHER: "Other",
}

# Registrable domains per platform, including the short-link and mobile hosts
# that appear in real search results.
PLATFORM_DOMAINS: dict[Platform, tuple[str, ...]] = {
    Platform.FACEBOOK: ("facebook.com", "fb.com", "fb.watch"),
    Platform.X: ("x.com", "twitter.com", "t.co"),
    Platform.THREADS: ("threads.net", "threads.com"),
    Platform.LINKEDIN: ("linkedin.com", "lnkd.in"),
    Platform.YOUTUBE: ("youtube.com", "youtu.be"),
    Platform.INSTAGRAM: ("instagram.com", "instagr.am"),
    Platform.TIKTOK: ("tiktok.com",),
    Platform.REDDIT: ("reddit.com", "redd.it"),
}

# The platforms this project sets out to cover, in the order they are reported.
TARGET_PLATFORMS: tuple[Platform, ...] = (
    Platform.FACEBOOK,
    Platform.X,
    Platform.THREADS,
    Platform.LINKEDIN,
    Platform.YOUTUBE,
    Platform.INSTAGRAM,
    Platform.TIKTOK,
)

# Everything treated as social media when ranking. Reddit is not one of the
# seven requested platforms but is unambiguously social, so a verified Reddit
# post still counts as satisfying the requirement.
SOCIAL_PLATFORMS: frozenset[Platform] = frozenset(
    set(TARGET_PLATFORMS) | {Platform.REDDIT}
)

# Domain -> platform, built once from the table above.
_DOMAIN_INDEX: dict[str, Platform] = {
    domain: platform
    for platform, domains in PLATFORM_DOMAINS.items()
    for domain in domains
}


def hostname_of(url: str) -> str | None:
    """Return the lowercased hostname of a URL, without any `www.` prefix."""
    try:
        parsed = urlparse(url if "//" in url else f"//{url}")
    except ValueError:
        return None

    host = (parsed.hostname or "").lower().strip(".")
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def classify(url: str) -> Platform:
    """Identify the platform a URL belongs to.

    A host matches a platform when it *is* the registrable domain or is a
    subdomain of it, so `m.facebook.com` matches while `notfacebook.com` and
    `facebook.com.evil.net` do not.
    """
    host = hostname_of(url)
    if host is None:
        return Platform.OTHER

    for domain, platform in _DOMAIN_INDEX.items():
        if host == domain or host.endswith(f".{domain}"):
            return platform
    return Platform.OTHER


def is_social(url: str) -> bool:
    """Whether a URL points at a social media platform.

    A redirect wrapper is deliberately not social, even when it leads to one.
    Google Lens sometimes returns `google.com/goto?url=<opaque token>` in place
    of a permalink; the destination really may be a Facebook post, but the
    wrapper is an opaque, expiring URL that nobody can independently check.
    Anchoring one as evidence would be worse than finding nothing, so it stays
    classified as OTHER and can never be selected as the social match. Do not
    "fix" this by reading the platform out of the result's display label.
    """
    return classify(url) in SOCIAL_PLATFORMS


def site_query_domain(platform: Platform) -> str:
    """The domain to use in a `site:` search operator for a platform."""
    return PLATFORM_DOMAINS[platform][0]
