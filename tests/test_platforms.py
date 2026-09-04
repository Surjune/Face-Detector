"""Recognising which platform a URL belongs to.

The whole social-media requirement rests on this: if classification is wrong,
a news article gets anchored as a "social media post", or a genuine post is
passed over. The lookalike cases matter as much as the happy path.
"""

from __future__ import annotations

import pytest

from src.search.platforms import (
    SOCIAL_PLATFORMS,
    TARGET_PLATFORMS,
    Platform,
    classify,
    hostname_of,
    is_social,
    site_query_domain,
)


class TestClassify:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.facebook.com/someone/posts/123", Platform.FACEBOOK),
            ("https://fb.watch/abc123/", Platform.FACEBOOK),
            ("https://x.com/user/status/1969339515467653471", Platform.X),
            ("https://twitter.com/user/status/123", Platform.X),
            ("https://www.threads.net/@user/post/abc", Platform.THREADS),
            ("https://www.linkedin.com/posts/someone_activity-123", Platform.LINKEDIN),
            ("https://www.youtube.com/watch?v=pDdrA46xBRk", Platform.YOUTUBE),
            ("https://youtu.be/pDdrA46xBRk", Platform.YOUTUBE),
            ("https://www.instagram.com/reel/Dbsr3pBoOfj/", Platform.INSTAGRAM),
            ("https://www.tiktok.com/@user/video/123", Platform.TIKTOK),
            ("https://www.reddit.com/r/technology/comments/abc/", Platform.REDDIT),
            ("https://en.wikipedia.org/wiki/Sundar_Pichai", Platform.OTHER),
            ("https://www.reuters.com/technology/article", Platform.OTHER),
        ],
    )
    def test_classifies_known_urls(self, url: str, expected: Platform) -> None:
        assert classify(url) is expected

    def test_matches_subdomains(self) -> None:
        assert classify("https://m.facebook.com/story.php?id=1") is Platform.FACEBOOK

    @pytest.mark.parametrize(
        "url",
        [
            "https://notfacebook.com/someone",
            "https://facebook.com.phishing.example/login",
            "https://x.company/about",
            "https://myinstagram.net/p/abc",
            "https://example.com/redirect?target=https://instagram.com/p/abc",
        ],
    )
    def test_rejects_lookalike_domains(self, url: str) -> None:
        """A substring test would accept every one of these."""
        assert classify(url) is Platform.OTHER
        assert not is_social(url)

    def test_a_google_redirect_wrapper_is_not_social(self) -> None:
        """Lens returns these in place of permalinks for some results.

        The destination may genuinely be a Facebook post, but the wrapper is an
        opaque expiring token nobody can independently verify, so it must never
        be anchored as evidence of a social media post.
        """
        url = "https://www.google.com/goto?url=CAESyAEB6zswFTSRcOlG6eEbGPA9W"
        assert classify(url) is Platform.OTHER
        assert not is_social(url)

    def test_handles_unparseable_input(self) -> None:
        assert classify("") is Platform.OTHER
        assert classify("not a url at all") is Platform.OTHER


class TestHostname:
    def test_strips_www(self) -> None:
        assert hostname_of("https://www.example.com/path") == "example.com"

    def test_lowercases(self) -> None:
        assert hostname_of("https://EXAMPLE.com/path") == "example.com"

    def test_returns_none_for_no_host(self) -> None:
        assert hostname_of("") is None


class TestSocialSet:
    def test_every_target_platform_counts_as_social(self) -> None:
        for platform in TARGET_PLATFORMS:
            assert platform in SOCIAL_PLATFORMS

    def test_reddit_counts_as_social(self) -> None:
        """Not among the seven requested, but unambiguously a social post."""
        assert Platform.REDDIT in SOCIAL_PLATFORMS

    def test_other_is_not_social(self) -> None:
        assert Platform.OTHER not in SOCIAL_PLATFORMS

    def test_every_target_platform_has_a_site_query_domain(self) -> None:
        for platform in TARGET_PLATFORMS:
            domain = site_query_domain(platform)
            assert "." in domain
            assert classify(f"https://{domain}/x") is platform
