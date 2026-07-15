"""Deterministic site identity, source classification, and relevance scoring."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from schemas import SiteIdentity, SourceClassification, TopicName


TOPIC_TERMS: dict[TopicName, frozenset[str]] = {
    "submission_policy": frozenset(
        {
            "job",
            "jobs",
            "slurm",
            "sbatch",
            "submit",
            "submission",
            "partition",
            "queue",
            "allocation",
            "account",
            "walltime",
            "memory",
            "policy",
        }
    ),
    "networking_policy": frozenset(
        {
            "network",
            "networking",
            "firewall",
            "port",
            "ports",
            "tcp",
            "socket",
            "login",
            "compute",
            "worker",
            "architecture",
            "outbound",
            "egress",
            "faq",
            "faqs",
            "policy",
            "policies",
        }
    ),
}

ROOT_TERMS = frozenset({"user", "guide", "documentation", "docs", "overview"})
PATH_SCOPE_MARKERS = frozenset({"knowledge", "userguides", "clusters", "systems"})
GENERIC_PATH_TOKENS = frozenset(
    {
        "hpc",
        "docs",
        "documentation",
        "guide",
        "guides",
        "userguide",
        "userguides",
        "general",
        "shared",
        "policies",
        "policy",
    }
)


def derive_site_identity(
    *,
    display_name: str,
    organization_domains: list[str],
    aliases: list[str] | None = None,
    preferred_path_tokens: list[str] | None = None,
    excluded_site_tokens: list[str] | None = None,
) -> SiteIdentity:
    """Build a reproducible identity with generic defaults from the site name."""

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", display_name)
    short_alias = words[-1] if words else display_name.strip()
    resolved_aliases = _dedupe([display_name, short_alias, *(aliases or [])])
    resolved_tokens = _dedupe(
        [
            *(
                preferred_path_tokens
                or [re.sub(r"[^a-z0-9]+", "-", short_alias.lower()).strip("-")]
            )
        ]
    )
    resolved_domains = _dedupe(
        [
            (urlparse(x if "://" in x else f"https://{x}").hostname or "")
            .lower()
            .strip(".")
            for x in organization_domains
        ]
    )
    return SiteIdentity(
        display_name=display_name,
        aliases=resolved_aliases,
        organization_domains=resolved_domains,
        preferred_path_tokens=[x.lower() for x in resolved_tokens if x],
        excluded_site_tokens=[
            x.lower() for x in _dedupe(excluded_site_tokens or []) if x
        ],
    )


def classify_source(
    *,
    identity: SiteIdentity,
    url: str,
    title: str = "",
    heading: str = "",
    text: str = "",
    linked_from_canonical_root: bool = False,
) -> SourceClassification:
    """Classify and score one source without model judgment."""

    parsed = urlparse(url)
    path = parsed.path.lower()
    title_lower = title.lower()
    heading_lower = heading.lower()
    prominent_text = text[:3000].lower()
    combined_strong = f"{path} {title_lower} {heading_lower}"
    aliases = [x.lower() for x in identity.aliases]
    alias_tokens = {_normalize_token(x) for x in aliases}
    preferred_tokens = {_normalize_token(x) for x in identity.preferred_path_tokens}

    matched_aliases = [
        original
        for original, lowered in zip(identity.aliases, aliases)
        if _contains_phrase(title_lower, lowered)
        or _contains_phrase(heading_lower, lowered)
        or _contains_phrase(prominent_text, lowered)
    ]
    conflicting = set(
        token
        for token in identity.excluded_site_tokens
        if _contains_token(combined_strong, token)
    )

    scoped_token = _scoped_path_token(path)
    if (
        scoped_token
        and _normalize_token(scoped_token) not in preferred_tokens
        and _normalize_token(scoped_token) not in alias_tokens
        and scoped_token not in GENERIC_PATH_TOKENS
    ):
        conflicting.add(scoped_token)

    title_site = _title_site_token(title_lower)
    if (
        title_site
        and _normalize_token(title_site) not in preferred_tokens
        and _normalize_token(title_site) not in alias_tokens
        and title_site not in GENERIC_PATH_TOKENS
    ):
        conflicting.add(title_site)

    score = 0.0
    reasons: list[str] = []
    target_path = any(_path_contains_token(path, token) for token in preferred_tokens)
    title_alias = any(_contains_phrase(title_lower, alias) for alias in aliases)
    heading_alias = any(_contains_phrase(heading_lower, alias) for alias in aliases)
    prominent_text_alias = _prominently_identifies_target(prominent_text, aliases)
    if target_path:
        score += 10
        reasons.append("URL path contains a target-site token (+10)")
    if title_alias:
        score += 8
        reasons.append("page title contains a target-site alias (+8)")
    if heading_alias:
        score += 8
        reasons.append("main heading contains a target-site alias (+8)")
    if prominent_text_alias:
        score += 4
        reasons.append("page text prominently identifies the target site (+4)")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    official = any(
        hostname == domain or hostname.endswith("." + domain)
        for domain in identity.organization_domains
    )
    if official:
        score += 3
        reasons.append("URL belongs to an approved organization domain (+3)")
    if linked_from_canonical_root:
        score += 5
        reasons.append("page is linked from the canonical site root (+5)")
    if _topic_word_count(f"{path} {title_lower}"):
        score += 3
        reasons.append("topic words appear in the title or path (+3)")

    organization_general = official and _looks_organization_general(
        title_lower, heading_lower, prominent_text
    )
    target_evidence = (
        target_path or title_alias or heading_alias or prominent_text_alias
    )

    if conflicting:
        score -= 100
        reasons.append("URL or title identifies a sibling/other site (-100)")
        scope = "other_site"
    elif target_evidence:
        scope = "target_site"
    elif organization_general:
        score += 1
        reasons.append("page appears to be organization-wide documentation (+1)")
        scope = "organization_general"
    else:
        score -= 10
        reasons.append("no target-site evidence appears in URL, title, or text (-10)")
        scope = "unrelated"

    return SourceClassification(
        site_scope=scope,
        matched_aliases=matched_aliases,
        conflicting_site_tokens=sorted(conflicting),
        score=score,
        reasons=reasons,
    )


def generate_discovery_queries(identity: SiteIdentity) -> dict[str, list[str]]:
    """Generate bounded canonical-root and topic searches with a site alias."""

    alias = shortest_site_alias(identity)
    domains = " OR ".join(f"site:{domain}" for domain in identity.organization_domains)
    site_filter = f" {domains}" if domains else ""
    return {
        "canonical_root": [
            f"{alias} official user guide{site_filter}",
            f"{alias} documentation user guide{site_filter}",
        ],
        "submission_policy": [
            f"{alias} Slurm submitting jobs account partition{site_filter}",
            f"{alias} required sbatch options walltime queue{site_filter}",
        ],
        "networking_policy": [
            f"{alias} compute login node network firewall TCP ports{site_filter}",
            f"{alias} worker networking outbound compute nodes{site_filter}",
        ],
    }


def repair_query(query: str, identity: SiteIdentity) -> tuple[str, bool]:
    if any(_contains_phrase(query.lower(), alias.lower()) for alias in identity.aliases):
        return query, False
    return f"{shortest_site_alias(identity)} {query}".strip(), True


def shortest_site_alias(identity: SiteIdentity) -> str:
    return min(identity.aliases, key=lambda value: (len(value.split()), len(value)))


def topic_matches(value: str, topic: TopicName) -> int:
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    return len(tokens & TOPIC_TERMS[topic])


def root_matches(value: str) -> int:
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    return len(tokens & ROOT_TERMS)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _contains_phrase(haystack: str, needle: str) -> bool:
    return bool(needle) and needle.lower() in haystack


def _contains_token(haystack: str, token: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", haystack))


def _path_contains_token(path: str, token: str) -> bool:
    segments = [_normalize_token(x) for x in path.split("/") if x]
    return _normalize_token(token) in segments


def _scoped_path_token(path: str) -> str | None:
    segments = [x for x in path.split("/") if x]
    for index, segment in enumerate(segments[:-1]):
        if segment in PATH_SCOPE_MARKERS:
            return segments[index + 1]
    return None


def _title_site_token(title: str) -> str | None:
    match = re.search(r"\b([a-z0-9._-]+)\s+user\s+guide\b", title)
    return match.group(1) if match else None


def _topic_word_count(value: str) -> int:
    return sum(topic_matches(value, topic) for topic in TOPIC_TERMS)


def _looks_organization_general(title: str, heading: str, text: str) -> bool:
    combined = f"{title} {heading} {text[:1000]}"
    phrases = (
        "all clusters",
        "all systems",
        "organization-wide",
        "hpc policy",
        "slurm basics",
        "cluster job submission",
        "research computing",
    )
    return any(phrase in combined for phrase in phrases)


def _prominently_identifies_target(text: str, aliases: list[str]) -> bool:
    """Avoid treating a navigation mention or disclaimer as target identity."""

    for alias in aliases:
        escaped = re.escape(alias)
        patterns = (
            rf"\b{escaped}\s+(?:user\s+guide|documentation|cluster|system|supercomputer|compute)\b",
            rf"\b(?:documentation|user\s+guide)\s+(?:for|of)\s+{escaped}\b",
            rf"\b(?:jobs?|polic(?:y|ies)|networking)\s+on\s+{escaped}\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            return True
    return False
