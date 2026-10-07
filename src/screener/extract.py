"""Deterministic extraction: name, email, GitHub username, matched skills.

None of this needs an LLM, so it stays in plain code where it can be tested.
"""

from __future__ import annotations

import re
from collections import Counter

from . import taxonomy
from .models import ParsedResume
from .sections import detect_heading

EMAIL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_GH_URL = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))(/[A-Za-z0-9._\-]+)?",
    re.I,
)
_GH_PAGES = re.compile(r"(?:https?://)?([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\.github\.io", re.I)
_GH_LABEL = re.compile(r"github\s*:\s*@?([A-Za-z0-9](?:[A-Za-z0-9-]{1,38}))(?![A-Za-z0-9.@/])", re.I)
_USERNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
_RESERVED = {
    "features", "topics", "orgs", "sponsors", "about", "pricing", "login", "join", "settings",
    "marketplace", "explore", "collections", "events", "apps", "http", "https", "www", "com",
    "linkedin", "leetcode", "portfolio", "profile", "link", "repo", "repository", "source",
}
_NOT_A_NAME = {
    "resume", "curriculum", "vitae", "cv", "engineer", "developer", "summary", "profile", "email",
    "phone", "mobile", "linkedin", "github", "portfolio", "software", "full", "stack", "fresher",
    "experience", "education", "skills", "objective", "contact", "address", "analyst", "intern",
}


# --------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------
def extract_email(parsed: ParsedResume) -> str | None:
    for link in parsed.links:
        if link.lower().startswith("mailto:"):
            match = EMAIL_RE.search(link[7:])
            if match:
                return match.group(0)
    match = EMAIL_RE.search(parsed.text)
    return match.group(0) if match else None


# --------------------------------------------------------------------------
# Name
# --------------------------------------------------------------------------
def _looks_like_name(value: str) -> bool:
    value = value.strip()
    if not 3 <= len(value) <= 40 or detect_heading(value):
        return False
    if re.search(r"[\d@|/\\:_#+]", value):
        return False
    tokens = value.replace(",", " ").split()
    if not 1 <= len(tokens) <= 5:
        return False
    if any(token.lower().strip(".") in _NOT_A_NAME for token in tokens):
        return False
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", token) for token in tokens):
        return False
    return any(len(token.strip(".")) >= 3 for token in tokens)


def _tidy_name(value: str) -> str:
    value = " ".join(value.split())
    if value.isupper() or value.islower():
        # initials such as "K R" or "KM" stay upper case
        value = " ".join(token.capitalize() if len(token.strip(".")) > 2 else token.upper()
                         for token in value.split())
    return value


def extract_name(parsed: ParsedResume, email: str | None = None) -> str | None:
    """Largest text on page one, else the first name-shaped line, else email."""
    lines = [line.strip() for line in parsed.text.split("\n") if line.strip()]
    if parsed.name_hint and _looks_like_name(parsed.name_hint):
        name = parsed.name_hint
        # A surname set on its own line in a slightly different size.
        if len(name.split()) == 1 and name in lines[:6]:
            following = lines[lines.index(name) + 1] if lines.index(name) + 1 < len(lines) else ""
            if re.fullmatch(r"[A-Za-z][A-Za-z.'\-]{1,20}", following) and _looks_like_name(f"{name} {following}"):
                name = f"{name} {following}"
        return _tidy_name(name)
    for line in lines[:8]:
        candidate = re.split(r"\s{2,}|\s[|•]\s", line.strip())[0]
        if _looks_like_name(candidate):
            return _tidy_name(candidate)
    if email:
        local = re.sub(r"[\d_.+-]+", " ", email.split("@")[0]).strip()
        if len(local) >= 3:
            return _tidy_name(local)
    return None


# --------------------------------------------------------------------------
# GitHub username
# --------------------------------------------------------------------------
def _valid_username(value: str) -> bool:
    return bool(_USERNAME.match(value)) and value.lower() not in _RESERVED


def github_candidates(parsed: ParsedResume) -> list[str]:
    """Possible GitHub usernames, most trustworthy first.

    Order: profile links, owners of linked repositories, then github.com URLs
    written in the text (re-joined when a username wraps onto the next line).
    Only if none of those exist does it fall back to `user.github.io` sites
    and a "GitHub: name" label. The enrichment step validates the candidates
    against the API; most resumes yield exactly one.
    """
    profile_links: list[str] = []
    repo_owners: Counter[str] = Counter()
    pages: list[str] = []
    for link in parsed.links:
        link = link.strip()
        match = _GH_URL.search(link)
        if match:
            owner, rest = match.group(1), match.group(2)
            if rest and rest.strip("/"):
                repo_owners[owner] += 1
            else:
                profile_links.append(owner)
            continue
        match = _GH_PAGES.search(link)
        if match:
            pages.append(match.group(1))

    text_users: list[str] = []
    lines = parsed.text.split("\n")
    for index, line in enumerate(lines):
        for match in _GH_URL.finditer(line):
            owner, rest = match.group(1), match.group(2)
            at_line_end = match.end() >= len(line.rstrip())
            if at_line_end and not rest and index + 1 < len(lines):
                tail = lines[index + 1].strip()
                # A username broken across two lines continues in lower case;
                # the joined form replaces the truncated one.
                if re.fullmatch(r"[a-z0-9-]{1,8}", tail) and not detect_heading(tail):
                    owner = owner + tail
            text_users.append(owner)
        for match in _GH_PAGES.finditer(line):
            pages.append(match.group(1))

    ordered = profile_links + [owner for owner, _ in repo_owners.most_common()] + text_users
    if not ordered:
        # Weaker hints, used only when the resume has no github.com URL at all.
        ordered = pages + [m.group(1) for m in _GH_LABEL.finditer(parsed.text)]
    seen: set[str] = set()
    result: list[str] = []
    for username in ordered:
        key = username.lower()
        if key in seen or not _valid_username(username):
            continue
        seen.add(key)
        result.append(username)
    return result


# --------------------------------------------------------------------------
# Skills
# --------------------------------------------------------------------------
def extract_skills(text: str) -> list[str]:
    cleaned = taxonomy.AI_TOOL_PHRASES.sub(" ", text)
    return [name for name, pattern in taxonomy.SKILLS.items() if pattern.search(cleaned)]
