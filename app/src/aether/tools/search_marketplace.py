"""Search-marketplace tool — search the skills.sh marketplace for installable skills.

skills.sh has no public JSON search API — the leaderboard is server-rendered HTML.
We fetch the homepage, parse skill entries from the HTML, and filter by keyword.

Each result includes the install shorthand so the LLM can immediately call
install_skill(source="owner/repo@skill-name") to install a match.

Parsing strategy:
  The skills.sh leaderboard renders skill cards. We look for patterns like:
    - Skill name links / headings
    - owner/repo references
  We use a lightweight regex approach (no heavy HTML parser dependency).
"""

from __future__ import annotations

import logging
import re

import httpx

from aether.tools.base import AetherTool, ToolParam, ToolResult

logger = logging.getLogger(__name__)

SKILLS_SH_URL = "https://skills.sh"

# Matches skill entries in the skills.sh leaderboard HTML.
# The page renders cards with data attributes or anchor hrefs like /skills/owner/repo
_SKILL_HREF_RE = re.compile(
    r'href=["\']/?skills/([^/"\']+)/([^/"\']+)["\']', re.IGNORECASE
)

# Skill name from card heading — <h2 ...>skill-name</h2> or similar
_HEADING_RE = re.compile(r"<h[123][^>]*>\s*([^<]{2,80})\s*</h[123]>", re.IGNORECASE)

# Install count — e.g. "1,234 installs" or "1234 installs"
_INSTALL_COUNT_RE = re.compile(r"([\d,]+)\s+install", re.IGNORECASE)


def _parse_leaderboard(html: str) -> list[dict]:
    """Extract skill entries from skills.sh HTML.

    Returns a list of dicts with keys: owner, repo, name, installs.
    Deduplicates by owner/repo.
    """
    seen: set[str] = set()
    skills: list[dict] = []

    # Find all owner/repo pairs from href="/skills/owner/repo" links
    for match in _SKILL_HREF_RE.finditer(html):
        owner = match.group(1)
        repo = match.group(2).rstrip("/")
        key = f"{owner}/{repo}"
        if key in seen:
            continue
        seen.add(key)

        # Derive a human-readable name from the repo slug
        name = repo.replace("-", " ").replace("_", " ").title()

        # Try to find an install count near this match (within 500 chars after)
        snippet = html[match.start() : match.start() + 500]
        count_match = _INSTALL_COUNT_RE.search(snippet)
        installs = count_match.group(1).replace(",", "") if count_match else "?"

        skills.append(
            {
                "owner": owner,
                "repo": repo,
                "name": name,
                "installs": installs,
                "source": f"{owner}/{repo}",
            }
        )

    return skills


def _filter_skills(skills: list[dict], query: str) -> list[dict]:
    """Filter skills by keyword overlap against name/owner/repo."""
    if not query.strip():
        return skills

    query_words = set(query.lower().split())
    results = []
    for skill in skills:
        haystack = f"{skill['name']} {skill['owner']} {skill['repo']}".lower()
        if any(word in haystack for word in query_words):
            results.append(skill)
    return results


class SearchMarketplaceTool(AetherTool):
    """Search the skills.sh marketplace for installable skills."""

    name = "search_marketplace"
    description = (
        "Search the skills.sh marketplace for installable skills. "
        "Returns matching skills with their install shorthand. "
        "Use install_skill(source=...) to install a result. "
        "Leave query empty to browse the top skills."
    )
    status_text = "Searching marketplace..."
    parameters = [
        ToolParam(
            name="query",
            type="string",
            description=(
                "Keyword(s) to search for "
                "(e.g. 'nextjs', 'coding', 'gmail', 'calendar'). "
                "Leave empty to list top skills from the leaderboard."
            ),
            required=False,
            default="",
        ),
    ]

    async def execute(self, query: str = "", **_) -> ToolResult:
        query = (query or "").strip()

        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(SKILLS_SH_URL)
        except httpx.RequestError as e:
            logger.error("Failed to fetch skills.sh: %s", e)
            return ToolResult.fail(
                f"Could not reach skills.sh marketplace: {e}. "
                "Check your network connection and try again."
            )

        if resp.status_code != 200:
            return ToolResult.fail(
                f"skills.sh returned HTTP {resp.status_code}. "
                "The marketplace may be temporarily unavailable."
            )

        all_skills = _parse_leaderboard(resp.text)

        if not all_skills:
            return ToolResult.fail(
                "Could not parse any skills from skills.sh. "
                "The site layout may have changed. "
                "Try browsing https://skills.sh directly."
            )

        matched = _filter_skills(all_skills, query) if query else all_skills

        if not matched:
            return ToolResult.success(
                f"No skills found matching '{query}' on skills.sh.\n\n"
                f"Try a broader search term, or browse https://skills.sh directly.\n\n"
                f"**Top skills available:**\n"
                + "\n".join(
                    f"- `{s['source']}` — {s['name']} ({s['installs']} installs)"
                    for s in all_skills[:5]
                ),
                query=query,
                total_found=0,
            )

        lines = [
            f"Found **{len(matched)}** skill(s)"
            + (f" matching '{query}'" if query else " on skills.sh")
            + ":"
        ]
        for skill in matched[:10]:  # cap at 10 results
            installs_str = (
                f" ({skill['installs']} installs)" if skill["installs"] != "?" else ""
            )
            lines.append(
                f"- **{skill['name']}**{installs_str}\n"
                f'  Install: `install_skill(source="{skill["source"]}")`'
            )

        if len(matched) > 10:
            lines.append(
                f"\n_...and {len(matched) - 10} more. Refine your search for better results._"
            )

        lines.append(
            '\nUse `install_skill(source="owner/repo")` to install, '
            'or `install_skill(source="owner/repo@skill-name")` for a specific skill.'
        )

        logger.info(
            "Marketplace search: query=%r → %d/%d results",
            query,
            len(matched),
            len(all_skills),
        )
        return ToolResult.success(
            "\n".join(lines),
            query=query,
            total_found=len(matched),
            total_available=len(all_skills),
        )
