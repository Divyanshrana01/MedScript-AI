"""Download NICE guideline pages and save each section as clean, tagged JSON.

Feeds the RAG knowledge base and synthetic_generation.py's seed corpus.
Uses the firecrawl CLI, not NICE's Syndication API -- that needs a licence
application approved first.
"""

import json
import re
import subprocess
import time

from pathlib import Path

OUTPUT_DIR = Path("data/raw/nice")

# Curated set of NICE guidelines covering common CDS presentations.
# Broader coverage (PLAB-adjacent specialties, full guidance index) is
# deferred per the agreed 6-8 week scope.
GUIDELINE_IDS = [
    "ng136",  # Hypertension in adults
    "ng28",   # Type 2 diabetes in adults
    "ng245",  # Asthma: diagnosis, monitoring and chronic asthma management (replaces ng80)
    "ng115",  # Chronic obstructive pulmonary disease in over 16s
    "ng222",  # Depression in adults
    "cg113",  # Generalised anxiety disorder and panic disorder
    "ng109",  # Urinary tract infection (lower): antimicrobial prescribing
    "ng203",  # Chronic kidney disease
    "ng106",  # Chronic heart failure in adults
    "ng196",  # Atrial fibrillation: diagnosis and management
    "ng59",   # Low back pain and sciatica in over 16s
    "ng217",  # Epilepsies in children, young people and adults
    "ng246",  # Obesity: identification, assessment and management (replaces cg189)
    "ng253",  # Suspected sepsis in people aged 16 and over (replaces ng51)
    "ng128",  # Stroke and transient ischaemic attack
]

# Matches numbered section headings, e.g. "### 1.1 Measuring blood pressure".
# Split here, not at a fixed token size -- chunk_size re-splitting for
# embeddings happens later at RAG ingest time.
SECTION_HEADING_RE = re.compile(r"^### (\d+\.\d+) (.+)$", re.MULTILINE)

# Matches a guidance chapter page, capturing its slug, e.g.
# ".../guidance/ng28/chapter/Blood-glucose-management" -> "Blood-glucose-management".
# Excludes "/resources/.../chapter/..." pages (separate implementation-advice
# documents, not the guideline body).
CHAPTER_URL_RE = re.compile(
    r"^https://www\.nice\.org\.uk/guidance/[a-z0-9]+/chapter/([^/?#]+)$",
    re.IGNORECASE,
)

# Chapter slugs that are boilerplate/meta, not clinical content -- skip
# scraping them to save credits.
NON_CLINICAL_CHAPTER_SLUGS = {
    "using-this-guideline",
    "update-information",
    "terms-used-in-this-guideline",
    "finding-more-information-and-committee-details",
    "context",
}


def fetch_guideline_ids() -> list[str]:
    """Return NICE guideline IDs to download, e.g. ["ng136", ...]."""
    return GUIDELINE_IDS


def _run_firecrawl(*args: str, retries: int = 3) -> str:
    """Run a firecrawl CLI subcommand and return its stdout.

    Firing dozens of calls back to back trips the API's rate limit
    intermittently -- a fixed pacing delay plus exponential backoff on
    failure clears it without needing to slow down the happy path much.
    """
    time.sleep(1.5)

    for attempt in range(retries + 1):
        result = subprocess.run(
            ["npx", "-y", "firecrawl-cli@1.19.6", *args],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
        if attempt == retries:
            print(f"firecrawl failed, stderr:\n{result.stderr}")
            result.check_returncode()
        time.sleep(5 * (2 ** attempt))


def _list_chapter_urls(guideline_id: str) -> list[str]:
    """Discover a guideline's own clinical-content chapter pages.

    Guidelines are structured inconsistently -- some use one consolidated
    "recommendations" chapter (ng136), others split it across several
    topic chapters (ng28). Mapping the site handles both.
    """
    base_url = f"https://www.nice.org.uk/guidance/{guideline_id}"
    stdout = _run_firecrawl("map", base_url)

    # Keyed by lowercased slug: NICE's map sometimes lists the same chapter
    # twice under different casing (e.g. "chapter/recommendations" and
    # "chapter/Recommendations"), which would otherwise scrape and emit
    # duplicate sections.
    urls_by_slug: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        match = CHAPTER_URL_RE.match(line)
        if not match:
            continue
        slug = match.group(1).lower()
        if slug in NON_CLINICAL_CHAPTER_SLUGS:
            continue
        urls_by_slug.setdefault(slug, line)

    return list(urls_by_slug.values())


def _scrape(url: str) -> str:
    """Fetch a URL's main content as markdown via the firecrawl CLI."""
    return _run_firecrawl("scrape", url, "--only-main-content")


def fetch_and_clean(guideline_id: str) -> list[dict]:
    """Fetch every clinical-content chapter and split each into sections.

    Each dict: {"guideline_id", "title", "section", "text"}.
    Chapters with no numbered subheadings (e.g. a research-recommendations
    page) simply contribute no sections.
    """
    sections = []
    title = None

    for url in _list_chapter_urls(guideline_id):
        markdown = _scrape(url)

        if title is None:
            title = markdown.splitlines()[0].lstrip("#").strip()

        headings = list(SECTION_HEADING_RE.finditer(markdown))
        for i, heading in enumerate(headings):
            section_number, section_title = heading.groups()
            start = heading.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)

            sections.append({
                "guideline_id": guideline_id,
                "title": title,
                "section": f"{section_number} {section_title}",
                "text": markdown[start:end].strip(),
            })

    return sections


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for guideline_id in fetch_guideline_ids():
        # Checkpoint: a guideline already has output files from a prior run
        # (this script gets re-run often while iterating on the scraper --
        # no need to re-spend firecrawl credits re-fetching guidelines that
        # already succeeded).
        if list(OUTPUT_DIR.glob(f"{guideline_id}_*.json")):
            print(f"{guideline_id}: already downloaded, skipping")
            continue

        sections = fetch_and_clean(guideline_id)

        for section in sections:
            slug = re.sub(r"[^a-z0-9]+", "-", section["section"].lower()).strip("-")
            out_path = OUTPUT_DIR / f"{guideline_id}_{slug}.json"
            out_path.write_text(json.dumps(section, indent=2))

        print(f"{guideline_id}: wrote {len(sections)} sections")


if __name__ == "__main__":
    main()
