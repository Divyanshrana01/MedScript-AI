"""Download NICE guideline pages and save each section as clean, tagged JSON.

Output feeds two things: the RAG knowledge base (backend/app/rag/ingest.py,
Week 4) and the synthetic QA seed corpus (synthetic_generation.py).
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("data/raw/nice")


def fetch_guideline_ids() -> list[str]:
    """Return NICE guideline IDs to download, e.g. ["NG138", ...].

    TODO: scrape nice.org.uk's guidance index, or use a curated list.
    """
    raise NotImplementedError


def fetch_and_clean(guideline_id: str) -> list[dict]:
    """Fetch one guideline's page(s) and return cleaned section dicts.

    Each dict: {"guideline_id", "title", "section", "text"}.
    TODO: fetch HTML, strip boilerplate, split at section headings.
    """
    raise NotImplementedError


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for guideline_id in fetch_guideline_ids():
        for section in fetch_and_clean(guideline_id):
            out_path = OUTPUT_DIR / f"{guideline_id}_{section['section']}.json"
            out_path.write_text(json.dumps(section, indent=2))


if __name__ == "__main__":
    main()
