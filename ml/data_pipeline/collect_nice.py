"""Download NICE guideline pages and save as clean, metadata-tagged JSON.

Output: one JSON file per guideline in data/raw/nice/, shape:
    {"guideline_id": "NG138", "title": ..., "section": ..., "text": ...}

This corpus doubles as the RAG knowledge base (Chapter 7) and the seed
corpus for synthetic QA generation (synthetic_generation.py), so it is
worth getting the cleaning right here rather than downstream.
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("data/raw/nice")


def fetch_guideline_ids() -> list[str]:
    # TODO: enumerate target NICE guideline IDs (e.g. from nice.org.uk sitemap
    # or a curated list) rather than crawling the whole site.
    raise NotImplementedError


def fetch_and_clean(guideline_id: str) -> list[dict]:
    # TODO: fetch the guideline page(s), strip HTML/nav/boilerplate, split
    # into sections, return one dict per section with guideline_id/title/
    # section/text metadata.
    raise NotImplementedError


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for guideline_id in fetch_guideline_ids():
        for section in fetch_and_clean(guideline_id):
            out_path = OUTPUT_DIR / f"{guideline_id}_{section['section']}.json"
            out_path.write_text(json.dumps(section, indent=2))


if __name__ == "__main__":
    main()
