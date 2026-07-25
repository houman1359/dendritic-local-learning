#!/usr/bin/env python3
"""Assemble exact review text and submission-ready responses into four packets."""

from __future__ import annotations

import re
from pathlib import Path


DRAFT_DIR = Path(__file__).resolve().parent.parent
REBUTTAL_DIR = DRAFT_DIR / "rebbutal"
REVIEWS_PATH = REBUTTAL_DIR / "rev1.txt"

MARKERS = {
    "QPha": "Official Review of Submission29735 by Reviewer QPha",
    "s2E9": "Official Review of Submission29735 by Reviewer s2E9",
    "QgDY": "Official Review of Submission29735 by Reviewer QgDY",
}

RESPONSE_PATHS = {
    "AC": REBUTTAL_DIR / "rebuttal_AC.txt",
    "QPha": REBUTTAL_DIR / "rebuttal_QPha.txt",
    "s2E9": REBUTTAL_DIR / "rebuttal_s2E9.txt",
    "QgDY": REBUTTAL_DIR / "rebuttal_QgDY.txt",
}


def _clean_review(text: str) -> str:
    """Remove the UI's trailing `Add:` separator without altering review text."""
    return re.sub(r"\nAdd:\s*$", "", text.rstrip())


def main() -> None:
    raw = REVIEWS_PATH.read_text(encoding="utf-8")
    qpha_start = raw.index(MARKERS["QPha"])
    s2e9_start = raw.index(MARKERS["s2E9"])
    qgdy_start = raw.index(MARKERS["QgDY"])

    reviews = {
        "AC": _clean_review(raw[:qpha_start]),
        "QPha": _clean_review(raw[qpha_start:s2e9_start]),
        "s2E9": _clean_review(raw[s2e9_start:qgdy_start]),
        "QgDY": _clean_review(raw[qgdy_start:]),
    }

    for reviewer, review in reviews.items():
        response = RESPONSE_PATHS[reviewer].read_text(encoding="utf-8").rstrip()
        if len(response) >= 10_000:
            raise ValueError(
                f"{reviewer} response is {len(response)} characters; limit is 10,000"
            )
        packet = (
            "================ ORIGINAL REVIEW ================\n\n"
            f"{review}\n\n"
            "================ AUTHOR RESPONSE ================\n\n"
            f"{response}\n"
        )
        output = REBUTTAL_DIR / f"final_{reviewer}_review_and_response.txt"
        output.write_text(packet, encoding="utf-8")
        print(
            f"{output.name}: response={len(response)} characters, "
            f"packet={len(packet.rstrip())} characters"
        )


if __name__ == "__main__":
    main()
