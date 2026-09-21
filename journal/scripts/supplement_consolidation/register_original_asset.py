#!/usr/bin/env python3
"""Re-register an upstream supplementary render after it has been rebuilt.

``build.py`` refuses to paste from a render whose bytes differ from the
sha256 recorded in ``original_assets.json``, and it cuts each panel from the
bold panel-letter spans recorded there.  That registry was written once, when
the consolidated supplement was frozen, and nothing in the repository updated
it afterwards, so every upstream rebuild stranded the sheet that pastes from
it.  This script re-derives the four recorded facts -- sha256, page size and
the bold single-letter spans A-H with their boxes and origins -- from the
render as it now is, and rewrites only the entries named on the command line.

    python3 scripts/supplement_consolidation/register_original_asset.py S31 S18

Use ``--check`` to compare without writing.  A render whose letter set changed
(a panel added or removed) is reported and NOT written unless ``--force`` is
given, because ``specification.py`` maps sheets to letters by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
REGISTRY = HERE / "original_assets.json"
LETTER = re.compile(r"^[A-H]$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def letter_spans(page) -> list[dict]:
    """Bold, single-letter A-H spans, the way the frozen registry records them."""
    found = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                bold = "bold" in span["font"].lower() or span["flags"] & 16
                if LETTER.match(text) and bold and span["size"] >= 8.0:
                    x0, y0, x1, y1 = span["bbox"]
                    found.append({"letter": text, "bbox": [x0, y0, x1, y1],
                                  "origin": list(span["origin"])})
    # one span per letter: keep the top-left-most if a letter is drawn twice
    best = {}
    for s in sorted(found, key=lambda s: (s["bbox"][1], s["bbox"][0])):
        best.setdefault(s["letter"], s)
    return [best[k] for k in sorted(best)]


def describe(record: dict) -> dict:
    path = JOURNAL / record["path"]
    with fitz.open(path) as doc:
        page = doc[0]
        return {"path": record["path"], "sha256": sha256(path),
                "old_label": record.get("old_label", ""),
                "width": float(page.rect.width), "height": float(page.rect.height),
                "letters": letter_spans(page) if record.get("letters") else []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("keys", nargs="+", help="registry keys such as S31")
    parser.add_argument("--check", action="store_true", help="report, do not write")
    parser.add_argument("--force", action="store_true",
                        help="write even when the letter set changed")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    changed = []
    for key in args.keys:
        if key not in registry:
            raise SystemExit(f"{key} is not in {REGISTRY.name}")
        old, new = registry[key], describe(registry[key])
        # compared as sets: build.py sorts the spans itself, so order is
        # not a fact the registry carries
        old_letters = "".join(sorted(l["letter"] for l in old.get("letters", [])))
        new_letters = "".join(sorted(l["letter"] for l in new["letters"]))
        same_bytes = old["sha256"] == new["sha256"]
        print(f"{key}: {old['path']}")
        print(f"   bytes {'unchanged' if same_bytes else 'CHANGED'}; "
              f"letters {old_letters or '*'} -> {new_letters or '*'}; "
              f"size {old['width']:.1f}x{old['height']:.1f} -> {new['width']:.1f}x{new['height']:.1f}")
        if old_letters != new_letters and not args.force:
            print(f"   letter set changed; not written (use --force after updating specification.py)")
            continue
        if not same_bytes or old_letters != new_letters or args.force:
            registry[key] = new
            changed.append(key)
    if args.check or not changed:
        print("nothing written" if not changed else f"would write: {changed}")
        return
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"wrote {REGISTRY.name}: {changed}")


if __name__ == "__main__":
    main()
