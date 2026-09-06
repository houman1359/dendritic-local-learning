"""Combine reading copies without losing or colliding named PDF destinations.

Resolve links in each original document before merging, convert named PDF
coordinates to MuPDF coordinates, and recreate explicit offset GoTo links.
Original PDFs are read only. The combined output is replaced only after its
links, outlines, pages and extracted text pass a save/reopen audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import fitz

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_destination(source: fitz.Document, item: dict, offset: int) -> dict:
    """Return an explicit GoTo, accounting for named destinations' PDF axes."""
    kind = item["kind"]
    if kind not in (fitz.LINK_GOTO, fitz.LINK_NAMED):
        raise ValueError(f"Unsupported internal destination kind: {kind}")
    page_number = item.get("page", -1)
    destination = item.get("to")
    if page_number is None or page_number < 0 or destination is None:
        raise ValueError(f"Unresolved source destination: {item}")
    if page_number >= len(source):
        raise ValueError(f"Source destination page out of range: {item}")
    point = fitz.Point(destination)
    if kind == fitz.LINK_NAMED:
        # get_links()/get_toc(False) leave resolved named coordinates in the
        # PDF bottom-left system; ordinary GoTo points are already top-left.
        point = point * source[page_number].transformation_matrix
    return dict(kind=fitz.LINK_GOTO, page=offset + page_number,
                to=point, zoom=float(item.get("zoom", 0) or 0))


def signature(item: dict) -> tuple:
    """Stable, geometry-aware signature for independent save/reopen checks."""
    kind = item["kind"]
    rectangle = tuple(round(float(v), 3) for v in item.get("from", ()))
    if kind == fitz.LINK_GOTO:
        return (kind, rectangle, item["page"],
                tuple(round(float(v), 3) for v in item["to"]),
                round(float(item.get("zoom", 0) or 0), 4))
    if kind == fitz.LINK_URI:
        return (kind, rectangle, item["uri"])
    raise ValueError(f"Unexpected output link type: {kind}")


def combine_pdfs(input_paths: list[Path], output_path: Path,
                 validation_path: Path | None = None) -> dict:
    input_paths = [Path(p).resolve() for p in input_paths]
    output_path = Path(output_path).resolve()
    if output_path in input_paths or len(set(input_paths)) != len(input_paths):
        raise ValueError("Inputs must be distinct and output must not replace an input")
    input_hashes = {str(p): sha256(p) for p in input_paths}
    output = fitz.open()
    expected_links: list[list[dict]] = []
    expected_text: list[str] = []
    expected_outlines: list[list] = []
    sources = []
    for path in input_paths:
        with fitz.open(path) as source:
            if source.needs_pass:
                raise ValueError(f"Encrypted source requires a password: {path}")
            offset = len(output)
            output.insert_pdf(source, links=False, annots=True)
            counts = Counter()
            for page_number, page in enumerate(source):
                links = []
                for link in page.get_links():
                    counts[link["kind"]] += 1
                    if link["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                        resolved = explicit_destination(source, link, offset)
                    elif link["kind"] == fitz.LINK_URI:
                        resolved = dict(kind=fitz.LINK_URI, uri=link["uri"])
                    else:
                        raise ValueError(f"Unsupported source link action in {path}: {link}")
                    resolved["from"] = fitz.Rect(link["from"])
                    output[offset + page_number].insert_link(resolved)
                    links.append(resolved)
                expected_links.append(links)
                expected_text.append(page.get_text())
            source_toc = source.get_toc(simple=False)
            for level, title, _page, destination in source_toc:
                if destination["kind"] not in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                    raise ValueError(f"Unsupported source outline action: {destination}")
                resolved = explicit_destination(source, destination, offset)
                for key in ("color", "bold", "italic", "collapse"):
                    if key in destination:
                        resolved[key] = destination[key]
                expected_outlines.append([level, title, resolved["page"] + 1, resolved])
            sources.append(dict(filename=path.name, sha256=input_hashes[str(path)],
                pages=len(source), page_offset=offset, internal_links=counts[fitz.LINK_GOTO]+counts[fitz.LINK_NAMED],
                named_links_converted=counts[fitz.LINK_NAMED], uri_links=counts[fitz.LINK_URI],
                bookmarks=len(source_toc)))
    if expected_outlines:
        output.set_toc(expected_outlines)
    output.set_metadata({"title": "Dendritic morphology as a dictionary for local credit assignment",
                         "subject": "Main manuscript and Supplementary Information; resolved internal links"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".combined-links-", suffix=".pdf", dir=output_path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        output.save(temporary, deflate=True)
        output.close()
        with fitz.open(temporary) as checked:
            assert len(checked) == len(expected_links)
            observed_counts = Counter()
            for page_number, page in enumerate(checked):
                observed = page.get_links()
                assert Counter(map(signature, observed)) == Counter(map(signature, expected_links[page_number])), page_number
                assert page.get_text() == expected_text[page_number], ("page text changed", page_number)
                for link in observed:
                    observed_counts[link["kind"]] += 1
                    if link["kind"] == fitz.LINK_GOTO:
                        assert 0 <= link["page"] < len(checked)
            outlines = checked.get_toc(simple=False)
            assert len(outlines) == len(expected_outlines)
            for actual, expected in zip(outlines, expected_outlines):
                assert actual[:3] == expected[:3]
                assert signature(actual[3]) == signature(expected[3])
            assert observed_counts[fitz.LINK_NAMED] == 0
        assert all(sha256(path) == input_hashes[str(path)] for path in input_paths)
        os.replace(temporary, output_path)
    finally:
        if not output.is_closed:
            output.close()
        temporary.unlink(missing_ok=True)
    report = dict(status="passed", sources=sources, pages=len(expected_links),
        internal_links=observed_counts[fitz.LINK_GOTO], uri_links=observed_counts[fitz.LINK_URI],
        unresolved_or_named_output_links=0, bookmarks=len(expected_outlines),
        all_link_rectangles_destinations_and_uris_verified=True, all_page_text_preserved=True,
        input_pdfs_unchanged=True, named_coordinates_transformed=True,
        output_sha256=sha256(output_path), pymupdf_version=fitz.VersionBind)
    if validation_path is not None:
        validation_path = Path(validation_path)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=ROOT / "main.pdf")
    parser.add_argument("--supplement", type=Path, default=ROOT / "supplementary/supplementary.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "main_with_supplementary.pdf")
    parser.add_argument("--validation", type=Path, default=ROOT / "analysis/completion_20260905/combined_pdf_link_validation.json")
    args = parser.parse_args()
    print(json.dumps(combine_pdfs([args.main, args.supplement], args.output, args.validation), indent=2))


if __name__ == "__main__":
    main()
