#!/usr/bin/env python3
"""Move the publication panel map onto the frozen 36-figure supplement.

`configs/credit_first_provenance/panel_sources.json` states which numerical
source backs which displayed panel.  Its `assets` block was already moved onto
the consolidated supplement of 2026-09-10 (SI_NUMBERING), but its `records`
block still names the earlier 35-figure sequence, so the released Source Data
inventory kept describing panels that no longer exist.

Two frozen tables below carry the whole renumbering.  `FIGURE_LINEAGE` maps
every earlier supplementary number onto its current one; the two merges
(checkpoint geometry into the utility sheet, measured topology into the
transfer-geometry sheet) appear as two earlier numbers sharing one target.
`PANEL_LINEAGE` lists only the panels whose letter moved, including the split
that promoted the error-field panels of the earlier S7 into the new S8.

Panel letters of a source that the curation manifest actually declares are
taken from that manifest, which is authoritative for the displayed panels; the
lineage table only places supporting inputs that the manifest does not name.
Sources for the two genuinely new sheets, S1 and S33, have no predecessor and
are declared explicitly in `NEW_FIGURE_RECORDS`.  One earlier panel has no
supplementary successor at all and is listed in `RETIRED_PANELS`.

The same pass restates two other things the consolidation left behind: each
supplementary asset row's panel range, which had become a uniform `a-c`
placeholder, and the historical display labels that still named sheets above
S36 without saying they were historical.

Run with --check to verify the config without rewriting it.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

JOURNAL = Path(__file__).resolve().parents[2]
MAP_PATH = JOURNAL / "configs/credit_first_provenance/panel_sources.json"
CURATION = JOURNAL / "configs/supplement_consolidation/manifest.json"

# Earlier supplementary number -> current supplementary number.
FIGURE_LINEAGE = {
    1: 2, 2: 3, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 10, 10: 11, 11: 12,
    12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21,
    21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30,
    30: 31, 31: 31, 32: 32, 33: 34, 34: 35, 35: 36,
}

# (earlier figure, earlier panel) -> [(current figure, current panel), ...].
# Panels absent here keep their letter under FIGURE_LINEAGE.  Every entry below
# is the crop record the curation manifest keeps for the current panel
# (`source_asset` and `source_panel`), so the table states which earlier panel
# each current panel was cut from rather than inferring it.  The handful of
# current panels that were re-cropped from the original frozen sheet instead of
# from the consolidated one carry no such record; those earlier panels are
# placed by their declared sources.
PANEL_LINEAGE = {
    # Checkpoint geometry is absorbed by the utility sheet: its three mechanism
    # panels become one merged checkpoint panel, its alignment panel one
    # alignment panel.
    (3, "a"): [(3, "e")],
    (3, "b"): [(3, "e")],
    (3, "c"): [(3, "e")],
    (3, "d"): [(3, "f")],
    # The error-field panels of the earlier image sheet become their own sheet.
    (7, "e"): [(8, "a")],
    (7, "f"): [(8, "b")],
    (7, "g"): [(8, "c")],
    # Branch-conflict controls lose a panel: the two coverage panels merge and
    # the three conflict panels shift up one letter.
    (9, "b"): [(10, "a")],
    (9, "c"): [(10, "b")],
    (9, "d"): [(10, "c")],
    (9, "e"): [(10, "d")],
    # Ancestry coefficients lose a panel; its last two are re-cropped from the
    # original sheet, so the crop record names no predecessor for them.
    (10, "d"): [(11, "b"), (11, "c")],
    # Scalar-tree capacity gains two restored panels ahead of these two.
    (11, "c"): [(12, "e")],
    (11, "d"): [(12, "f")],
    # Fixed-profile budget interleaves two restored extension panels.
    (15, "a"): [(16, "b")],
    (15, "b"): [(16, "d")],
    (15, "c"): [(16, "e")],
    (15, "d"): [(16, "f")],
    # Conductance optimization gains a restored robustness panel at c.
    (19, "c"): [(20, "d")],
    (19, "d"): [(20, "e")],
    # Shunt sensitivity gains two restored panels at b-d, so the typed-direct
    # panels move to e and f.
    (28, "b"): [(29, "e")],
    (28, "c"): [(29, "f")],
    # Measured topology is absorbed by the transfer-geometry sheet.
    (30, "b"): [(31, "a")],
    (30, "c"): [(31, "a")],
    (30, "d"): [(31, "e")],
    (31, "a"): [(31, "b"), (31, "c"), (31, "d")],
    # Finite horizon gains a restored prediction panel at d.
    (34, "d"): [(35, "e"), (35, "f")],
}

# Earlier panels with no supplementary successor.  Old S28 panel D was the
# normalized-dose comparison, which the consolidation promoted into main
# Figure 8D; releasing its table under S29 would claim a panel that sheet does
# not draw.  Its records stay under Methods, where the complete table already is.
RETIRED_PANELS = {
    (28, "d"),
}

_CURATED_ROLE = "Complete numerical source of selected input panel"
_CURATED_GENERATOR = "scripts/supplement_consolidation/build.py"

# The two sheets with no predecessor in the earlier sequence.
NEW_FIGURE_RECORDS = (
    ("figS1", "abcde", "source_data/figure2/path_gain_cv_runs.csv",
     "paired independent training seed (n=5 per architecture)",
     "Mechanistic chain sheet, frozen numbering of 2026-09-10 (SI_NUMBERING)."
     " Complete table; it can contain other conditions and implies no"
     " additional replication."),
    ("figS33", "ab", "source_data/animal_learning_francioni/animal_common_signed_modes.csv",
     "animal (n=6)",
     "Reanalysis of published recordings; no new animal experiment."
     " Complete table; it can contain other conditions and implies no"
     " additional replication."),
    ("figS33", "ab", "source_data/animal_learning_francioni/animal_signed_contrasts.csv",
     "animal (n=6)",
     "Reanalysis of published recordings; no new animal experiment."
     " Complete table; it can contain other conditions and implies no"
     " additional replication."),
    ("figS33", "ab", "source_data/animal_learning_francioni/neuron_sd_residual_distributions.csv",
     "neuron nested within published condition; descriptive only",
     "Reanalysis of published recordings; no new animal experiment."
     " Complete table; it can contain other conditions and implies no"
     " additional replication."),
)

LAYOUT = ("Nine credit-first main figures; focused empirical capture, coefficient"
          " learning and gate presentation; consolidated 36-figure supplement.")


def declared_panels(curation):
    """(current figure number, source) -> panel letters the manifest declares."""
    result = collections.defaultdict(set)
    for asset in curation["assets"]:
        number = int(asset["figure"].removeprefix("S"))
        for panel in asset["panels"]:
            for declared in panel["numerical_source_paths"]:
                source = "source_data/" + declared.split("source_data/", 1)[1]
                result[(number, source)].add(panel["panel"].lower())
    return result


def targets(figure, panel, path, declared):
    """Current (figure, panel) homes of one earlier panel record."""
    number = int(figure.removeprefix("figS"))
    if (number, panel) in RETIRED_PANELS:
        return
    lineage = PANEL_LINEAGE.get((number, panel), [(FIGURE_LINEAGE[number], panel)])
    grouped = collections.defaultdict(list)
    for new_figure, new_panel in lineage:
        grouped[new_figure].append(new_panel)
    for new_figure, fallback in grouped.items():
        for letter in sorted(declared.get((new_figure, path)) or fallback):
            yield f"figS{new_figure}", letter


def renumbered_records(records, declared):
    result, seen = [], set()

    def emit(record):
        key = (record["figure"], record["panel"], record["path"])
        if key in seen:
            return
        seen.add(key)
        result.append(record)

    for record in records:
        if not record["figure"].startswith("figS"):
            emit(record)
            continue
        for figure, panel in targets(record["figure"], record["panel"].lower(),
                                     record["path"], declared):
            emit(dict(record, figure=figure, panel=panel))
    for figure, panels, path, unit, notes in NEW_FIGURE_RECORDS:
        for panel in panels:
            emit({"figure": figure, "panel": panel, "path": path,
                  "role": _CURATED_ROLE, "generator": _CURATED_GENERATOR,
                  "replication_unit": unit, "notes": notes})
    return result


def panel_range(asset):
    """The asset's own panel letters, as the `a-h` range the ledger prints."""
    letters = [panel["panel"].lower() for panel in asset["panels"]]
    expected = [chr(ord(letters[0]) + k) for k in range(len(letters))]
    if letters != expected:
        raise SystemExit(f"{asset['id']} panel letters are not contiguous: {letters}")
    return letters[0] if len(letters) == 1 else f"{letters[0]}-{letters[-1]}"


def restated_assets(assets, curation):
    """Give every supplementary asset row the panel range its sheet draws."""
    ranges = {asset["figure"].replace("S", "figS", 1): panel_range(asset)
              for asset in curation["assets"]}
    return [dict(asset, panel=ranges[asset["figure"]])
            if asset["figure"] in ranges else asset
            for asset in assets]


def restated_history(records):
    """Say that a display label naming a retired sheet is historical.

    These rows are Methods-associated; their panel column records where the
    source used to be displayed, under a supplementary sequence that no longer
    reaches those numbers.
    """
    result = []
    for record in records:
        panel = record["panel"]
        if (record["figure"] == "methods" and panel.startswith("supporting figS")
                and int(re.search(r"figS(\d+)", panel).group(1)) > 36):
            panel = "supporting; previous display " + panel.removeprefix("supporting ")
        result.append(dict(record, panel=panel))
    return result


def verify(records, curation, declared):
    numbers = {int(r["figure"].removeprefix("figS"))
               for r in records if r["figure"].startswith("figS")}
    expected = set(range(1, len(curation["assets"]) + 1))
    if numbers != expected:
        raise SystemExit(f"supplementary records are not 1..{len(expected)}: "
                         f"missing={sorted(expected - numbers)} extra={sorted(numbers - expected)}")
    stale = [r["panel"] for r in records
             if r["panel"].startswith("supporting figS")
             and int(re.search(r"figS(\d+)", r["panel"]).group(1)) > 36]
    if stale:
        raise SystemExit(f"{len(stale)} display labels still name a retired sheet, e.g. {stale[:3]}")
    present = {(r["figure"], r["panel"], r["path"]) for r in records}
    missing = [(f"figS{number}", letter, source)
               for (number, source), letters in declared.items()
               for letter in letters
               if (f"figS{number}", letter, source) not in present]
    if missing:
        raise SystemExit(f"{len(missing)} declared panel sources are unreleased, "
                         f"e.g. {missing[:3]}")
    return len(numbers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the stored map without rewriting it")
    args = parser.parse_args()
    layout = json.loads(MAP_PATH.read_text())
    curation = json.loads(CURATION.read_text())
    declared = declared_panels(curation)
    if args.check:
        verify(layout["records"], curation, declared)
        if layout["assets"] != restated_assets(layout["assets"], curation):
            raise SystemExit("supplementary asset panel ranges do not match the curation manifest")
        print(json.dumps({"status": "current", "records": len(layout["records"])}, indent=2))
        return
    before = len(layout["records"])
    records = restated_history(renumbered_records(layout["records"], declared))
    figures = verify(records, curation, declared)
    layout["assets"] = restated_assets(layout["assets"], curation)
    layout["records"] = records
    layout["layout"] = LAYOUT
    MAP_PATH.write_text(json.dumps(layout, indent=2) + "\n")
    print(json.dumps({"supplementary_figures": figures,
                      "records_before": before,
                      "records_after": len(records)}, indent=2))


if __name__ == "__main__":
    main()
