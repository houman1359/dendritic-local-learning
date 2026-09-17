# Figure overhaul: how to resume (written 2026-09-10 09:30 EDT)

Paper: "Dendritic morphology as a dictionary for local credit assignment" (Nature Communications). Everything lives on branch
`figure-overhaul-20260908` in the worktree `drafts/.dendritic-figure-overhaul-20260908/journal` (base: the author's main at 55bf9b8, merged).
Latest commit: 7590591. The author's checkout `drafts/dendritic-local-learning` was NOT modified except for two untracked
reports in `journal/analysis/` (FIGURE_REVIEW_20260908.md, REFEREE_REPORT_20260907.md).

## Done
1. Review (24-agent panel): `analysis/figure_overhaul_20260908/review_20260908/FIGURE_REVIEW_20260908.md` (+ reviewer_results.json).
2. Library upgrade to the Nature-standard spec (Nimbus Sans as TrueType in `figures/fonts/`, three type sizes 7/8/9, stroke cap,
   series/anatomy palette registers with a colour-blind gate, grid-locked letters, forest helper, glyph rules, new strict audits).
3. Plans: `v2/figN/PLAN.md`, `v2/AMENDMENTS.md`, `v2/DECISIONS.md` (author-side rulings), `v2/TEXT_REPAIRS.md`, `v2/SI_PLAN.md`,
   `v2/SI_NUMBERING.md` (frozen 36-figure supplement).
4. All nine main figures rebuilt to the plans, two QA/fix rounds + a residual pass; every `figures/main/figure_0N.pdf` passes
   `python3 scripts/figure_canvas.py --audit <pdf> --strict` and the three layout audits. Captions/callouts for each figure are in
   `v2/figN/TEXT.md` (NOT yet integrated into main.tex).
5. Supplement repaired by the consolidation agent (36 curated figures rebuilt, restorations, renumbering); result and its nine open
   issues in `v2/SUPPLEMENT_RESULT.json` (the first: supplementary.tex textwidth 495.77 pt vs 518.4 pt SI PDFs needs a ruling:
   either set geometry so \textwidth = 518.4 pt or scale the includegraphics).

## Remaining
A. Integration into main.tex: captions from v2/figN/TEXT.md, body callouts, the TEXT_REPAIRS.md edits (string-matched, per DECISIONS G6),
   SI citations per v2/SI_NUMBERING.md section 3, MAIN_PANEL_INVENTORY in tests/test_manuscript_crossrefs.py (letters:
   1 ABCDEFG, 2 ABCDEFGH, 3 ABCDEF, 4 ABCDEFGH, 5 ABCDEFG, 6 ABCDEFGH, 7 ABCDEFGH, 8 ABCDEFGH, 9 ABCDEF), panel_sources.json,
   figures/README.md builder map + rebuild_final_publication_figures.py mapping (overhaul builders are production), then
   `python3 scripts/rebuild_final_publication_figures.py --main-only`, strict audit, `make audit` (rehash with
   `python3 scripts/update_provenance_hashes.py` if only hashes fail), `make combined`, `python3 -m pytest tests -q`.
B. Verification (three agents: cross-figure consistency, text-figure agreement, number audit) and a final fix pass.
C. `make combined` -> review `main_with_supplementary.pdf`; then merge the branch into the author's main (the author decides).

## Exact resume command (Claude Code, Workflow tool)
Workflow({ scriptPath: "<worktree>/analysis/figure_overhaul_20260908/v2/figure-build-v2.js",
           resumeFromRunId: "wf_0f5a03fe-4f4",
           args: {"model": "opus", "skip_build": true,
                  "panels": {"1":"ABCDEFG","2":"ABCDEFGH","3":"ABCDEF","4":"ABCDEFGH","5":"ABCDEFG","6":"ABCDEFGH","7":"ABCDEFGH","8":"ABCDEFGH","9":"ABCDEF"}} })
The build phase replays as stubs and the supplement result replays from the journal; the run continues with `integrate`, then the
three verifiers. If the journal is unavailable, launch WITHOUT resumeFromRunId with the same args: the supplement agent then re-runs
on the already-repaired supplement (it will mostly confirm), then integrate and verify run.
Session limits reset roughly every five hours (seen at 3:50pm, 8:50pm, 5:30am ET); a run killed by the limit is resumed the same way.

## Conventions
Commit only as Houman Safaai, one-sentence messages, no Claude trailer; never touch the author's checkout; never bare `git stash`.
