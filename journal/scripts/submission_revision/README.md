# Bounded SGD development check

The exact-gradient rule was tested at six fixed rates, with the three original
rescue development seeds, 32,768 updates, and validation-only selection.
`sgd_check.py` freezes the stopping rule before execution: only a rate reaching
validation NMSE <= 0.001 in all three seeds permits a fresh paired comparison.
All 18 fits completed; none passed. No fresh protocol or fresh fits were made.
`analyze_sgd.py` checks every hash and independently replays the selected and
endpoint states before exporting Tables S19 and its source tables.

The original cluster runner expects the immutable archive layout and import
paths. For a portable replay of any grid condition, use
`code/population_replay/launch.py --sgd-rate RATE --rule exact --seed SEED`.
The same launcher covers Figure 6 rescue and task-sensitivity conditions.
