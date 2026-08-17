# Exploratory coupling-by-depth result

Completed: 12 August 2026.  Status: positive seed-consistent interaction;
all-depth accessibility gate failed.

## Mean held-out accuracy

| Child conductance | D1 `[8]` | D2 `[2,3]` | D3 `[2,1,2]` | D3 minus D1 |
|---:|---:|---:|---:|---:|
| 1 | 0.5393 | 0.5426 | 0.5042 | -0.0352 |
| 4 | 0.5362 | 0.5464 | 0.5836 | +0.0474 |
| 16 | 0.5351 | 0.5505 | 0.6313 | +0.0962 |
| 64 | 0.5354 | 0.5483 | 0.6237 | +0.0884 |

The prespecified `(D3-D1 at 64) - (D3-D1 at 1)` interaction is +0.1215
and +0.1256 in seeds 10120 and 10121 (mean +0.1235).  Stronger axial access
therefore reverses the sign of the physical-depth effect in the production
shunting model.  The dose response is not simply monotone global improvement:
D1 remains near 0.535 across coupling, while D3 rises from 0.504 at coupling 1
to 0.631 at 16 and 0.624 at 64.

This is consistent with serial attenuation: deeper composition becomes usable
only when intermediate compartments have sufficient access to their parents.
It does not establish that depth is universally helpful, nor that shunting is
better than a matched grouped-point implementation.

## Gate and decision

No coupling passes the frozen all-depth accessibility gate because D1 and D2
remain below 0.60.  The interaction is therefore retained as exploratory and
does not authorize the sufficient-seed control factorial.  A final transparent
signal-strength calibration at the pre-existing stable coupling 16 will ask
whether all depths can occupy a learnable, non-ceiling regime while retaining
the depth contrast.

## Provenance

- Run:
  `nonlinear_physical_depth_runs/journal_exploratory_physical_depth_coupling_bp_20260812000751`
- Frozen manifest SHA256:
  `e1ef8b4986631d2233870ae115a5fa8cd1dded36b4ffb5190530cae836a5edf1`
- Row-level source:
  `source_data/nonlinear_physical_depth_coupling/bp_seed_rows.csv`
