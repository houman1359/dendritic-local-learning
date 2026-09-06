# Publication source map

`panel_sources.json` declares the actual displayed panel inputs and figure
builders for eight main figures and the added supplementary figures. Its
`records` list is suitable for a Source Data exporter: each record gives a
path, figure, panel, role, independent unit and description. Supporting
experimental outputs have a Methods association unless a plotted association
is explicitly declared.

`retained_supplementary.tsv` preserves established supplementary panel
assignments and supporting sources. Figure S45 is the byte-identical former
image-diagnostic sheet, so its panel assignments are preserved exactly.
Historical main-only numerical files remain supporting sources; obsolete
main-only asset records do not designate current publication figures.

Run `python scripts/update_provenance_hashes.py` after the final assets and
experimental exports are complete. It verifies canonical PDF identity against
builder outputs, checks recorded figure-input hashes, and produces the
canonical manifest and flat `source_data/credit_first_provenance/source_inventory.tsv`.
`--check` checks those outputs without rewriting them. `--prepare` writes only
an explicitly incomplete draft under `analysis/`.

The physical-depth source export contains consolidated histories, endpoints,
configurations and raw-file hash inventories. Per-epoch raw JSON trees,
checkpoints, execution logs and internal revision prose are not automatically
included in the public Source Data inventory.
