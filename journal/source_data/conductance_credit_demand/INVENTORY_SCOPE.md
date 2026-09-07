# Inventory scope

`science_handoff_inventory_20260906.tsv` preserves the original scientific handoff byte for byte. The portable launcher uses its fixed SHA256 to anchor the original source and protocol entries. This historical inventory also lists presentation files whose current versions may have changed during final figure or release preparation; it is not a statement that every current file still has its historical hash.

`study_source_inventory.tsv` records the current local study scripts, outcomes, analyses, figure artifacts and documentation at the final handoff, excluding interpreter caches and the inventory itself. It includes the historical inventory, portable launcher and validation, and the separately frozen post hoc expanded-rate screen. No experimental freeze is rewritten when this current inventory is refreshed.

The submission's canonical archive inventories identify the bytes actually released. Source Data may omit duplicated TeX, PDF and PNG presentation files that are supplied in the manuscript or software package. Thus this full local study inventory provides provenance and is not a promise that every presentation duplicate is shipped under the same Source Data path.

`validation_audit.json` is the historical scientific audit. `final_validation_audit.json` adds the completed expanded-rate screen and portable checks, distinguishing original fits, continuations, repeated seed blocks and installation checks. Portable checks are excluded from paper outcomes.
