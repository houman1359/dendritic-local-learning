PYTHON ?= python
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
PDFUNITE ?= pdfunite

.PHONY: paper supplement combined figures canonical-figures presentation-pdf refresh-regular-tree-source test reproducibility software-release format-audit citation-audit overlap-audit submission-bundle overleaf-bundle audit clean clean-aux

paper:
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

supplement:
	cd supplementary && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary.tex
	cd supplementary && $(BIBTEX) supplementary
	cd supplementary && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary.tex
	cd supplementary && $(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary.tex

combined: paper supplement
	$(PDFUNITE) main.pdf supplementary/supplementary.pdf main_with_supplementary.pdf

presentation-pdf:
	$(PYTHON) ../presentations/legacy_decks/build_pdf_presentation.py

figures:
	$(PYTHON) scripts/analyze_prospective_followup_results.py
	$(PYTHON) scripts/analyze_bandwidth_matched_routing.py
	$(PYTHON) scripts/build_prospective_input_validity_audit.py
	$(PYTHON) scripts/analyze_spatial_topology_audit.py
	$(PYTHON) scripts/analyze_prospective_learning_results.py --audit-csv source_data/prospective_learning/seed_outcomes.csv
	$(PYTHON) scripts/build_trained_subtree_address_figure.py
	$(PYTHON) scripts/analyze_trained_partition_residual.py
	$(PYTHON) scripts/build_trained_partition_residual_figure.py
	$(PYTHON) scripts/build_focal_selectivity_figure.py
	$(PYTHON) scripts/analyze_focal_gradient_shapley.py
	$(PYTHON) scripts/run_alignment_controlled_learning.py
	$(PYTHON) scripts/analyze_reciprocal_routing_controls.py
	$(PYTHON) scripts/analyze_physical_cable_sensitivity.py
	$(PYTHON) scripts/build_journal_figures.py
	$(PYTHON) scripts/build_regular_tree_regime_figure.py
	$(PYTHON) scripts/build_microns_v661_replication_figure.py
	$(PYTHON) scripts/analyze_microns_inhibitory_routes.py
	$(PYTHON) scripts/analyze_francioni_signed_credit.py
	$(PYTHON) scripts/build_alignment_animal_figure.py
	$(PYTHON) scripts/analyze_branch_credit_interference.py
	$(PYTHON) scripts/build_new_confirmatory_figures.py
	$(PYTHON) scripts/build_credit_phase_figure.py
	$(PYTHON) scripts/analyze_credit_phase_spectral_bound.py
	$(PYTHON) scripts/analyze_positive_conductance_reliability.py
	$(PYTHON) scripts/build_positive_conductance_reliability_figure.py
	$(PYTHON) scripts/build_adaptive_conductance_reliability_figure.py
	$(PYTHON) scripts/analyze_irregular_tree_wavelets.py
	$(PYTHON) scripts/build_irregular_tree_wavelet_figure.py
	$(PYTHON) scripts/analyze_same_span_coefficient_learning.py
	$(PYTHON) scripts/build_same_span_coefficient_figure.py
	PYTHONPATH=../../..:scripts $(PYTHON) scripts/diagnose_nonlinear_physical_depth.py
	$(PYTHON) scripts/analyze_nonlinear_physical_depth_confirmatory.py
	$(PYTHON) scripts/render_nonlinear_physical_depth_calibration.py
	$(PYTHON) scripts/analyze_point_dendrite_credit_controls.py
	$(PYTHON) scripts/analyze_physical_alignment_dose.py
	$(PYTHON) scripts/analyze_remaining_physical_experiments.py
	$(PYTHON) scripts/analyze_fashion_feedback_ladder.py
	$(PYTHON) scripts/build_main_figure_schematics.py
	$(PYTHON) scripts/build_main_physical_control_panels.py
	$(PYTHON) scripts/sync_canonical_figures.py
	$(PYTHON) scripts/assemble_compact_main_figures.py

canonical-figures:
	$(PYTHON) scripts/sync_canonical_figures.py
	$(PYTHON) scripts/assemble_compact_main_figures.py

refresh-regular-tree-source:
	$(PYTHON) scripts/export_regular_tree_source_data.py

test:
	pytest -q tests

reproducibility:
	$(PYTHON) code/regular_tree/test_additive_reference.py
	$(PYTHON) reproducibility/audit_reproducibility.py

software-release:
	$(PYTHON) scripts/build_software_release.py --force

format-audit:
	$(PYTHON) scripts/audit_nature_communications_format.py --strict

citation-audit:
	$(PYTHON) scripts/audit_citations.py --strict

overlap-audit:
	$(PYTHON) scripts/audit_neurips_text_overlap.py --strict

submission-bundle: combined
	$(PYTHON) scripts/build_submission_bundle.py --force --software-archive submission/Dendritic_credit_assignment_software.zip

overleaf-bundle: combined
	$(PYTHON) scripts/build_overleaf_bundle.py --force

audit: paper supplement test reproducibility format-audit citation-audit overlap-audit
	$(PYTHON) scripts/audit_figure_style_lineage.py
	$(PYTHON) scripts/audit_submission.py --strict-pending
	$(PYTHON) scripts/audit_submission.py --strict-pending --manuscript supplementary/supplementary.tex

clean-aux:
	rm -f main.aux main.blg main.log main.out main.toc
	rm -f supplementary/supplementary.aux supplementary/supplementary.blg supplementary/supplementary.log supplementary/supplementary.out supplementary/supplementary.toc

clean: clean-aux
	rm -f main.bbl main.pdf main_with_supplementary.pdf
	rm -f supplementary/supplementary.bbl supplementary/supplementary.pdf
