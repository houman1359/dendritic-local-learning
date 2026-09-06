#!/usr/bin/env python3
"""Rebuild the credit-first eight main and 47 SI figures from frozen source tables.

No experiment runner, model fit, selection, or historical all-analysis pipeline
is invoked. Unchanged historical vector source components are retained inputs.
Native replacements are drawn last; the legacy sync/compositor entry points are
intentionally bypassed because they restore superseded supplementary sheets.
"""
from __future__ import annotations
from tex_sources import expanded_tex
import argparse
import importlib
import json
from pathlib import Path
import re
import runpy
import shutil
import pandas as pd
from assemble_compact_main_figures import (COMPONENTS, ROOT, SUPP, Panel, Slot,
                                           compose, emit_native, emit_physical_depth_supplement, panel)


def run_script(name, args=()):
    # Isolate generic module names (notably each experiment's run.py).
    import subprocess, sys
    print(f"Rendering {name}", flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts'/name), *map(str,args)],
                   cwd=ROOT, check=True)


def copy_generated(stem,target):
    shutil.copyfile(ROOT/'figures/generated'/f'{stem}.pdf',SUPP/target)


def compose_final_supplements():
    # S17–S20 are native frozen-data redraws; they are emitted below.
    # S22 is drawn directly from frozen tables; cropping the ten-panel
    # historical sheet clipped axis labels and exposed neighboring glyphs.
    import build_journal_figures as historical
    historical._figure5_final_supplement()
    compose(SUPP/'figure_S28_panels_A-B.pdf',[
        panel('figure_05_panels_G-L.pdf','J',2,3),
        panel('figure_05_panels_G-L.pdf','K',2,3)],
        ['Credit-coordinate ladder','Pairwise credit comparisons'],rows=1,cols=2,height=220)


def supplementary():
    run_script('rebuild_review_visual_corrections.py')
    # The following function calls render existing tables only. Never invoke
    # these analysis modules' main functions, which may recompute experiments.
    import run_alignment_controlled_learning as controlled
    controlled.make_figure(pd.read_csv(ROOT/'source_data/alignment_controlled/alignment_controlled_curves.csv'),
                           pd.read_csv(ROOT/'source_data/alignment_controlled/cell_alignment_metrics.csv'),
                           ROOT/'figures/generated')
    import build_journal_figures as historical
    historical._figure5_detailed()
    run_script('analyze_microns_inhibitory_routes.py',['--plot-only'])
    copy_generated('fig_microns_inhibitory_routes','figure_S12_panels_A-J.pdf')
    for name in ['build_focal_selectivity_figure.py','build_same_span_coefficient_figure.py',
                 'render_nonlinear_physical_depth_calibration.py','build_irregular_tree_wavelet_figure.py',
                 'build_positive_conductance_reliability_figure.py','build_adaptive_conductance_reliability_figure.py',
                 'build_interior_optimum_figure.py','build_trained_partition_residual_figure.py']:
        run_script(name)
    for stem,target in [
        ('fig6_alignment_controlled','figure_S05_panels_A-H.pdf'),
        ('fig_focal_selectivity_matrix','figure_S11_panels_A-C.pdf'),
        ('fig_positive_conductance_reliability','figure_S13_panels_A-F.pdf'),
        ('fig_same_span_coefficient_learning','figure_S14_panels_A-F.pdf'),
        ('fig_supp_nonlinear_depth_calibration','figure_S15_panels_A-D.pdf'),
        ('fig_interior_optimum','figure_S16_panels_A-D.pdf'),
        ('fig_trained_partition_residual','figure_S23_panels_A-C.pdf'),
        ('fig_adaptive_conductance_reliability','figure_S24_panels_A-D.pdf'),
        ('fig_irregular_tree_wavelets','figure_S25_panels_A-D.pdf')]:
        copy_generated(stem,target)
    compose_final_supplements()
    for name in ['build_supplementary_figure_s01_native.py','build_supplementary_figure_s02_native.py',
                 'build_supplementary_figure_s03_native.py','build_supplementary_figure_s09_native.py',
                 'build_supplementary_figures_s17_s20_native.py',
                 'build_supplementary_figure_s27_native.py','build_supplementary_figure_s28_native.py','build_supplementary_figure_s21_native.py','build_path_necessity_fashion_figure.py',
                 'build_supplementary_figure_s30_native.py','build_review_completion_figures.py',
                 'build_review_response_baselines_figure.py']:
        run_script(name)
    run_script('build_supplementary_figure_s04_native.py',
               ['--confirmatory-analysis-dir',ROOT/'source_data/cifar10_additive_feedback_ladder_confirmatory'])
    # S26 retains its previously audited vector source asset; it is an unchanged
    # frozen component of this rendering-only rebuild.
    emit_physical_depth_supplement()
    run_script('build_supplementary_figure_s35_native.py')
    run_script('build_morphology_followup_figures.py')
    run_script('build_morphology_credit_figure_tables.py')
    run_script('build_morphology_calibration_figure_tables.py')
    run_script('build_morphology_bridge_figures.py')
    run_script('build_boolean_morphology_figures.py')
    run_script('build_main_figure_02.py')
    shutil.copyfile(ROOT/'figures/components/main_figure_02_native.pdf',SUPP/'figure_S45_image_diagnostics.pdf')
    run_script('build_utility_supplement.py')
    run_script('shunt_ancestry_gain/build_figure.py')
    shutil.copyfile(ROOT/'source_data/shunt_ancestry_gain/figures/weak_channel_linearization_check.pdf',
                    SUPP/'figure_shunt_weak_channels.pdf')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-emit-main',action='store_true',help='Build native components and SI, leaving canonical main PDFs for the final manuscript owner to emit.')
    parser.add_argument('--supplement-only',action='store_true')
    parser.add_argument('--main-only',action='store_true')
    args=parser.parse_args()
    if args.supplement_only and args.main_only:parser.error('Choose at most one partial build.')
    if not args.supplement_only:
        for name in ["credit_first_figures/build_framework.py", "build_main_figure_04.py",
                     "credit_first_figures/build_ancestry.py", "credit_rule_bridge/build_figure.py",
                     "physical_depth_budget/build_main_figure5.py", "credit_first_figures/build_anatomy.py",
                     "shunt_ancestry_gain/build_figure.py", "credit_first_figures/build_measured.py"]:
            run_script(name)
    if not args.main_only:supplementary()
    if not args.no_emit_main and not args.supplement_only:
        mapping={
            1:"figures/components/credit_first_figure_01.pdf",
            2:"figures/components/main_figure_04_native.pdf",
            3:"figures/components/credit_first_figure_03.pdf",
            4:"source_data/credit_rule_bridge/figures/credit_interaction_bridge_native.pdf",
            5:"source_data/physical_depth_budget/figure/figure5_physical_depth_budget.pdf",
            6:"figures/components/credit_first_figure_06.pdf",
            7:"source_data/shunt_ancestry_gain/figures/shunt_ancestry_gain_native.pdf",
            8:"figures/components/credit_first_figure_08.pdf",
        }
        for number,source in mapping.items():
            shutil.copyfile(ROOT/source, ROOT/f"figures/main/figure_{number:02d}.pdf")
    paths=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',expanded_tex(ROOT/'supplementary/supplementary.tex'))
    paths=[(ROOT/'figures'/p).resolve() for p in paths if p.startswith('supplementary/figure')]
    if len(paths)!=47:raise ValueError(f'Expected 47 SI includes, found {len(paths)}')
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError('\n'.join(missing))
    print('Final figure assets ready: eight main figures and 47 supplementary figures.',flush=True)

if __name__=='__main__':main()
