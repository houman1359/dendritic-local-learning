"""Keep current figure labels and displayed seed identities tied to their data."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

JOURNAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
sys.path.insert(0, str(JOURNAL / "scripts/credit_first_figures"))

import build_ancestry as ancestry
import build_framework as framework
import build_main_figure_04 as conflict


@pytest.fixture(scope="module")
def framework_data():
    conditions, seeds, paired, differences = framework.read_fresh()
    cohorts = framework.cohort_contrasts(paired, differences)
    return conditions, seeds, paired, cohorts


@pytest.mark.parametrize("cohort, folder, column, conditions", [
    ("mnist_dfa", "mnist_between_within_factorial", "within",
     {"identity": ("neuron specific", "scalar broadcast"),
      "exact": ("exact path", "neuron specific")}),
    ("fashion", "fashion_feedback_ladder", "feedback",
     {"identity": ("neuron indexed", "scalar fallback"),
      "exact": ("exact path", "neuron indexed")}),
    ("cifar10", "cifar10_additive_feedback_ladder_confirmatory", "feedback",
     {"identity": ("neuron specific", "strict scalar"),
      "exact": ("exact path", "neuron specific")}),
])
def test_cohort_seed_ids_pair_with_frozen_outcomes(
        framework_data, cohort, folder, column, conditions):
    cohorts = framework_data[-1]
    source = pd.read_csv(JOURNAL / "source_data" / folder / "seed_outcomes.csv",
                         float_precision="round_trip")
    if cohort == "mnist_dfa":
        source = source[source.between.eq("dfa")]
    for _, row in cohorts[cohorts.cohort.eq(cohort)].iterrows():
        selected = source
        if "architecture" in selected:
            selected = selected[selected.architecture.eq(row.architecture)]
        paired = selected.pivot(index="seed", columns=column, values="test_accuracy")
        lhs, rhs = conditions[row.kind]
        expected = 100 * (paired[lhs] - paired[rhs]).sort_index()
        assert row.seed_ids == expected.index.astype(int).tolist()
        # CIFAR's frozen fan values have nine-decimal fractional precision.
        np.testing.assert_allclose(row.seed_pp, expected.to_numpy(), rtol=0, atol=5e-8)


def test_figure1_export_preserves_actual_seed_ids(tmp_path, monkeypatch, framework_data):
    conditions, seeds, paired, cohorts = framework_data
    cap_seed, cap_summary = framework.read_capture()
    dictionaries = np.load(framework.RECORDS / "figure_01_illustrative_dictionaries.npz")
    monkeypatch.setattr(framework, "CURATED", tmp_path / "figure_01_plotted.csv")
    output = framework.write_curated(
        conditions, seeds, paired, {}, cohorts, framework.read_bp_equivalence(),
        cap_seed, cap_summary, dictionaries)
    exported = pd.read_csv(output)
    drawn = exported[exported.panel.isin(["E", "F"]) & exported.record.eq("seed")]
    assert len(drawn) == 180
    assert not drawn.duplicated(["panel", "architecture", "cohort", "seed"]).any()
    for _, row in cohorts.iterrows():
        selected = drawn[drawn.cohort.eq(row.cohort)
                         & drawn.architecture.eq(row.architecture)
                         & drawn.contrast.eq(row.source_contrast)].sort_values("seed")
        assert selected.seed.astype(int).tolist() == row.seed_ids
        np.testing.assert_allclose(selected.value, row.seed_pp, rtol=0, atol=1e-12)
    fresh = drawn[drawn.cohort.eq("mnist_fresh")]
    assert set(fresh.seed.astype(int)) == set(seeds.seed.astype(int))
    assert set(fresh.seed.astype(int)) == set(range(50300, 50310))


def test_control_heatmap_uses_disclosed_row_normalization():
    supports = ancestry.control_supports()
    actual = ancestry.normalized_control_supports(supports)
    original = supports[[f"b{i}" for i in range(1, 9)]].to_numpy(float)
    np.testing.assert_allclose(actual, original / np.abs(original).max(axis=1)[:, None])
    np.testing.assert_allclose(np.linalg.norm(original, axis=1), 1)
    np.testing.assert_allclose(actual.max(axis=1), 1)
    assert not np.allclose(actual, np.abs(original))
    assert np.count_nonzero(actual < 0) == np.count_nonzero(original < 0) == 3
    fig = plt.figure(figsize=(295.6 / 72, 124 / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    try:
        ancestry.panel_dictionaries(ax, ancestry.coefficient_prediction(), supports)
        assert "Φ / max |Φ|" in {text.get_text() for text in ax.texts}
    finally:
        plt.close(fig)


def test_derangement_is_cyclic_and_caption_defines_last_branch_wrap():
    fig = plt.figure(figsize=(169.5 / 72, 124 / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    try:
        conflict.backward_credit_schematic(ax)
        labels = {text.get_text() for text in ax.texts}
        assert "Next branch only" in labels
        assert "Cyclic reassignment" in labels
        assert " = δ 1[b ≡ b* + 1]" not in labels
        assert "b → b + 1" not in labels
        manuscript = (JOURNAL / "main.tex").read_text()
        caption = manuscript.split(r"\label{fig:branchconflict}")[0].rsplit(
            r"\caption{", 1)[1]
        assert r"\delta\mathbf{1}[b\equiv b^\star+1\pmod B]" in caption
        assert "cyclic indexing" in caption
    finally:
        plt.close(fig)


def test_source_data_readme_describes_current_figure1c():
    readme = (JOURNAL / "source_data/curated_publication/README.md").read_text()
    template = (JOURNAL / "scripts/rebuild_final_publication_figures.py").read_text()
    assert "Figure 1C shows the indicator" in readme
    assert "Figure 1C shows the indicator" in template
    assert "illustrative evaluation of the one-step bound" not in readme
    assert "illustrative evaluation of the one-step bound" not in template
