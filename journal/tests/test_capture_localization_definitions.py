"""Protect the coordinate and denominator distinctions in the reading copy."""
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]


def test_main_defines_capture_scales_before_interpreting_them():
    text = (JOURNAL / "main.tex").read_text()
    definition = r"$C_{\rm res}=(C-C_0)/(1-C_0)$"
    assert definition in text
    assert text.index(definition) < text.index("Varying $K$ separates the common signal")
    assert "within each cell before averaging across cells" in text


def test_focal_contrast_is_between_localization_indices_not_factors():
    text = (JOURNAL / "main.tex").read_text()
    contrast = (r"\Delta\Lambda_k=\Lambda_k(\bm q'\odot\bm d')"
                r"-\Lambda_k(\bm q\odot\bm d')")
    assert text.count(contrast) == 1  # Full equation in Results; caption explains the operation.
    caption = text.split(r"\label{fig:focal}")[0].rsplit(r"\caption{", 1)[1]
    assert "adjoint is additionally substituted while driving force remains post-shunt" in caption
    assert "not a subtraction of gradient factors" in caption
    assert "always relative to the same baseline" in text
    assert "unprimed factors remain at baseline" in text
    assert "q'd'-d'" not in text


def test_notation_preserves_section_specific_adjoint_and_update_dictionary():
    text = (JOURNAL / "supplementary/curated/si_notation.tex").read_text()
    assert r"$g_n^{\rm tot},R_n^{\rm tot}$" in text
    assert r"q_n=\partial u_0/\partial u_n" in text
    assert r"J_V^{\mathsf T}\bm q=\nabla_{\bm V}\mathcal L" in text
    assert r"D_eA_T\operatorname{diag}(\bm\beta)S_K" in text
    assert "Distinct from the compartmental matrix" in text
