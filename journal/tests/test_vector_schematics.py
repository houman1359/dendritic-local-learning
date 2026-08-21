from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMATICS = ROOT / "figures" / "schematics"


def test_figure1_schematic_exports_are_editable_vectors():
    expected = {
        "credit_assignment_gap.svg",
        "point_vs_dendritic.svg",
        "credit_information_ladder.svg",
        "eligibility_transport.svg",
        "evidence_boundary_path.svg",
    }
    assert expected <= {path.name for path in SCHEMATICS.glob("*.svg")}

    for name in expected | {"figure_01.svg"}:
        path = ROOT / "figures" / "main" / name if name == "figure_01.svg" else SCHEMATICS / name
        text = path.read_text(encoding="utf-8")
        assert "<svg" in text
        assert "<text" in text, f"{name} should retain editable text"
        assert "<image" not in text, f"{name} should not embed raster artwork"
