from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


JOURNAL = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mesh_edges_include_faces_and_links_without_duplicates() -> None:
    module = _module(
        JOURNAL / "scripts" / "prepare_pinky_v185_replication.py", "pinky_prepare"
    )
    faces = np.asarray([[0, 1, 2], [2, 1, 3]])
    links = np.asarray([[0, 3], [3, 0]])
    edges = module.face_edges(faces, links)
    assert {tuple(row) for row in edges} == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }


def test_k4_capture_contrast_uses_control_minus_morphology_residual() -> None:
    module = _module(
        JOURNAL / "scripts" / "analyze_pinky_v185_replication.py", "pinky_analyze"
    )
    rows = []
    for root_id in (1, 2):
        for method, residual in {
            "morphology-aware paths": 0.30,
            "random paths": 0.60,
            "depth-only bins": 0.50,
            "shuffled ancestry": 0.55,
        }.items():
            rows.append(
                {
                    "root_id": root_id,
                    "channels": 4,
                    "method": method,
                    "residual": residual,
                }
            )
    contrasts, summary = module.k4_contrasts(pd.DataFrame(rows))
    random = contrasts[contrasts["control"].eq("random paths")]
    assert np.allclose(random["morphology_capture_advantage"], 0.30)
    assert summary["random paths"]["positive_cells"] == 2


def test_reference_summary_can_skip_combinatorial_sign_flip() -> None:
    module = _module(
        JOURNAL / "scripts" / "analyze_pinky_v185_replication.py", "pinky_analyze_skip"
    )
    rows = []
    for root_id in range(30):
        for method, residual in {
            "morphology-aware paths": 0.30,
            "random paths": 0.60,
            "depth-only bins": 0.50,
            "shuffled ancestry": 0.55,
        }.items():
            rows.append(
                {
                    "root_id": root_id,
                    "channels": 4,
                    "method": method,
                    "residual": residual,
                }
            )
    _contrasts, summary = module.k4_contrasts(
        pd.DataFrame(rows), compute_exact_descriptive_p=False
    )
    assert summary["random paths"]["descriptive_cell_sign_flip_p_two_sided"] is None
    assert "not recomputed" in summary["random paths"]["descriptive_cell_sign_flip_method"]
