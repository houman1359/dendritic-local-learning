from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]
SCRIPTS = JOURNAL / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_submission_bundle", SCRIPTS / "build_submission_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def included_figures(tex_path: Path) -> set[str]:
    return set(GRAPHIC.findall(tex_path.read_text(encoding="utf-8")))


def test_bundle_figure_allowlist_matches_compiled_manuscripts() -> None:
    main = included_figures(JOURNAL / "main.tex")
    supplementary = included_figures(JOURNAL / "supplementary" / "supplementary.tex")
    assert set(bundle.MAIN_FIGURES) == main
    assert set(bundle.SUPPLEMENTARY_FIGURES) == supplementary


def test_bundle_figure_assets_exist() -> None:
    missing = [
        relative
        for relative in bundle.MAIN_FIGURES + bundle.SUPPLEMENTARY_FIGURES
        if not (JOURNAL / "figures" / relative).is_file()
    ]
    assert not missing
