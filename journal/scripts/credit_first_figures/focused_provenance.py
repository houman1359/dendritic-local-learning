"""Write focused figure outputs without changing their frozen study inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd


JOURNAL = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(number, output, rows, sources, builders, panels, *, emit_main=False,
            layout_findings=(), notes=""):
    """Record current displayed values and their complete source identities."""
    table_dir = JOURNAL / "source_data/curated_publication"
    record_dir = JOURNAL / "figures/provenance/credit_clarity_20260908"
    table_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    table = table_dir / f"figure_{number:02d}_plotted.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    shutil.copyfile(table, record_dir / table.name)
    builders = list(dict.fromkeys([*builders, Path(__file__)]))
    record = {
        "figure": f"Figure {number}", "output": str(output.relative_to(JOURNAL)),
        "output_sha256": digest(output), "plotted_table": str(table.relative_to(JOURNAL)),
        "plotted_table_sha256": digest(table), "panel_sources": panels,
        "source_sha256": {str(p.relative_to(JOURNAL)): digest(p) for p in sources},
        "builders_sha256": {str(p.relative_to(JOURNAL)): digest(p) for p in builders},
        "source_data_immutable": True, "new_experiments": 0,
        "layout_findings": layout_findings, "scope": notes,
    }
    (record_dir / f"figure_{number:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
    if emit_main:
        canonical = JOURNAL / "figures/main" / f"figure_{number:02d}.pdf"
        shutil.copyfile(output, canonical)
        assert digest(canonical) == digest(output)
    return record
