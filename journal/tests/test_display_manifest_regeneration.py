"""A full rebuild must retain display records from direct-output builders."""
import hashlib
import importlib.util
import json
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]


def test_display_export_covers_all_nine_without_overwriting_direct_tables(tmp_path, monkeypatch):
    path = JOURNAL / "scripts/rebuild_final_publication_figures.py"
    spec = importlib.util.spec_from_file_location("display_rebuild", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "source_data/curated_publication"
    output.mkdir(parents=True)
    expected = {}
    for number in range(1, 10):
        data = f"panel,value\nA,{number}\n".encode()
        expected[number] = data
        if number in module.PROVENANCE_DIR:
            source = tmp_path / "figures/provenance" / module.PROVENANCE_DIR[number] / f"figure_{number:02d}_plotted.csv"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(data)
        else:
            (output / f"figure_{number:02d}_plotted.csv").write_bytes(data)
    module.export_display_tables()
    records = json.loads((output / "manifest.json").read_text())["records"]
    assert [record["figure"] for record in records] == list(range(1, 10))
    for record in records:
        data = expected[record["figure"]]
        assert (tmp_path / record["path"]).read_bytes() == data
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    text = (output / "README.md").read_text()
    assert "Figure 1 panels C-G" in text and "Figure 2 panels C-H" in text
