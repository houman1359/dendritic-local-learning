"""Nested TeX inputs retain the compilation root, as pdfLaTeX does."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tex_sources import expanded_tex, tex_sources


def test_nested_fragment_resolves_from_master_directory(tmp_path):
    (tmp_path / 'curated').mkdir()
    master = tmp_path / 'supplementary.tex'
    a = tmp_path / 'curated/a.tex'
    b = tmp_path / 'curated/b.tex'
    master.write_text(r'Before \input{curated/a} After')
    a.write_text(r'First \input{curated/b}')
    b.write_text('Second')
    assert list(tex_sources(master)) == [master, a, b]
    assert expanded_tex(master) == 'Before First Second After'
    b.write_text(r'\input{curated/a}')
    with pytest.raises(ValueError, match='Cyclic'):
        list(tex_sources(master))
