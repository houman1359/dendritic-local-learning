"""Protect scientific exclusions and ordering through publication rebuilds."""
from pathlib import Path
import json,re,sys
import pandas as pd
J=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(J/'scripts'))
from build_submission_bundle import SUPPLEMENTARY_FIGURES
from tex_sources import tex_sources

def test_unresolved_historical_conditions_are_not_drawn():
    manifest=json.loads((J/'configs/supplement_consolidation/manifest.json').read_text())
    by_id={a['id']:a for a in manifest['assets']}
    assert [p['source_panel'] for p in by_id['mechanistic_chain']['panels']]==list('ABC')
    assert len(by_id['input_coverage_depth']['panels'])==4
    assert not any(p['source']=='S8' for a in manifest['assets'] for p in a['panels'])
    assert not any(p['source']=='S12' and p['source_panel']=='F' for a in manifest['assets'] for p in a['panels'])
    displayed=pd.read_csv(J/'source_data/curated_publication/si_mnist_coverage_plotted.csv')
    assert set(displayed.task)=={'mnist'}
    seeds=displayed[displayed.record.eq('paired seed difference')]
    assert seeds.groupby('feedback').size().eq(10).all() and len(seeds)==40

def test_extension_order_agrees_between_manuscript_and_release():
    source=(J/'supplementary/supplementary.tex').read_text()
    fragments=['si_05_conductance_learning_figures','si_11_checkpoint_computation','si_12_optional_extensions','si_06_physical_depth','si_10_selection_statistics']
    places=[source.index(r'\input{curated/'+name+'}') for name in fragments]
    assert places==sorted(places)
    assert SUPPLEMENTARY_FIGURES[21:23]==('supplementary/curated/checkpoint_computation.pdf','supplementary/curated/optional_extensions.pdf')
    # Every manuscript figure is included in exactly the archive's display order.
    drawn=[]
    for p in tex_sources(J/'supplementary/supplementary.tex'):
        drawn += re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{(supplementary/[^}]+)\}',p.read_text())
    assert tuple(drawn)==SUPPLEMENTARY_FIGURES

def test_reference_identifiers_are_available_without_duplicate_arxiv_urls():
    bib=(J/'references.bib').read_text()
    for key in ('safaai2026localcredit','safaai2026gainload'):
        entry=re.search(r'@\w+\{'+key+r',.*?\n\}',bib,re.S)[0]
        assert 'eprint =' in entry and 'url =' not in entry
    entry=re.search(r'@article\{greedy2026celltype,.*?\n\}',bib,re.S)[0]
    assert 'https://doi.org/10.64898/2026.06.16.732595' in entry
