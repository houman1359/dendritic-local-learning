"""Inline science must not disappear from a legend-length estimate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_nature_communications_format import prose_words


def test_inline_dollar_and_parenthesis_math_count_the_same():
    assert prose_words(r'Gain $h_p \alpha$ changes.') == prose_words(r'Gain \(h_p \alpha\) changes.')
    assert len(prose_words(r'Gain $h_p \alpha$ changes.')) > len(prose_words('Gain changes.'))


def test_displayed_equations_and_labels_do_not_count_as_prose():
    assert prose_words(r'Before. \begin{equation} x+y=z \end{equation} After.') == ['Before', 'After']
    assert prose_words(r'Before. \label{long_internal_label} After.') == ['Before', 'After']


def test_citation_keys_are_not_words_but_rendered_reference_counts():
    words = prose_words(r'Prior work \citep{one,two,three}.')
    assert words == ['Prior', 'work', 'reference']


def test_caption_math_can_trigger_limit_previously_hidden():
    text = ' '.join(['word'] * 340) + r' $a_b+c_d+e_f+g_h+i_j+k_l$'
    assert len(prose_words(text)) > 350
