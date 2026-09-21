"""The launcher must bind the sibling calibration runner, not historical run.py."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_serialization_fix_v2 import run


def test_explicit_dispatch_binds_correct_frozen_protocol_and_output_directory():
    here=Path(__file__).resolve().parent
    assert Path(run.__file__).resolve()==here/"run.py"
    assert run.OUT==here.parents[1]/"source_data/morphology_calibration"
    assert len(run.CONFIRMATORY_SEEDS)==20
    assert run.frozen_protocol()["selected_alpha_scale"]==.15
