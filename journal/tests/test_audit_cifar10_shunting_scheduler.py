import importlib.util
from pathlib import Path

R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scheduler_audit', R/'scripts/audit_cifar10_shunting_scheduler.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def accounting():
    return ('JobID|State|ExitCode|Partition|QOS|NodeList|AllocTRES|Constraints|Restarts\n'
            '47767554_0|COMPLETED|0:0|kempner_requeue|priority|node1|cpu=8,gres/gpu:nvidia_rtx_pro_6000_blackwell_server_edition=1,gres/gpu=1,mem=64G,node=1|rtx6000pro|0\n'
            '47767554_1|COMPLETED|0:0|kempner_requeue|normal|node2|cpu=8,gres/gpu:nvidia_rtx_pro_6000_blackwell_server_edition=1,gres/gpu=1,mem=64G,node=1|rtx6000pro|1\n')


def test_identical_hardware_across_partitions_and_completed_restart_are_valid():
    audit = M.validate(accounting(), '47767554', 2)
    assert audit['valid']
    assert audit['requeued_tasks'] == [1]


def test_mixed_hardware_and_missing_constraint_are_rejected():
    assert not M.validate(accounting().replace('nvidia_rtx_pro_6000_blackwell_server_edition', 'nvidia_h200'), '47767554', 2)['valid']
    assert not M.validate(accounting().replace('|rtx6000pro|', '||'), '47767554', 2)['valid']


def test_incomplete_duplicate_or_failed_jobs_are_rejected():
    raw = accounting()
    assert not M.validate(raw, '47767554', 3)['valid']
    assert not M.validate(raw + raw.splitlines()[-1] + '\n', '47767554', 2)['valid']
    assert not M.validate(raw.replace('|COMPLETED|', '|FAILED|'), '47767554', 2)['valid']
