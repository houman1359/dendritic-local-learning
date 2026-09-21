"""CPU and process-resource helpers for DataLoader tuning."""

from __future__ import annotations

import os

_LOCAL_WORLD_SIZE_ENV_VARS = (
    "LOCAL_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MV2_COMM_WORLD_LOCAL_SIZE",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_STEP_TASKS_PER_NODE",
)


def visible_cpu_count() -> int:
    """Return CPUs visible to the process, respecting scheduler affinity."""
    available_cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except OSError:
            # Affinity queries can fail in restricted runtimes; keep os.cpu_count().
            pass
    return max(1, available_cpus)


def _leading_positive_int(raw_value: str) -> int | None:
    digits = []
    for char in raw_value.strip():
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    value = int("".join(digits))
    return value if value > 0 else None


def _parse_local_process_count(raw_value: str) -> int | None:
    """Parse local process counts from simple integers or SLURM node specs."""
    values = []
    for chunk in raw_value.split(","):
        value = _leading_positive_int(chunk)
        if value is not None:
            values.append(value)
    return max(values) if values else None


def local_process_count() -> int:
    """Return the number of training processes sharing this node, if known."""
    for env_name in _LOCAL_WORLD_SIZE_ENV_VARS:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        value = _parse_local_process_count(raw_value)
        if value is not None:
            return value
    return 1


def auto_worker_cpu_budget(available_cpus: int) -> int:
    """Return the CPU budget for auto worker selection on the current process."""
    local_processes = local_process_count()
    if local_processes <= 1:
        return max(1, available_cpus)
    return max(0, available_cpus // local_processes)


__all__ = [
    "auto_worker_cpu_budget",
    "local_process_count",
    "visible_cpu_count",
]
