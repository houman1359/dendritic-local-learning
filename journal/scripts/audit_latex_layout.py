#!/usr/bin/env python3
"""Fail when compiled manuscript logs contain page- or box-overflow warnings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_LOGS = (Path("main.log"), Path("supplementary/supplementary.log"))
PROHIBITED = (
    re.compile(r"Float too large for page", re.IGNORECASE),
    re.compile(r"Overfull \\vbox", re.IGNORECASE),
    re.compile(r"Overfull \\hbox", re.IGNORECASE),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LaTeX logs for warnings that can clip publication content."
    )
    parser.add_argument("logs", nargs="*", type=Path, default=list(DEFAULT_LOGS))
    args = parser.parse_args()

    failures: list[str] = []
    for log_path in args.logs:
        if not log_path.is_file():
            failures.append(f"{log_path}: missing log; compile the document first")
            continue
        for line_number, line in enumerate(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if any(pattern.search(line) for pattern in PROHIBITED):
                failures.append(f"{log_path}:{line_number}: {line.strip()}")

    if failures:
        print("LaTeX layout audit failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    joined = ", ".join(str(path) for path in args.logs)
    print(f"LaTeX layout audit passed: no float or box overflows in {joined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
