#!/usr/bin/env python
"""
Modular Result Analyzer for Dendritic Modeling Sweeps
======================================================

Analyzes parameter sweep results and generates comprehensive plots.
Automatically detects sweep type and applies appropriate analysis.

USAGE
-----

Basic usage (auto-detect sweep type):
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results

Specify sweep type explicitly:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results --type ei
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results --type local_learning
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results --type branch
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results --type general

Custom output directory:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results --output /custom/analysis/dir

List available sweep types:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --list-types

Verbose logging:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results --verbose

DIRECTORY STRUCTURE
-------------------

Expected input structure:
    sweep_results/
    ├── configs/
    │   ├── config_0.yaml
    │   ├── config_1.yaml
    │   └── ...
    └── results/
        ├── config_0/
        │   ├── performance/final.json
        │   ├── information_analysis/final
        │   ├── weight_analysis/final
        │   └── ...
        └── config_1/
            └── ...

Generated output structure:
    plots/
    ├── {sweep_type}_processed_data.csv
    ├── {sweep_type}_aggregated_data.csv
    ├── performance/
    ├── information/
    ├── weights/
    ├── ablation/
    ├── noise/
    └── {sweep_specific}/

SWEEP TYPES
-----------

ei              - E/I ratio sweeps (groups by ee_value, ie_value, use_shunting)
local_learning  - Local learning parameter sweeps (groups by rule_variant, rho_mode, etc.)
branch          - Architecture sweeps (groups by layer sizes, branch factors, plots vs nparams)
general         - General parameter sweeps (fallback for any other sweep)

Note: Ablation and noise analyses are automatically included when data is available.

EXAMPLES
--------

Analyze E/I sweep:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/ei_sweep_results

Analyze local learning sweep with verbose output:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/local_sweep --type local_learning --verbose

Analyze any sweep with auto-detection:
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/unknown_sweep
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from dendritic_modeling.scripts.sweeps.analyzers import (
    ANALYZER_REGISTRY,
    auto_detect_sweep_type,
    get_analyzer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_sweep(
    results_dir: Path,
    sweep_type: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Analyze a parameter sweep."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if sweep_type is None:
        logger.info("Auto-detecting sweep type...")
        sweep_type = auto_detect_sweep_type(results_dir)
        if sweep_type is None:
            logger.warning("Using 'general' analyzer")
            sweep_type = "general"
        else:
            logger.info(f"Detected: {sweep_type}")

    analyzer = get_analyzer(sweep_type)
    results = analyzer.analyze(results_dir, output_dir)

    print(f"\nAnalysis Complete: {results['sweep_type']}")
    print(f"Configs: {results['n_configs']}, Plots: {len(results['plot_paths'])}")
    print(f"Output: {results['output_dir']}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Result Analyzer for Sweep Analysis")
    parser.add_argument(
        "results_dir", nargs="?", type=Path, help="Sweep results directory"
    )
    parser.add_argument(
        "--type", "-t", dest="sweep_type", choices=list(ANALYZER_REGISTRY.keys())
    )
    parser.add_argument("--output", "-o", dest="output_dir", type=Path)
    parser.add_argument(
        "--list-types", action="store_true", help="List available sweep types"
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_types:
        print("\nAvailable Sweep Types:", ", ".join(ANALYZER_REGISTRY.keys()))
        return 0

    if not args.results_dir:
        print("Error: results_dir required (or use --list-types)")
        return 1

    try:
        analyze_sweep(args.results_dir, args.sweep_type, args.output_dir)
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
