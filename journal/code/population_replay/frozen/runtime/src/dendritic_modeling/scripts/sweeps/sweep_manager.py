#!/usr/bin/env python
"""
Unified Sweep Manager for Dendritic Modeling Framework
=====================================================

This is the main entry point for all parameter sweeps. It coordinates
the generation of configurations, job submission, and result analysis
for ALL sweep types through YAML configuration alone.

The sweep type (EI, branch, noise, general, mixed) is determined by
the sweep_config and filter_config in the YAML file.

Usage:
    # Generate and run any sweep (default behavior)
    python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/my_sweep.yaml

    # Generate configs only (no job submission)
    python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/my_sweep.yaml --generate-only

    # Analyze existing results
    python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results

    # Generate with job splitting
    python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/my_sweep.yaml --split-jobs 10

    # Dry run (show what would be submitted)
    python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/my_sweep.yaml --dry-run
"""

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

# Add the repo root/src to the path (so `import dendritic_modeling` works when run as a script)
# File is: <repo>/src/dendritic_modeling/scripts/sweeps/sweep_manager.py
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "src"))

# Use absolute imports to avoid relative import issues
try:
    from dendritic_modeling.scripts.sweeps.control_plane import (
        SCHEDULER_PLACEHOLDER,
        validate_scheduler_profile,
        write_frozen_sweep_manifest,
    )
    from dendritic_modeling.scripts.sweeps.unified.base_sweep import SweepJobManager
    from dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep import (
        UnifiedSweepGenerator,
    )
except ImportError:
    # Fallback for when run as script
    sys.path.insert(0, str(Path(__file__).parent))
    from control_plane import (  # type: ignore[no-redef]
        SCHEDULER_PLACEHOLDER,
        validate_scheduler_profile,
        write_frozen_sweep_manifest,
    )
    from unified.base_sweep import SweepJobManager
    from unified.sweep_types.unified_sweep import UnifiedSweepGenerator


class SweepManager:
    """
    Main orchestrator for all sweep operations.

    Uses a unified generator that handles all sweep types (EI, branch, noise, general, mixed)
    through YAML configuration. No need to specify sweep type - it's determined by the config.
    """

    def __init__(self):
        # Use unified generator for all sweep types
        self.generator = UnifiedSweepGenerator()
        # Analysis is handled by separate result_analyzer.py to avoid circular imports
        self.sweep_type = "unified"  # For backward compatibility in output

    @staticmethod
    def _load_composed_config(config_path: str) -> tuple[object, list[Path]]:
        """Load a sweep YAML with an optional relative ``extends`` chain.

        Composition happens before validation and generation. Child mappings
        deep-merge over their parent, while lists replace parent lists. Every
        source path is returned for the frozen provenance manifest.
        """

        def _load(path: Path, stack: tuple[Path, ...]):
            resolved = path.expanduser().resolve()
            if resolved in stack:
                cycle = " -> ".join(str(item) for item in (*stack, resolved))
                raise ValueError(f"Sweep config extends cycle: {cycle}")
            raw = OmegaConf.load(resolved)
            parent_reference = raw.get("extends")
            if "extends" in raw:
                del raw["extends"]
            if parent_reference is None:
                return raw, [resolved]
            if not isinstance(parent_reference, str) or not parent_reference.strip():
                raise TypeError("Sweep config extends must be one non-empty path")
            parent_path = Path(parent_reference)
            if not parent_path.is_absolute():
                parent_path = resolved.parent / parent_path
            parent, sources = _load(parent_path, (*stack, resolved))
            return OmegaConf.merge(parent, raw), [*sources, resolved]

        return _load(Path(config_path), ())

    @staticmethod
    def _expected_config_count(config, observed_count: int) -> int:
        """Resolve and enforce an optional predeclared factorial size."""
        declared = OmegaConf.select(config, "sweep_contract.expected_config_count")
        if declared is None:
            declared = config.get("expected_config_count")
        if declared is None:
            return int(observed_count)
        if isinstance(declared, bool):
            raise ValueError("expected_config_count must be an integer")
        try:
            expected = int(declared)
        except (TypeError, ValueError) as error:
            raise ValueError("expected_config_count must be an integer") from error
        if expected < 0:
            raise ValueError("expected_config_count must be non-negative")
        if expected != int(observed_count):
            raise ValueError(
                "Generated config count does not match sweep_contract: "
                f"expected {expected}, observed {observed_count}"
            )
        return expected

    def generate_sweep(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        split_jobs: Optional[int] = None,
        use_array_jobs: bool = True,
        exclude_nodes: Optional[list[str]] = None,
        slurm_account: Optional[str] = None,
        slurm_partition: Optional[str] = None,
        generate_only: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """
        Generate sweep configurations and job scripts.

        Args:
            config_path: Path to sweep configuration file
            output_dir: Output directory (None to auto-generate)
            split_jobs: Number of jobs to split into (None for single job)

        Returns:
            Dictionary with sweep information
        """
        print(f"Generating sweep from {config_path}")

        # Load configuration
        config, composed_config_sources = self._load_composed_config(config_path)

        # Ensure slurm section exists and normalize account/partition defaults.
        if "slurm_config" not in config or config.slurm_config is None:
            config.slurm_config = OmegaConf.create({})

        allow_scheduler_placeholder = generate_only or dry_run

        if slurm_account is not None:
            config.slurm_config.account = slurm_account
        elif not config.slurm_config.get("account"):
            if allow_scheduler_placeholder:
                config.slurm_config.account = SCHEDULER_PLACEHOLDER
            else:
                raise ValueError(
                    "SLURM account not set. Provide 'account' in slurm_config "
                    "or pass --slurm-account on the command line."
                )

        if slurm_partition is not None:
            config.slurm_config.partition = slurm_partition
        elif not config.slurm_config.get("partition"):
            if allow_scheduler_placeholder:
                config.slurm_config.partition = SCHEDULER_PLACEHOLDER
            else:
                raise ValueError(
                    "SLURM partition not set. Provide 'partition' in slurm_config "
                    "or pass --slurm-partition on the command line."
                )

        scheduler_profile = validate_scheduler_profile(
            config.slurm_config.account,
            config.slurm_config.partition,
            allow_placeholder=allow_scheduler_placeholder,
        )
        # Preserve normalized values in both generated scripts and provenance.
        config.slurm_config.account = scheduler_profile.account
        config.slurm_config.partition = scheduler_profile.partition

        # Validate configuration
        if not self.generator.validate_config(config):
            raise ValueError("Configuration validation failed")

        # Create output directory - check multiple possible locations
        if output_dir:
            base_dir = output_dir
        else:
            # Try different locations for output directory in priority order
            base_dir = (
                config.get("output_dir")  # Top-level output_dir (old format)
                or config.get("outputs", {}).get(
                    "dir"
                )  # Top-level outputs.dir (new format)
                or config.get("base_config", {})
                .get("outputs", {})
                .get("results_dir")  # base_config.outputs.results_dir
                or "output"  # Default fallback
            )

        # Get run_name for directory naming
        run_name = (
            config.get("outputs", {}).get("run_name")
            or config.get("base_config", {}).get("outputs", {}).get("run_name")
            or "unified_sweep"
        )

        sweep_output_dir = self.generator.create_output_directory(
            base_dir, prefix=run_name
        )
        print(f"Created output directory: {sweep_output_dir}")

        # Generate configurations
        print("Generating sweep configurations...")
        sweep_configs = self.generator.generate_configs(config)
        print(f"Generated {len(sweep_configs)} configurations")
        expected_config_count = self._expected_config_count(
            config,
            observed_count=len(sweep_configs),
        )

        # Save configurations
        execution_results_dir = config.get("execution_results_dir")
        if execution_results_dir is not None:
            if not isinstance(execution_results_dir, str) or not execution_results_dir:
                raise TypeError("execution_results_dir must be one non-empty path")
            execution_results_dir = os.path.abspath(
                os.path.expanduser(execution_results_dir)
            )
        config_paths = self.generator.save_configs(
            sweep_configs,
            sweep_output_dir,
            execution_results_dir=execution_results_dir,
        )
        print(f"Saved configurations to: {os.path.join(sweep_output_dir, 'configs')}")
        if execution_results_dir is not None:
            print(f"Execution results root: {execution_results_dir}")

        # Create job scripts
        print("Creating job scripts...")
        slurm_config = config.get("slurm_config", {})
        # Add run_name to slurm_config so it can be used for job naming
        slurm_config["run_name"] = run_name
        if execution_results_dir is not None:
            slurm_config["execution_results_dir"] = execution_results_dir

        # Override exclude_nodes from command line if provided
        if exclude_nodes:
            slurm_config["exclude_nodes"] = exclude_nodes
            print(f"Excluding nodes from command line: {exclude_nodes}")

        job_manager = SweepJobManager(slurm_config)
        job_scripts = job_manager.create_job_scripts(
            config_paths, sweep_output_dir, split_jobs, use_array_jobs
        )
        print(f"Created {len(job_scripts)} job scripts")

        # Save original config for reference
        original_config_path = os.path.join(sweep_output_dir, "original_config.yaml")
        OmegaConf.save(config, original_config_path, resolve=True)

        generation_mode = (
            "generate_only" if generate_only else "dry_run" if dry_run else "submission"
        )
        generator_source = inspect.getsourcefile(type(self.generator))
        manifest_path = write_frozen_sweep_manifest(
            output_dir=sweep_output_dir,
            input_config_path=config_path,
            resolved_config_path=original_config_path,
            generated_config_paths=config_paths,
            expected_config_count=expected_config_count,
            scheduler_profile=scheduler_profile,
            repo_root=repo_root,
            source_files=[
                Path(__file__),
                Path(__file__).with_name("control_plane.py"),
                repo_root
                / "src/dendritic_modeling/scripts/training/train_experiments.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/dendritic/initialize_reactivation.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/dendritic/initialize_methods.py",
                repo_root / "src/dendritic_modeling/config/reactivation.py",
                repo_root / "src/dendritic_modeling/config/model_aliases.py",
                repo_root / "src/dendritic_modeling/networks/architectures/factory.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/ei_network.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/ei_layer.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/dendritic/dendrinet.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/dendritic/branch_layer.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/dendritic/synapse_config.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/synapse/factory.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/synapse/topk.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/synapse/indexed_sparse.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/synapse/spatial_morphology.py",
                repo_root / "src/dendritic_modeling/networks/architectures/"
                "excitation_inhibition/synapse/structured_mask.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_broadcast_mixin.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_broadcast_transport_mixin.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_rule_mixin.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_topk_mixin.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_signals.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_post_factors_mixin.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_gradient_utils.py",
                repo_root / "src/dendritic_modeling/training/strategies/"
                "local_learning_parts/local_learning_epoch.py",
                *([generator_source] if generator_source else []),
                *composed_config_sources,
            ],
            generator_identity=(
                f"{type(self.generator).__module__}.{type(self.generator).__qualname__}"
            ),
            generation_mode=generation_mode,
        )

        sweep_info = {
            "sweep_type": self.sweep_type,
            "output_dir": sweep_output_dir,
            "num_configs": len(sweep_configs),
            "config_paths": config_paths,
            "execution_results_dir": execution_results_dir,
            "job_scripts": job_scripts,
            "original_config": config_path,
            "frozen_manifest": str(manifest_path),
            "scheduler_profile": scheduler_profile.as_dict(),
        }

        print("Sweep generation complete!")
        print(f"   Output directory: {sweep_output_dir}")
        print(f"   Configurations: {len(sweep_configs)}")
        print(f"   Job scripts: {len(job_scripts)}")
        print(f"   Frozen manifest: {manifest_path}")

        return sweep_info

    def submit_jobs(self, sweep_info: dict, dry_run: bool = False) -> bool:
        """
        Submit SLURM jobs for the sweep.

        Args:
            sweep_info: Sweep information from generate_sweep
            dry_run: If True, show what would be submitted without submitting

        Returns:
            True if successful
        """
        job_scripts = sweep_info["job_scripts"]
        sweep_info["output_dir"]
        scheduler_profile = sweep_info.get("scheduler_profile", {})

        if not dry_run and scheduler_profile.get("placeholder", False):
            raise ValueError(
                "Cannot submit a sweep generated with scheduler placeholders; "
                "regenerate it with one approved account/partition pair"
            )

        if dry_run:
            print("DRY RUN - Would submit the following jobs:")
            for script in job_scripts:
                if not script.endswith("submit_all_jobs.sh"):
                    print(f"   sbatch {script}")
            return True

        print(f"Submitting {len(job_scripts)} jobs...")

        # Find the main submission script
        main_script = None
        for script in job_scripts:
            if script.endswith("submit_all_jobs.sh"):
                main_script = script
                break

        if main_script:
            print(f"Running main submission script: {main_script}")
            os.system(f"bash {main_script}")
        else:
            print("No main submission script found, submitting individual jobs...")
            for script in job_scripts:
                if script.endswith(".sh") and not script.endswith("submit_all_jobs.sh"):
                    print(f"Submitting {script}")
                    os.system(f"sbatch {script}")

        print("Jobs submitted successfully!")
        print("Monitor job progress with: squeue -u $USER")

        return True

    def analyze_results(
        self, results_dir: str, output_dir: Optional[str] = None
    ) -> dict:
        """
        Analyze results from a completed sweep.

        Args:
            results_dir: Directory containing sweep results
            output_dir: Directory to save analysis (defaults to results_dir/plots)

        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing sweep results from {results_dir}")

        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # For unified sweeps, we recommend using the separate result_analyzer.py
        # to avoid circular dependencies and provide full analysis capabilities
        print("For comprehensive analysis, use:")
        print(
            f"python -m src.dendritic_modeling.scripts.sweeps.result_analyzer --results-dir {results_dir} --auto-detect"
        )

        plot_output_dir = output_dir or results_dir

        # Basic analysis info
        analysis_info = {
            "sweep_type": self.sweep_type,
            "results_dir": results_dir,
            "output_dir": plot_output_dir,
            "plot_paths": [],
            "report_path": None,
            "results": {},
        }

        print(
            "Basic analysis complete! Use result_analyzer.py for detailed plots and reports."
        )
        return analysis_info

    def run_full_sweep(
        self,
        config_path: str,
        split_jobs: Optional[int] = None,
        dry_run: bool = False,
        exclude_nodes: Optional[list[str]] = None,
        slurm_account: Optional[str] = None,
        slurm_partition: Optional[str] = None,
    ) -> dict:
        """
        Run a complete sweep: generate configs, submit jobs.

        Args:
            config_path: Path to sweep configuration file
            split_jobs: Number of jobs to split into
            dry_run: If True, generate configs and show jobs without submitting

        Returns:
            Dictionary with sweep information
        """
        print("Running full sweep")

        # Generate sweep
        sweep_info = self.generate_sweep(
            config_path,
            split_jobs=split_jobs,
            exclude_nodes=exclude_nodes,
            slurm_account=slurm_account,
            slurm_partition=slurm_partition,
            dry_run=dry_run,
        )

        # Submit jobs
        if not dry_run:
            self.submit_jobs(sweep_info, dry_run=dry_run)
        else:
            print("DRY RUN - Skipping job submission")
            self.submit_jobs(sweep_info, dry_run=True)

        return sweep_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Sweep Manager for Dendritic Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and run any sweep (default behavior)
  python sweep_manager.py --config configs/my_sweep.yaml

  # Generate configs only (no job submission)
  python sweep_manager.py --config configs/my_sweep.yaml --generate-only

  # Analyze existing results
  python sweep_manager.py --analyze --results-dir output/sweep_20240101120000

  # Generate with job splitting
  python sweep_manager.py --config configs/my_sweep.yaml --split-jobs 10

  # Dry run (show what would be submitted)
  python sweep_manager.py --config configs/my_sweep.yaml --dry-run

  # Override SLURM account/partition at submission time
  python sweep_manager.py --config configs/my_sweep.yaml --slurm-account my_account --slurm-partition gpu_partition
""",
    )

    # Action group - mutually exclusive
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--config", help="Path to sweep configuration file (for generation)"
    )
    action_group.add_argument(
        "--analyze", action="store_true", help="Analyze existing results"
    )

    # Options for generation
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate configs only, don't submit jobs",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Generate configs and submit jobs (default behavior - kept for backward compatibility)",
    )
    parser.add_argument(
        "--split-jobs", type=int, help="Number of jobs to split sweep into"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for sweep (auto-generated if not specified)",
    )
    parser.add_argument(
        "--exclude-nodes",
        nargs="+",
        help="List of nodes to exclude from job scheduling (e.g., --exclude-nodes node1 node2)",
    )
    parser.add_argument(
        "--slurm-account",
        help="Override slurm_config.account in sweep config (e.g., my_account)",
    )
    parser.add_argument(
        "--slurm-partition",
        help="Override slurm_config.partition with one approved compute partition",
    )

    # Options for analysis
    parser.add_argument(
        "--results-dir", help="Directory containing sweep results to analyze"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Create sweep manager
        manager = SweepManager()

        if args.analyze:
            # Analysis mode
            if not args.results_dir:
                print("Error: --results-dir required for analysis mode")
                sys.exit(1)

            manager.analyze_results(args.results_dir, args.output_dir)

        else:
            # Generation mode
            if not args.config:
                print("Error: --config required for generation mode")
                sys.exit(1)

            if args.generate_only:
                # Generate only
                sweep_info = manager.generate_sweep(
                    args.config,
                    output_dir=args.output_dir,
                    split_jobs=args.split_jobs,
                    exclude_nodes=args.exclude_nodes,
                    slurm_account=args.slurm_account,
                    slurm_partition=args.slurm_partition,
                    generate_only=True,
                )

                if sweep_info["scheduler_profile"]["placeholder"]:
                    print(
                        "\nGenerated with non-submitting scheduler placeholders. "
                        "Regenerate with an approved account/partition pair before "
                        "submission."
                    )
                else:
                    print("\nTo submit jobs, run:")
                    print(f"   bash {sweep_info['output_dir']}/submit_all_jobs.sh")
                    print("   OR run without --generate-only flag to auto-submit")
            else:
                # Default behavior: generate and submit (--run is now default)
                manager.run_full_sweep(
                    args.config,
                    split_jobs=args.split_jobs,
                    dry_run=args.dry_run,
                    exclude_nodes=args.exclude_nodes,
                    slurm_account=args.slurm_account,
                    slurm_partition=args.slurm_partition,
                )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
