"""
Base classes for the unified sweep framework.

This module provides abstract base classes that define the interface
for different sweep types (EI, branch, noise, general).
"""

import os
import shlex
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity


def _execution_source_guard(repo_root: Path) -> str:
    """Return a shell preflight that binds a job to its generation checkout."""
    identity = discover_source_identity(repo_root)
    git_identity = identity.get("git", {})
    if not git_identity.get("available"):
        return (
            "# Source guard unavailable: generation directory is not a Git checkout.\n"
        )

    expected_head = str(git_identity["commit"])
    expected_diff = git_identity.get("tracked_diff_sha256")
    if not expected_diff:
        raise RuntimeError("could not fingerprint the tracked generation worktree")

    quoted_root = shlex.quote(str(repo_root))
    return (
        "# Refuse execution after the source checkout changes.\n"
        f"REPOSITORY_ROOT={quoted_root}\n"
        f'EXPECTED_REPOSITORY_HEAD="{expected_head}"\n'
        f'EXPECTED_TRACKED_DIFF_SHA256="{expected_diff}"\n'
        'ACTUAL_REPOSITORY_HEAD="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"\n'
        'ACTUAL_TRACKED_DIFF_SHA256="$(git -C "$REPOSITORY_ROOT" '
        "diff --binary HEAD | sha256sum | awk '{print $1}')\"\n"
        'if [ "$ACTUAL_REPOSITORY_HEAD" != "$EXPECTED_REPOSITORY_HEAD" ]; then\n'
        '  echo "Source commit changed after sweep generation: expected '
        '$EXPECTED_REPOSITORY_HEAD, observed $ACTUAL_REPOSITORY_HEAD" >&2\n'
        "  exit 66\n"
        "fi\n"
        'if [ "$ACTUAL_TRACKED_DIFF_SHA256" != "$EXPECTED_TRACKED_DIFF_SHA256" ]; then\n'
        '  echo "Tracked source changes differ from the frozen generation worktree" >&2\n'
        "  exit 67\n"
        "fi\n"
    )


class BaseSweepGenerator(ABC):
    """
    Abstract base class for sweep configuration generators.

    Each sweep type (EI, branch, noise, general) should inherit from this
    and implement the required methods.
    """

    def __init__(self, sweep_type: str):
        self.sweep_type = sweep_type
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    @abstractmethod
    def generate_configs(self, base_config: DictConfig) -> list[DictConfig]:
        """
        Generate sweep configurations from base configuration.

        Args:
            base_config: Base configuration with sweep parameters

        Returns:
            List of configurations for each sweep condition
        """

    @abstractmethod
    def validate_config(self, config: DictConfig) -> bool:
        """
        Validate that the configuration is valid for this sweep type.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, raises ValueError if invalid
        """

    def create_output_directory(
        self, base_dir: str, prefix: Optional[str] = None
    ) -> str:
        """
        Create timestamped output directory for sweep results.

        Args:
            base_dir: Base directory for outputs
            prefix: Optional prefix for directory name

        Returns:
            Path to created output directory
        """
        if prefix is None:
            prefix = self.sweep_type

        output_dir = os.path.join(base_dir, f"{prefix}_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Create standard subdirectories
        for subdir in ["configs", "results", "plots", "jobs"]:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        return output_dir

    def save_configs(
        self,
        configs: list[DictConfig],
        output_dir: str,
        execution_results_dir: Optional[str] = None,
    ) -> list[str]:
        """
        Save generated configurations to files.

        Args:
            configs: List of configurations to save
            output_dir: Directory to save configurations
            execution_results_dir: Optional root used by training jobs for
                checkpoints and metrics. This permits lightweight sweep
                manifests and job scripts to be staged separately from large
                execution outputs.

        Returns:
            List of paths to saved configuration files
        """
        config_dir = os.path.join(output_dir, "configs")
        results_base_dir = execution_results_dir or os.path.join(output_dir, "results")
        if execution_results_dir is not None:
            staged_results = Path(output_dir) / "results"
            if staged_results.is_symlink():
                if os.readlink(staged_results) != execution_results_dir:
                    raise FileExistsError(
                        f"staged results link has a different target: {staged_results}"
                    )
            else:
                if staged_results.exists():
                    if any(staged_results.iterdir()):
                        raise FileExistsError(
                            "cannot replace non-empty staged results directory: "
                            f"{staged_results}"
                        )
                    staged_results.rmdir()
                staged_results.symlink_to(
                    execution_results_dir, target_is_directory=True
                )
        config_paths = []

        for i, config in enumerate(configs):
            # Set the results_dir for this config to point to its specific subdirectory
            config_results_dir = os.path.join(results_base_dir, f"config_{i}")

            # Ensure outputs section exists and set results_dir
            if "outputs" not in config:
                config["outputs"] = {}
            config["outputs"]["results_dir"] = config_results_dir

            # Save the config
            config_path = os.path.join(config_dir, f"{self.sweep_type}_config_{i}.yaml")
            OmegaConf.save(config, config_path)
            config_paths.append(config_path)

        # Save metadata
        metadata = {
            "sweep_type": self.sweep_type,
            "timestamp": self.timestamp,
            "num_configs": len(configs),
            "config_files": [os.path.basename(p) for p in config_paths],
        }
        metadata_path = os.path.join(config_dir, "sweep_metadata.yaml")
        OmegaConf.save(OmegaConf.create(metadata), metadata_path)

        return config_paths


class BaseSweepAnalyzer(ABC):
    """
    Abstract base class for sweep result analyzers.
    """

    def __init__(self, sweep_type: str):
        self.sweep_type = sweep_type

    @abstractmethod
    def collect_results(self, results_dir: str) -> dict[str, Any]:
        """
        Collect and aggregate results from sweep experiments.

        Args:
            results_dir: Directory containing sweep results

        Returns:
            Dictionary with aggregated results
        """

    @abstractmethod
    def generate_plots(self, results: dict[str, Any], output_dir: str) -> list[str]:
        """
        Generate analysis plots for sweep results.

        Args:
            results: Aggregated results from collect_results
            output_dir: Directory to save plots

        Returns:
            List of paths to generated plot files
        """

    def find_best_configs(
        self, results: dict[str, Any], metric: str = "test_acc"
    ) -> list[dict[str, Any]]:
        """
        Find the best performing configurations.

        Args:
            results: Aggregated results
            metric: Metric to optimize (default: test_acc)

        Returns:
            List of best configurations sorted by performance
        """
        # Default implementation - can be overridden by subclasses
        if "configs" not in results or metric not in results:
            return []

        configs_with_scores = []
        for i, config in enumerate(results["configs"]):
            if i < len(results[metric]):
                score = results[metric][i]
                configs_with_scores.append(
                    {"config": config, "score": score, "config_id": i}
                )

        # Sort by score (descending for accuracy, may need to adjust for loss)
        configs_with_scores.sort(key=lambda x: x["score"], reverse=True)
        return configs_with_scores[:10]  # Return top 10

    def generate_summary_report(self, results: dict[str, Any], output_dir: str) -> str:
        """
        Generate a summary report of sweep results.

        Args:
            results: Aggregated results
            output_dir: Directory to save report

        Returns:
            Path to generated report file
        """
        report_path = os.path.join(
            output_dir, "plots", f"{self.sweep_type}_sweep_summary.txt"
        )

        with open(report_path, "w") as f:
            f.write(f"{self.sweep_type.upper()} Sweep Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sweep type: {self.sweep_type}\n\n")

            if "configs" in results:
                f.write(f"Total configurations: {len(results['configs'])}\n")

            # Find best configs if possible
            try:
                best_configs = self.find_best_configs(results)
                if best_configs:
                    f.write("\nTop 5 configurations:\n")
                    f.write("-" * 30 + "\n")
                    for i, config_info in enumerate(best_configs[:5]):
                        f.write(
                            f"{i + 1}. Score: {config_info['score']:.4f} "
                            f"(Config ID: {config_info['config_id']})\n"
                        )
            except Exception as e:
                f.write(f"\nCould not determine best configurations: {e}\n")

        return report_path


class SweepJobManager:
    """
    Manages SLURM job creation and submission for sweeps.
    """

    def __init__(self, slurm_config: DictConfig):
        self.slurm_config = slurm_config

    def create_job_scripts(
        self,
        config_paths: list[str],
        output_dir: str,
        split_jobs: Optional[int] = None,
        use_array_jobs: bool = True,
    ) -> list[str]:
        """
        Create SLURM job scripts for sweep configurations.

        Args:
            config_paths: List of configuration file paths
            output_dir: Output directory for sweep
            split_jobs: Number of jobs to split into (None for single job or array)
            use_array_jobs: If True, use SLURM array jobs for parallel execution

        Returns:
            List of paths to created job scripts
        """
        jobs_dir = os.path.join(output_dir, "jobs")
        job_scripts = []

        if use_array_jobs and len(config_paths) > 1:
            # Use SLURM array jobs for parallel execution
            script_path = os.path.join(jobs_dir, "run_array_sweep.sh")
            self._create_array_job_script(script_path, config_paths, output_dir)
            job_scripts.append(script_path)
        elif split_jobs is None or split_jobs <= 1:
            # Single job script (sequential execution)
            script_path = os.path.join(jobs_dir, "run_sweep.sh")
            self._create_single_job_script(script_path, config_paths, output_dir)
            job_scripts.append(script_path)
        else:
            # Split into multiple jobs
            configs_per_job = max(1, len(config_paths) // split_jobs)

            for i in range(split_jobs):
                start_idx = i * configs_per_job
                end_idx = min((i + 1) * configs_per_job, len(config_paths))

                if start_idx >= len(config_paths):
                    break

                job_configs = config_paths[start_idx:end_idx]
                script_path = os.path.join(jobs_dir, f"run_sweep_{i}.sh")
                self._create_single_job_script(script_path, job_configs, output_dir)
                job_scripts.append(script_path)

        # Create main submission script
        main_script = self._create_main_submission_script(job_scripts, output_dir)

        return [*job_scripts, main_script]

    def _create_single_job_script(
        self, script_path: str, config_paths: list[str], output_dir: str
    ):
        """Create a single SLURM job script."""
        repo_root = Path(__file__).resolve().parents[5]
        account = self.slurm_config.get("account")
        partition = self.slurm_config.get("partition")
        qos = self.slurm_config.get("qos")
        if not account:
            raise ValueError(
                "SLURM account not set. Provide 'account' in slurm_config."
            )
        if not partition:
            raise ValueError(
                "SLURM partition not set. Provide 'partition' in slurm_config."
            )
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(
                f"#SBATCH --job-name={self.slurm_config.get('run_name') or self.slurm_config.get('job_name', 'sweep')}\n"
            )
            f.write(f"#SBATCH --account={account}\n")
            f.write(f"#SBATCH --partition={partition}\n")
            if qos:
                f.write(f"#SBATCH --qos={qos}\n")
            f.write(f"#SBATCH --chdir={repo_root}\n")
            constraint = self.slurm_config.get("constraint", None)
            if constraint:
                f.write(f"#SBATCH --constraint={constraint}\n")
            f.write(f"#SBATCH --time={self.slurm_config.get('time', '02:00:00')}\n")
            f.write(f"#SBATCH --nodes={self.slurm_config.get('nodes', 1)}\n")
            f.write(
                f"#SBATCH --ntasks-per-node={self.slurm_config.get('ntasks_per_node', 1)}\n"
            )
            f.write(f"#SBATCH --gres=gpu:{self.slurm_config.get('gpus_per_node', 1)}\n")
            f.write(
                f"#SBATCH --cpus-per-task={self.slurm_config.get('cpus_per_task', 4)}\n"
            )
            f.write(f"#SBATCH --mem={self.slurm_config.get('mem', '32G')}\n")

            # Exclude specific nodes if specified
            exclude_nodes = self.slurm_config.get("exclude_nodes", [])
            if exclude_nodes:
                exclude_list = ",".join(exclude_nodes)
                f.write(f"#SBATCH --exclude={exclude_list}\n")

            f.write(f"#SBATCH --output={output_dir}/jobs/%j.out\n")
            f.write(f"#SBATCH --error={output_dir}/jobs/%j.err\n\n")

            f.write("set -euo pipefail\n\n")
            f.write(_execution_source_guard(repo_root))
            f.write("\n")

            # Load modules
            modules = self.slurm_config.get("modules_to_load", [])
            for module in modules:
                f.write(f"module load {module}\n")

            # Set up Python environment
            conda_env_path = self.slurm_config.get("conda_env_path")
            if conda_env_path:
                # Use direct path to Python interpreter in conda environment
                python_path = f"{conda_env_path}/bin/python"
                f.write(f"# Using conda environment at: {conda_env_path}\n")
                f.write(f'PYTHON_BIN="{python_path}"\n')
                f.write("\n")
                # Add LD_LIBRARY_PATH to ensure conda libs are found
                f.write("# Set library path for conda environment\n")
                f.write(
                    f'export LD_LIBRARY_PATH="{conda_env_path}/lib:${{LD_LIBRARY_PATH:-}}"\n'
                )
                f.write("\n")
                f.write("# Verify environment works\n")
                f.write(
                    "$PYTHON_BIN -c \"import torch; print(f'PyTorch version: {torch.__version__}')\"\n"
                )
            else:
                # Fallback to system python
                f.write("# Using system python\n")
                f.write('PYTHON_BIN="python"\n')
            f.write('export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"\n')

            # Mount canary: a wedged NFS mount on the allocated node turns
            # the first path stat inside python into a silent multi-hour
            # hang (observed on holylfs06, 2026-08-15). Probe the mount
            # roots named in the sweep configs and fail loudly instead.
            f.write("\n# Mount canary (fail fast on wedged NFS mounts)\n")
            f.write(
                "for fs_root in $(grep -rhoE '/n/[a-zA-Z0-9_-]+' "
                f'"{output_dir}/configs" | sort -u); do\n'
            )
            f.write('    if ! timeout 60 stat -t "$fs_root" > /dev/null 2>&1; then\n')
            f.write(
                '        echo "MOUNT CANARY FAILED on $(hostname): '
                '$fs_root unreachable" >&2\n'
            )
            f.write("        exit 75\n")
            f.write("    fi\n")
            f.write("done\n")
            f.write('echo "Mount canary passed on $(hostname)"\n')

            f.write("\n# Run experiments\n")

            # Training script
            training_script = self.slurm_config.get(
                "training_script", "src/dendritic_modeling/scripts/train_experiments.py"
            )
            launcher = str(self.slurm_config.get("launcher", "python")).lower()
            if launcher not in {"python", "torchrun"}:
                raise ValueError("slurm_config.launcher must be 'python' or 'torchrun'")
            launch_prefix = "$PYTHON_BIN"
            if launcher == "torchrun":
                launch_prefix += (
                    " -u -m torch.distributed.run --standalone "
                    f"--nproc_per_node={self.slurm_config.get('gpus_per_node', 1)}"
                )

            for config_path in config_paths:
                rel_config_path = os.path.relpath(
                    os.path.abspath(config_path), start=repo_root
                )
                f.write(f"{launch_prefix} {training_script} {rel_config_path}\n")

        # Make executable
        os.chmod(script_path, 0o755)

    def _create_array_job_script(
        self, script_path: str, config_paths: list[str], output_dir: str
    ):
        """Create a SLURM array job script for parallel execution."""
        repo_root = Path(__file__).resolve().parents[5]
        account = self.slurm_config.get("account")
        partition = self.slurm_config.get("partition")
        if not account:
            raise ValueError(
                "SLURM account not set. Provide 'account' in slurm_config."
            )
        if not partition:
            raise ValueError(
                "SLURM partition not set. Provide 'partition' in slurm_config."
            )
        num_configs = len(config_paths)
        max_concurrent = self.slurm_config.get("max_concurrent_jobs", 10)

        # Format time properly
        time_value = self.slurm_config.get("time", "02:00:00")
        if isinstance(time_value, (int, float)) or (
            isinstance(time_value, str) and time_value.isdigit()
        ):
            seconds = int(time_value)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            formatted_time = time_value

        qos = self.slurm_config.get("qos")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(
                f"#SBATCH --job-name={self.slurm_config.get('run_name') or self.slurm_config.get('job_name', 'sweep')}\n"
            )
            f.write(f"#SBATCH --account={account}\n")
            f.write(f"#SBATCH --partition={partition}\n")
            if qos:
                f.write(f"#SBATCH --qos={qos}\n")
            f.write(f"#SBATCH --chdir={repo_root}\n")
            f.write(f"#SBATCH --time={formatted_time}\n")
            f.write(f"#SBATCH --nodes={self.slurm_config.get('nodes', 1)}\n")
            f.write(
                f"#SBATCH --ntasks-per-node={self.slurm_config.get('ntasks_per_node', 1)}\n"
            )
            f.write(f"#SBATCH --gres=gpu:{self.slurm_config.get('gpus_per_node', 1)}\n")
            f.write(
                f"#SBATCH --cpus-per-task={self.slurm_config.get('cpus_per_task', 4)}\n"
            )
            f.write(f"#SBATCH --mem={self.slurm_config.get('mem', '32G')}\n")

            # Exclude specific nodes if specified
            exclude_nodes = self.slurm_config.get("exclude_nodes", [])
            if exclude_nodes:
                exclude_list = ",".join(exclude_nodes)
                f.write(f"#SBATCH --exclude={exclude_list}\n")

            # Array job with optional concurrent limit
            if num_configs > max_concurrent:
                f.write(f"#SBATCH --array=0-{num_configs - 1}%{max_concurrent}\n")
            else:
                f.write(f"#SBATCH --array=0-{num_configs - 1}\n")

            # Use the existing jobs directory directly.  Some Slurm site
            # wrappers create `%A_%a` directories automatically, but vanilla
            # Slurm opens output files before the job script can create them.
            f.write(f"#SBATCH --output={output_dir}/jobs/output_%A_%a.out\n")
            f.write(f"#SBATCH --error={output_dir}/jobs/error_%A_%a.err\n\n")

            # IMPORTANT: SBATCH directives must appear before any shell commands.
            f.write("set -euo pipefail\n\n")
            f.write(_execution_source_guard(repo_root))
            f.write("\n")

            f.write(
                "# ================================================================\n"
            )
            f.write("# This file has been generated by the unified sweep system\n")
            f.write("#                            ***\n")
            f.write("#       Manual changes to this file may be overwritten.\n")
            f.write(
                "# ================================================================\n\n"
            )

            # Load modules
            f.write("module purge\n")
            modules = self.slurm_config.get("modules_to_load", [])
            for module in modules:
                f.write(f"module load {module}\n")
            f.write("\n")

            # Set up Python environment
            conda_env_path = self.slurm_config.get("conda_env_path")
            if conda_env_path:
                python_path = f"{conda_env_path}/bin/python"
                f.write(f"# Using conda environment at: {conda_env_path}\n")
                f.write(f'PYTHON="{python_path}"\n\n')
                # Add LD_LIBRARY_PATH to ensure conda libs are found
                f.write("# Set library path for conda environment\n")
                f.write(
                    f'export LD_LIBRARY_PATH="{conda_env_path}/lib:${{LD_LIBRARY_PATH:-}}"\n\n'
                )
                f.write("# Verify environment works\n")
                f.write(
                    "$PYTHON -c \"import torch; print(f'PyTorch version: {torch.__version__}')\"\n\n"
                )
            else:
                f.write("# Using system python\n")
                f.write('PYTHON="python"\n\n')
            f.write('export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"\n\n')

            # Define config directory and files
            config_dir = (
                os.path.dirname(config_paths[0])
                if config_paths
                else os.path.join(output_dir, "configs")
            )
            f.write("# Path to the config file directory\n")
            f.write(f'CONFIG_DIR="{config_dir}"\n\n')

            # Mount canary: a wedged NFS mount on the allocated node turns
            # the first path stat inside python into a silent multi-hour
            # hang (observed on holylfs06, 2026-08-15). Probe the mount
            # roots named in the sweep configs and fail loudly instead.
            f.write("# Mount canary (fail fast on wedged NFS mounts)\n")
            f.write(
                "for fs_root in $(grep -rhoE '/n/[a-zA-Z0-9_-]+' "
                '"$CONFIG_DIR" | sort -u); do\n'
            )
            f.write('    if ! timeout 60 stat -t "$fs_root" > /dev/null 2>&1; then\n')
            f.write(
                '        echo "MOUNT CANARY FAILED on $(hostname): '
                '$fs_root unreachable" >&2\n'
            )
            f.write("        exit 75\n")
            f.write("    fi\n")
            f.write("done\n")
            f.write('echo "Mount canary passed on $(hostname)"\n\n')

            f.write("# Path to output directory\n")
            f.write(f'OUTPUT_DIR="{output_dir}"\n\n')

            execution_results_dir = self.slurm_config.get("execution_results_dir")
            if execution_results_dir:
                f.write("# External root for checkpoints and metrics\n")
                f.write(f'EXECUTION_RESULTS_DIR="{execution_results_dir}"\n\n')

            # Array job config selection
            f.write("# Array job: select config based on SLURM_ARRAY_TASK_ID\n")
            f.write("CONFIG_FILES=(\n")
            for config_path in config_paths:
                rel_config_path = os.path.relpath(
                    os.path.abspath(config_path), start=repo_root
                )
                f.write(f'    "{rel_config_path}"\n')
            f.write(")\n\n")

            f.write("# Get the config file for this array task\n")
            f.write('CONFIG="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"\n')
            f.write('RUN_NAME="sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"\n\n')

            f.write("# Path to the folder to save outputs for this specific run\n")
            if execution_results_dir:
                f.write(
                    'OUTPUT_FOLDER_PATH="${EXECUTION_RESULTS_DIR}/config_'
                    '${SLURM_ARRAY_TASK_ID}"\n\n'
                )
            else:
                f.write(
                    'OUTPUT_FOLDER_PATH="${OUTPUT_DIR}/results/config_'
                    '${SLURM_ARRAY_TASK_ID}"\n\n'
                )

            f.write("# Create output directory\n")
            f.write("mkdir -p ${OUTPUT_FOLDER_PATH}\n\n")

            f.write('echo "Starting training job at $(date)"\n')
            f.write('echo "Array task $SLURM_ARRAY_TASK_ID processing: $CONFIG"\n')
            f.write("start_time=$(date +%s)\n\n")

            # Training script
            training_script = self.slurm_config.get(
                "training_script",
                "src/dendritic_modeling/scripts/training/train_experiments.py",
            )
            launcher = str(self.slurm_config.get("launcher", "python")).lower()
            if launcher not in {"python", "torchrun"}:
                raise ValueError("slurm_config.launcher must be 'python' or 'torchrun'")
            launch_prefix = "$PYTHON -u"
            if launcher == "torchrun":
                launch_prefix += (
                    " -m torch.distributed.run --standalone "
                    f"--nproc_per_node={self.slurm_config.get('gpus_per_node', 1)}"
                )

            f.write("# Run experiment with srun\n")
            f.write("srun \\\n")
            f.write("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\\n")
            f.write("  --kill-on-bad-exit \\\n")

            if training_script.startswith("/"):
                f.write(
                    f'  {launch_prefix} {training_script} "$CONFIG" --output_dir "$OUTPUT_FOLDER_PATH"\n\n'
                )
            else:
                f.write(
                    f'  {launch_prefix} $(pwd)/{training_script} "$CONFIG" --output_dir "$OUTPUT_FOLDER_PATH"\n\n'
                )

            f.write("end_time=$(date +%s)\n")
            f.write('echo "Done with training job at $(date)"\n')
            f.write('echo "Total duration: $((end_time - start_time)) seconds."\n')

        # Make executable
        os.chmod(script_path, 0o755)

    def _create_main_submission_script(
        self, job_scripts: list[str], output_dir: str
    ) -> str:
        """Create main script to submit all jobs."""
        main_script = os.path.join(output_dir, "submit_all_jobs.sh")
        repo_root = Path(__file__).resolve().parents[5]

        with open(main_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -euo pipefail\n")
            f.write("# Main script to submit all sweep jobs\n\n")
            f.write(f"# Sweep output directory: {output_dir}\n\n")
            f.write(f'REPO_ROOT="{repo_root}"\n')
            f.write('SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')

            f.write('echo "Submitting sweep jobs..."\n')
            f.write('echo "- sweep_dir: ${SWEEP_DIR}"\n')
            f.write('echo "- repo_root: ${REPO_ROOT}"\n')
            for script in job_scripts:
                if script.endswith("submit_all_jobs.sh"):
                    continue  # Skip self
                script_basename = os.path.basename(script)
                # NOTE: This is a Python f-string; we must escape bash `${VAR}` braces as `${{VAR}}`
                # so Python doesn't try to interpolate `{VAR}` as a Python variable.
                f.write(f'echo "Submitting ${{SWEEP_DIR}}/jobs/{script_basename}"\n')
                f.write(
                    f'sbatch --chdir "${{REPO_ROOT}}" "${{SWEEP_DIR}}/jobs/{script_basename}"\n'
                )
                f.write("sleep 2  # Brief delay between submissions\n")

            f.write('\necho "Submitted. Check status with: squeue -u $USER"\n')

        os.chmod(main_script, 0o755)
        return main_script
