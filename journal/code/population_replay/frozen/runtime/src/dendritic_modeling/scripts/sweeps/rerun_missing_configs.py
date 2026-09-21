#!/usr/bin/env python
"""
rerun_missing_configs.py
------------------------
Script to check which configurations from a unified sweep don't have results
and generate a new SLURM job script to run only the missing configurations.

This script has been adapted for the new unified sweep system structure.

Usage:
    python src/dendritic_modeling/scripts/sweeps/rerun_missing_configs.py --folder /path/to/unified_TIMESTAMP --output-file rerun_missing.sh
    python src/dendritic_modeling/scripts/sweeps/rerun_missing_configs.py --timestamp TIMESTAMP --output-file rerun_missing.sh
    python src/dendritic_modeling/scripts/sweeps/rerun_missing_configs.py --timestamp TIMESTAMP --config configs/sweep_config.yaml
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for missing results in a unified sweep and create a batch job for them"
    )

    # Either provide a full path or a timestamp
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="Full path to the unified_TIMESTAMP folder")
    group.add_argument("--timestamp", help="Timestamp of the unified_TIMESTAMP folder")

    # Add config file parameter
    parser.add_argument(
        "--config",
        default="configs/sweep_config.yaml",
        help="Path to the sweep configuration file to get output_dir",
    )

    # Where to find the base folder if only timestamp provided and no config file is specified
    parser.add_argument(
        "--base-dir",
        default="",
        help="Base directory where unified_TIMESTAMP folders are located (overridden if config file has output_dir)",
    )

    # Output options
    parser.add_argument(
        "--output-file",
        default="rerun_missing.sh",
        help="Path to save the generated SLURM script",
    )

    # Job settings
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=40,
        help="Maximum number of concurrent jobs",
    )

    # Add verbose mode
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about configs and results",
    )

    # Add option to check log content
    parser.add_argument(
        "--check-log-content",
        action="store_true",
        help="Check log content to verify task completion rather than just file existence",
    )

    # Add option to rerun all configurations with any failed attempts
    parser.add_argument(
        "--rerun-partial-failures",
        action="store_true",
        help="Rerun configs that have any failed attempts, even if they also have successful runs",
    )

    # Add option to specify exactly which config IDs to rerun
    parser.add_argument(
        "--force-config-ids",
        type=str,
        help="Comma-separated list of config IDs to forcibly rerun (e.g., '8,15,23')",
    )

    # Add option to identify any task directory that needs to be rerun
    parser.add_argument(
        "--identify-failed-directories",
        action="store_true",
        help="Identify specific job directories with failed tasks",
    )

    return parser.parse_args()


def get_config_ids(configs_dir, verbose=False):
    """Get all config IDs from the configs directory."""
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    config_ids = []

    if verbose:
        print(f"Looking for config files in {configs_dir}")
        print(f"Found {len(config_files)} yaml files")

    for config_file in config_files:
        # Extract the config ID from the filename (e.g., sweep_config_42.yaml -> 42)
        filename = os.path.basename(config_file)

        # Handle different naming patterns
        # Look for patterns like: sweep_config_42.yaml, config_42.yaml, unified_config_42.yaml
        if filename.startswith("sweep_metadata"):
            continue  # Skip metadata files

        parts = filename.replace(".yaml", "").split("_")
        for part in reversed(parts):  # Check from the end for the ID
            try:
                config_id = int(part)
                config_ids.append(config_id)
                if verbose:
                    print(f"Found config file: {filename} with ID {config_id}")
                break
            except ValueError:
                continue

    return sorted(config_ids)


def check_result_for_config(
    results_dir, config_id, verbose=False, check_log_content=False
):
    """Check if results exist for a specific config ID.

    This function looks for "Training experiment completed successfully" in output.log files
    within the config-specific directory structure used by the unified sweep system.

    Returns:
        (is_completed, _, _): Tuple containing completion status and placeholders
        for backward compatibility.
    """
    if verbose:
        print(f"\n===== Checking completion for config ID {config_id} =====")
        print(f"Looking in results directory: {results_dir}")

    # Check for config-specific directory in unified sweep structure
    config_dir = os.path.join(results_dir, f"config_{config_id}")

    if os.path.isdir(config_dir):
        if verbose:
            print(f"Found config directory: {config_dir}")

        # Check output log for completion message
        # output_log = os.path.join(config_dir, "output.log")
        output_log = os.path.join(config_dir, "dendritic_modeling.log")
        if os.path.exists(output_log):
            if verbose:
                print(f"Found output log: {output_log}")

            # Check if log contains success message
            try:
                with open(output_log) as f:
                    content = f.read()
                    # if "Training experiment completed successfully" in content:
                    if "Saved model to" in content:
                        if verbose:
                            print(
                                f"FOUND: 'Training experiment completed successfully' in {output_log}"
                            )
                        return True, False, []
                    elif verbose:
                        print(f"NOT FOUND: Success message not in {output_log}")
            except Exception as e:
                if verbose:
                    print(f"Error reading output log: {e}")
        elif verbose:
            print(f"No output log found at {output_log}")

    # If we reach here, no successful completion was found
    if verbose:
        print(
            f"Config {config_id} incomplete: no successful completion message found\n"
        )
    return False, True, []


def find_missing_configs(
    configs_dir,
    results_dir,
    verbose=False,
    check_log_content=False,
    rerun_partial_failures=False,
    force_config_ids=None,
):
    """Find all configs that don't have successful completion messages in output logs."""
    # Get all config IDs
    config_ids = get_config_ids(configs_dir, verbose)

    if verbose:
        print(f"\nFound {len(config_ids)} total config files in {configs_dir}")

    # If force_config_ids is specified, only consider those IDs
    if force_config_ids:
        forced_ids = [int(id_str.strip()) for id_str in force_config_ids.split(",")]
        config_ids = [cid for cid in config_ids if cid in forced_ids]
        if verbose:
            print(f"Limiting check to forced config IDs: {forced_ids}")

    # Check each config for completion (has successful completion message)
    completed_ids = set()
    missing_ids = []

    if verbose:
        print(
            f"\nChecking {len(config_ids)} configurations for successful completion..."
        )

    for config_id in config_ids:
        # Check if this config has a successful completion message
        is_completed, _, _ = check_result_for_config(
            results_dir, config_id, verbose, check_log_content
        )

        if is_completed:
            completed_ids.add(config_id)
            if verbose:
                print(
                    f"Config {config_id} is COMPLETED (successful completion message found) and will not be rerun"
                )
        else:
            missing_ids.append(config_id)
            if verbose:
                print(
                    f"Config {config_id} is INCOMPLETE (no successful completion message) and will be rerun"
                )

    # Also add forced config IDs that weren't already included
    if force_config_ids:
        for config_id in forced_ids:
            if config_id not in missing_ids and config_id in config_ids:
                missing_ids.append(config_id)
                if verbose:
                    print(
                        f"Config {config_id} is being forcibly added to the rerun list"
                    )

    if verbose:
        print("\nCompletion check summary:")
        print(f"- Total configs checked: {len(config_ids)}")
        print(f"- Completed configs: {len(completed_ids)} - {sorted(completed_ids)}")
        print(
            f"- Missing/incomplete configs: {len(missing_ids)} - {sorted(missing_ids)}"
        )

    return config_ids, completed_ids, sorted(missing_ids), []


def get_base_dir_from_config(config_file, verbose=False):
    """Extract the output_dir from the sweep configuration file."""
    if verbose:
        print(f"Reading base directory from config file: {config_file}")

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)

            # Check for output_dir at top level (new unified structure)
            if "output_dir" in config:
                base_dir = config["output_dir"]
                if base_dir:  # Check if it's not empty
                    if verbose:
                        print(f"Found base directory in config: {base_dir}")
                    return base_dir

            # Fallback to old structure
            if "slurm_config" in config and "output_dir" in config["slurm_config"]:
                base_dir = config["slurm_config"]["output_dir"]
                if base_dir:  # Check if it's not empty
                    if verbose:
                        print(f"Found base directory in slurm_config: {base_dir}")
                    return base_dir
                else:
                    if verbose:
                        print("Base directory in config file is empty")
            else:
                if verbose:
                    if "slurm_config" not in config:
                        print("Config file does not have 'slurm_config' section")
                    elif "output_dir" not in config["slurm_config"]:
                        print(
                            "Config file has 'slurm_config' but no 'output_dir' field"
                        )
    except Exception as e:
        print(f"Error reading config file {config_file}: {e}")

    if verbose:
        print("No base directory found in config file or it's empty")

    return None


def get_original_slurm_config(sweep_dir):
    """Get the original SLURM configuration used for this sweep."""
    # Try to find the sweep_params.yaml file (old structure)
    # sweep_params_path = os.path.join(sweep_dir, "configs", "sweep_params.yaml")
    sweep_params_path = os.path.join(sweep_dir, "original_config.yaml")

    # Also try sweep_metadata.yaml (new structure)
    metadata_path = os.path.join(sweep_dir, "configs", "sweep_metadata.yaml")

    config_file = None
    if os.path.exists(sweep_params_path):
        config_file = sweep_params_path
    elif os.path.exists(metadata_path):
        config_file = metadata_path
    else:
        print(f"Error: Could not find sweep configuration in {sweep_dir}/configs")
        return None

    # Load the sweep config file
    with open(config_file) as f:
        try:
            config = yaml.safe_load(f)
            if "slurm_config" in config:
                return config["slurm_config"]
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    return None


def generate_rerun_script(
    missing_ids,
    sweep_dir,
    output_file,
    max_concurrent,
    failed_task_dirs=None,
    identify_failed_directories=False,
):
    """Generate a SLURM script to rerun missing configurations."""
    # Get the original SLURM settings
    slurm_config = get_original_slurm_config(sweep_dir)
    if not slurm_config:
        print("Error: Could not retrieve original SLURM configuration")
        return False

    # Get script name details
    sweep_timestamp = os.path.basename(sweep_dir)
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Start generating the script
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rerun_{sweep_timestamp}",
        f"#SBATCH --account={slurm_config['account']}",
    ]

    # Output paths - use config-based directories
    if len(missing_ids) > 1:
        lines.extend(
            [
                f"#SBATCH --output={sweep_dir}/results/config_%a/output.log",
                f"#SBATCH --error={sweep_dir}/results/config_%a/error.log",
            ]
        )
    else:
        # For single config, hardcode the config ID
        config_id = missing_ids[0] if missing_ids else 0
        lines.extend(
            [
                f"#SBATCH --output={sweep_dir}/results/config_{config_id}/output.log",
                f"#SBATCH --error={sweep_dir}/results/config_{config_id}/error.log",
            ]
        )

    # Format time to HH:MM:SS if it's not already
    time_value = slurm_config.get("time", "1:00:00")
    if isinstance(time_value, (int, float)) or (
        isinstance(time_value, str) and time_value.isdigit()
    ):
        # Convert seconds to HH:MM:SS
        seconds = int(float(time_value))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
    elif isinstance(time_value, str):
        if ":" not in time_value:
            # Assume it's hours if no colons
            formatted_time = f"{time_value}:00:00"
        else:
            formatted_time = time_value
    else:
        # Default to 3 hours if unknown format
        formatted_time = "3:00:00"

    # Resources
    lines.extend(
        [
            f"#SBATCH --nodes={slurm_config.get('nodes', 1)}",
            f"#SBATCH --ntasks-per-node={slurm_config.get('ntasks_per_node', 1)}",
            f"#SBATCH --gres=gpu:{slurm_config.get('gpus_per_node', 1)}",
            f"#SBATCH --cpus-per-task={slurm_config.get('cpus_per_task', 10)}",
            f"#SBATCH --time={formatted_time}",
        ]
    )

    # Properly handle memory specification - use only one method
    # Check if the memory specification has a unit suffix
    mem_value = slurm_config.get("mem", "64GB")
    if isinstance(mem_value, str):
        # If it has a unit suffix (GB, MB, etc.), use it as is for total memory
        if any(unit in mem_value.upper() for unit in ["GB", "MB", "G", "M", "K", "KB"]):
            lines.append(f"#SBATCH --mem={mem_value}")
        else:
            # If no unit, assume it's in MB and add the unit
            lines.append(f"#SBATCH --mem={mem_value}MB")
    else:
        # If it's a number, assume it's in MB
        lines.append(f"#SBATCH --mem={mem_value}MB")

    lines.append(f"#SBATCH --partition={slurm_config['partition']}")
    if slurm_config.get("qos"):
        lines.append(f"#SBATCH --qos={slurm_config['qos']}")

    # Handle array job configuration
    if len(missing_ids) > 1:
        # Format array indices as comma-separated list: 1,2,3,4,5
        array_indices = ",".join(map(str, missing_ids))
        lines.append(f"#SBATCH --array={array_indices}%{max_concurrent}")
    else:
        # Single task, no need for array
        single_id = missing_ids[0] if missing_ids else 0
        lines.append(f"#SBATCH --array={single_id}")

    # Add metadata
    lines.extend(
        [
            "",
            "# ================================================================ ",
            "# This file has been generated by rerun_missing_configs.py script ",
            "#                            ***                                   ",
            f"# Rerunning missing configs from: {sweep_timestamp}              ",
            f"# Generated on: {current_timestamp}                              ",
            "# ================================================================ ",
        ]
    )

    # Add module loading and conda environment
    lines.extend(
        [
            "module purge",
        ]
    )

    for module in slurm_config.get("modules_to_load", []):
        lines.append(f"module load {module}")

    lines.extend(
        [
            "",
            f"conda activate {slurm_config.get('conda_env_path', '/path/to/conda/env')}",
            "",
        ]
    )

    # Set up directories
    lines.extend(
        [
            "# Path to the config file directory",
            f"CONFIG_DIR={sweep_dir}/configs",
            "",
            "# Path to main output directory",
            f"MAIN_OUTPUT_DIR={sweep_dir}",
            "",
            "# Subdirectories",
            "RESULTS_DIR=${MAIN_OUTPUT_DIR}/results",
            "PLOTS_DIR=${MAIN_OUTPUT_DIR}/plots",
            "",
            "# Config file naming pattern (adapted for unified sweep)",
            f"CONFIG_FILE_PATTERN={slurm_config.get('config_filename_pattern', 'sweep_config')}",
            f"SC_RUN_NAME=rerun_{slurm_config.get('run_name', 'unified_sweep')}",
            "",
            "#Training script: ",
            f"TRAINING_SCRIPT={slurm_config.get('training_script', 'src/dendritic_modeling/scripts/training/train_experiments.py')}",
            "",
            "#Python path: ",
            f"PYTHON={slurm_config.get('conda_env_path', '/path/to/conda/env')}/bin/python",
            "",
        ]
    )

    # Configure task execution - using the config-based directory structure
    lines.extend(
        [
            "# Find the config file for this array task ID",
            'CONFIG_FILE=$(find ${CONFIG_DIR} -name "*_${SLURM_ARRAY_TASK_ID}.yaml" | head -1)',
            'if [ -z "$CONFIG_FILE" ]; then',
            '    echo "Error: Could not find config file for task ID ${SLURM_ARRAY_TASK_ID}"',
            "    exit 1",
            "fi",
            "",
            "CONFIG_IDX=${SLURM_ARRAY_TASK_ID}",
            "",
            "# Path to the config-specific results folder",
            "CONFIG_FOLDER_PATH=${RESULTS_DIR}/config_${CONFIG_IDX}",
            "PLOT_FOLDER_PATH=${PLOTS_DIR}/config_${CONFIG_IDX}",
            "",
            "# Create output directory",
            "mkdir -p ${CONFIG_FOLDER_PATH}",
            "mkdir -p ${PLOT_FOLDER_PATH}",
            "",
            'echo "Starting running a training job at $(date)"',
            "start_time=$(date +%s)",
            "",
            "# Unset conflicting SLURM memory environment variables to avoid conflicts",
            "unset SLURM_MEM_PER_CPU",
            "unset SLURM_MEM_PER_GPU",
            "unset SLURM_MEM_PER_NODE",
            "",
        ]
    )

    # Execute command
    lines.extend(
        [
            "srun \\",
            "  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\",
            "  --kill-on-bad-exit \\",
            "  --export=ALL,SLURM_MEM_PER_CPU=,SLURM_MEM_PER_GPU=,SLURM_MEM_PER_NODE= \\",
        ]
    )

    # Training command
    training_script = slurm_config.get("training_script", "")
    # if training_script.startswith("/"):
    #     lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG_FILE}")
    # else:
    #     lines.append("  ${PYTHON} -u $(pwd)/${TRAINING_SCRIPT} ${CONFIG_FILE}")
    if training_script.startswith("/"):
        lines.append(
            '  ${PYTHON} -u ${TRAINING_SCRIPT} "${CONFIG_FILE}" --output_dir "${CONFIG_FOLDER_PATH}"'
        )
    else:
        lines.append(
            '  ${PYTHON} -u $(pwd)/${TRAINING_SCRIPT} "${CONFIG_FILE}" --output_dir "${CONFIG_FOLDER_PATH}"'
        )

    # Finish
    lines.extend(
        [
            "end_time=$(date +%s)",
            'echo "Done with running a training job at $(date)"',
            'echo "Total duration: $((end_time - start_time)) seconds."',
            'echo "Copying logs to results directory"',
            "cp ${CONFIG_FILE} ${CONFIG_FOLDER_PATH}/config_used.yaml",
        ]
    )

    # Write the script
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    # Make the script executable
    os.chmod(output_file, 0o755)

    return True


def main():
    args = parse_args()

    # Process force-config-ids if provided
    force_config_ids = None
    if args.force_config_ids:
        force_config_ids = args.force_config_ids
        print(f"Will forcibly rerun the following config IDs: {force_config_ids}")

    # Get base directory from config if provided
    base_dir = args.base_dir
    if os.path.exists(args.config):
        config_base_dir = get_base_dir_from_config(args.config, args.verbose)
        if config_base_dir:
            base_dir = config_base_dir

    if not base_dir:
        # Default if not provided in args or config
        base_dir = "./results"

    # Determine the sweep directory
    if args.folder:
        sweep_dir = args.folder
    else:
        # Look for unified sweep directories (adapted from ei_ to unified_)
        sweep_dir = os.path.join(base_dir, f"unified_{args.timestamp}")

        # Fallback: also check for timestamped directories without prefix
        if not os.path.isdir(sweep_dir):
            potential_dirs = [
                os.path.join(base_dir, f"sweep_{args.timestamp}"),
                os.path.join(base_dir, args.timestamp),
            ]
            for potential_dir in potential_dirs:
                if os.path.isdir(potential_dir):
                    sweep_dir = potential_dir
                    break

    print(f"Checking sweep directory: {sweep_dir}")

    # Ensure the directory exists
    if not os.path.isdir(sweep_dir):
        print(f"ERROR: Sweep directory not found: {sweep_dir}")
        sys.exit(1)

    # Check configuration files and results
    configs_dir = os.path.join(sweep_dir, "configs")
    results_dir = os.path.join(sweep_dir, "results")

    if not os.path.isdir(configs_dir) or not os.path.isdir(results_dir):
        print("ERROR: Could not find configs or results directories")
        sys.exit(1)

    # Find missing configurations
    config_ids, completed_ids, missing_ids, _ = find_missing_configs(
        configs_dir,
        results_dir,
        verbose=args.verbose,
        check_log_content=args.check_log_content,
        rerun_partial_failures=args.rerun_partial_failures,
        force_config_ids=force_config_ids,
    )

    print("\nSUMMARY:")
    print(f"Found {len(config_ids)} total configurations")
    print(
        f"Found {len(completed_ids)} completed configurations (with successful completion messages)"
    )
    print(f"Found {len(missing_ids)} configurations to rerun (missing or failed)")

    if not missing_ids:
        print("\nAll configurations have completed successfully. Nothing to do.")
        sys.exit(0)

    # Generate rerun script
    success = generate_rerun_script(
        missing_ids, sweep_dir, args.output_file, args.max_concurrent
    )

    if success:
        print(f"Successfully generated rerun script: {args.output_file}")
        print(f"To run missing configurations, use: sbatch {args.output_file}")
    else:
        print("Failed to generate rerun script")
        sys.exit(1)


if __name__ == "__main__":
    main()
