#!/bin/bash
#SBATCH --job-name=mnist_gpu_capacity
#SBATCH --account=kempner_dev
#SBATCH --partition=serial_requeue
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
set -euo pipefail
scontrol update JobId=44924013 ArrayTaskThrottle=12
