#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=64G
# We need ~200GB of storage on the local disk of each node to extract ImageNet into.
# `--tmp` is honoured on the Mila cluster, but it might not be on other clusters.
#SBATCH --tmp=200G

# `cluv submit` runs `sbatch --chdir=<project dir>`, so the job starts in this project's
# folder on the cluster, and the rest of the work is shared with the other clusters:
exec bash scripts/train.sh "$@"
