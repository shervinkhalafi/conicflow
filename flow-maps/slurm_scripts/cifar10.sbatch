#!/bin/bash
#SBATCH --job-name=loss_method_comp
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:4  # 4 H100s for training
#SBATCH --mem=64GB
#SBATCH --output=/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/lmm/cifar10/logs/loss_method_%A_%a.out
#SBATCH --error=/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/lmm/cifar10/logs/loss_method_%A_%a.err
#SBATCH --array=0-3  # 4 experiments: LSD convex, PSD uniform, PSD midpoint, ESD full

## Create output directories if they don't exist
mkdir -p /n/holylabs/LABS/albergo_lab/Everyone/nmboffi/lmm/cifar10/logs

## Configuration
PY_DIR=/n/home04/nboffi/flow-maps/py
CONFIG_PATH="configs.cifar10"
DATASET_LOCATION="/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/datasets"
OUTPUT_FOLDER="/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/lmm/cifar10"

# Map array task ID to experiment
SLURM_ID=${SLURM_ARRAY_TASK_ID}

# Get experiment details for logging
case ${SLURM_ID} in
    0)
        METHOD="LSD"
        STOPGRAD="convex"
        SAMPLING="uniform_upper_triangle"
        ;;
    1)
        METHOD="PSD uniform"
        STOPGRAD="convex"
        SAMPLING="uniform_upper_triangle"
        ;;
    2)
        METHOD="PSD midpoint"
        STOPGRAD="convex"
        SAMPLING="uniform_upper_triangle"
        ;;
    3)
        METHOD="ESD (Eulerian)"
        STOPGRAD="full"
        SAMPLING="uniform_upper_triangle"
        ;;
esac

# Print job info
echo "=========================================="
echo "Loss Method Comparison Training"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: ${SAMPLING}"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

# Load modules
module purge
module load gcc/12.2.0-fasrc01

# Hook conda/mamba into the shell and activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax_3_17_25

# Fix library path to prioritize conda environment libraries over system ones
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Change to project directory
cd $PY_DIR || exit

echo "Starting training..."
echo "Time limit: 24 hours"
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"

# Run training with the specific configuration
python launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --dataset_location "${DATASET_LOCATION}" \
    --output_folder "${OUTPUT_FOLDER}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: fmm2"
echo "=========================================="

exit