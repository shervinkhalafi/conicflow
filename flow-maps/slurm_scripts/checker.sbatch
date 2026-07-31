#!/bin/bash
#SBATCH --job-name=checker_loss_comp
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner  # Regular kempner partition with A100s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00  # 12 hours should be enough for checker
#SBATCH --gres=gpu:1  # Single A100 GPU
#SBATCH --mem=32GB  # Less memory needed for 2D problem
#SBATCH --output=/n/netscratch/albergo_lab/Everyone/nmboffi/checker/logs/loss_method_%A_%a.out
#SBATCH --error=/n/netscratch/albergo_lab/Everyone/nmboffi/checker/logs/loss_method_%A_%a.err
#SBATCH --array=0-3  # 4 experiments: LSD convex, PSD uniform, PSD midpoint, ESD full

## Create output directories if they don't exist (using netscratch for more space)
mkdir -p /n/netscratch/albergo_lab/Everyone/nmboffi/checker/logs

## Configuration
PY_DIR=/n/home04/nboffi/flow-maps/py
CONFIG_PATH="configs.checker"
OUTPUT_FOLDER="/n/netscratch/albergo_lab/Everyone/nmboffi/checker"

# Map array task ID to experiment
SLURM_ID=${SLURM_ARRAY_TASK_ID}

# Get experiment details for logging
case ${SLURM_ID} in
    0)
        METHOD="LSD"
        PFMM_TYPE="N/A"
        STOPGRAD="convex"
        ;;
    1)
        METHOD="PSD uniform"
        PFMM_TYPE="uniform"
        STOPGRAD="convex"
        ;;
    2)
        METHOD="PSD midpoint"
        PFMM_TYPE="midpoint"
        STOPGRAD="convex"
        ;;
    3)
        METHOD="ESD (Eulerian)"
        PFMM_TYPE="N/A"
        STOPGRAD="full"
        ;;
esac

# Print job info
echo "=========================================="
echo "Checker Loss Method Comparison Training"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "PFMM Type: ${PFMM_TYPE}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: uniform_upper_triangle (half plane)"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Batch Size: 100,000"
echo "Total Steps: 100,000"
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
echo "Time limit: 12 hours"
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Dataset: Checker (2D, 10M samples)"
echo "Network: 4-layer MLP with 512 neurons"
echo "GPU: Single A100"

# Run training with the specific configuration
# Note: dataset_location not needed for checker, but keeping for consistency
python launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --dataset_location "" \
    --output_folder "${OUTPUT_FOLDER}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: fmm2"
echo "=========================================="

exit