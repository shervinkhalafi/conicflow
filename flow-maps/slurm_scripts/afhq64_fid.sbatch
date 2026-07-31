#!/bin/bash
#SBATCH --job-name=afhq64_diag75_fid
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-00:30
#SBATCH --gres=gpu:1  # Single A100 per job
#SBATCH --mem=64GB
#SBATCH --output=/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/fids/logs/afhq64_fid_%A_%a.out
#SBATCH --error=/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/fids/logs/afhq64_fid_%A_%a.err
#SBATCH --array=0-209  # 3 methods × 14 checkpoints × 5 sampling steps = 210 jobs

## Create output directories if they don't exist
mkdir -p /n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/fids/logs

## Configuration
PY_DIR=/n/home04/nboffi/flow-maps/py
STATS_PATH="/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/datasets/afhq64/afhq64_fid_stats.npz"
OUTPUT_DIR="/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/fids"
CHECKPOINT_BASE="/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64"

# Define the three experiments from the config
# 0: LSD convex
# 1: PSD uniform
# 2: PSD midpoint
EXPERIMENTS=(0 1 2)

# Checkpoints from 6 to 45 in increments of 3
CHECKPOINT_ITERS=(6 9 12 15 18 21 24 27 30 33 36 39 42 45)
SAMPLING_STEPS=(1 2 4 8 16)
EMA_FAC=0.9999

# Calculate dimensions for decomposition
N_EXPERIMENTS=${#EXPERIMENTS[@]}
N_CHECKPOINT=${#CHECKPOINT_ITERS[@]}
N_STEPS=${#SAMPLING_STEPS[@]}

# Decompose array task ID into three indices
EXPERIMENT_IDX=$((SLURM_ARRAY_TASK_ID / (N_CHECKPOINT * N_STEPS)))
REMAINING=$((SLURM_ARRAY_TASK_ID % (N_CHECKPOINT * N_STEPS)))
CHECKPOINT_IDX=$((REMAINING / N_STEPS))
STEP_IDX=$((REMAINING % N_STEPS))

# Get specific parameters
EXPERIMENT="${EXPERIMENTS[$EXPERIMENT_IDX]}"
CHECKPOINT_ITER="${CHECKPOINT_ITERS[$CHECKPOINT_IDX]}"
N_STEPS="${SAMPLING_STEPS[$STEP_IDX]}"

# Map experiment index to checkpoint naming convention and config
case $EXPERIMENT in
0)
    # LSD convex
    CHECKPOINT_NAME="afhq64_lsd_stopgrad_convex_diag75_bs256_9_30"
    CONFIG_PATH="configs.9_30_25_afhq64_loss_comparison"
    SLURM_ID=0
    METHOD_DESC="LSD convex"
    ;;
1)
    # PSD uniform
    CHECKPOINT_NAME="afhq64_pfmm_uniform_stopgrad_convex_diag75_bs256_9_30"
    CONFIG_PATH="configs.9_30_25_afhq64_loss_comparison"
    SLURM_ID=1
    METHOD_DESC="PSD uniform"
    ;;
2)
    # PSD midpoint
    CHECKPOINT_NAME="afhq64_pfmm_shortcut_stopgrad_convex_diag75_bs256_9_30"
    CONFIG_PATH="configs.9_30_25_afhq64_loss_comparison"
    SLURM_ID=2
    METHOD_DESC="PSD midpoint"
    ;;
esac

# Construct checkpoint path
CHECKPOINT_PATH="${CHECKPOINT_BASE}/${CHECKPOINT_NAME}_${CHECKPOINT_ITER}.pkl"

# FID calculation parameters
N_IMAGES=50000
BATCH_SIZE=256

# Print job info
echo "=========================================="
echo "AFHQ64 diag75 FID Calculation"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD_DESC}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint Name: ${CHECKPOINT_NAME}"
echo "Checkpoint Iteration: ${CHECKPOINT_ITER}k steps"
echo "Sampling Steps: ${N_STEPS}"
echo "EMA Factor: ${EMA_FAC}"
echo "Checkpoint Path: ${CHECKPOINT_PATH}"
echo "=========================================="

# Load modules
module purge
module load gcc/12.2.0-fasrc01

# Hook conda/mamba into the shell and activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax_3_17_25

# Fix library path to prioritize conda environment libraries over system ones
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Suppress duplicate CUDA plugin registration warnings
export TF_CPP_VMODULE=cuda_dnn=0,cuda_blas=0

# Enable Python unbuffered output for better debugging
export PYTHONUNBUFFERED=1

# Change to project directory
cd $PY_DIR || exit

echo "Starting FID calculation..."
echo "Dataset: AFHQ64 (64x64)"
echo "Number of samples: ${N_IMAGES}"
echo "Batch size: ${BATCH_SIZE}"

# Run FID calculation
python launchers/sample_and_calc_fid.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --checkpoint "${CHECKPOINT_PATH}" \
    --ema_fac ${EMA_FAC} \
    --n_steps ${N_STEPS} \
    --n_images ${N_IMAGES} \
    --bs ${BATCH_SIZE} \
    --stats "${STATS_PATH}" \
    --out_dir "${OUTPUT_DIR}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  FID values will be saved with checkpoint and step info"
echo "=========================================="

exit
