#!/bin/bash
#SBATCH --job-name=celeba_fid
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-00:30
#SBATCH --gres=gpu:1  # Single A100 per job
#SBATCH --mem=64GB
#SBATCH --output=/n/netscratch/albergo_lab/Everyone/nmboffi/celeba/fids/logs/celeba_fid_%A_%a.out
#SBATCH --error=/n/netscratch/albergo_lab/Everyone/nmboffi/celeba/fids/logs/celeba_fid_%A_%a.err
#SBATCH --array=0-314  # 3 methods × 21 checkpoints × 5 sampling steps × 1 EMA = 315 jobs

## Create output directories if they don't exist
mkdir -p /n/netscratch/albergo_lab/Everyone/nmboffi/celeba/fids/logs

## Configuration
PY_DIR=/n/home04/nboffi/flow-maps/py
CONFIG_PATH="configs.9_8_25_celeba_loss_method_resume"
STATS_PATH="/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/datasets/celeb_a/celeba_stats.npz"
OUTPUT_DIR="/n/netscratch/albergo_lab/Everyone/nmboffi/celeba/fids"
CHECKPOINT_BASE="/n/netscratch/albergo_lab/Everyone/nmboffi/celeba"

# Sweep parameters - loss method comparison
# Matches the experiments array in the config:
# 0: LSD convex
# 1: PSD uniform
# 2: PSD midpoint
LOSS_METHODS=(0 1 2)
# Checkpoints from 30 to 130 in increments of 5
CHECKPOINT_ITERS=(30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130)
SAMPLING_STEPS=(1 2 4 8 16)
EMA_FACTORS=(0.9999)

# Calculate dimensions for decomposition
N_METHODS=${#LOSS_METHODS[@]}
N_CHECKPOINT=${#CHECKPOINT_ITERS[@]}
N_STEPS=${#SAMPLING_STEPS[@]}
N_EMA=${#EMA_FACTORS[@]}

# Decompose array task ID into four indices
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / (N_CHECKPOINT * N_STEPS * N_EMA)))
REMAINING=$((SLURM_ARRAY_TASK_ID % (N_CHECKPOINT * N_STEPS * N_EMA)))
CHECKPOINT_IDX=$((REMAINING / (N_STEPS * N_EMA)))
REMAINING2=$((REMAINING % (N_STEPS * N_EMA)))
STEP_IDX=$((REMAINING2 / N_EMA))
EMA_IDX=$((REMAINING2 % N_EMA))

# Get specific parameters
LOSS_METHOD="${LOSS_METHODS[$METHOD_IDX]}"
CHECKPOINT_ITER="${CHECKPOINT_ITERS[$CHECKPOINT_IDX]}"
N_STEPS="${SAMPLING_STEPS[$STEP_IDX]}"
EMA_FAC="${EMA_FACTORS[$EMA_IDX]}"

# Map loss method index to checkpoint naming convention
# Based on config naming: celeba_{method_str}_stopgrad_{stopgrad_type}_diag75_bs256_9_8_resumed
case $LOSS_METHOD in
0)
    CHECKPOINT_NAME="celeba_lsd_stopgrad_convex_diag75_bs256_9_8_resumed"
    METHOD_DESC="LSD convex"
    ;;
1)
    CHECKPOINT_NAME="celeba_pfmm_uniform_stopgrad_convex_diag75_bs256_9_8_resumed"
    METHOD_DESC="PSD uniform"
    ;;
2)
    CHECKPOINT_NAME="celeba_pfmm_shortcut_stopgrad_convex_diag75_bs256_9_8_resumed"
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
echo "CelebA Loss Method Comparison FID Calculation"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Loss Method: ${METHOD_DESC}"
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
echo "Dataset: CelebA (64x64)"
echo "Number of samples: ${N_IMAGES}"
echo "Batch size: ${BATCH_SIZE}"

# Run FID calculation
# The slurm_id maps to the loss method index (0-2)
python launchers/sample_and_calc_fid.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${LOSS_METHOD} \
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