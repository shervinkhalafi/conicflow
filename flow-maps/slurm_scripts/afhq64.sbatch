#!/bin/bash
#SBATCH --job-name=afhq64_bs256
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4  # Request 4 H100s
#SBATCH --time=01-12:00  # 36 hours for AFHQ64
#SBATCH --mem=128GB  # 128GB total for the node
#SBATCH --output=/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/logs/bs256_%A_%a.out
#SBATCH --error=/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/logs/bs256_%A_%a.err
#SBATCH --array=0-3  # 4 experiments: LSD convex, PSD uniform, PSD midpoint, ESD full

## Create output directories if they don't exist
mkdir -p /n/netscratch/albergo_lab/Everyone/nmboffi/afhq64/logs

## Configuration with updated paths
PY_DIR=/n/home04/nboffi/flow-maps/py
CONFIG_PATH="configs.afhq64"
DATASET_LOCATION="/n/holylabs/LABS/albergo_lab/Everyone/nmboffi/datasets"
OUTPUT_FOLDER="/n/netscratch/albergo_lab/Everyone/nmboffi/afhq64"

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
echo "AFHQ64 Loss Method Comparison (Single-Node 4x GPU)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "PFMM Type: ${PFMM_TYPE}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: uniform_upper_triangle (half plane)"
echo "Batch Size: 256 (64 per GPU × 4 GPUs)"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="
echo "Path Configuration:"
echo "  Dataset: ${DATASET_LOCATION}"
echo "  Output: ${OUTPUT_FOLDER}"
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

echo "Starting training from scratch..."
echo "Time limit: 36 hours"
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Dataset: AFHQ64 (64x64)"
echo "Batch size: 256 total (64 per GPU)"
echo "Model channels: 128"
echo "Channel multipliers: [1, 2, 3, 4]"
echo "Attention resolutions: [16, 8]"
echo "Training samples: ~204.8 million"

# Run training with the specific configuration (single node, no srun needed)
python launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --dataset_location "${DATASET_LOCATION}" \
    --output_folder "${OUTPUT_FOLDER}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  Log folder: ${OUTPUT_FOLDER}/logs"
echo "  WandB project: fmm2"
echo "=========================================="

exit
