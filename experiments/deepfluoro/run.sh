#!/bin/bash
#SBATCH --job-name=deepfluoro
#SBATCH --output=logs/deepfluoro_%A_%a.out
#SBATCH --error=logs/deepfluoro_%A_%a.err
#SBATCH --array=0-17
#SBATCH --partition=polina-all
#SBATCH --qos=vision-polina-main
#SBATCH --account=vision-polina
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Define arrays for subject IDs and model names
SUBJECT_IDS=(1 2 3 4 5 6)
MODEL_NAMES=(polypose densexyz densese3)

# Calculate indices from the job array task ID
# Total combinations: 3 models × 6 subjects = 18
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
SUBJECT_INDEX=$((SLURM_ARRAY_TASK_ID % 6))

# Get the actual values
SUBJECT_ID=${SUBJECT_IDS[$SUBJECT_INDEX]}
MODEL_NAME=${MODEL_NAMES[$MODEL_INDEX]}

# Print job information
echo "Running job array task: $SLURM_ARRAY_TASK_ID"
echo "Subject ID: $SUBJECT_ID"
echo "Model Name: $MODEL_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

# Run the command
uv run python run.py --subject_id $SUBJECT_ID --model_name $MODEL_NAME

echo "Finished at: $(date)"