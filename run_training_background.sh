#!/bin/bash
# run_training_background.sh
# 
# Usage: ./run_training_background.sh [model_type]
# Example: ./run_training_background.sh baseline

# Default to 'baseline' if no argument is provided
MODEL=${1:-baseline}

# Define the log file name
LOG_FILE="${MODEL}_training.log"

echo "=========================================================="
echo "Starting background training for model: $MODEL"
echo "Logs will be saved to: $LOG_FILE"
echo "To view live progress, run: tail -f $LOG_FILE"
echo "=========================================================="

# Run with nohup and redirect stdout/stderr to the log file
nohup python train.py --model "$MODEL" > "$LOG_FILE" 2>&1 &

echo "Training process started in the background (PID: $!)."
echo "You can now safely close your terminal."
