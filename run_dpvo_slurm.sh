#!/bin/bash
#SBATCH --job-name=dpvo_array_job          # Job name
#SBATCH --output=logs/dpvo_array_job_%A_%a.out   # Standard output and error log
#SBATCH --error=logs/dpvo_array_job_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mem=8G                          # Total memory
#SBATCH --gres=gpu:1                       # Number of GPUs per node
#SBATCH --time=2:00:00                    # Time limit hrs:min:sec
#SBATCH --array=0-499                       # Array range (e.g., 100 jobs)


# Define variables
TOTAL_JOBS=500
JOB_INDEX=${SLURM_ARRAY_TASK_ID}
# JOB_INDEX=0 # for testing

# Define paths
PYTHON_SCRIPT=dpvo_slurm.py
VIDEO_DIR=dataset/citywalk_2min/videos
CALIB_FILE=calib/citywalk.txt
OUTPUT_DIR=dataset/citywalk_2min/poses

# Create logs directory if not exists
mkdir -p logs

conda activate dpvo;
python $PYTHON_SCRIPT --total_jobs $TOTAL_JOBS --job_index $JOB_INDEX --videodir $VIDEO_DIR --calib $CALIB_FILE --output_dir $OUTPUT_DIR --stride 6 --save_trajectory
