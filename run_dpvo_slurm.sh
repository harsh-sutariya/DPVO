#!/bin/bash
#SBATCH --job-name=dpvo_array_job          # Job name
#SBATCH --output=/scratch/hs5580/citywalker/logs/dpvo_array_job_%A_%a.out   # Standard output and error log
#SBATCH --error=/scratch/hs5580/citywalker/logs/dpvo_array_job_%A_%a.err
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --mem=8G                          # Total memory
#SBATCH --gres=gpu:1                       # Number of GPUs per node
#SBATCH --time=2:00:00                    # Time limit hrs:min:sec
#SBATCH --array=187,348                       # Test specific failed jobs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Define variables
TOTAL_JOBS=500
JOB_INDEX=${SLURM_ARRAY_TASK_ID}
# JOB_INDEX=0 # for testing

REPO=/scratch/hs5580/citywalker/CityWalker
OVERLAY=/scratch/hs5580/singularity/citywalker.ext3
TEMP_OVERLAY=/tmp/temp_overlay_${SLURM_JOB_ID}.ext3
CONDA_IMAGE=/scratch/work/public/singularity/anaconda3-2024.06-1.sqf
OS_IMAGE=/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif

# Define paths
PYTHON_SCRIPT=dpvo_slurm.py
VIDEO_DIR=/vast/hs5580/data/citywalker/citywalk_2min
CALIB_FILE=$REPO/thirdparty/DPVO/calib/citywalk.txt
OUTPUT_DIR=/vast/hs5580/data/citywalker/citywalk_2min/poses

singularity overlay create --size 1024 $TEMP_OVERLAY

# Create logs directory if not exists
mkdir -p /scratch/hs5580/citywalker/logs

singularity exec --nv \
  --overlay $OVERLAY:ro \
  --overlay $TEMP_OVERLAY:rw \
  --overlay $CONDA_IMAGE:ro \
  $OS_IMAGE /bin/bash -c "
        source /ext3/env.sh
        conda activate dpvo_legacy
        cd $REPO/thirdparty/DPVO
        python $PYTHON_SCRIPT --total_jobs $TOTAL_JOBS --job_index $JOB_INDEX --videodir $VIDEO_DIR --calib $CALIB_FILE --output_dir $OUTPUT_DIR --stride 6 --save_trajectory
        "

rm -f $TEMP_OVERLAY
