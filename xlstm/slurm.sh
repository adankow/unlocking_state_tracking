#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx3080 #  partition (queue)
#SBATCH -t 01-00:00:00 # time (D-HH:MM)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  # reserves four GPUs
#SBATCH -D /home/siemsj/projects/xlstm # Change working_dir
#SBATCH -o log_slurm/log_$USER_%Y-%m-%d.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log_slurm/err_$USER_%Y-%m-%d.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J xlstm  # sets the job name. If not specified, the file name will be used as job name
# #SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate conda environment
source ~/.bashrc
conda activate nanogpt

PYTHONPATH=$PWD python experiments/main.py --config $1 --seed $2

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";