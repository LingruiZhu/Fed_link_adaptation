#!/bin/bash
#SBATCH -p wiss
#SBATCH -t 02-05:00:00
#SBATCH --ntasks 1
#SBATCH --nice=10
#SBATCH --gpus=1
#SBATCH --mem=80000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhu@ant.uni-bremen.de
#SBATCH --job-name=olla_LUT
#SBATCH -o slurm_outputs/lstm_model_train.out
#SBATCH -e slurm_errors/lstm_model_train.err
echo “$(date): Start jobs for user $(whoami) on node $(hostname)!”
echo “Run jobID=$SLURM_JOBID on slurmNode=$SLURMD_NODENAME NodeList=[$SLURM_NODELIST]”
srun --ntasks=1 python3 recurrent_model/lstm_model.py
wait
echo “$(date): Finished jobs!”