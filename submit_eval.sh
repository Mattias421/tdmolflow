#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

module load Anaconda3/2019.07

source activate tdmolflow

export WANDB_ENTITY=mattias421
export WANDB_PROJECT=tdmolflow

model_folder=$1

python generate.py --model_folder $model_folder

python evaluate.py --model_folder $model_folder

echo "jobs done"
