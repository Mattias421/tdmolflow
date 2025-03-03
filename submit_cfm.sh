#!/bin/bash
#SBATCH --job-name=tdflow
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=168:00:00
#SBATCH --array=0-0

module load Anaconda3/2019.07

source activate tdmolflow

export WANDB_ENTITY=mattias421
export WANDB_PROJECT=tdmolflow

sigmas=(0.1 0.2 0.3)

sigma=${sigmas[$SLURM_ARRAY_TASK_ID]}

python train.py --workers 8 --sample 50 --batch 64 --lr 0.00003 --ema 0.5 \
    --observed 0,0,0,1,1,1,1,1,1 --exist 1,1,1,1,1,1,1,1,1 --snap 25 --dump 25 \
    --precond eps --data_class QM9Dataset --qm9dataset_shuffle_node_ordering True \
    --qm9dataset_condition_on_alpha False --qm9dataset_only_second_half False \
    --qm9dataset_atom_type_norm 0.25 --loss_class JumpLossFinalDim \
    --jumplossfinaldim_rate_function_name step --jumplossfinaldim_rate_cut_t 0.1 \
    --jumplossfinaldim_mean_or_sum_over_dim mean --jumplossfinaldim_noise_schedule_name cfm_ode \
    --jumplossfinaldim_vp_sde_beta_min $sigma \
    --jumplossfinaldim_x0_logit_ce_loss_weight 1.0 --jumplossfinaldim_nearest_atom_pred True \
    --sampler_class JumpSampler --jumpsampler_sample_near_atom True --network_class EGNNMultiHeadJump \
    --egnnmultiheadjump_detach_last_layer True --egnnmultiheadjump_rate_use_x0_pred True \
    --egnnmultiheadjump_n_attn_blocks 8 --egnnmultiheadjump_n_heads 8 \
    --egnnmultiheadjump_transformer_dim 128 --grad_conditioner_class MoleculeJump \
    --moleculejump_grad_norm_clip 1.0 --wandb_dir ./results/vanilla \
    --outdir ./results/vanilla/training-runs
