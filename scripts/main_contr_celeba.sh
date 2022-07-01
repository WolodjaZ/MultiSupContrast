#!/bin/bash
#SBATCH --job-name=mutlisupcontrast_contr
#SBATCH --partition=tesla
#SBATCH --qos=tesla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=25:00:00
#SBATCH --mail-user=241368@student.pwr.edu.pl

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH="/home/wolodja5/celeba"
EXPERIMENT_PATH="/home/wolodja5/results/experiments"
mkdir -p $EXPERIMENT_PATH
RUN_ID=0

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label singularity run --nv pytorch.sif \
    python -m  torch.distributed.launch --nproc_per_node=1 tmp/MultiSupContrast/main_contr.py \
    --data=$DATASET_PATH \
    --data-name=CELEBA \
    --model-name=tresnet_s \
    --num-classes=40 \
    --image-size=64 \
    --method=MultiSupCon \
    --temp=0.1 \
    --feat-dim=128 \
    --c_treshold=0.2 \
    --learning_rate=0.25 \
    --lr_decay_epochs="70,80,90" \
    --cosine \
    --batch-size=128 \
    --epochs=100 \
    --sync_bn=False \
    --workers=4 \
    --dump_path=$EXPERIMENT_PATH
    --run=$RUN_ID