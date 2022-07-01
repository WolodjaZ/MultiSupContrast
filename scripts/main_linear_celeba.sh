#!/bin/bash
#SBATCH --job-name=mutlisupcontrast_linear
#SBATCH --partition=tesla
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
EXPERIMENT_PATH="/home/wolodja5/results/experiments_linear"
MODEL_PATH="/home/wolodja5/results/experiments/run_0/best_checkpoint.pth.tar"
mkdir -p $EXPERIMENT_PATH
RUN_ID=0

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label singularity run --nv pytorch.sif \
    python -m  torch.distributed.launch --nproc_per_node=1 tmp/MultiSupContrast/main_linear.py \
    --data=$DATASET_PATH \
    --data-name=CELEBA \
    --model-name=tresnet_s \
    --model-path=$MODEL_PATH \
    --num-classes=40 \
    --image-size=64 \
    --learning_rate=1 \
    --lr_decay_epochs="60,75,90" \
    --method=Asymetric
    --freeze=True
    --batch-size=256 \
    --epochs=100 \
    --sync_bn=False \
    --feat-dim=128 \
    --workers=4 \
    --method_used=MultiSupCon \
    --dump_path=$EXPERIMENT_PATH
    --run=$RUN_ID