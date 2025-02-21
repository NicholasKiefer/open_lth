#!/bin/bash

#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -p normal
#SBATCH --gres=gpu:full:4
#SBATCH --output=/hkfs/work/workspace_haic/scratch/vq6575-gen_param/open_lth/logs/job_log.txt
#SBATCH --error=/hkfs/work/workspace_haic/scratch/vq6575-gen_param/open_lth/logs/job_log_e.txt
#SBATCH -J lottery_bnn
#SBATCH --open-mode=append
cd /hkfs/work/workspace_haic/scratch/vq6575-gen_param/
source venv/bin/activate
cd open_lth/
export CUDA_VISIBLE_DEVICES=0,1,2,3

python open_lth.py lottery --default_hparams cifar_resnet_bnn --levels 3 --platform haicore
