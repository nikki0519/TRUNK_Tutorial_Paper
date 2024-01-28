#!/bin/sh -l
# FILENAME:  comparisons_job.sh

#SBATCH -A standby
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --constraint A100

#SBATCH --time=04:00:00
#SBATCH --job-name deit_cifar

#SBATCH --output=/home/ravi30/logs/stdout/deit_cifar_thop.out
#SBATCH --error=/home/ravi30/logs/stderr/deit_cifar_thop.err

module load cuda
module load cudnn
module load anaconda
cd /home/ravi30/TRUNK_Tutorial_Paper/Monolithic\ Architectures/
conda activate mnn
python comparisons.py --dataset cifar10 --model deit