#!/bin/bash
#SBATCH -J ice # Job name
#SBATCH -p mit_normal_gpu # Partition(s) (separate with
# commas if using multiple)
#SBATCH --ntasks=1 # Number of cores
#SBATCH --gpus=1
#SBATCH -t 0-01:00:00 # Time (D-HH:MM:SS)
#SBATCH --mem=40G # Memory
#SBATCH -o icepy_%j.o # Name of standard output file
#SBATCH -e icepy_%j.e # Name of standard error file
#SBATCH --signal=USR1@4
#SBATCH --signal=USR1@2
#SBATCH --mail-user=orealao@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# FINally get email notifications

# first use version of script with only 2 trials and dataset size 100, ten increase requested time
# pixi add wandb for final
module load miniforge/24.3.0-0

cd  /home/orealao/orcd/pool/nrgw

pixi run -e gpu python src/hypertune1D/scriptt.py --batch_size=50 --steps=100 --temp=199 --num_trials=3 --num_train_samples=100 --num_test_samples=100 --num_val_samples=100 --lattice_size=32