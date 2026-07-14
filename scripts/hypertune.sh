#!/bin/bash
#SBATCH --account=iaifi_lab
#SBATCH -J ice # Job name
#SBATCH -p iaifi_gpu # Partition(s) (separate with
# commas if using multiple)
#SBATCH --ntasks= 32 # Number of cores
#SBATCH -t 0-01:30:00 # Time (D-HH:MM:SS)
#SBATCH --mem=80G # Memory
#SBATCH -o icepy_%j.o # Name of standard output file
#SBATCH -e icepy_%j.e # Name of standard error file
#SBATCH --signal=USR1@4
#SBATCH --signal=USR1@2
#SBATCH --mail-user=orealao@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# FINally get email notifications

# first use version of script with only 2 trials and dataset size 100, ten increase requested time
# pixi add wandb for final
module load miniforge

cd  /n/holystore01/LABS/iaifi_lab/Users/oalao/nrgw

python src/hypertune1D/scripttmini.py