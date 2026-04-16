#!/bin/bash
#SBATCH --job-name=ps_rl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/users/YOUR_SUNETID/projects/reinvent_photoswitch/logs/rl2_%j.out
#SBATCH --error=/home/users/YOUR_SUNETID/projects/reinvent_photoswitch/logs/rl2_%j.err

module load micromamba
export PYTHONPATH=/home/users/YOUR_SUNETID/projects/reinvent_photoswitch/plugins:$PYTHONPATH

cd /home/users/YOUR_SUNETID/projects/reinvent_photoswitch/code/REINVENT4

micromamba run -n reinvent4 reinvent -d cpu \
  -l /home/users/YOUR_SUNETID/projects/reinvent_photoswitch/logs/rl_stage2.log \
  /home/users/YOUR_SUNETID/projects/reinvent_photoswitch/outputs/rl_stage2/stage2.toml
