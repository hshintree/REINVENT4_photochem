#!/bin/bash
#SBATCH --job-name=ps_rl3_xtb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=/home/users/YOUR_SUNETID/projects/reinvent_photoswitch/logs/rl3_%j.out
#SBATCH --error=/home/users/YOUR_SUNETID/projects/reinvent_photoswitch/logs/rl3_%j.err

module load micromamba
export PYTHONPATH=/home/users/YOUR_SUNETID/projects/reinvent_photoswitch/plugins:$PYTHONPATH

# xTB uses OpenMP threading — match OMP_NUM_THREADS to cpus-per-task
export OMP_NUM_THREADS=16

cd /home/users/YOUR_SUNETID/projects/reinvent_photoswitch/code/REINVENT4

micromamba run -n reinvent4 reinvent -d cpu \
  -l /home/users/YOUR_SUNETID/projects/reinvent_photoswitch/logs/rl_stage3.log \
  /home/users/YOUR_SUNETID/projects/reinvent_photoswitch/outputs/rl_stage3/stage3.toml
