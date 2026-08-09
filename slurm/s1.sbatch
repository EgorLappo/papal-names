#!/bin/bash
#SBATCH --job-name=popes
#SBATCH -p hns,normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=elappo@stanford.edu
#SBATCH --mail-type=ALL

# to monitor: sacct -X --format=jobid,elapsed,ncpus,ntasks,state --jobs $jobid

module load system
module load devel
module load gcc/14.2.0

pixi run python figure_s1_compute.py


