#! /bin/sh
#SBATCH --job-name=runadv
#SBATCH --partition gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output output-file-%j.out
#SBATCH --error error-file%j.err
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate base
conda activate finalnsn
python runadvdata.py
