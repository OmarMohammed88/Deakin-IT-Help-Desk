#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --time=0-72:00
#SBATCH --mem=60G
#SBATCH --output=%N-%j.out
#SBATCH --qos=batch-short
#SBATCH --output=logs/%N-%j.out


module load Anaconda3
source activate
# conda activate llava2
conda activate /home/s223795137/.conda/envs/llava



# gradio  gradio_run.py

# gradio gradio_re_order.py


gradio citation_run.py

# gradio Climate_Disclosures.py