#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --time=0-120:00
#SBATCH --mem=60G
#SBATCH --output=%N-%j.out
#SBATCH --qos=batch-long
#SBATCH --output=logs/%N-%j.out



module load Anaconda3
source activate
conda activate rag_app




gradio citation_run.py

