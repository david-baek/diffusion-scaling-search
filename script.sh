#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 16

export CUDA_VISIBLE_DEVICES=5
export GEMINI_API_KEY=AIzaSyB9KFR3xYXteXK8Nbe51WlC8hYagR4fRNM

python main.py \
  --num_prompts=100