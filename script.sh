#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 16

export GEMINI_API_KEY=YOUR_API_KEY_HERE

python main.py \
  --num_prompts=5