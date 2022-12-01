#!/bin/bash
#SBATCH --job-name=supcon_fedavg_beta_point1
#SBATCH -n 16 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-24:00 # Runtime in D-HH:MM
#SBATCH --partition=seas_gpu_requeue,seas_gpu,seas_dgx1,gpu_requeue
#SBATCH --constraint="a100|v100|p100|titanx"
#SBATCH --gres=gpu:1
#SBATCH --mem=24G # Memory pool for all cores in MB

#conda activate ltl_new
python load_model_eng2ltl_t5.py \
 -c config/eng2ltl_t5_load_data.json \
 -n eng2ltl_para_gen_5_11_28_word_infix \
 -o para2.txt \
 -g 0
