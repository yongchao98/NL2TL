#!/bin/bash
#SBATCH --job-name=supcon_fedavg_beta_point1
#SBATCH -n 16 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-24:00 # Runtime in D-HH:MM
#SBATCH --partition=seas_gpu_requeue,seas_gpu,seas_dgx1,gpu_requeue
#SBATCH --constraint="a100|v100|p100|titanx"
#SBATCH --gres=gpu:1
#SBATCH --mem=24G # Memory pool for all cores in MB
#python FedAvg
#python main.py --dataset=cifar10     --model=simple-cnn     --alg=fedavg     --lr=0.01     --mu=5     --epochs=10     --comm_round=30     --n_parties=3     --partition=noniid     --beta=0.5     --logdir='./logs/'     --datadir='./data/'

python run_trained_model.py

#### call your code below
#python main.py --dataset=cifar10     --model=simple-cnn     --alg=moon     --lr=0.01     --mu=5     --epochs=10     --comm_round=1     --n_parties=10     --partition=noniid     --beta=0.5     --logdir='./logs/'     --datadir='./data/'

