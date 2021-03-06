#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 10000 # 2GB solicitados.
#SBATCH -D /home/grupo07/week3 # working directory
#SBATCH -p mhigh,mlow # or mlow Partition to submit to
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o out/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e out/%x_%u_%j.err # File to which STDERR will be written

cd week3
python mlp_manu.py
