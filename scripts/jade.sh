#!/bin/bash


#### Set-up ressources #####

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=01:00:00

# set name of job
#SBATCH --job-name=nodepint

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
##SBATCH --mail-type=ALL

# send mail to this address
##SBATCH --mail-user=gb21553@bristol.ac.uk

# Set the folder for output
#SBATCH --output ./graphpint/tests/data/%j.out

## JAX version that works here: 0.4.7 with cudnn11.2

## Some commands to quickly try this on JADE
# sbatch scripts/jade.sh  # Submit the script to JADE
# sacct -u rrn27-wwp02    # Monitor my jobs
# srun --job-name "GraphPinT" --gres=gpu:1 --time 0:30:00 --pty bash -i # Run an interactive job

## Load the tensorflow GPU container
# /jmain02/apps/docker/tensorflow -c

## Activate Conda (Make sure dependencies are in-there)
# source /etc/profile.d/modules.sh
# module load tensorflow/2.9.1
# module load cuda/11.1-gcc-9.1.0
# module load python/anaconda3
# source activate base
# conda activate jaxenv

## Steps to follow
# 1. source /etc/profile.d/modules.sh
# 2. module load python/anaconda3
# 3. source activate base
# 4. conda activate jaxenv
# 5. srun --job-name "NodeBias" --gres=gpu:1 --time 7:30:00 --pty bash -i
# 6. module load cuda/11.1-gcc-9.1.0
# 7. Enjoy! (nohup python main.py > tmp/nohup.log 2>&1 &)

## Run Python script
# python3 ./scripts/jax_test.py
nvidia-smi
python3 graphpint/tests/jax_test.py
