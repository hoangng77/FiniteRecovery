#!/bin/bash
#SBATCH --job-name=sac_cartpole
#SBATCH --partition=hpg-turin       
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                
#SBATCH --mem=16G                    
#SBATCH --time=12:00:00             
#SBATCH --output=sac_cartpole_%j.log

module load python/3.11
cd /home/ngochoangnguyen/ondemand/data/sys/dashboard/batch_connect/sys/vscode

source cartPoleVenv/bin/activate

python3 ppo.py