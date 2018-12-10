#!/bin/bash
#BSUB -J QNET_TR
#BSUB -o quaternet_train_gen_%J.out
#BSUB -q gpuv100
#BSUB -W 06:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"

module load cuda/10.0
module load python3
source ../bin/activate
python3 -u train_subject_net_gen.py
